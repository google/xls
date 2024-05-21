// Copyright 2024 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLS_JIT_FUNCTION_BASE_JIT_WRAPPER_H_
#define XLS_JIT_FUNCTION_BASE_JIT_WRAPPER_H_

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/events.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_view.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/function_jit.h"
#include "xls/public/ir_parser.h"

namespace xls {

// This class provides the underlying implementation for shared aspects of
// jit-wrappers. Specifically it implements the constructor and the actual calls
// to the underlying jit. This allows the wrapper to basically just implement
// (1) some type conversions for primitive calls and (2) the right number of
// arguments.
class BaseFunctionJitWrapper {
 public:
  FunctionJit* jit() { return jit_.get(); }

 protected:
  BaseFunctionJitWrapper(std::unique_ptr<Package> package,
                         std::unique_ptr<FunctionJit> jit,
                         bool needs_fake_token)
      : package_(std::move(package)),
        jit_(std::move(jit)),
        needs_fake_token_(needs_fake_token) {}

  template <typename RealType>
  static absl::StatusOr<std::unique_ptr<RealType>> Create(
      std::string_view ir_text, std::string_view function_name,
      absl::Span<uint8_t const> aot_entrypoint_proto_bin,
      JitFunctionType unpacked_entrypoint, JitFunctionType packed_entrypoint)
    requires(std::is_base_of_v<BaseFunctionJitWrapper, RealType>)
  {
    XLS_ASSIGN_OR_RETURN(auto package,
                         ParsePackage(ir_text, /*filename=*/std::nullopt));
    XLS_ASSIGN_OR_RETURN(auto function, package->GetFunction(function_name));
    AotEntrypointProto proto;
    // NB We could fallback to real jit here maybe?
    XLS_RET_CHECK(proto.ParseFromArray(aot_entrypoint_proto_bin.data(),
                                       aot_entrypoint_proto_bin.size()))
        << "Unable to parse aot information.";
    XLS_ASSIGN_OR_RETURN(
        auto jit, FunctionJit::CreateFromAot(
                      function, proto, unpacked_entrypoint, packed_entrypoint));
    return std::unique_ptr<RealType>(
        new RealType(std::move(package), std::move(jit),
                     MatchesImplicitToken(function->GetType()->parameters())));
  }

  // Matches the parameter signature for an "implicit token/activation taking"
  // function.
  static bool MatchesImplicitToken(absl::Span<Type* const> params) {
    if (params.size() < 2) {
      return false;
    }
    return params[0]->IsToken() && params[1]->IsBits() &&
           params[1]->GetFlatBitCount() == 1;
  }

  // Run the jitted function using values.
  absl::StatusOr<Value> RunInternal(absl::Span<Value const> args) {
    if (needs_fake_token_) {
      std::vector<Value> ext_args;
      ext_args.reserve(args.size() + 2);
      ext_args.push_back(Value::Token());
      ext_args.push_back(Value::Bool(true));
      absl::c_copy(args, std::back_inserter(ext_args));
      XLS_ASSIGN_OR_RETURN(auto retval,
                           DropInterpreterEvents(jit_->Run(ext_args)));
      return retval.element(1);
    }
    return DropInterpreterEvents(jit_->Run(args));
  }

  // Run the jitted function using packed views
  template <typename... Args>
  absl::Status RunInternalPacked(Args... args) {
    if (needs_fake_token_) {
      uint8_t token_value = 0;
      uint8_t activated_value = 1;
      return jit_->RunWithPackedViews(
          xls::PackedBitsView<0>(&token_value, 0),
          xls::PackedBitsView<1>(&activated_value, 0), args...);
    }
    return jit_->RunWithPackedViews(args...);
  }

  // Run the jitted function using unpacked views
  template <typename... Args>
  absl::Status RunInternalUnpacked(Args... args) {
    if (needs_fake_token_) {
      uint8_t token_value = 0;
      uint8_t activated_value = 1;
      return jit_->RunWithUnpackedViews(xls::BitsView<0>(&token_value),
                                        xls::BitsView<1>(&activated_value),
                                        args...);
    }
    return jit_->RunWithUnpackedViews(args...);
  }

  std::unique_ptr<Package> package_;
  std::unique_ptr<FunctionJit> jit_;
  const bool needs_fake_token_;
};

}  // namespace xls

#endif  // XLS_JIT_FUNCTION_BASE_JIT_WRAPPER_H_
