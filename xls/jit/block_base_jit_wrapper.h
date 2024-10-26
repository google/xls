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

#ifndef XLS_JIT_BLOCK_BASE_JIT_WRAPPER_H_
#define XLS_JIT_BLOCK_BASE_JIT_WRAPPER_H_

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/ir/block.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/block_jit.h"
#include "xls/jit/function_base_jit.h"
#include "xls/public/ir_parser.h"

namespace xls {

class BaseBlockJitWrapper;
// Helper for block-aot wrappers.
//
// TODO(allight): Giving more direct access to underlying bits repr might be
// worthwhile.
class BaseBlockJitWrapperContinuation {
 public:
  BlockJitContinuation* inner_continuation() const { return inner_.get(); }

  absl::Status SetInputPorts(absl::Span<const Value> values) {
    XLS_RET_CHECK(to_set_inputs_.empty())
        << "Cannot use both 'set...' and all-inputs set in a single cycle.";
    return inner_->SetInputPorts(values);
  }
  absl::Status SetInputPorts(std::initializer_list<const Value> values) {
    return SetInputPorts(
        absl::Span<const Value>(values.begin(), values.size()));
  }
  // Overwrite all input-ports with given values.
  absl::Status SetInputPorts(
      const absl::flat_hash_map<std::string, Value>& inputs) {
    XLS_RET_CHECK(to_set_inputs_.empty())
        << "Cannot use both 'set...' and all-inputs set in a single cycle.";
    return inner_->SetInputPorts(inputs);
  }
  // Overwrite all registers with given values.
  absl::Status SetRegisters(absl::Span<const Value> values) {
    XLS_RET_CHECK(to_set_registers_.empty())
        << "Cannot use both 'set...' and all-inputs set in a single cycle.";
    saved_output_registers_.clear();
    return inner_->SetRegisters(values);
  }
  // Overwrite all registers with given values.
  absl::Status SetRegisters(
      const absl::flat_hash_map<std::string, Value>& regs) {
    XLS_RET_CHECK(to_set_registers_.empty())
        << "Cannot use both 'set...' and all-inputs set in a single cycle.";
    saved_output_registers_.clear();
    return inner_->SetRegisters(regs);
  }

  std::vector<Value> GetOutputPorts() const { return inner_->GetOutputPorts(); }
  const absl::flat_hash_map<std::string, Value>& GetOutputPortsMap() const;
  std::vector<Value> GetRegisters() const { return inner_->GetRegisters(); }
  const absl::flat_hash_map<std::string, Value>& GetRegistersMap() const;

 protected:
  Value GetOutputByName(std::string_view name) const;

 private:
  explicit BaseBlockJitWrapperContinuation(
      std::unique_ptr<BlockJitContinuation> cont)
      : inner_(std::move(cont)) {}

  absl::Status PrepareForCycle();
  std::unique_ptr<BlockJitContinuation> inner_;

  absl::flat_hash_map<std::string, Value> to_set_inputs_;
  absl::flat_hash_map<std::string, Value> to_set_registers_;
  mutable absl::flat_hash_map<std::string, Value> saved_outputs_;
  mutable absl::flat_hash_map<std::string, Value> saved_output_registers_;

  friend class BaseBlockJitWrapper;
};

class BaseBlockJitWrapper {
 public:
  BlockJit* jit() { return jit_.get(); }

  absl::Status RunOneCycle(BaseBlockJitWrapperContinuation& cont) {
    XLS_RETURN_IF_ERROR(cont.PrepareForCycle());
    return jit_->RunOneCycle(*cont.inner_continuation());
  }

 protected:
  template <typename RealContinuation>
    requires(
        std::is_base_of_v<BaseBlockJitWrapperContinuation, RealContinuation>)
  std::unique_ptr<RealContinuation> NewContinuationImpl() {
    return std::unique_ptr<RealContinuation>(
        new RealContinuation(jit_->NewContinuation()));
  }

  explicit BaseBlockJitWrapper(std::unique_ptr<Package> pkg,
                               std::unique_ptr<BlockJit> jit)
      : pkg_(std::move(pkg)), jit_(std::move(jit)) {}

  template <typename RealType>
  static absl::StatusOr<std::unique_ptr<RealType>> Create(
      std::string_view inlined_ir_text, std::string_view top_name,
      absl::Span<uint8_t const> aot_proto, JitFunctionType entrypoint,
      const EvaluatorOptions& options)
    requires(std::is_base_of_v<BaseBlockJitWrapper, RealType>)
  {
    if (options.support_observers()) {
      return absl::UnimplementedError("Observers not supported by AOT code.");
    }
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<Package> package,
        ParsePackage(inlined_ir_text, /*filename=*/std::nullopt));
    XLS_ASSIGN_OR_RETURN(Block * block, package->GetBlock(top_name));
    AotPackageEntrypointsProto proto;
    XLS_RET_CHECK(proto.ParseFromArray(aot_proto.data(), aot_proto.size()));
    XLS_RET_CHECK_EQ(proto.entrypoint_size(), 1)
        << "Wrapper should only have a single XLS block compiled.";
    XLS_ASSIGN_OR_RETURN(
        auto jit, BlockJit::CreateFromAot(block, proto.entrypoint(0),
                                          proto.data_layout(), entrypoint));
    return std::unique_ptr<RealType>(
        new RealType(std::move(package), std ::move(jit)));
  }

 private:
  std::unique_ptr<Package> pkg_;
  std::unique_ptr<BlockJit> jit_;
};

}  // namespace xls

#endif  // XLS_JIT_BLOCK_BASE_JIT_WRAPPER_H_
