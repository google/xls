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

#ifndef XLS_JIT_BASE_JIT_WRAPPER_H_
#define XLS_JIT_BASE_JIT_WRAPPER_H_

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/proc_runtime.h"
#include "xls/ir/events.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/ir/value_view.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_proc_runtime.h"
#include "xls/jit/jit_runtime.h"
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
      std::string_view ir_text, std::string_view function_name)
    requires(std::is_base_of_v<BaseFunctionJitWrapper, RealType>)
  {
    XLS_ASSIGN_OR_RETURN(auto package,
                         ParsePackage(ir_text, /*filename=*/std::nullopt));
    XLS_ASSIGN_OR_RETURN(auto function, package->GetFunction(function_name));
    XLS_ASSIGN_OR_RETURN(auto jit, FunctionJit::Create(function));
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

class BaseProcJitWrapper {
 public:
  template <typename RealType>
  static absl::StatusOr<std::unique_ptr<RealType>> Create(
      std::string_view ir_text, std::string_view function_name)
    requires(std::is_base_of_v<BaseProcJitWrapper, RealType>)
  {
    XLS_ASSIGN_OR_RETURN(auto package,
                         ParsePackage(ir_text, /*filename=*/std::nullopt));
    XLS_ASSIGN_OR_RETURN(Proc * proc, package->GetProc(function_name));
    XLS_RET_CHECK_EQ(proc, package->GetTop().value_or(nullptr))
        << "Only top proc supported right now.";
    XLS_ASSIGN_OR_RETURN(auto jit, CreateJitSerialProcRuntime(package.get()));

    XLS_ASSIGN_OR_RETURN(auto* man, jit->GetJitChannelQueueManager());
    JitRuntime& runtime = man->runtime();

    return std::unique_ptr<RealType>(
        new RealType(std::move(package), proc, std::move(jit), runtime));
  }

  ProcRuntime* runtime() const { return runtime_.get(); }

  // Reset the state of all of the procs to their initial state.
  void Reset() { runtime_->ResetState(); }

  // Get the current 'state' of the (top) proc.
  absl::flat_hash_map<std::string, Value> state() const {
    std::vector<Value> state = runtime_->ResolveState(proc_);
    absl::flat_hash_map<std::string, Value> res;
    res.reserve(state.size());
    auto it = state.cbegin();
    for (Param* p : proc_->StateParams()) {
      res[p->name()] = *it;
      ++it;
    }
    return res;
  }

  // Execute (up to) a single iteration of every proc in the package. If no
  // conditional send/receive nodes exist in the package then calling Tick will
  // execute exactly one iteration for all procs in the package. If conditional
  // send/receive nodes do exist, then some procs may be blocked in a state
  // where the iteration is partially complete. In this case, the call to Tick()
  // will not execute a complete iteration of the proc. Calling Tick() again
  // will resume these procs from their partially executed state. Returns an
  // error if no progress can be made due to a deadlock.
  absl::Status Tick() { return runtime_->Tick(); }

  // Tick until all procs with IO (send or receive nodes) are blocked on receive
  // operations. `max_ticks` is the maximum number of ticks of the proc network
  // before returning an error. Note: some proc networks are not guaranteed to
  // block even if given no inputs. `max_ticks` is the maximum number of ticks
  // of the proc network before returning an error.
  absl::StatusOr<int64_t> TickUntilBlocked(
      std::optional<int64_t> max_ticks = std::nullopt) {
    return runtime_->TickUntilBlocked(max_ticks);
  }

  // Add 'v' onto the queue of things to be sent to the proc on the given
  // channel.
  absl::Status SendToChannel(std::string_view chan_name, xls::Value v) {
    XLS_ASSIGN_OR_RETURN(auto* man, runtime_->GetJitChannelQueueManager());
    XLS_ASSIGN_OR_RETURN(auto* queue, man->GetQueueByName(chan_name));
    return queue->Write(v);
  }

  // Remove and return the oldest element in the channels queue.
  absl::StatusOr<std::optional<xls::Value>> ReceiveFromChannel(
      std::string_view chan_name) {
    XLS_ASSIGN_OR_RETURN(auto* man, runtime_->GetJitChannelQueueManager());
    XLS_ASSIGN_OR_RETURN(auto* queue, man->GetQueueByName(chan_name));
    return queue->Read();
  }

 protected:
  BaseProcJitWrapper(std::unique_ptr<Package> package, Proc* proc,
                     std::unique_ptr<ProcRuntime> runtime,
                     JitRuntime& jit_runtime)
      : package_(std::move(package)),
        proc_(proc),
        runtime_(std::move(runtime)),
        jit_runtime_(jit_runtime) {}

  template <typename PackedView>
  absl::Status SendToChannelPacked(std::string_view chan_name,
                                   PackedView view) {
    XLS_ASSIGN_OR_RETURN(auto* man, runtime_->GetJitChannelQueueManager());
    XLS_ASSIGN_OR_RETURN(auto* queue, man->GetQueueByName(chan_name));
    return queue->Write(
        jit_runtime_.UnpackBuffer(view.buffer(), queue->channel()->type()));
  }

  template <typename PackedView>
  absl::StatusOr<bool> ReceiveFromChannelPacked(std::string_view chan_name,
                                                PackedView memory) {
    XLS_ASSIGN_OR_RETURN(std::optional<Value> v, ReceiveFromChannel(chan_name));
    if (!v) {
      return false;
    }
    XLS_RETURN_IF_ERROR(jit_runtime_.PackArgs(
        {*v}, {package_->GetTypeForValue(*v)}, {memory.mutable_buffer()}));
    return true;
  }

  std::unique_ptr<Package> package_;
  Proc* proc_;
  std::unique_ptr<ProcRuntime> runtime_;
  JitRuntime& jit_runtime_;
};

}  // namespace xls

#endif  // XLS_JIT_BASE_JIT_WRAPPER_H_
