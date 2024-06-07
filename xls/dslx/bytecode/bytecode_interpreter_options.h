// Copyright 2023 The XLS Authors
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

#ifndef XLS_DSLX_BYTECODE_BYTECODE_INTERPRETER_OPTIONS_H_
#define XLS_DSLX_BYTECODE_BYTECODE_INTERPRETER_OPTIONS_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

using PostFnEvalHook = std::function<absl::Status(
    const Function* f, absl::Span<const InterpValue> args, const ParametricEnv*,
    const InterpValue& got)>;

using TraceHook =
    std::function<void(/*entry=*/std::string_view, /*file=*/std::string_view,
                       /*line=*/int)>;

using RolloverHook = std::function<void(const Span&)>;

class BytecodeInterpreterOptions {
 public:
  // Callback to invoke after a DSLX function is evaluated by the interpreter.
  // This is useful for e.g. externally-implementing and hooking-in comparison
  // to the JIT execution mode.
  BytecodeInterpreterOptions& post_fn_eval_hook(PostFnEvalHook hook) {
    post_fn_eval_hook_ = std::move(hook);
    return *this;
  }
  const PostFnEvalHook& post_fn_eval_hook() const { return post_fn_eval_hook_; }

  // Callback to invoke when a trace operation executes. The callback argument
  // is the trace string.
  BytecodeInterpreterOptions& trace_hook(TraceHook hook) {
    trace_hook_ = std::move(hook);
    return *this;
  }
  const TraceHook& trace_hook() const { return trace_hook_; }

  BytecodeInterpreterOptions& rollover_hook(RolloverHook hook) {
    rollover_hook_ = std::move(hook);
    return *this;
  }
  const RolloverHook& rollover_hook() const { return rollover_hook_; }

  // Whether to log values sent and received on channels as trace messages.
  BytecodeInterpreterOptions& trace_channels(bool value) {
    trace_channels_ = value;
    return *this;
  }
  bool trace_channels() const { return trace_channels_; }

  // When executing procs, this is the maximum number of ticks which will
  // execute executed before a DeadlineExceeded error is returned. If nullopt
  // no limit is imposed.
  BytecodeInterpreterOptions& max_ticks(std::optional<int64_t> value) {
    max_ticks_ = value;
    return *this;
  }
  std::optional<int64_t> max_ticks() const { return max_ticks_; }

  void set_validate_final_stack_depth(bool enabled) {
    validate_final_stack_depth_ = enabled;
  }
  bool validate_final_stack_depth() const {
    return validate_final_stack_depth_;
  }

  // The format preference to use when one is not otherwise specified. This is
  // used for `{}` in `trace_fmt`, in `assert_eq` messages, with the
  // `trace_channels` options and elsewhere.
  BytecodeInterpreterOptions& format_preference(FormatPreference value) {
    format_preference_ = value;
    return *this;
  }
  FormatPreference format_preference() const { return format_preference_; }

 private:
  PostFnEvalHook post_fn_eval_hook_ = nullptr;
  TraceHook trace_hook_ = nullptr;
  RolloverHook rollover_hook_ = nullptr;
  bool trace_channels_ = false;
  std::optional<int64_t> max_ticks_;
  bool validate_final_stack_depth_ = true;
  FormatPreference format_preference_ = FormatPreference::kDefault;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_BYTECODE_INTERPRETER_OPTIONS_H_
