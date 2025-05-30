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

#ifndef XLS_DSLX_BYTECODE_BUILTINS_H_
#define XLS_DSLX_BYTECODE_BUILTINS_H_

#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/bytecode/frame.h"
#include "xls/dslx/bytecode/interpreter_stack.h"
#include "xls/dslx/frontend/proc_id.h"

namespace xls::dslx {

absl::Status RunBuiltinArraySlice(const Bytecode& bytecode,
                                  InterpreterStack& stack);

absl::Status RunBuiltinUpdate(const Bytecode& bytecode,
                              InterpreterStack& stack);

absl::Status RunBuiltinBitSlice(const Bytecode& bytecode,
                                InterpreterStack& stack);

absl::Status RunBuiltinBitSliceUpdate(const Bytecode& bytecode,
                                      InterpreterStack& stack);

absl::Status RunBuiltinGate(const Bytecode& bytecode, InterpreterStack& stack);

absl::Status RunBuiltinEncode(const Bytecode& bytecode,
                              InterpreterStack& stack);

absl::Status RunBuiltinOneHot(const Bytecode& bytecode,
                              InterpreterStack& stack);

absl::Status RunBuiltinSMulp(const Bytecode& bytecode, InterpreterStack& stack);

absl::Status RunBuiltinUMulp(const Bytecode& bytecode, InterpreterStack& stack);

absl::Status RunBuiltinSignex(const Bytecode& bytecode,
                              InterpreterStack& stack);

absl::Status RunBuiltinPrioritySel(const Bytecode& bytecode,
                                   InterpreterStack& stack);

absl::Status RunBuiltinOneHotSel(const Bytecode& bytecode,
                                 InterpreterStack& stack);

// Common handler for the range bytecode and builtin range() fn.
absl::Status BuiltinRangeInternal(InterpreterStack& stack);

absl::Status RunBuiltinAndReduce(const Bytecode& bytecode,
                                 InterpreterStack& stack);
// ceil(log2(x))
absl::Status RunBuiltinCeilLog2(const Bytecode& bytecode,
                                InterpreterStack& stack);
absl::Status RunBuiltinClz(const Bytecode& bytecode, InterpreterStack& stack);
absl::Status RunBuiltinCover(const Bytecode& bytecode, InterpreterStack& stack);
absl::Status RunBuiltinCtz(const Bytecode& bytecode, InterpreterStack& stack);
absl::Status RunBuiltinEnumerate(const Bytecode& bytecode,
                                 InterpreterStack& stack);
absl::Status RunBuiltinOrReduce(const Bytecode& bytecode,
                                InterpreterStack& stack);
absl::Status RunBuiltinRange(const Bytecode& bytecode, InterpreterStack& stack);
absl::Status RunBuiltinRev(const Bytecode& bytecode, InterpreterStack& stack);
absl::Status RunBuiltinZip(const Bytecode& bytecode, InterpreterStack& stack);
absl::Status RunBuiltinArrayRev(const Bytecode& bytecode,
                                InterpreterStack& stack);
absl::Status RunBuiltinArraySize(const Bytecode& bytecode,
                                 InterpreterStack& stack);
absl::Status RunBuiltinXorReduce(const Bytecode& bytecode,
                                 InterpreterStack& stack);

absl::Status RunBuiltinAssertEq(const Bytecode& bytecode,
                                InterpreterStack& stack, const Frame& frame,
                                const BytecodeInterpreterOptions& options,
                                const std::optional<ProcId>& caller_proc_id);
absl::Status RunBuiltinAssertLt(const Bytecode& bytecode,
                                InterpreterStack& stack, const Frame& frame,
                                const BytecodeInterpreterOptions& options,
                                const std::optional<ProcId>& caller_proc_id);

// Returns a differences string in the style of Rust's pretty_assertions.
// All lines are indented by two positions.
// Whenever two values differ, the lhs value is prefixed with <, the rhs with >.
std::string HighlightLineByLineDifferences(std::string_view lhs,
                                           std::string_view rhs);

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_BUILTINS_H_
