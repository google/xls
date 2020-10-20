// Copyright 2020 The XLS Authors
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

// Contains implementations of builtin DSLX callables that are usable from the
// DSLX interpreter.

#ifndef XLS_DSLX_BUILTINS_H_
#define XLS_DSLX_BUILTINS_H_

#include "absl/status/statusor.h"
#include "xls/dslx/cpp_ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Implements 'fail!' builtin function.
absl::StatusOr<InterpValue> BuiltinFail(absl::Span<const InterpValue> args,
                                        const Span& span, Invocation* expr,
                                        SymbolicBindings* symbolic_bindings);

// Implements 'assert_eq' builtin function.
absl::StatusOr<InterpValue> BuiltinAssertEq(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings);

// Implements 'assert_lt' builtin function.
absl::StatusOr<InterpValue> BuiltinAssertLt(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings);

// Implements 'and_reduce' builtin function.
absl::StatusOr<InterpValue> BuiltinAndReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings);

// Implements 'or_reduce' builtin function.
absl::StatusOr<InterpValue> BuiltinOrReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings);

// Implements 'xor_reduce' builtin function.
absl::StatusOr<InterpValue> BuiltinXorReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings);

// Implements 'rev' builtin function.
absl::StatusOr<InterpValue> BuiltinRev(absl::Span<const InterpValue> args,
                                       const Span& span, Invocation* expr,
                                       SymbolicBindings* symbolic_bindings);

// Implements 'bit_slice' builtin function.
absl::StatusOr<InterpValue> BuiltinBitSlice(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings);

// Implements 'slice' builtin function.
absl::StatusOr<InterpValue> BuiltinSlice(absl::Span<const InterpValue> args,
                                         const Span& span, Invocation* expr,
                                         SymbolicBindings* symbolic_bindings);

// Implements 'add_with_carry' builtin function.
absl::StatusOr<InterpValue> BuiltinAddWithCarry(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings);

// Implements 'clz' builtin function.
absl::StatusOr<InterpValue> BuiltinClz(absl::Span<const InterpValue> args,
                                       const Span& span, Invocation* expr,
                                       SymbolicBindings* symbolic_bindings);

// Implements 'ctz' builtin function.
absl::StatusOr<InterpValue> BuiltinCtz(absl::Span<const InterpValue> args,
                                       const Span& span, Invocation* expr,
                                       SymbolicBindings* symbolic_bindings);

// Implements 'one_hot' builtin function.
absl::StatusOr<InterpValue> BuiltinOneHot(absl::Span<const InterpValue> args,
                                          const Span& span, Invocation* expr,
                                          SymbolicBindings* symbolic_bindings);

// Implements 'one_hot_sel' builtin function.
absl::StatusOr<InterpValue> BuiltinOneHotSel(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    SymbolicBindings* symbolic_bindings);

// Implements 'signex' builtin function.
absl::StatusOr<InterpValue> BuiltinSignex(absl::Span<const InterpValue> args,
                                          const Span& span, Invocation* expr,
                                          SymbolicBindings* symbolic_bindings);

}  // namespace xls::dslx

#endif  // XLS_DSLX_BUILTINS_H_
