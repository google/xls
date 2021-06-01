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
#include "xls/dslx/abstract_interpreter.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Enumerates signed comparisons available via builtin functionality.
enum SignedCmp {
  kLt,
  kLe,
  kGe,
  kGt,
};

std::string SignedCmpToString(SignedCmp cmp);

// Traces the given expression/result to stderr if it meets the bar for "not too
// noisy an AST node" (and is not itself a trace).
void OptionalTrace(Expr* expr, const InterpValue& result,
                   AbstractInterpreter* interp);

// NOTE on the signature for builtin functions: symbolic_bindings may be null
// e.g. in the entry function.
//
// TODO(leary): 2020-11-24 Would be nice to eliminate this and use const ref
// everywhere.

// Implements 'map' builtin function.
//
// Map is special in that it's one of the only builtins that re-enters back into
// interpretation, most builtins can work just using the values that have been
// provided as parameters (i.e. they are not higher order functions or lazy
// evaluators).
absl::StatusOr<InterpValue> BuiltinMap(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings, AbstractInterpreter* interp);

// Implements 'trace!' builtin function.
absl::StatusOr<InterpValue> BuiltinTrace(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings, AbstractInterpreter* interp);

// Implements 'fail!' builtin function.
absl::StatusOr<InterpValue> BuiltinFail(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'cover!' builtin function.
absl::StatusOr<InterpValue> BuiltinCover(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'update' builtin function.
absl::StatusOr<InterpValue> BuiltinUpdate(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'assert_eq' builtin function.
absl::StatusOr<InterpValue> BuiltinAssertEq(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'assert_lt' builtin function.
absl::StatusOr<InterpValue> BuiltinAssertLt(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'and_reduce' builtin function.
absl::StatusOr<InterpValue> BuiltinAndReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'or_reduce' builtin function.
absl::StatusOr<InterpValue> BuiltinOrReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'xor_reduce' builtin function.
absl::StatusOr<InterpValue> BuiltinXorReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'rev' builtin function.
absl::StatusOr<InterpValue> BuiltinRev(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'enumerate' builtin function.
absl::StatusOr<InterpValue> BuiltinEnumerate(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'range' builtin function.
absl::StatusOr<InterpValue> BuiltinRange(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'bit_slice' builtin function.
absl::StatusOr<InterpValue> BuiltinBitSlice(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'bit_slice_update' builtin function.
absl::StatusOr<InterpValue> BuiltinBitSliceUpdate(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'slice' builtin function.
absl::StatusOr<InterpValue> BuiltinSlice(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'add_with_carry' builtin function.
absl::StatusOr<InterpValue> BuiltinAddWithCarry(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'clz' builtin function.
absl::StatusOr<InterpValue> BuiltinClz(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'ctz' builtin function.
absl::StatusOr<InterpValue> BuiltinCtz(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'one_hot' builtin function.
absl::StatusOr<InterpValue> BuiltinOneHot(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'one_hot_sel' builtin function.
absl::StatusOr<InterpValue> BuiltinOneHotSel(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Implements 'signex' builtin function.
absl::StatusOr<InterpValue> BuiltinSignex(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings);

// Helper that creates a stylized error status that represents a FailureError --
// when it propagates to the pybind11 boundary it should be thrown as an
// exception.
absl::Status FailureErrorStatus(const Span& span, absl::string_view message);

}  // namespace xls::dslx

#endif  // XLS_DSLX_BUILTINS_H_
