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

#ifndef XLS_DSLX_CPP_EVALUATE_H_
#define XLS_DSLX_CPP_EVALUATE_H_

#include "absl/status/statusor.h"
#include "xls/dslx/abstract_interpreter.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/interp_value.h"
#include "xls/ir/bits.h"

namespace xls::dslx {

// Returns whether 'value' conforms to the given concrete type.
absl::StatusOr<bool> ConcreteTypeAcceptsValue(const ConcreteType& type,
                                              const InterpValue& value);

// Returns whether the value is compatible with type (recursively).
//
// This compatibility test is used for e.g. casting validity purposes.
absl::StatusOr<bool> ValueCompatibleWithType(const ConcreteType& type,
                                             const InterpValue& value);

// Converts "value" into a value of "type".
//
// Args:
//  type: Type to convert value into.
//  value: Value to convert into "type".
//  span: The span of the expression performing the conversion (used for error
//    reporting).
//  enum_values: If type is an enum, provides the evaluated values for it.
//  enum_signed: If type is an enum, provides whether it is signed.
absl::StatusOr<InterpValue> ConcreteTypeConvertValue(
    const ConcreteType& type, const InterpValue& value, const Span& span,
    absl::optional<std::vector<InterpValue>> enum_values,
    absl::optional<bool> enum_signed);

// Evaluates the user defined function fn as an invocation against args.
//
// Args:
//   fn: The user-defined function to evaluate.
//   args: The argument with which the user-defined function is being invoked.
//   span: The source span of the invocation.
//   symbolic_bindings: Tuple containing the symbolic bindings to use in
//     the evaluation of this function body (computed by the typechecker)
//
// Returns:
//   The value that results from evaluating the function on the arguments.
//
// Raises:
//   EvaluateError: If the types annotated on either parameters or the return
//     type do not match with the values presented as arguments / the value
//     resulting from the function evaluation.
absl::StatusOr<InterpValue> EvaluateFunction(
    Function* f, absl::Span<const InterpValue> args, const Span& span,
    const SymbolicBindings& symbolic_bindings, AbstractInterpreter* interp);

// Note: all interpreter "node evaluators" have the same signature.

absl::StatusOr<InterpValue> EvaluateConstRef(ConstRef* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context,
                                             AbstractInterpreter* interp);

absl::StatusOr<InterpValue> EvaluateNameRef(NameRef* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context,
                                            AbstractInterpreter* interp);

absl::StatusOr<InterpValue> EvaluateColonRef(ColonRef* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context,
                                             AbstractInterpreter* interp);

absl::StatusOr<InterpValue> EvaluateWhile(While* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp);

absl::StatusOr<InterpValue> EvaluateCarry(Carry* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp);

// Evaluates a Number AST node to a value.
//
// Args:
//  expr: Number AST node.
//  bindings: Name bindings for this evaluation.
//  type_context: Type context for evaluating this number; since numbers
//    literals are agnostic of their bit width this allows us to create the
//    proper-width value.
//
// Returns:
//   The resulting interpreter value.
//
// Raises:
//   EvaluateError: If the type context is missing or inappropriate (e.g. a
//     tuple cannot be the type for a number).
absl::StatusOr<InterpValue> EvaluateNumber(Number* expr,
                                           InterpBindings* bindings,
                                           ConcreteType* type_context,
                                           AbstractInterpreter* interp);

// Evaluates a string node down to its flat representation as an array of u8s.
absl::StatusOr<InterpValue> EvaluateString(String* expr,
                                           InterpBindings* bindings,
                                           ConcreteType* type_context,
                                           AbstractInterpreter* interp);

// Evaluates a struct instance expression; e.g. `Foo { field: stuff }`.
absl::StatusOr<InterpValue> EvaluateStructInstance(StructInstance* expr,
                                                   InterpBindings* bindings,
                                                   ConcreteType* type_context,
                                                   AbstractInterpreter* interp);

// Evaluates a struct instance expression;
// e.g. `Foo { field: stuff, ..other_foo }`.
absl::StatusOr<InterpValue> EvaluateSplatStructInstance(
    SplatStructInstance* expr, InterpBindings* bindings,
    ConcreteType* type_context, AbstractInterpreter* interp);

// Evaluates a tuple expression; e.g. `(x, y)`.
absl::StatusOr<InterpValue> EvaluateXlsTuple(XlsTuple* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context,
                                             AbstractInterpreter* interp);

// Evaluates a let expression; e.g. `let x = y in z`
absl::StatusOr<InterpValue> EvaluateLet(Let* expr, InterpBindings* bindings,
                                        ConcreteType* type_context,
                                        AbstractInterpreter* interp);

// Evaluates a for expression.
absl::StatusOr<InterpValue> EvaluateFor(For* expr, InterpBindings* bindings,
                                        ConcreteType* type_context,
                                        AbstractInterpreter* interp);

// Evaluates a cast expression; e.g. `x as u32`.
absl::StatusOr<InterpValue> EvaluateCast(Cast* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         AbstractInterpreter* interp);

// Evaluates a unary operation expression; e.g. `-x`.
absl::StatusOr<InterpValue> EvaluateUnop(Unop* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         AbstractInterpreter* interp);

// Evaluates an array expression; e.g. `[a, b, c]`.
absl::StatusOr<InterpValue> EvaluateArray(Array* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp);

// Evaluates a binary operation expression; e.g. `x + y`.
absl::StatusOr<InterpValue> EvaluateBinop(Binop* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp);

// Evaluates a ternary expression; e.g. `foo if bar else baz`.
absl::StatusOr<InterpValue> EvaluateTernary(Ternary* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context,
                                            AbstractInterpreter* interp);

// Evaluates an attribute expression; e.g. `x.y`.
absl::StatusOr<InterpValue> EvaluateAttr(Attr* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         AbstractInterpreter* interp);

// Evaluates a match expression; e.g. `match x { ... }`.
absl::StatusOr<InterpValue> EvaluateMatch(Match* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp);

// Evaluates an index expression; e.g. `a[i]`.
absl::StatusOr<InterpValue> EvaluateIndex(Index* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp);

// Get-or-creates the top level bindings for a given module (with respect to the
// interpreter's ImportData as storage).
//
// Note that we may not be able to create a *complete* set of bindings in the
// return value if we've re-entered this routine; e.g. in evaluating a top-level
// constant we recur to ask "what enums (or similar) are available in the module
// scope?" -- in those cases we populate as many top level bindings as we can
// before we reach the work-in-progress point.
//
// Args:
//   module: The top-level module to make bindings for.
//   interp: Provides ability to call back into the interpreter facilities
//    e.g. on import or for evaluating constant value expressions.
//
// Implementation note: the work-in-progress (tracking for re-entrancy as
// described above) is kept track of via the
// AbstractInterpreter::{IsWip,NoteWip} functions.
absl::StatusOr<const InterpBindings*> GetOrCreateTopLevelBindings(
    Module* module, AbstractInterpreter* interp);

using ConcretizeVariant = absl::variant<TypeAnnotation*, EnumDef*, StructDef*>;

// Resolve "type" into a concrete type via expression evaluation.
absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeType(
    ConcretizeVariant type, InterpBindings* bindings,
    AbstractInterpreter* interp);

// As above, but specifically for concretizing TypeAnnotation nodes.
absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeTypeAnnotation(
    TypeAnnotation* type, InterpBindings* bindings,
    AbstractInterpreter* interp);

// Resolves (parametric) dimensions from deduction vs the current bindings.
absl::StatusOr<int64_t> ResolveDim(
    absl::variant<Expr*, int64_t, ConcreteTypeDim> dim,
    InterpBindings* bindings);

// The result of dereferencing a type definition. It can either be an enum, a
// struct, or a TypeAnnotation which can be e.g. a tuple.
using DerefVariant = absl::variant<TypeAnnotation*, EnumDef*, StructDef*>;

// Returns the type_definition dereferenced into a Struct or Enum or
// TypeAnnotation.
//
// Will produce TypeAnnotation in the case we bottom out in a tuple, for
// example.
//
// Args:
//   node: Node to resolve to a struct/enum/annotation.
//   bindings: Current bindings for evaluating the node.
//   parametrics: (Outparam) List of encountered parametric assignment
//       expressions during type traversal. As new parametrics are encountered,
//       they're added to the back of this vector.
absl::StatusOr<DerefVariant> EvaluateToStructOrEnumOrAnnotation(
    TypeDefinition type_definition, InterpBindings* bindings,
    AbstractInterpreter* interp, std::vector<Expr*>* parametrics);

}  // namespace xls::dslx

#endif  // XLS_DSLX_CPP_EVALUATE_H_
