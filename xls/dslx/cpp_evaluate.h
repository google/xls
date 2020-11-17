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
#include "xls/dslx/cpp_ast.h"
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

// Note: all interpreter "node evaluators" have the same signature.

absl::StatusOr<InterpValue> EvaluateConstRef(ConstRef* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context);

absl::StatusOr<InterpValue> EvaluateNameRef(NameRef* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context);
// Callback used to determine if a constant definition (at the module scope) is
// in the process of being evaluated -- this lets us detect re-entry (i.e. a top
// level constant that wants our top-level bindings to do the evaluation needs
// to make forward progress using definitions previous to it in the file).
using IsWipFn = std::function<bool(ConstantDef*)>;

// Callback used to note that a constant evaluation is "work in progress"
// (underway) -- this is noted by passing nullopt before a call to evaluate it.
// Once the constant is fully evaluated, the callback will be invoked with the
// given value. If this callback returns a non-nullopt value, the constant had
// already been evaluated (and was cached).
using NoteWipFn = std::function<absl::optional<InterpValue>(
    ConstantDef*, absl::optional<InterpValue>)>;

// Callback used to evaluate an expression (e.g. a constant at the top level).
using EvaluateFn = std::function<absl::StatusOr<InterpValue>(
    Expr*, InterpBindings*, std::unique_ptr<ConcreteType>)>;

// Callback used to retrieve type information (from the interpreter state).
using GetTypeFn = std::function<std::shared_ptr<TypeInfo>()>;

// Bundles up the above callbacks so they can be passed around as a unit.
struct InterpCallbackData {
  TypecheckFn typecheck;
  EvaluateFn eval_fn;
  IsWipFn is_wip;
  NoteWipFn note_wip;
  GetTypeFn get_type_info;
  ImportCache* cache;

  // Wraps eval_fn to provide a default argument for type_context (as it is
  // often nullptr).
  absl::StatusOr<InterpValue> Eval(
      Expr* expr, InterpBindings* bindings,
      std::unique_ptr<ConcreteType> type_context = nullptr) {
    return this->eval_fn(expr, bindings, std::move(type_context));
  }
};

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
                                           InterpCallbackData* callbacks);

// Evaluates a struct instance expression; e.g. `Foo { field: stuff }`.
absl::StatusOr<InterpValue> EvaluateStructInstance(
    StructInstance* expr, InterpBindings* bindings, ConcreteType* type_context,
    InterpCallbackData* callbacks);

// Evaluates a struct instance expression;
// e.g. `Foo { field: stuff, ..other_foo }`.
absl::StatusOr<InterpValue> EvaluateSplatStructInstance(
    SplatStructInstance* expr, InterpBindings* bindings,
    ConcreteType* type_context, InterpCallbackData* callbacks);

// Evaluates an enum reference expression; e.g. `Foo::BAR`.
absl::StatusOr<InterpValue> EvaluateEnumRef(EnumRef* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context,
                                            InterpCallbackData* callbacks);

// Evaluates a tuple expression; e.g. `(x, y)`.
absl::StatusOr<InterpValue> EvaluateXlsTuple(XlsTuple* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context,
                                             InterpCallbackData* callbacks);

// Evaluates a let expression; e.g. `let x = y in z`
absl::StatusOr<InterpValue> EvaluateLet(Let* expr, InterpBindings* bindings,
                                        ConcreteType* type_context,
                                        InterpCallbackData* callbacks);

// Evaluates a unary operation expression; e.g. `-x`.
absl::StatusOr<InterpValue> EvaluateUnop(Unop* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         InterpCallbackData* callbacks);

// Evaluates a binary operation expression; e.g. `x + y`.
absl::StatusOr<InterpValue> EvaluateBinop(Binop* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          InterpCallbackData* callbacks);

// Evaluates a ternary expression; e.g. `foo if bar else baz`.
absl::StatusOr<InterpValue> EvaluateTernary(Ternary* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context,
                                            InterpCallbackData* callbacks);

// Evaluates an attribute expression; e.g. `x.y`.
absl::StatusOr<InterpValue> EvaluateAttr(Attr* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         InterpCallbackData* callbacks);

// Evaluates a match expression; e.g. `match x { ... }`.
absl::StatusOr<InterpValue> EvaluateMatch(Match* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          InterpCallbackData* callbacks);

// Evaluates an index expression; e.g. `a[i]`.
absl::StatusOr<InterpValue> EvaluateIndex(Index* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          InterpCallbackData* callbacks);

// Creates the top level bindings for a given module. We may not be able to
// create a *complete* set of bindings if we've re-entered this routine; e.g. in
// evaluating a top-level constant we recur to ask what enums (or similar) are
// available in the module scope -- in those cases we populate as many top level
// bindings as we can before we reach the work-in-progress point.
//
// Args:
//   module: The top-level module to make bindings for.
//   callbacks: Provide ability to call back into the interpreter facilities
//    e.g. on import or for evaluating constant value expressions.
absl::StatusOr<InterpBindings> MakeTopLevelBindings(
    const std::shared_ptr<Module>& module, InterpCallbackData* callbacks);

using ConcretizeVariant = absl::variant<TypeAnnotation*, EnumDef*, StructDef*>;

// Resolve "type" into a concrete type via expression evaluation.
absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeType(
    ConcretizeVariant type, InterpBindings* bindings,
    InterpCallbackData* callbacks);

// As above, but specifically for concretizing TypeAnnotation nodes.
absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeTypeAnnotation(
    TypeAnnotation* type, InterpBindings* bindings,
    InterpCallbackData* callbacks);

// Resolves (parametric) dimensions from deduction vs the current bindings.
absl::StatusOr<int64> ResolveDim(
    absl::variant<Expr*, int64, ConcreteTypeDim> dim, InterpBindings* bindings);

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
absl::StatusOr<DerefVariant> EvaluateToStructOrEnumOrAnnotation(
    TypeDefinition type_definition, InterpBindings* bindings,
    InterpCallbackData* callbacks);

}  // namespace xls::dslx

#endif  // XLS_DSLX_CPP_EVALUATE_H_
