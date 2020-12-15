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

#ifndef XLS_DSLX_DEDUCE_H_
#define XLS_DSLX_DEDUCE_H_

#include "xls/common/status/ret_check.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/deduce_ctx.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"

namespace xls::dslx {

// Checks that the given "number" AST node fits inside the bitwidth of "type".
//
// After type inference we may have determined the type of some numbers, and we
// need to then make sure they actually fit within that inferred type's
// bit width.
absl::Status CheckBitwidth(const Number& number, const ConcreteType& type);

// -- Deduction rules, determine the concrete type of the node and return it.

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceUnop(Unop* node,
                                                         DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceParam(Param* node,
                                                          DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceConstantDef(
    ConstantDef* node, DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceNumber(Number* node,
                                                           DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTypeRef(TypeRef* node,
                                                            DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTypeDef(TypeDef* node,
                                                            DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceXlsTuple(XlsTuple* node,
                                                             DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTernary(Ternary* node,
                                                            DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceBinop(Binop* node,
                                                          DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceEnumDef(EnumDef* node,
                                                            DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceLet(Let* node,
                                                        DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceFor(For* node,
                                                        DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceCast(Cast* node,
                                                         DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceStructDef(StructDef* node,
                                                              DeduceCtx* ctx);

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceArray(Array* node,
                                                          DeduceCtx* ctx);

// See Attr class in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceAttr(Attr* node,
                                                         DeduceCtx* ctx);

// See ConstantArray in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceConstantArray(
    ConstantArray* node, DeduceCtx* ctx);

// See ColonRef in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceColonRef(ColonRef* node,
                                                             DeduceCtx* ctx);

// See Index in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceIndex(Index* node,
                                                          DeduceCtx* ctx);

// See Match in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceMatch(Match* node,
                                                          DeduceCtx* ctx);

// See StructInstance in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceStructInstance(
    StructInstance* node, DeduceCtx* ctx);

// See SplatStructInstance in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceSplatStructInstance(
    SplatStructInstance* node, DeduceCtx* ctx);

// See BuiltinTypeAnnotation in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceBuiltinTypeAnnotation(
    BuiltinTypeAnnotation* node, DeduceCtx* ctx);

// See ArrayTypeAnnotation in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceArrayTypeAnnotation(
    ArrayTypeAnnotation* node, DeduceCtx* ctx);

// See TupleTypeAnnotation in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTupleTypeAnnotation(
    TupleTypeAnnotation* node, DeduceCtx* ctx);

// See TypeRefTypeAnnotation in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTypeRefTypeAnnotation(
    TypeRefTypeAnnotation* node, DeduceCtx* ctx);

// See MatchArm in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceMatchArm(MatchArm* node,
                                                             DeduceCtx* ctx);

// See While in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceWhile(While* node,
                                                          DeduceCtx* ctx);

// See Carry in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceCarry(Carry* node,
                                                          DeduceCtx* ctx);

// See Invocation in the AST.
absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceInvocation(Invocation* node,
                                                               DeduceCtx* ctx);

// Resolves "type_" via provided symbolic bindings.
//
// Uses the symbolic bindings of the function we're currently inside of to
// resolve parametric types.
//
// Args:
//  type: Type to resolve any contained dims for.
//  ctx: Deduction context to use in resolving the dims.
//
// Returns:
//  "type" with dimensions resolved according to bindings in "ctx".
absl::StatusOr<std::unique_ptr<ConcreteType>> Resolve(const ConcreteType& type,
                                                      DeduceCtx* ctx);

}  // namespace xls::dslx

#endif  // XLS_DSLX_DEDUCE_H_
