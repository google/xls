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

#include "xls/dslx/deduce.h"

namespace xls::dslx {

absl::Status CheckBitwidth(const Number& number, const ConcreteType& type) {
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bits_dim, type.GetTotalBitCount());
  XLS_RET_CHECK(absl::holds_alternative<int64>(bits_dim.value()));
  int64 bit_count = absl::get<int64>(bits_dim.value());
  absl::StatusOr<Bits> bits = number.GetBits(bit_count);
  if (!bits.ok()) {
    return absl::InternalError(
        absl::StrFormat("TypeInferenceError: %s %s Value '%s' does not fit in "
                        "the bitwidth of a %s (%d)",
                        number.span().ToString(), type.ToString(),
                        number.text(), type.ToString(), bit_count));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceUnop(Unop* node,
                                                         DeduceCtx* ctx) {
  return ctx->Deduce(node->operand());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceParam(Param* node,
                                                          DeduceCtx* ctx) {
  return ctx->Deduce(node->type());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceConstantDef(
    ConstantDef* node, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> result,
                       ctx->Deduce(node->value()));
  ctx->type_info()->SetItem(node->name_def(), *result);
  ctx->type_info()->NoteConstant(node->name_def(), node);
  return result;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTypeRef(TypeRef* node,
                                                            DeduceCtx* ctx) {
  return ctx->Deduce(ToAstNode(node->type_definition()));
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTypeDef(TypeDef* node,
                                                            DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                       ctx->Deduce(node->type()));
  ctx->type_info()->SetItem(node->name_def(), *type);
  return type;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceXlsTuple(XlsTuple* node,
                                                             DeduceCtx* ctx) {
  std::vector<std::unique_ptr<ConcreteType>> members;
  for (Expr* e : node->members()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> m, ctx->Deduce(e));
    members.push_back(std::move(m));
  }
  return absl::make_unique<TupleType>(std::move(members));
}

}  // namespace xls::dslx
