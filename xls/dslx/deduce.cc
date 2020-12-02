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

absl::StatusOr<std::unique_ptr<ConcreteType>> Resolve(const ConcreteType& type,
                                                      DeduceCtx* ctx) {
  XLS_RET_CHECK(!ctx->fn_stack().empty());
  const FnStackEntry& entry = ctx->fn_stack().back();
  const SymbolicBindings& fn_symbolic_bindings = entry.symbolic_bindings;
  return type.MapSize([&fn_symbolic_bindings](ConcreteTypeDim dim)
                          -> absl::StatusOr<ConcreteTypeDim> {
    if (absl::holds_alternative<ConcreteTypeDim::OwnedParametric>(
            dim.value())) {
      const auto& parametric =
          absl::get<ConcreteTypeDim::OwnedParametric>(dim.value());
      ParametricExpression::Env env;
      for (const SymbolicBinding& binding : fn_symbolic_bindings.bindings()) {
        env[binding.identifier] = binding.value;
      }
      return ConcreteTypeDim(parametric->Evaluate(env));
    }
    return dim;
  });
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceNumber(Number* node,
                                                           DeduceCtx* ctx) {
  if (node->type() == nullptr) {
    switch (node->kind()) {
      case NumberKind::kBool:
        return BitsType::MakeU1();
      case NumberKind::kCharacter:
        return BitsType::MakeU8();
      default:
        break;
    }
    return absl::InternalError(
        absl::StrFormat("TypeInferenceError: %s <> Could not infer a type for "
                        "this number, please annotate a type.",
                        node->span().ToString()));
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_type,
                       ctx->Deduce(node->type()));
  XLS_ASSIGN_OR_RETURN(concrete_type, Resolve(*concrete_type, ctx));
  XLS_RETURN_IF_ERROR(CheckBitwidth(*node, *concrete_type));
  return concrete_type;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> DeduceTernary(Ternary* node,
                                                            DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> test_type,
                       ctx->Deduce(node->test()));
  XLS_ASSIGN_OR_RETURN(test_type, Resolve(*test_type, ctx));
  auto test_want = BitsType::MakeU1();
  if (*test_type != *test_want) {
    return absl::InternalError(absl::StrFormat(
        "XlsTypeError: %s %s %s Test type for conditional expression is not "
        "\"bool\"",
        node->span().ToString(), test_type->ToString(), test_want->ToString()));
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> consequent_type,
                       ctx->Deduce(node->consequent()));
  XLS_ASSIGN_OR_RETURN(consequent_type, Resolve(*consequent_type, ctx));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> alternate_type,
                       ctx->Deduce(node->alternate()));
  XLS_ASSIGN_OR_RETURN(alternate_type, Resolve(*alternate_type, ctx));

  if (*consequent_type != *alternate_type) {
    return absl::InternalError(absl::StrFormat(
        "XlsTypeError: %s %s %s Ternary consequent type (in the 'then' clause) "
        "did not match alternative type (in the 'else' clause)",
        node->span().ToString(), consequent_type->ToString(),
        alternate_type->ToString()));
  }
  return consequent_type;
}

}  // namespace xls::dslx
