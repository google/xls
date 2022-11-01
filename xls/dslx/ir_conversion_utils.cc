// Copyright 2021 The XLS Authors
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
#include "xls/dslx/ir_conversion_utils.h"

#include "xls/dslx/deduce_ctx.h"

namespace xls::dslx {

absl::StatusOr<ConcreteTypeDim> ResolveDim(ConcreteTypeDim dim,
                                           const SymbolicBindings& bindings) {
  while (
      std::holds_alternative<ConcreteTypeDim::OwnedParametric>(dim.value())) {
    ParametricExpression& original =
        *std::get<ConcreteTypeDim::OwnedParametric>(dim.value());
    ParametricExpression::Evaluated evaluated =
        original.Evaluate(ToParametricEnv(bindings));
    dim = ConcreteTypeDim(std::move(evaluated));
  }
  return dim;
}

absl::StatusOr<int64_t> ResolveDimToInt(const ConcreteTypeDim& dim,
                                        const SymbolicBindings& bindings) {
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim resolved, ResolveDim(dim, bindings));
  if (std::holds_alternative<InterpValue>(resolved.value())) {
    return std::get<InterpValue>(resolved.value()).GetBitValueInt64();
  }
  return absl::InternalError(absl::StrFormat(
      "Expected resolved dimension of %s to be an integer, got: %s",
      dim.ToString(), resolved.ToString()));
}

absl::StatusOr<xls::Type*> TypeToIr(Package* package,
                                    const ConcreteType& concrete_type,
                                    const SymbolicBindings& bindings) {
  XLS_VLOG(5) << "Converting concrete type to IR: " << concrete_type;
  if (auto* array_type = dynamic_cast<const ArrayType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(
        xls::Type * element_type,
        TypeToIr(package, array_type->element_type(), bindings));
    XLS_ASSIGN_OR_RETURN(int64_t element_count,
                         ResolveDimToInt(array_type->size(), bindings));
    xls::Type* result = package->GetArrayType(element_count, element_type);
    XLS_VLOG(5) << "Converted type to IR; concrete type: " << concrete_type
                << " ir: " << result->ToString()
                << " element_count: " << element_count;
    return result;
  }
  if (auto* bits_type = dynamic_cast<const BitsType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                         ResolveDimToInt(bits_type->size(), bindings));
    return package->GetBitsType(bit_count);
  }
  if (auto* enum_type = dynamic_cast<const EnumType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, enum_type->size().GetAsInt64());
    return package->GetBitsType(bit_count);
  }
  if (dynamic_cast<const TokenType*>(&concrete_type) != nullptr) {
    return package->GetTokenType();
  }
  std::vector<xls::Type*> members;
  if (auto* struct_type = dynamic_cast<const StructType*>(&concrete_type)) {
    for (const std::unique_ptr<ConcreteType>& m : struct_type->members()) {
      XLS_ASSIGN_OR_RETURN(xls::Type * type, TypeToIr(package, *m, bindings));
      members.push_back(type);
    }
    return package->GetTupleType(std::move(members));
  }
  auto* tuple_type = dynamic_cast<const TupleType*>(&concrete_type);
  XLS_RET_CHECK(tuple_type != nullptr) << concrete_type;
  for (const std::unique_ptr<ConcreteType>& m : tuple_type->members()) {
    XLS_ASSIGN_OR_RETURN(xls::Type * type, TypeToIr(package, *m, bindings));
    members.push_back(type);
  }
  return package->GetTupleType(std::move(members));
}

}  // namespace xls::dslx
