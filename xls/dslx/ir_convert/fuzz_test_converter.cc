// Copyright 2026 The XLS Authors
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

#include "xls/dslx/ir_convert/fuzz_test_converter.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/text_format.h"
#include "xls/common/attribute_data.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::dslx {

namespace {

absl::Status LowerRange(const InterpValue& min_val, const InterpValue& max_val,
                        PackageInterfaceProto::FuzzTestDomain& proto) {
  XLS_ASSIGN_OR_RETURN(Value ir_min, min_val.ConvertToIr());
  XLS_ASSIGN_OR_RETURN(Value ir_max, max_val.ConvertToIr());
  XLS_ASSIGN_OR_RETURN(ValueProto min_proto, ir_min.AsProto());
  XLS_ASSIGN_OR_RETURN(ValueProto max_proto, ir_max.AsProto());

  auto* range_proto = proto.mutable_range();
  *range_proto->mutable_min() = std::move(min_proto);
  *range_proto->mutable_max() = std::move(max_proto);
  return absl::OkStatus();
}

absl::Status LowerArray(const std::vector<InterpValue>& elements,
                        PackageInterfaceProto::FuzzTestDomain& proto) {
  if (elements.empty()) {
    return absl::InvalidArgumentError(
        "Empty arrays are unsupported as fuzztest domains");
  }
  auto* element_of_proto = proto.mutable_element_of();
  for (const auto& element : elements) {
    XLS_ASSIGN_OR_RETURN(Value ir_val, element.ConvertToIr());
    XLS_ASSIGN_OR_RETURN(ValueProto val_proto, ir_val.AsProto());
    *element_of_proto->add_values() = std::move(val_proto);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status FuzzTestConverter::LowerTuple(
    const Type* param_type, const std::vector<InterpValue>& elements,
    PackageInterfaceProto::FuzzTestDomain& proto) {
  if (elements.empty()) {
    // Cannot send enums into this function; must use LowerArbitraryType
    XLS_RET_CHECK(param_type == nullptr || !param_type->IsEnum());
    proto.set_arbitrary(true);
    return absl::OkStatus();
  }
  auto* tuple_proto = proto.mutable_tuple();
  const TupleType* tuple_type = nullptr;
  const StructType* struct_type = nullptr;
  if (param_type != nullptr) {
    if (param_type->IsTuple()) {
      tuple_type = &param_type->AsTuple();
    } else if (param_type->IsStruct()) {
      struct_type = &param_type->AsStruct();
    }
  }
  for (size_t i = 0; i < elements.size(); ++i) {
    const Type* member_type = nullptr;
    if (tuple_type != nullptr) {
      member_type = tuple_type->members()[i].get();
    } else if (struct_type != nullptr) {
      member_type = &struct_type->GetMemberType(i);
    }
    const InterpValue& element = elements[i];
    XLS_RETURN_IF_ERROR(
        LowerConstant(member_type, element, *tuple_proto->add_elements()));
  }
  return absl::OkStatus();
}

absl::Status FuzzTestConverter::LowerStructInstanceDomain(
    const StructType& struct_type, const StructInstance& struct_domain,
    PackageInterfaceProto::FuzzTestDomain& proto) {
  auto* tuple_proto = proto.mutable_tuple();

  for (int64_t i = 0; i < struct_type.size(); ++i) {
    std::string_view field_name = struct_type.GetMemberName(i);
    const Type& field_type = struct_type.GetMemberType(i);
    PackageInterfaceProto::FuzzTestDomain* element_proto =
        tuple_proto->add_elements();

    absl::StatusOr<Expr*> specified_domain = struct_domain.GetExpr(field_name);
    if (specified_domain.ok()) {
      XLS_RETURN_IF_ERROR(
          LowerDomainExpr(&field_type, *specified_domain, *element_proto));
    } else if (absl::IsNotFound(specified_domain.status())) {
      XLS_RETURN_IF_ERROR(LowerArbitraryType(&field_type, *element_proto));
    } else {
      return specified_domain.status();
    }
  }
  return absl::OkStatus();
}

absl::Status FuzzTestConverter::LowerArbitraryEnum(
    const Type* param_type, PackageInterfaceProto::FuzzTestDomain& proto) {
  XLS_RET_CHECK(param_type != nullptr && param_type->HasEnum());
  const EnumType& enum_type = param_type->AsEnum();

  auto* element_of_proto = proto.mutable_element_of();
  //  1. Get the enum values
  for (const auto& enum_member : enum_type.nominal_type().values()) {
    Expr* value_expr = enum_member.value;
    XLS_ASSIGN_OR_RETURN(
        InterpValue const_value,
        ConstexprEvaluator::EvaluateToValue(import_data_, current_type_info_,
                                            /*warning_collector=*/nullptr,
                                            /*bindings=*/{}, value_expr));
    XLS_ASSIGN_OR_RETURN(ValueProto val_proto, const_value.AsProto());
    //  2. Add them to the element_of_proto
    *element_of_proto->add_values() = std::move(val_proto);
  }

  return absl::OkStatus();
}

absl::Status FuzzTestConverter::LowerArbitraryType(
    const Type* param_type, PackageInterfaceProto::FuzzTestDomain& proto) {
  if (param_type->IsEnum()) {
    return LowerArbitraryEnum(param_type, proto);
  }
  if (param_type->IsTuple()) {
    auto* tuple_proto = proto.mutable_tuple();
    const TupleType& tuple_type = param_type->AsTuple();
    for (const auto& member_type : tuple_type.members()) {
      XLS_RETURN_IF_ERROR(
          LowerArbitraryType(member_type.get(), *tuple_proto->add_elements()));
    }
    return absl::OkStatus();
  }
  proto.set_arbitrary(true);
  return absl::OkStatus();
}

absl::Status FuzzTestConverter::LowerConstant(
    const Type* param_type, const InterpValue& val,
    PackageInterfaceProto::FuzzTestDomain& proto) {
  if (val.is_range()) {
    // InterpValues that originated as ranges are stored as an array of
    // elements, so we need to get the first and last entries in the array
    // (the min and max of the range).
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         val.GetValues());
    if (elements->empty()) {
      return absl::InvalidArgumentError(
          "Empty ranges are unsupported as fuzztest domains");
    }
    return LowerRange(elements->front(), elements->back(), proto);
  }
  if (val.IsArray()) {
    // Indicates an ElementOf domain, NOT an array parameter.
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         val.GetValues());
    return LowerArray(*elements, proto);
  }
  if (val.IsTuple()) {
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         val.GetValues());

    // Option B desugared domain Tuple decoding
    if (!elements->empty() && elements->front().IsBits()) {
      XLS_ASSIGN_OR_RETURN(int64_t tag, elements->front().GetBitValueViaSign());
      if (tag == 0 && elements->size() == 1) {
        // TAG_ARBITRARY
        proto.set_arbitrary(true);
        return absl::OkStatus();
      }
      if (tag == 1 && elements->size() == 3) {
        // TAG_RANGE
        return LowerRange((*elements)[1], (*elements)[2], proto);
      }
    }

    // If the parameter is an enum, or contains enums, with an arbitrary domain,
    // we need to lower it so it is represented as an ElementOf domain (or tuple
    // thereof), rather than a RangeOf the underlying bits type. This is because
    // some enumerations don't cover their whole bits range, and so it would be
    // wrong to use the whole bits range as a fuzztest domain.
    XLS_RET_CHECK(param_type != nullptr);
    if (param_type->HasEnum() && elements->empty()) {
      return LowerArbitraryType(param_type, proto);
    }
    return LowerTuple(param_type, *elements, proto);
  }
  return absl::UnimplementedError(absl::StrCat(
      "Unsupported constant fuzztest domain type: ", val.ToString()));
}

absl::Status FuzzTestConverter::LowerDomainExpr(
    const Type* param_type, const Expr* expr,
    PackageInterfaceProto::FuzzTestDomain& proto) {
  if (expr->kind() == AstNodeKind::kStructInstance) {
    XLS_RET_CHECK(param_type != nullptr && param_type->IsStruct());
    const StructInstance* struct_domain =
        absl::down_cast<const StructInstance*>(expr);
    const StructType& struct_type = param_type->AsStruct();
    return LowerStructInstanceDomain(struct_type, *struct_domain, proto);
  }

  if (expr->kind() == AstNodeKind::kRange) {
    // Ranges get expanded into arrays by the constexpr evaluator, so if you
    // have a range of u32:0..u32:FFFFFFFF, it will try to turn it into an array
    // of 2^32 elements, which fills memory. So for ranges we perform the
    // lowering directly from the AST, without turning to InterpValue.
    const Range* range_node = absl::down_cast<const Range*>(expr);
    return LowerRangeExpr(range_node, proto);
  }
  XLS_ASSIGN_OR_RETURN(
      InterpValue const_value,
      ConstexprEvaluator::EvaluateToValue(import_data_, current_type_info_,
                                          /*warning_collector=*/nullptr,
                                          /*bindings=*/{}, expr));
  return LowerConstant(param_type, const_value, proto);
}

absl::Status FuzzTestConverter::LowerRangeExpr(
    const Range* range_node, PackageInterfaceProto::FuzzTestDomain& proto) {
  XLS_ASSIGN_OR_RETURN(InterpValue min_val,
                       ConstexprEvaluator::EvaluateToValue(
                           import_data_, current_type_info_,
                           /*warning_collector=*/nullptr,
                           /*bindings=*/{}, range_node->start()));
  XLS_ASSIGN_OR_RETURN(
      InterpValue max_val,
      ConstexprEvaluator::EvaluateToValue(import_data_, current_type_info_,
                                          /*warning_collector=*/nullptr,
                                          /*bindings=*/{}, range_node->end()));
  if (!range_node->inclusive_end()) {
    XLS_ASSIGN_OR_RETURN(InterpValue ge_val, min_val.Ge(max_val));
    if (ge_val.IsTrue()) {
      return absl::InvalidArgumentError(
          "Empty ranges are unsupported as fuzztest domains");
    }
    // DSLX range syntax is [start, end), so we need to decrement the end value
    // to make it inclusive for the InRange domain. Example: u32:0..10
    // does not include 10, so we have to specify InRange(0, 9) to get the same
    // range of 0 to 9 inclusive.
    std::optional<InterpValue> decremented = max_val.Decrement();
    if (!decremented.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Could not decrement range end value: ", max_val.ToString()));
    }
    max_val = *decremented;
  }
  return LowerRange(min_val, max_val, proto);
}

absl::StatusOr<std::optional<AttributeData>>
FuzzTestConverter::LowerFuzzTestDomains(const Function* node) {
  if (node->parent() == nullptr ||
      node->parent()->kind() != AstNodeKind::kFuzzTestFunction) {
    return std::nullopt;
  }
  const FuzzTestFunction* ft =
      absl::down_cast<const FuzzTestFunction*>(node->parent());

  if (ft->domains().has_value()) {
    const XlsTuple* domains_tuple = *ft->domains();
    // We use a dummy Function proto here solely to get the
    // `parameter_domains` field name wrapper in the serialized text proto.
    // This will allow clients to easily parse the string back into a Function
    // proto and recover the domains therein.
    PackageInterfaceProto::Function temp_func;

    for (int i = 0; i < domains_tuple->members().size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          const Type* param_type,
          current_type_info_->GetItemOrError(node->params()[i]));
      const Expr* domain_expr = domains_tuple->members().at(i);

      PackageInterfaceProto::FuzzTestDomain* domain_proto =
          temp_func.add_parameter_domains();
      XLS_RETURN_IF_ERROR(
          LowerDomainExpr(param_type, domain_expr, *domain_proto));
    }

    std::string proto_str;
    google::protobuf::TextFormat::Printer printer;
    printer.SetSingleLineMode(true);
    XLS_RET_CHECK(printer.PrintToString(temp_func, &proto_str));

    std::vector<AttributeData::Argument> args;
    args.push_back(
        AttributeData::StringKeyValueArgument{.first = "domains",
                                              .second = std::move(proto_str),
                                              .is_backticked = true});

    return AttributeData(AttributeKind::kFuzzTest, std::move(args));
  }
  return std::nullopt;
}

}  // namespace xls::dslx
