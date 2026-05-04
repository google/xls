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

#include <optional>
#include <string>
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
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/interp_value.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::dslx {

absl::Status FuzzTestConverter::LowerDomainExpr(
    Expr* expr, PackageInterfaceProto::FuzzTestDomain* proto) {
  if (expr->kind() == AstNodeKind::kArray) {
    return LowerArrayExpr(absl::down_cast<Array*>(expr), proto);
  }
  if (expr->kind() == AstNodeKind::kRange) {
    Range* range_node = absl::down_cast<Range*>(expr);
    return LowerRangeExpr(range_node, proto);
  }
  if (expr->kind() == AstNodeKind::kXlsTuple) {
    return LowerTupleExpr(absl::down_cast<XlsTuple*>(expr), proto);
  }
  if (expr->kind() == AstNodeKind::kNameRef) {
    NameRef* name_ref = absl::down_cast<NameRef*>(expr);

    absl::StatusOr<ConstantDef*> const_def =
        module_->GetMemberOrError<ConstantDef>(name_ref->identifier());
    if (const_def.ok()) {
      // TODO: davidplass - This fails if there's a const that refers to another
      // const, or calls a function, or references an imported symbol. Instead,
      // we could tag the InterpValue of such arrays as actually ranges, use
      // GetConstExpr to get the range instead.
      Expr* value_expr = (*const_def)->value();
      return LowerDomainExpr(value_expr, proto);
    }
  }
  return absl::UnimplementedError(
      absl::StrCat("Unsupported fuzztest domain type: ", expr->ToString()));
}

absl::Status FuzzTestConverter::LowerRangeExpr(
    Range* range_node, PackageInterfaceProto::FuzzTestDomain* proto) {
  XLS_ASSIGN_OR_RETURN(InterpValue min_val,
                       current_type_info_->GetConstExpr(range_node->start()));
  XLS_ASSIGN_OR_RETURN(InterpValue max_val,
                       current_type_info_->GetConstExpr(range_node->end()));

  XLS_ASSIGN_OR_RETURN(Value ir_min, min_val.ConvertToIr());
  XLS_ASSIGN_OR_RETURN(Value ir_max, max_val.ConvertToIr());

  XLS_ASSIGN_OR_RETURN(ValueProto min_proto, ir_min.AsProto());
  XLS_ASSIGN_OR_RETURN(ValueProto max_proto, ir_max.AsProto());

  auto* range_proto = proto->mutable_range();
  *range_proto->mutable_min() = std::move(min_proto);
  *range_proto->mutable_max() = std::move(max_proto);
  return absl::OkStatus();
}

absl::Status FuzzTestConverter::LowerArrayExpr(
    Array* array_node, PackageInterfaceProto::FuzzTestDomain* proto) {
  auto* element_of_proto = proto->mutable_element_of();
  for (Expr* member : array_node->members()) {
    XLS_ASSIGN_OR_RETURN(InterpValue val,
                         current_type_info_->GetConstExpr(member));
    XLS_ASSIGN_OR_RETURN(Value ir_val, val.ConvertToIr());
    XLS_ASSIGN_OR_RETURN(ValueProto val_proto, ir_val.AsProto());
    *element_of_proto->add_values() = std::move(val_proto);
  }
  return absl::OkStatus();
}

absl::Status FuzzTestConverter::LowerTupleExpr(
    XlsTuple* tuple_node, PackageInterfaceProto::FuzzTestDomain* proto) {
  if (tuple_node->members().empty()) {
    proto->set_arbitrary(true);
    return absl::OkStatus();
  }
  auto* tuple_proto = proto->mutable_tuple();
  for (Expr* member : tuple_node->members()) {
    XLS_RETURN_IF_ERROR(LowerDomainExpr(member, tuple_proto->add_elements()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<AttributeData>>
FuzzTestConverter::LowerFuzzTestDomains(Function* node) {
  if (node->parent() == nullptr ||
      node->parent()->kind() != AstNodeKind::kFuzzTestFunction) {
    return std::nullopt;
  }
  FuzzTestFunction* ft = absl::down_cast<FuzzTestFunction*>(node->parent());

  if (ft->domains().has_value()) {
    XlsTuple* domains_tuple = *ft->domains();
    // We use a dummy Function proto here solely to get the
    // `parameter_domains` field name wrapper in the serialized text proto.
    // This will allow clients to easily parse the string back into a Function
    // proto and recover the domains therein.
    PackageInterfaceProto::Function temp_func;

    for (Expr* domain_expr : domains_tuple->members()) {
      PackageInterfaceProto::FuzzTestDomain* domain_proto =
          temp_func.add_parameter_domains();

      XLS_RETURN_IF_ERROR(LowerDomainExpr(domain_expr, domain_proto));
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
