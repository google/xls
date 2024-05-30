// Copyright 2023 The XLS Authors
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

#include "xls/dslx/ir_convert/convert_format_macro.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/make_value_format_descriptor.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/function_builder.h"

namespace xls::dslx {
namespace {

struct ConvertContext {
  std::vector<FormatStep>& fmt_steps;
  std::vector<BValue>& ir_args;
  BuilderBase& fn_builder;
};

// Forward decl for recursion.
absl::Status Flatten(const ValueFormatDescriptor& vfd, const BValue& v,
                     ConvertContext& ctx);

absl::Status FlattenTuple(const TupleFormatDescriptor& tfd, const BValue& v,
                          ConvertContext& ctx) {
  ctx.fmt_steps.push_back("(");
  for (size_t i = 0; i < tfd.elements().size(); ++i) {
    BValue item = ctx.fn_builder.TupleIndex(v, i);
    XLS_RETURN_IF_ERROR(Flatten(*tfd.elements().at(i), item, ctx));
    if (i + 1 != tfd.elements().size()) {
      ctx.fmt_steps.push_back(", ");
    }
  }
  if (tfd.elements().size() == 1) {
    // Singleton tuple -- we put a trailing comma on the item.
    ctx.fmt_steps.push_back(",");
  }
  ctx.fmt_steps.push_back(")");
  return absl::OkStatus();
}

absl::Status FlattenStruct(const StructFormatDescriptor& sfd, const BValue& v,
                           ConvertContext& ctx) {
  ctx.fmt_steps.push_back(absl::StrCat(sfd.struct_name(), "{{"));
  size_t fieldno = 0;
  for (size_t i = 0; i < sfd.elements().size(); ++i) {
    const StructFormatDescriptor::Element& e = sfd.elements().at(i);
    std::string leader = absl::StrCat(e.field_name, ": ");
    if (i != 0) {
      leader = absl::StrCat(", ", leader);
    }
    ctx.fmt_steps.push_back(leader);
    BValue field_value = ctx.fn_builder.TupleIndex(v, fieldno++);
    XLS_RETURN_IF_ERROR(Flatten(*e.fmt, field_value, ctx));
  }
  ctx.fmt_steps.push_back("}}");
  return absl::OkStatus();
}

absl::Status FlattenArray(const ArrayFormatDescriptor& sfd, const BValue& v,
                          ConvertContext& ctx) {
  ctx.fmt_steps.push_back("[");
  for (int64_t i = 0; i < sfd.size(); ++i) {
    BValue index = ctx.fn_builder.Literal(UBits(i, /*bit_count=*/32));
    BValue elem = ctx.fn_builder.ArrayIndex(v, {index});
    if (i != 0) {
      ctx.fmt_steps.push_back(", ");
    }
    XLS_RETURN_IF_ERROR(Flatten(sfd.element_format(), elem, ctx));
  }
  ctx.fmt_steps.push_back("]");
  return absl::OkStatus();
}

absl::Status FlattenEnum(const EnumFormatDescriptor& efd, const BValue& v,
                         ConvertContext& ctx) {
  // We currently don't have a way to give a "map of value=>string" to the IR.
  //
  // We could make a different trace IR op for every possible enum value but
  // that doesn't scale with number of operands or size of enums. So for now, at
  // the IR level we just trace the value itself.
  ctx.fmt_steps.push_back(absl::StrCat(efd.enum_name(), "::"));
  ctx.fmt_steps.push_back(FormatPreference::kDefault);
  ctx.ir_args.push_back(v);
  return absl::OkStatus();
}

class FlattenVisitor : public ValueFormatVisitor {
 public:
  FlattenVisitor(BValue ir_value, ConvertContext& ctx)
      : ir_value_(ir_value), ctx_(ctx) {}

  ~FlattenVisitor() override = default;

  absl::Status HandleArray(const ArrayFormatDescriptor& d) override {
    return FlattenArray(d, ir_value_, ctx_);
  }
  absl::Status HandleStruct(const StructFormatDescriptor& d) override {
    return FlattenStruct(d, ir_value_, ctx_);
  }
  absl::Status HandleEnum(const EnumFormatDescriptor& d) override {
    return FlattenEnum(d, ir_value_, ctx_);
  }
  absl::Status HandleTuple(const TupleFormatDescriptor& d) override {
    return FlattenTuple(d, ir_value_, ctx_);
  }
  absl::Status HandleLeafValue(const LeafValueFormatDescriptor& d) override {
    ctx_.fmt_steps.push_back(d.format());
    ctx_.ir_args.push_back(ir_value_);
    return absl::OkStatus();
  }

 private:
  BValue ir_value_;
  ConvertContext& ctx_;
};

absl::Status Flatten(const ValueFormatDescriptor& vfd, const BValue& v,
                     ConvertContext& ctx) {
  FlattenVisitor visitor(v, ctx);
  return vfd.Accept(visitor);
}

}  // namespace

absl::StatusOr<BValue> ConvertFormatMacro(const FormatMacro& node,
                                          const BValue& entry_token,
                                          const BValue& control_predicate,
                                          absl::Span<const BValue> arg_vals,
                                          const TypeInfo& current_type_info,
                                          BuilderBase& function_builder) {
  // This is the data we build up for the final "trace" operation we place in
  // the IR -- there is a sequence of format steps and a corresponding set of
  // ir_args that are interpolated into the format steps.
  std::vector<FormatStep> fmt_steps;
  std::vector<BValue> ir_args;
  ConvertContext ctx{.fmt_steps = fmt_steps,
                     .ir_args = ir_args,
                     .fn_builder = function_builder};

  size_t next_argno = 0;
  for (size_t node_format_index = 0; node_format_index < node.format().size();
       ++node_format_index) {
    const FormatStep& step = node.format().at(node_format_index);
    if (std::holds_alternative<std::string>(step)) {
      fmt_steps.push_back(step);
    } else {
      XLS_RET_CHECK(std::holds_alternative<FormatPreference>(step));
      FormatPreference preference = std::get<FormatPreference>(step);
      const BValue& arg_val = arg_vals.at(next_argno);
      const Expr* arg_expr = node.args().at(next_argno);

      std::optional<Type*> maybe_type = current_type_info.GetItem(arg_expr);
      XLS_RET_CHECK(maybe_type.has_value());
      Type* type = maybe_type.value();
      XLS_ASSIGN_OR_RETURN(auto value_format_descriptor,
                           MakeValueFormatDescriptor(*type, preference));
      XLS_RETURN_IF_ERROR(Flatten(*value_format_descriptor, arg_val, ctx));
      next_argno += 1;
    }
  }

  return function_builder.Trace(entry_token, control_predicate, ir_args,
                                fmt_steps);
}

}  // namespace xls::dslx
