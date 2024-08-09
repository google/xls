// Renders the `op.cc` source file from our programmatic description tables.

#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "xls/ir/op_specification.h"

namespace xls {

static const std::string_view kSourceTemplate = R"(#include "xls/ir/op.h"

#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"

namespace xls {

{OP_FROM_OP_PROTO}

{OP_TO_OP_PROTO}

{OP_TO_STRING}

{STRING_TO_OP}

{OP_IS_COMPARE}

{OP_IS_ASSOCIATIVE}

{OP_IS_COMMUTATIVE}

{OP_IS_BIT_WISE}

{OP_IS_SIDE_EFFECTING}

{IS_OP_CLASS_TEMPLATE_IMPLS}

std::ostream& operator<<(std::ostream& os, Op op) {
  os << OpToString(op);
  return os;
}

}  // namespace xls
)";

static std::string RenderFromOpProperty(absl::string_view name,
                                        Property target) {
  std::vector<std::string> lines = {
      absl::StrFormat("bool OpIs%s(Op op) {", name)};
  for (const Op& op : GetOpsSingleton()) {
    if (op.properties.contains(target)) {
      lines.push_back(
          absl::StrFormat("  if (op == Op::%s) return true;", op.enum_name));
    }
  }
  lines.push_back("  return false;");
  lines.push_back("}");
  return absl::StrJoin(lines, "\n");
}

static std::string RenderIsOpClassTemplateImpls() {
  std::vector<std::string> lines;
  for (const auto& [_name, op_class] : GetOpClassKindsSingleton()) {
    lines.push_back(absl::StrFormat("template<> bool IsOpClass<%s>(Op op) {",
                                    op_class.name()));
    for (const Op& op : GetOpsSingleton()) {
      if (&op.op_class == &op_class) {
        lines.push_back(
            absl::StrFormat("  if (op == Op::%s) return true;", op.enum_name));
      }
    }
    lines.push_back("  return false;");
    lines.push_back("}");
  }
  return absl::StrJoin(lines, "\n");
}

static std::string RenderOpFromOpProto() {
  std::vector<std::string> lines = {
      "Op FromOpProto(OpProto op_proto) {",
      "  switch (op_proto) {",
      "   case OP_INVALID: LOG(FATAL) << \"Cannot convert OP_INVALID proto "
      "value to a C++ Op\"; break;",
  };
  for (const Op& op : GetOpsSingleton()) {
    lines.push_back(absl::StrFormat("   case OP_%s: return Op::%s;",
                                    absl::AsciiStrToUpper(op.name),
                                    op.enum_name));
  }
  lines.push_back(
      "   // Note: since this is a proto enum there are sentinel values");
  lines.push_back(
      "   // defined in addition to the 'real' ones above, which is why");
  lines.push_back("   // the numeration above is not exhaustive.");
  lines.push_back("   default:");
  lines.push_back(
      "    LOG(FATAL) << \"Invalid OpProto: \" << "
      "static_cast<int64_t>(op_proto);");
  lines.push_back("  }");
  lines.push_back("}");
  return absl::StrJoin(lines, "\n");
}

static std::string RenderToOpProto() {
  std::vector<std::string> lines = {"OpProto ToOpProto(Op op) {"};
  lines.push_back("  switch (op) {");
  for (const Op& op : GetOpsSingleton()) {
    lines.push_back(absl::StrFormat("   case Op::%s: return OP_%s;",
                                    op.enum_name,
                                    absl::AsciiStrToUpper(op.name)));
  }
  lines.push_back("  }");
  lines.push_back(
      "  LOG(FATAL) << \"Invalid Op: \" << static_cast<int64_t>(op);");
  lines.push_back("}");
  return absl::StrJoin(lines, "\n");
}

static std::string RenderOpToString() {
  std::vector<std::string> lines = {"std::string OpToString(Op op) {"};
  lines.push_back("  switch (op) {");
  for (const Op& op : GetOpsSingleton()) {
    lines.push_back(absl::StrFormat("   case Op::%s: return \"%s\";",
                                    op.enum_name, op.name));
  }
  lines.push_back("  }");
  lines.push_back(
      "  LOG(FATAL) << \"Invalid Op: \" << static_cast<int64_t>(op);");
  lines.push_back("}");
  return absl::StrJoin(lines, "\n");
}

static std::string RenderStringToOp() {
  std::vector<std::string> lines = {
      "absl::StatusOr<Op> StringToOp(std::string_view s) {"};
  lines.push_back("  if (false) {}  // prologue");
  for (const Op& op : GetOpsSingleton()) {
    lines.push_back(absl::StrFormat("  else if (s == \"%s\") return Op::%s;",
                                    op.name, op.enum_name));
  }
  lines.push_back("  else { ");
  lines.push_back(
      "    return absl::InvalidArgumentError(absl::StrFormat(\"Unknown "
      "operation for string-to-op conversion: %s\", s));");
  lines.push_back("  }");
  lines.push_back("}");
  return absl::StrJoin(lines, "\n");
}

std::string RenderOpSource() {
  return absl::StrReplaceAll(
      kSourceTemplate,
      {
          {"{OP_FROM_OP_PROTO}", RenderOpFromOpProto()},
          {"{OP_TO_OP_PROTO}", RenderToOpProto()},
          {"{OP_TO_STRING}", RenderOpToString()},
          {"{STRING_TO_OP}", RenderStringToOp()},
          {"{OP_IS_COMPARE}",
           RenderFromOpProperty("Compare", Property::kComparison)},
          {"{OP_IS_ASSOCIATIVE}",
           RenderFromOpProperty("Associative", Property::kAssociative)},
          {"{OP_IS_COMMUTATIVE}",
           RenderFromOpProperty("Commutative", Property::kCommutative)},
          {"{OP_IS_BIT_WISE}",
           RenderFromOpProperty("BitWise", Property::kBitwise)},
          {"{OP_IS_SIDE_EFFECTING}",
           RenderFromOpProperty("SideEffecting", Property::kSideEffecting)},
          {"{IS_OP_CLASS_TEMPLATE_IMPLS}", RenderIsOpClassTemplateImpls()},
      });
}

}  // namespace xls
