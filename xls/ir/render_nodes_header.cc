#include "xls/ir/render_nodes_header.h"

#include <string>
#include <string_view>

#include "absl/strings/str_replace.h"

namespace xls {

static const std::string_view kSubclassTemplate =
    R"(class {OP_CLASS_NAME} final : public Node {
 public:
{OPERAND_INDICES}

{CONSTRUCTOR_DECL}

  absl::StatusOr<Node *>
  CloneInNewFunction(absl::Span<Node *const> new_operands,
                     FunctionBase *new_function) const final;

{OP_CLASS_METHODS}

{MANUAL_OPTIONAL_IMPLS}

{DEFINITELY_EQUAL_TO_DECL}

 private:
{OP_CLASS_DATA_MEMBERS}
};
)";

static const std::string_view kHeaderTemplate = R"(#ifndef XLS_IR_NODES_
#define XLS_IR_NODES_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/register.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

class Function;
class Instantiation;

struct SliceData {
  int64_t start;
  int64_t width;
};

{NODE_SUBCLASSES}

}  // namespace xls

#endif  // XLS_IR_NODES_
)";

static std::string RenderOpIndices(const OpClass& op_class) {
  std::vector<std::string> lines;
  for (const OperandInfo& op_set : op_class.GetFixedOperands()) {
    lines.push_back(
        absl::StrFormat("  static constexpr int64_t k%sOperand = %d;",
                        op_set.operand().CamelCaseName(), op_set.index()));
  }
  return absl::StrJoin(lines, "\n");
}

static std::string RenderMethods(const OpClass& op_class) {
  std::vector<std::string> lines;
  for (const Method& method : op_class.GetMethods()) {
    lines.push_back(absl::StrFormat("  %s %s(%s)%s", method.return_cpp_type(),
                                    method.name(), method.params(),
                                    (method.is_const() ? " const" : "")));
    if (method.expression().has_value()) {
      lines.back() += " {";
      if (method.expression_is_body()) {
        lines.push_back(absl::StrFormat("    %s", method.expression().value()));
      } else {
        lines.push_back(
            absl::StrFormat("    return %s;", method.expression().value()));
      }
      lines.push_back("  }");
    } else {
      lines.back() += ";";
    }
  }

  return absl::StrJoin(lines, "\n");
}

static std::string RenderDataMembers(const OpClass& op_class) {
  std::vector<std::string> lines;
  for (const DataMember& member : op_class.GetDataMembers()) {
    lines.push_back(absl::StrFormat("  %s %s;", member.cpp_type, member.name));
  }
  return absl::StrJoin(lines, "\n");
}

static std::string RenderOptionalOperands(const OpClass& op_class) {
  std::vector<std::string> lines;
  for (const OperandInfo op_set : op_class.GetOptionalOperands()) {
    if (!op_set.operand().AsOptionalOrDie().manual_optional_implementation()) {
      lines.push_back(absl::StrFormat(
          "  absl::StatusOr<int64_t> %s_operand_number() const {",
          op_set.operand().name()));
      lines.push_back(
          absl::StrFormat("    if (!has_%s_) { return absl::InternalError(\"%s "
                          "is not present\"); }",
                          op_set.operand().name(), op_set.operand().name()));
      lines.push_back(absl::StrFormat("    int64_t ret = %d;", op_set.index()));
      for (const OperandInfo other_optional : op_class.GetOptionalOperands()) {
        if (other_optional.index() < op_set.index()) {
          lines.push_back(absl::StrFormat("    if (!has_%s_) { ret--; }",
                                          other_optional.operand().name()));
        }
      }
      lines.push_back("    return ret;");
      lines.push_back("  }");
    }
  }
  return absl::StrJoin(lines, "\n");
}

std::string RenderNodeSubclass(const OpClass& op_class) {
  std::string constructor_decl = absl::StrFormat(
      "  %s(%s);", op_class.name(), op_class.GetConstructorArgsStr());

  std::string definitely_equal_to_decl;
  if (!op_class.GetDataMembers().empty()) {
    definitely_equal_to_decl =
        "  bool IsDefinitelyEqualTo(const Node *other) const final;";
  }

  return absl::StrReplaceAll(
      kSubclassTemplate,
      {
          {"{OP_CLASS_NAME}", op_class.name()},
          {"{OPERAND_INDICES}", RenderOpIndices(op_class)},
          {"{CONSTRUCTOR_DECL}", constructor_decl},
          {"{OP_CLASS_METHODS}", RenderMethods(op_class)},
          {"{OP_CLASS_DATA_MEMBERS}", RenderDataMembers(op_class)},
          {"{DEFINITELY_EQUAL_TO_DECL}", definitely_equal_to_decl},
          {"{MANUAL_OPTIONAL_IMPLS}", RenderOptionalOperands(op_class)},
      });
}

std::string RenderNodesHeader() {
  std::vector<std::string> subclasses;
  for (const auto& [name, op_class] : GetOpClassKindsSingleton()) {
    subclasses.push_back(RenderNodeSubclass(op_class));
  }
  std::string subclasses_str = absl::StrJoin(subclasses, "\n");

  return absl::StrReplaceAll(kHeaderTemplate,
                             {{"{NODE_SUBCLASSES}", subclasses_str}});
}

}  // namespace xls
