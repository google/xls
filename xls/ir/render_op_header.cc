// Renders the `op.h` header from our programmatic description tables.

#include "xls/ir/render_op_header.h"

#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "xls/ir/op_specification.h"

namespace xls {

std::string RenderEnumClassOp() {
  std::vector<std::string> lines = {
      "// Enumerates the operators for nodes in the IR.", "enum class Op {"};
  for (const Op& op : GetOpsSingleton()) {
    lines.push_back(absl::StrFormat("  %s,", op.enum_name));
  }
  lines.push_back("};");
  return absl::StrJoin(lines, "\n");
}

std::string RenderFnAllOps() {
  std::vector<std::string> lines = {
      "inline std::vector<Op> AllOps() {",
      "  return {",
  };
  for (const Op& op : GetOpsSingleton()) {
    lines.push_back(absl::StrFormat("    Op::%s,", op.enum_name));
  }

  lines.push_back("  };");
  lines.push_back("}");
  return absl::StrJoin(lines, "\n");
}

static const std::string_view kHeaderTemplate = R"(#ifndef XLS_IR_OP_H_
#define XLS_IR_OP_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/types/span.h"
#include "absl/status/statusor.h"
#include "xls/ir/op.pb.h"

namespace xls {

{ENUM_CLASS_OP}

{FN_ALL_OPS}

{OP_LIMIT}

// Converts an OpProto into an Op.
Op FromOpProto(OpProto op_proto);

// Converts an Op into an OpProto.
OpProto ToOpProto(Op op);

// Converts the "op" enumeration to a human readable string.
std::string OpToString(Op op);

// Converts a human readable op string into the "op" enumeration.
absl::StatusOr<Op> StringToOp(std::string_view op_str);

// Returns whether the operation is a compare operation.
bool OpIsCompare(Op op);

// Returns whether the operation is associative, eg., kAdd, or kOr.
bool OpIsAssociative(Op op);

// Returns whether the operation is commutative, eg., kAdd, or kEq.
bool OpIsCommutative(Op op);

// Returns whether the operation is a bitwise logical op, eg., kAnd or kOr.
bool OpIsBitWise(Op op);

// Returns whether the operation has side effects, eg., kAssert, kSend.
bool OpIsSideEffecting(Op op);

// Returns the delay of this operation in picoseconds.
// TODO(meheff): This value should be plugable and be derived from other aspects
// of Node, not just the op.
int64_t OpDelayInPs(Op op);

// Forward declare all `Op` classes (subclasses of `Node`).
{FORWARD_DECLS}

// Returns whether the given Op has the OpT node subclass.
class Node;

template<typename OpT>
bool IsOpClass(Op op) {
  static_assert(std::is_base_of<Node, OpT>::value, "OpT is not a Node subclass");
  return false;
}

{IS_OP_CLASS_SPECIALIZATIONS}

// Streams the string for "op" to the given output stream.
std::ostream& operator<<(std::ostream& os, Op op);

}  // namespace xls

#endif  // XLS_IR_OP_H_
)";

static std::string RenderForwardDecls() {
  std::vector<std::string> lines;
  for (const auto& [_name, op_class] : GetOpClassKindsSingleton()) {
    lines.push_back(absl::StrFormat("class %s;", op_class.name()));
  }
  return absl::StrJoin(lines, "\n");
}

static std::string RenderIsOpClassSpecializations() {
  std::vector<std::string> lines;
  for (const auto& [_name, op_class] : GetOpClassKindsSingleton()) {
    lines.push_back(absl::StrFormat("template<> bool IsOpClass<%s>(Op op);",
                                    op_class.name()));
  }
  return absl::StrJoin(lines, "\n");
}

std::string RenderOpHeader() {
  const std::string op_limit = absl::StrFormat(
      "const int64_t kOpLimit = static_cast<int64_t>(Op::%s)+1;",
      GetOpsSingleton().back().enum_name);

  return absl::StrReplaceAll(
      kHeaderTemplate,
      {
          {"{ENUM_CLASS_OP}", RenderEnumClassOp()},
          {"{FN_ALL_OPS}", RenderFnAllOps()},
          {"{OP_LIMIT}", op_limit},
          {"{FORWARD_DECLS}", RenderForwardDecls()},
          {"{IS_OP_CLASS_SPECIALIZATIONS}", RenderIsOpClassSpecializations()},
      });
}

}  // namespace xls
