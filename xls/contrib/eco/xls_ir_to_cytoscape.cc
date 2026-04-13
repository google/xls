// Copyright 2025 The XLS Authors
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

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "google/protobuf/util/json_util.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/visualization/ir_viz/node_attribute_visitor.h"

constexpr std::string_view kUsage = R"usage(
  Usage: xls_ir_to_cytoscape <input_ir_path>
)usage";

namespace xls {
namespace {
struct QuotingFormatter {
  void operator()(std::string* out, std::string_view str) const {
    absl::StrAppend(out, "\"", str, "\"");
  }
};
struct NodeDtypeFormatter {
  void operator()(std::string* out, const Node* node) const {
    absl::StrAppend(out, "\"", node->GetType()->ToString(), "\"");
  }
};

std::string Quote(std::string_view str) {
  return absl::StrCat("\"", str, "\"");
}

std::string_view ColorForNode(Node* node) {
  switch (node->op()) {
    case Op::kParam:
      return "#ffffff";
    case Op::kLiteral:
      return "#90ee90";
    default:
      return "#ffcccb";
  }
}

// Note that this includes fields the way networkx's cytoscape reader expects
// them to be. Networkx seems to make assumptions about some fields being
// present that cytoscape doesn't require.
class CytoscapeNodeVisitor final : public DfsVisitorWithDefault {
 public:
  CytoscapeNodeVisitor(std::ostream& os, std::string_view indent)
      : os_(os), indent_(indent) {}

  absl::Status DefaultHandler(Node* node) final {
    std::vector<std::pair<std::string, std::string>> attributes{
        {"id", absl::StrCat(node->id())},
        {"name", Quote(node->GetName())},
        {"value", Quote(node->GetName())},
        {"op", Quote(OpToString(node->op()))},
        {"data_type", Quote(node->GetType()->ToString())},
        {"color", Quote(ColorForNode(node))},
    };

    XLS_ASSIGN_OR_RETURN(auto cost_attributes, CostAttributes(node));

    if (!cost_attributes.empty()) {
      attributes.push_back(
          {"cost_attributes",
           absl::StrCat(
               "{",
               absl::StrJoin(cost_attributes, ", ",
                             absl::PairFormatter(QuotingFormatter(), ": ",
                                                 absl::AlphaNumFormatter())),
               "}")});
    }

    PrintNewline();
    os_ << absl::StrCat(
        "{\"data\": {",
        absl::StrJoin(attributes, ", ",
                      absl::PairFormatter(QuotingFormatter(), ": ",
                                          absl::AlphaNumFormatter())),
        "}}");
    return absl::OkStatus();
  }

 private:
  static absl::StatusOr<std::vector<std::pair<std::string, std::string>>>
  CostAttributes(Node* node) {
    std::vector<std::pair<std::string, std::string>> attributes{
        {"op", Quote(OpToString(node->op()))},
        {"dtype_str", Quote(node->GetType()->ToString())},
    };

    if (node->operand_count() > 0) {
      attributes.push_back(
          {"operand_dtype_strs",
           absl::StrCat(
               "[", absl::StrJoin(node->operands(), ", ", NodeDtypeFormatter()),
               "]")});
    }

    xls::AttributeVisitor visitor;
    XLS_RETURN_IF_ERROR(node->VisitSingleNode(&visitor));
    const viz::NodeAttributes& node_attributes = visitor.attributes();
    google::protobuf::util::JsonPrintOptions print_options;
    print_options.add_whitespace = false;
    print_options.preserve_proto_field_names = true;

    std::string json_node_attributes;
    XLS_RETURN_IF_ERROR(google::protobuf::util::MessageToJsonString(
        node_attributes, &json_node_attributes, print_options));
    if (!json_node_attributes.empty() && json_node_attributes != "{}") {
      attributes.push_back({"node_attributes", json_node_attributes});
    }
    return attributes;
  }

  void PrintNewline() {
    if (first_done_) {
      os_ << ",\n";
    }
    os_ << indent_;
    first_done_ = true;
  }
  std::ostream& os_;
  std::string_view indent_;
  bool first_done_ = false;
};

class CytoscapeEdgeVisitor : public DfsVisitorWithDefault {
 public:
  CytoscapeEdgeVisitor(std::ostream& os, std::string_view indent)
      : os_(os), indent_(indent) {}

  absl::Status DefaultHandler(Node* node) final {
    for (int64_t index = 0; index < node->operand_count(); ++index) {
      Node* operand = node->operand(index);

      XLS_ASSIGN_OR_RETURN(auto cost_attributes, CostAttributes(node, operand));

      std::vector<std::pair<std::string, std::string>> attributes{
          {"source", Quote(node->GetName())},
          {"target", Quote(operand->GetName())},
      };
      if (OpIsCommutative(node->op())) {
        attributes.push_back({"index", absl::StrCat(index)});
        cost_attributes.push_back({"index", absl::StrCat(index)});
      }
      if (!cost_attributes.empty()) {
        attributes.push_back(
            {"cost_attributes",
             absl::StrCat(
                 "{",
                 absl::StrJoin(cost_attributes, ", ",
                               absl::PairFormatter(QuotingFormatter(), ": ",
                                                   absl::AlphaNumFormatter())),
                 "}")});
      }

      PrintNewline();
      os_ << absl::StrCat(
          "{\"data\": {",
          absl::StrJoin(attributes, ", ",
                        absl::PairFormatter(QuotingFormatter(), ": ",
                                            absl::AlphaNumFormatter())),
          "}}");
    }

    return absl::OkStatus();
  }

 private:
  static absl::StatusOr<std::vector<std::pair<std::string, std::string>>>
  CostAttributes(Node* source, Node* sink) {
    std::vector<std::pair<std::string, std::string>> attributes{
        {"source_data_type", Quote(source->GetType()->ToString())},
        {"sink_data_type", Quote(sink->GetType()->ToString())},
    };
    return attributes;
  }

  void PrintNewline() {
    if (first_done_) {
      os_ << ",\n";
    }
    os_ << indent_;
    first_done_ = true;
  }
  std::ostream& os_;
  const std::string_view indent_;
  bool first_done_ = false;
};

absl::Status RealMain(std::string_view ir_path) {
  if (ir_path == "-") {
    ir_path = "/dev/stdin";
  }

  XLS_ASSIGN_OR_RETURN(std::string ir, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir, ir_path));
  std::optional<FunctionBase*> top_optional = package->GetTop();
  XLS_RET_CHECK(top_optional.has_value());
  FunctionBase* top = *top_optional;

  std::cout << R"({
  "data": [],
  "directed": true,
  "multigraph": true,
  "elements": {
    "nodes": [
)";

  CytoscapeNodeVisitor node_visitor(std::cout, "      ");
  CytoscapeEdgeVisitor edge_visitor(std::cout, "      ");
  XLS_RETURN_IF_ERROR(top->Accept(&node_visitor));
  std::cout << R"(
    ],
    "edges": [
)";
  XLS_RETURN_IF_ERROR(top->Accept(&edge_visitor));
  std::cout << R"(
    ]
  }
})";

  return absl::OkStatus();
}
}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_args =
      xls::InitXls(kUsage, argc, argv);

  LOG_IF(QFATAL, (positional_args.size() != 1))
      << "Expected a single positional argument (the IR path)";

  std::string_view ir_path = positional_args[0];

  return xls::ExitStatus(xls::RealMain(ir_path));
}
