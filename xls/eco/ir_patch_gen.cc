#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xls/eco/ged.h"
#include "xls/eco/graph.h"
#include "xls/eco/ir_patch.pb.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_value.pb.h"

namespace {

std::optional<xls::TypeProto> ParseTypeProto(absl::string_view type_str) {
  if (type_str.empty()) {
    return std::nullopt;
  }
  static auto* package = new xls::Package("ir_patch_gen_types");
  absl::StatusOr<xls::Type*> type_or =
      xls::Parser::ParseType(type_str, package);
  if (!type_or.ok()) {
    return std::nullopt;
  }
  return type_or.value()->ToProto();
}

std::unordered_map<std::string, std::string> ParseFields(
    const std::string& input) {
  std::unordered_map<std::string, std::string> result;
  std::stringstream ss(input);
  std::string token;

  // Each token looks like: key=value
  while (std::getline(ss, token, '|')) {
    auto pos = token.find('=');
    if (pos == std::string::npos) {
      continue;  // ignore malformed entries
    }
    std::string key = token.substr(0, pos);
    std::string value = token.substr(pos + 1);
    result[key] = value;
  }

  return result;
}

std::optional<int64_t> ParseInt64(absl::string_view text) {
  int64_t value = 0;
  return absl::SimpleAtoi(text, &value) ? std::optional<int64_t>(value)
                                        : std::nullopt;
}

std::optional<bool> ParseBool(absl::string_view text) {
  if (text.empty()) {
    return std::nullopt;
  }
  std::string lowered = std::string(absl::AsciiStrToLower(text));
  if (lowered == "true" || lowered == "1") {
    return true;
  }
  if (lowered == "false" || lowered == "0") {
    return false;
  }
  return std::nullopt;
}

std::optional<xls::ValueProto> ParseValueProto(absl::string_view value_str,
                                               absl::string_view type_str) {
  if (value_str.empty() || type_str.empty()) {
    return std::nullopt;
  }
  std::string typed = absl::StrCat(type_str, ":", value_str);
  absl::StatusOr<xls::Value> value_or = xls::Parser::ParseTypedValue(typed);
  if (!value_or.ok()) {
    return std::nullopt;
  }
  absl::StatusOr<xls::ValueProto> proto = value_or.value().AsProto();
  if (!proto.ok()) {
    return std::nullopt;
  }
  return *proto;
}

std::string ExtractValueToken(absl::string_view raw_value) {
  size_t data_pos = raw_value.find("data=");
  if (data_pos == absl::string_view::npos) {
    return std::string(absl::StripAsciiWhitespace(raw_value));
  }
  data_pos += 5;
  size_t end = raw_value.find_first_of(")|", data_pos);
  if (end == absl::string_view::npos) {
    end = raw_value.size();
  }
  return std::string(
      absl::StripAsciiWhitespace(raw_value.substr(data_pos, end - data_pos)));
}

std::vector<std::string> SplitTopLevel(absl::string_view input) {
  std::vector<std::string> parts;
  if (input.empty()) {
    return parts;
  }
  std::string current;
  int depth = 0;
  for (char c : input) {
    if (c == '(' || c == '[') {
      depth++;
    } else if (c == ')' || c == ']') {
      depth = std::max(0, depth - 1);
    }
    if (c == ',' && depth == 0) {
      parts.push_back(std::string(absl::StripAsciiWhitespace(current)));
      current.clear();
      continue;
    }
    current.push_back(c);
  }
  if (!current.empty()) {
    parts.push_back(std::string(absl::StripAsciiWhitespace(current)));
  }
  return parts;
}

void PopulateUniqueArgs(
    const std::unordered_map<std::string, std::string>& attrs,
    absl::string_view op_name, xls_eco::NodeProto* node_proto) {
  if (node_proto == nullptr) {
    return;
  }
  if (op_name == "literal") {
    auto value_str_it = attrs.find("value_str");
    if (value_str_it == attrs.end()) {
      value_str_it = attrs.find("value");
    }
    auto dtype_str_it = attrs.find("dtype_str");
    if (value_str_it != attrs.end() && dtype_str_it != attrs.end()) {
      if (auto value_proto =
              ParseValueProto(value_str_it->second, dtype_str_it->second);
          value_proto.has_value()) {
        node_proto->add_unique_args()->mutable_value()->CopyFrom(*value_proto);
      }
    }
  } else if (op_name == "bit_slice") {
    auto start_it = attrs.find("start");
    if (start_it != attrs.end()) {
      if (auto start_value = ParseInt64(start_it->second);
          start_value.has_value()) {
        node_proto->add_unique_args()->set_start(
            static_cast<uint64_t>(*start_value));
      }
    }
  } else if (op_name == "tuple_index") {
    auto index_it = attrs.find("index");
    if (index_it != attrs.end()) {
      if (auto index_value = ParseInt64(index_it->second);
          index_value.has_value()) {
        node_proto->add_unique_args()->set_index(
            static_cast<uint64_t>(*index_value));
      }
    }
  } else if (op_name == "array_index" || op_name == "array_update") {
    auto assumed_it = attrs.find("assumed_in_bounds");
    if (assumed_it != attrs.end()) {
      if (auto assumed = ParseBool(assumed_it->second); assumed.has_value()) {
        node_proto->add_unique_args()->set_assumed_in_bounds(*assumed);
      }
    }
  } else if (op_name == "one_hot") {
    auto lsb_it = attrs.find("lsb_prio");
    if (lsb_it != attrs.end()) {
      if (auto lsb = ParseBool(lsb_it->second); lsb.has_value()) {
        node_proto->add_unique_args()->set_lsb_prio(*lsb);
      }
    }
  } else if (op_name == "sign_ext") {
    auto count_it = attrs.find("new_bit_count");
    if (count_it != attrs.end()) {
      if (auto count = ParseInt64(count_it->second); count.has_value()) {
        node_proto->add_unique_args()->set_new_bit_count(
            static_cast<uint64_t>(*count));
      }
    }
  } else if (op_name == "sel" || op_name == "priority_sel") {
    auto default_it = attrs.find("has_default");
    if (default_it == attrs.end()) {
      default_it = attrs.find("has_default_value");
    }
    if (default_it != attrs.end()) {
      if (auto has_default = ParseBool(default_it->second);
          has_default.has_value()) {
        node_proto->add_unique_args()->set_has_default_value(*has_default);
      }
    }
  } else if (op_name == "receive") {
    auto channel_it = attrs.find("channel");
    if (channel_it != attrs.end()) {
      node_proto->add_unique_args()->set_channel(channel_it->second);
    }
    auto blocking_it = attrs.find("blocking");
    if (blocking_it != attrs.end()) {
      if (auto blocking = ParseBool(blocking_it->second);
          blocking.has_value()) {
        node_proto->add_unique_args()->set_blocking(*blocking);
      }
    }
  } else if (op_name == "send") {
    auto channel_it = attrs.find("channel");
    if (channel_it != attrs.end()) {
      node_proto->add_unique_args()->set_channel(channel_it->second);
    }
  } else if (op_name == "state_read") {
    auto index_it = attrs.find("index");
    if (index_it != attrs.end()) {
      if (auto index_value = ParseInt64(index_it->second);
          index_value.has_value()) {
        node_proto->add_unique_args()->set_index(
            static_cast<uint64_t>(*index_value));
      }
    }
    auto state_element_it = attrs.find("state_element");
    if (state_element_it != attrs.end()) {
      node_proto->add_unique_args()->set_state_element(
          state_element_it->second);
    }
    auto init_it = attrs.find("init");
    auto dtype_it = attrs.find("dtype_str");
    if (init_it != attrs.end() && dtype_it != attrs.end()) {
      std::string value_token = ExtractValueToken(init_it->second);
      if (auto init_proto = ParseValueProto(value_token, dtype_it->second);
          init_proto.has_value()) {
        node_proto->add_unique_args()->mutable_init()->CopyFrom(*init_proto);
      }
    }
  }
    }

void ExportNodeProto(const XLSNode& node, xls_eco::NodeProto* node_proto) {
  auto attribs = ParseFields(node.cost_attributes);
  node_proto->set_name(node.name);
  auto data_type_it = attribs.find("data_type");
  if (data_type_it == attribs.end()) {
    data_type_it = attribs.find("dtype_str");
  }
  if (data_type_it != attribs.end()) {
    if (auto type_proto = ParseTypeProto(data_type_it->second);
        type_proto.has_value()) {
      node_proto->mutable_data_type()->CopyFrom(*type_proto);
    }
  }
  auto op_it = attribs.find("op");
  if (op_it != attribs.end()) {
    node_proto->set_op(op_it->second);
  }

  auto add_operand_type = [&](absl::string_view type_str) {
    if (auto operand_type_proto = ParseTypeProto(type_str);
        operand_type_proto.has_value()) {
      node_proto->add_operand_data_types()->CopyFrom(*operand_type_proto);
    }
  };

  auto single_operand_it = attribs.find("operand_data_type");
  if (single_operand_it == attribs.end()) {
    single_operand_it = attribs.find("operand_dtype_str");
  }
  if (single_operand_it != attribs.end()) {
    add_operand_type(single_operand_it->second);
  }

  auto operand_list_it = attribs.find("operand_data_types");
  if (operand_list_it == attribs.end()) {
    operand_list_it = attribs.find("operand_dtype_strs");
  }
  if (operand_list_it != attribs.end()) {
    for (const std::string& operand_type_str :
         SplitTopLevel(operand_list_it->second)) {
      add_operand_type(operand_type_str);
    }
  }

  PopulateUniqueArgs(attribs, node_proto->op(), node_proto);
}

}  // namespace

xls_eco::IrPatchProto GenerateIrPatchProto(const XLSGraph& original_graph,
                                           const XLSGraph& modified_graph,
                                           const ged::GEDResult& ged_result) {
  xls_eco::IrPatchProto patch_proto;
  uint32_t id = 0;

  for (const auto& node_sub : ged_result.node_substitutions) {
    int original_node_id = node_sub.first;
    int modified_node_id = node_sub.second;
    auto edit_path_proto = patch_proto.add_edit_paths();
    edit_path_proto->set_id(id);
    edit_path_proto->set_operation(xls_eco::Operation::UPDATE);
    auto node_edit_path = edit_path_proto->mutable_node_edit_path();
    ExportNodeProto(original_graph.nodes[original_node_id],
                    node_edit_path->mutable_node());
    ExportNodeProto(modified_graph.nodes[modified_node_id],
                    node_edit_path->mutable_updated_node());
    id++;
  }
  for (int original_node_id : ged_result.node_deletions) {
    auto edit_path_proto = patch_proto.add_edit_paths();
    edit_path_proto->set_id(id);
    edit_path_proto->set_operation(xls_eco::Operation::DELETE);
    auto node_edit_path = edit_path_proto->mutable_node_edit_path();
    ExportNodeProto(original_graph.nodes[original_node_id],
                    node_edit_path->mutable_node());
    node_edit_path->mutable_node()->set_name(
        original_graph.nodes[original_node_id].name);
    id++;
  }
  for (int modified_node_id : ged_result.node_insertions) {
    auto edit_path_proto = patch_proto.add_edit_paths();
    edit_path_proto->set_id(id);
    edit_path_proto->set_operation(xls_eco::Operation::INSERT);
    auto node_edit_path = edit_path_proto->mutable_node_edit_path();
    ExportNodeProto(modified_graph.nodes[modified_node_id],
                    node_edit_path->mutable_node());
    node_edit_path->mutable_node()->set_name(
        modified_graph.nodes[modified_node_id].name);
    id++;
  }
  for (const auto& edge_sub : ged_result.edge_substitutions) {
    int original_edge_id = edge_sub.first;
    int modified_edge_id = edge_sub.second;
    auto edit_path_proto = patch_proto.add_edit_paths();
    edit_path_proto->set_id(id);
    edit_path_proto->set_operation(xls_eco::Operation::UPDATE);
    auto edge_edit_path = edit_path_proto->mutable_edge_edit_path();
    const XLSEdge& original_edge = original_graph.edges[original_edge_id];
    const XLSEdge& modified_edge = modified_graph.edges[modified_edge_id];
    auto edge_proto = edge_edit_path->mutable_edge();
    edge_proto->set_from_node(
        original_graph.nodes[original_edge.endpoints.first].name);
    edge_proto->set_to_node(
        original_graph.nodes[original_edge.endpoints.second].name);
    edge_proto->set_index(original_edge.index);
    auto updated_edge_proto = edge_edit_path->mutable_updated_edge();
    updated_edge_proto->set_from_node(
        modified_graph.nodes[modified_edge.endpoints.first].name);
    updated_edge_proto->set_to_node(
        modified_graph.nodes[modified_edge.endpoints.second].name);
    updated_edge_proto->set_index(modified_edge.index);
    id++;
  }
  for (int original_edge_id : ged_result.edge_deletions) {
    auto edit_path_proto = patch_proto.add_edit_paths();
    edit_path_proto->set_id(id);
    edit_path_proto->set_operation(xls_eco::Operation::DELETE);
    auto edge_edit_path = edit_path_proto->mutable_edge_edit_path();
    const XLSEdge& original_edge = original_graph.edges[original_edge_id];
    auto edge_proto = edge_edit_path->mutable_edge();
    edge_proto->set_from_node(
        original_graph.nodes[original_edge.endpoints.first].name);
    edge_proto->set_to_node(
        original_graph.nodes[original_edge.endpoints.second].name);
    edge_proto->set_index(original_edge.index);
    id++;
  }
  for (int modified_edge_id : ged_result.edge_insertions) {
    auto edit_path_proto = patch_proto.add_edit_paths();
    edit_path_proto->set_id(id);
    edit_path_proto->set_operation(xls_eco::Operation::INSERT);
    auto edge_edit_path = edit_path_proto->mutable_edge_edit_path();
    const XLSEdge& modified_edge = modified_graph.edges[modified_edge_id];
    auto edge_proto = edge_edit_path->mutable_edge();
    edge_proto->set_from_node(
        modified_graph.nodes[modified_edge.endpoints.first].name);
    edge_proto->set_to_node(
        modified_graph.nodes[modified_edge.endpoints.second].name);
    edge_proto->set_index(modified_edge.index);
    id++;
  }

  if (modified_graph.return_node_name.has_value() &&
      modified_graph.return_node_name != original_graph.return_node_name) {
    auto it =
        modified_graph.node_name_to_index.find(*modified_graph.return_node_name);
    if (it != modified_graph.node_name_to_index.end()) {
      xls_eco::NodeProto* ret_proto = patch_proto.mutable_return_node();
      ret_proto->set_name(*modified_graph.return_node_name);
      ExportNodeProto(modified_graph.nodes[it->second], ret_proto);
    }
  }

  return patch_proto;
}
