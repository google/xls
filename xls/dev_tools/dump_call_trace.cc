#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/ir/evaluator_result.pb.h"
#include "xls/ir/value.h"

static constexpr std::string_view kUsage = R"(
Reads a EvaluatorResultsProto and prints the trace call messages
in a nicely formatted form. Usage:

   interpreter_main --trace_calls -- output_results_proto=<PROTOFILE>
   dump_call_trace --input=<PROTOFILE>
)";

ABSL_FLAG(std::string, input, "", "Path to EvaluatorResultsProto file.");
ABSL_FLAG(std::string, function, "",
          "If set, only emit calls nested within this function; output is "
          "unindented relative to this function");
ABSL_FLAG(bool, sort_calls, false,
          "If set, sort sibling calls by source location (file, line, column) "
          "within each function scope");

namespace xls {
namespace {

constexpr int kIndentSpaces = 2;
static void PrintIndent(int spaces) { printf("%*s", spaces, ""); }

static void PrettyPrintValue(const Value& v, int indent) {
  if (v.IsTuple()) {
    PrintIndent(indent);
    printf("(\n");
    for (const Value& e : v.elements()) {
      PrettyPrintValue(e, indent + kIndentSpaces);
    }
    PrintIndent(indent);
    printf(")\n");
    return;
  }
  if (v.IsArray()) {
    PrintIndent(indent);
    printf("[\n");
    for (const Value& e : v.elements()) {
      PrettyPrintValue(e, indent + kIndentSpaces);
    }
    PrintIndent(indent);
    printf("]\n");
    return;
  }
  PrintIndent(indent);
  printf("%s\n", v.ToString().c_str());
}

static std::string LocationPrefix(const TraceMessageProto& tm) {
  if (!tm.has_location()) {
    return std::string("<unknown>: ");
  }
  const SourceLocationProto& loc = tm.location();
  std::string f =
      loc.has_filename() ? loc.filename() : std::string("<unknown>");
  std::string l =
      loc.has_line() ? std::to_string(loc.line()) : std::string("?");
  std::string c =
      loc.has_column() ? std::to_string(loc.column()) : std::string("?");
  return absl::StrFormat("%s:%s:%s: ", f, l, c);
}

struct CallNode {
  const TraceMessageProto* trace_msg;
  const TraceMessageProto* return_msg = nullptr;
  int64_t depth;
  std::vector<std::unique_ptr<CallNode>> children;
};

static void SortChildrenByLocation(CallNode* node) {
  auto location_key = [](const TraceMessageProto& tm) {
    int64_t line_no = std::numeric_limits<int64_t>::max();
    int64_t col_no = std::numeric_limits<int64_t>::max();
    if (tm.has_location()) {
      const SourceLocationProto& loc = tm.location();
      if (loc.has_line()) {
        line_no = loc.line();
      }
      if (loc.has_column()) {
        col_no = loc.column();
      }
    }
    return std::make_tuple(line_no, col_no);
  };

  std::stable_sort(node->children.begin(), node->children.end(),
                   [&](const std::unique_ptr<CallNode>& a,
                       const std::unique_ptr<CallNode>& b) {
                     return location_key(*a->trace_msg) <
                            location_key(*b->trace_msg);
                   });
  for (const auto& child : node->children) {
    SortChildrenByLocation(child.get());
  }
}

static void PrintCallPrettyAtDepth(const CallNode& node, int64_t depth) {
  const TraceMessageProto& tm = *node.trace_msg;
  const TraceCallProto& call = tm.call();
  std::string indent(depth * kIndentSpaces, ' ');
  std::string loc_prefix = LocationPrefix(tm);
  const std::string& fn = call.function_name();
  printf("%s%s%s(\n", indent.c_str(), loc_prefix.c_str(), fn.c_str());
  std::string arg_indent = indent + std::string(kIndentSpaces, ' ');
  for (const ValueProto& vp : call.args()) {
    absl::StatusOr<Value> v = Value::FromProto(vp);
    if (!v.ok()) {
      printf("%s<invalid value: %s>\n", arg_indent.c_str(),
             v.status().ToString().c_str());
      continue;
    }
    PrettyPrintValue(*v, /*indent=*/static_cast<int>(arg_indent.size()));
  }
  printf("%s)\n", indent.c_str());
}

static void PrintReturnPrettyAtDepth(const CallNode& node, int64_t depth) {
  if (node.return_msg == nullptr || !node.return_msg->has_call_return()) {
    return;
  }
  const TraceMessageProto& tm = *node.trace_msg;
  const TraceCallProto& call = tm.call();
  std::string indent(depth * kIndentSpaces, ' ');
  const std::string& fn = call.function_name();
  const TraceCallReturnProto& cr = node.return_msg->call_return();
  absl::StatusOr<Value> ret_v = Value::FromProto(cr.return_value());
  if (!ret_v.ok()) {
    printf("%s%s(...) => <invalid return: %s>\n", indent.c_str(), fn.c_str(),
           ret_v.status().ToString().c_str());
    return;
  }
  std::string ret_str = ret_v->ToString();
  printf("%s%s(...) => %s\n", indent.c_str(), fn.c_str(), ret_str.c_str());
}

static void PrintSubtree(const CallNode* node, int depth_adjustment) {
  PrintCallPrettyAtDepth(*node, node->depth + depth_adjustment);
  int child_adjustment = depth_adjustment;
  for (const auto& child : node->children) {
    PrintSubtree(child.get(), child_adjustment);
  }
  PrintReturnPrettyAtDepth(*node, node->depth + depth_adjustment);
}

// Only one subtree printer is needed; we control indentation via
// depth_adjustment.

static absl::Status RealMain(std::string_view input_path,
                             const std::string& filter_function,
                             bool sort_calls) {
  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(input_path));
  EvaluatorResultsProto results;
  QCHECK(google::protobuf::TextFormat::ParseFromString(contents, &results))
      << "Failed to parse EvaluatorResultsProto textproto from: " << input_path;

  for (const EvaluatorResultProto& result : results.results()) {
    const EvaluatorEventsProto& events = result.events();

    // Build a forest of call trees using call_depth nesting.
    std::vector<std::unique_ptr<CallNode>> roots;
    std::vector<CallNode*> stack;  // Stack of active calls by depth.
    for (const TraceMessageProto& tm : events.trace_msgs()) {
      if (tm.type_case() == TraceMessageProto::kCall) {
        const TraceCallProto& call = tm.call();
        int64_t depth = call.call_depth();
        while (!stack.empty() && static_cast<int64_t>(stack.size()) > depth) {
          stack.pop_back();
        }
        auto node = std::make_unique<CallNode>();
        node->trace_msg = &tm;
        node->depth = depth;
        CallNode* node_raw = node.get();
        if (stack.empty()) {
          roots.push_back(std::move(node));
        } else {
          stack.back()->children.push_back(std::move(node));
        }
        stack.push_back(node_raw);
      } else if (tm.type_case() == TraceMessageProto::kCallReturn) {
        const TraceCallReturnProto& cr = tm.call_return();
        int64_t depth = cr.call_depth();
        // When call_depth is N, there should be N+1 nodes on the stack, with
        // the top corresponding to this call.
        while (!stack.empty() &&
               static_cast<int64_t>(stack.size()) > depth + 1) {
          stack.pop_back();
        }
        if (!stack.empty() && static_cast<int64_t>(stack.size()) == depth + 1) {
          stack.back()->return_msg = &tm;
        }
      }
    }

    // Optionally sort siblings by source location within each function scope.
    if (sort_calls) {
      for (const auto& root : roots) {
        SortChildrenByLocation(root.get());
      }
    }

    if (filter_function.empty()) {
      for (const auto& root : roots) {
        PrintSubtree(root.get(), /*depth_adjustment=*/0);
      }
    } else {
      // Print each occurrence of the filter function with normalize
      // indentation.
      std::vector<const CallNode*> stack_nodes;
      stack_nodes.reserve(roots.size());
      // DFS through the forest.
      std::vector<const CallNode*> dfs;
      dfs.reserve(roots.size());
      for (const auto& root : roots) {
        dfs.push_back(root.get());
      }
      while (!dfs.empty()) {
        const CallNode* node = dfs.back();
        dfs.pop_back();
        if (node->trace_msg->call().function_name() == filter_function) {
          PrintSubtree(node,
                       /*depth_adjustment=*/-static_cast<int>(node->depth));
        }
        for (auto it = node->children.rbegin(); it != node->children.rend();
             ++it) {
          dfs.push_back(it->get());
        }
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(kUsage, argc, argv);
  std::string input = absl::GetFlag(FLAGS_input);
  QCHECK(!input.empty()) << "--input must be specified";
  std::string function = absl::GetFlag(FLAGS_function);
  bool sort_calls = absl::GetFlag(FLAGS_sort_calls);
  return xls::ExitStatus(xls::RealMain(input, function, sort_calls));
}
