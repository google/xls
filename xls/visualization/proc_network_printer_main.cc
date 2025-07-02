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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/state_element.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "re2/re2.h"

const char kUsage[] = R"(
Dump scheduling result to stdout in Graphviz's dot plain text format.
In contrast to sched_printer_main, this binary only shows pipeline stages and
channel operations.

Example invocation:
  proc_network_printer_main <ir_path> <schedule_path>
)";

ABSL_FLAG(bool, collapse_channels, false,
          "Collapse replicated channels (e.g. from arrays or "
          "multiply-instantiated procs) into one node.");
ABSL_FLAG(bool, collapse_functions, false,
          "Collapse replicated procs (e.g. from multiple instantiations) into "
          "one node.");
ABSL_FLAG(
    bool, strip_prefix, false,
    "Strip any prefix before and including '__'. This is path separator "
    "often adds a lot of clutter to the graph without disambiguating much.");

namespace xls {
namespace {

static constexpr LazyRE2 kChannelSuffixRegex{"__\\d+(_\\d+)*$"};
static constexpr LazyRE2 kProcSuffixRegex{"(_\\d+)*_next$"};

struct ChannelOperation {
  std::string function_name;
  std::string op_name;
  int64_t op_stage;
  Op op;  // kSend or kReceive
};

struct StateOperation {
  std::string function_name;
  std::string op_name;
  int64_t op_stage;
  Op op;  // kStateRead or kNext
};

// Get the name to label a function with stripping parts of the XLS name as
// specified by `collapse_functions` and `strip_prefix`.
std::string FunctionName(std::string_view function_name,
                         bool collapse_functions, bool strip_prefix) {
  std::string function_name_str = std::string(function_name);
  if (strip_prefix) {
    size_t pos = function_name_str.rfind("__");
    if (pos != std::string::npos) {
      function_name_str = function_name_str.substr(pos + 2);
    }
  }
  if (collapse_functions) {
    RE2::Replace(&function_name_str, *kProcSuffixRegex, "_next");
  }
  return function_name_str;
}

// For the function with `schedule`, dump the pipeline to `os`.
std::ostream& DumpPipeline(std::ostream& os,
                           const PipelineScheduleProto& schedule,
                           bool collapse_functions, bool strip_prefix) {
  std::string function_name =
      FunctionName(schedule.function(), collapse_functions, strip_prefix);
  auto stage_node_name = [&](int i) {
    return absl::StrCat(function_name, "_stage", i);
  };
  os << "  subgraph cluster_" << function_name << " {\n";
  os << "    label=\"" << function_name << "\";\n";
  os << "    rankdir=LR;\n";
  os << "    style=filled;\n";
  os << "    color=lightgrey;\n";
  os << "    node [style=filled, color=white, shape=square, fixedsize=true, "
        "width=1.25, height=1.25];\n";
  for (int i = 0; i < schedule.stages_size(); ++i) {
    os << "    " << stage_node_name(i) << " [label=\"Stage " << i << "\"];\n";
  }

  os << "  } // subgraph cluster_" << function_name << "\n";

  for (int i = 0; i < schedule.stages_size() - 1; ++i) {
    os << "    " << stage_node_name(i) << " -> " << stage_node_name(i + 1)
       << " [style=invis];\n";
  }

  os << '\n';

  return os;
}

// For the channels in `channel_operations`, dump the channel nodes and edges to
// the appropriate pipeline stages.
std::ostream& DumpChannels(
    std::ostream& os,
    const absl::btree_map<std::string, std::vector<ChannelOperation>>&
        channel_operations,
    bool collapse_functions, bool strip_prefix) {
  auto op_name = [collapse_functions,
                  strip_prefix](const ChannelOperation& operation) {
    return absl::StrCat(
        FunctionName(operation.function_name, collapse_functions, strip_prefix),
        "_stage", operation.op_stage);
  };
  absl::btree_map<std::pair<std::string, std::string>, int64_t> edge_count;

  for (const auto& [channel, ops] : channel_operations) {
    std::string channel_name = channel;
    os << "  " << channel_name << ";\n\n";
    for (const auto& op : ops) {
      std::string lhs, rhs;
      if (op.op == Op::kReceive) {
        lhs = channel_name;
        rhs = op_name(op);
      } else {
        lhs = op_name(op);
        rhs = channel_name;
      }
      ++edge_count[std::make_pair(lhs, rhs)];
    }
  }
  for (const auto& [edge, weight] : edge_count) {
    auto& [lhs, rhs] = edge;
    os << "  " << lhs << " -> " << rhs;
    if (weight > 1) {
      os << " [xlabel=" << weight << "]";
    }
    os << ";\n";
  }
  return os;
}

std::ostream& DumpStateOps(
    std::ostream& os,
    const absl::btree_map<std::string, std::vector<StateOperation>>&
        state_operations,
    bool collapse_functions, bool strip_prefix) {
  auto op_name = [collapse_functions,
                  strip_prefix](const StateOperation& operation) {
    return absl::StrCat(
        FunctionName(operation.function_name, collapse_functions, strip_prefix),
        "_stage", operation.op_stage);
  };
  absl::btree_map<std::pair<std::string, std::string>, int64_t> edge_count;

  for (const auto& [state, ops] : state_operations) {
    std::string state_name = absl::StrCat("\"", state, "\"");
    os << "  " << state_name << ";\n\n";
    for (const auto& op : ops) {
      std::string lhs, rhs;
      if (op.op == Op::kStateRead) {
        lhs = state_name;
        rhs = op_name(op);
      } else {
        lhs = op_name(op);
        rhs = state_name;
      }
      ++edge_count[std::make_pair(lhs, rhs)];
    }
  }
  for (const auto& [edge, weight] : edge_count) {
    auto& [lhs, rhs] = edge;
    os << "  " << lhs << " -> " << rhs;
    if (weight > 1) {
      os << " [xlabel=" << weight << "]";
    }
    os << ";\n";
  }

  return os;
}

std::ostream& DumpToDot(
    std::ostream& os, const PackageScheduleProto& schedule,
    const absl::btree_map<std::string, std::vector<ChannelOperation>>&
        channel_operations,
    const absl::btree_map<std::string, std::vector<StateOperation>>&
        state_operations,
    bool collapse_functions, bool strip_prefix) {
  os << "digraph {\n";
  os << "  concatenate=true;\n";
  os << "  fontname=\"Helvetica,Arial,sans-serif\"\n";
  os << "  node [fontname=\"Helvetica,Arial,sans-serif\"]\n";
  os << "  edge [fontname=\"Helvetica,Arial,sans-serif\"]\n";

  for (const auto& [name, schedule] : schedule.schedules()) {
    DumpPipeline(os, schedule, collapse_functions, strip_prefix);
  }
  DumpChannels(os, channel_operations, collapse_functions, strip_prefix);
  DumpStateOps(os, state_operations, collapse_functions, strip_prefix);

  // closing digraph.
  os << "} // digraph\n";
  return os;
}

absl::Status RealMain(std::string_view ir_path, std::string_view schedule_path,
                      bool collapse_channels, bool collapse_functions,
                      bool strip_prefix) {
  if (ir_path == schedule_path) {
    // Also catches the case where both paths are "-".
    return absl::InvalidArgumentError(
        "IR and schedule paths must be different.");
  }
  if (ir_path == "-") {
    ir_path = "/dev/stdin";
  }
  if (schedule_path == "-") {
    schedule_path = "/dev/stdin";
  }

  XLS_ASSIGN_OR_RETURN(std::string ir_contents, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p,
                       Parser::ParsePackage(ir_contents, ir_path));
  while (!p->blocks().empty()) {
    XLS_RETURN_IF_ERROR(p->RemoveBlock(p->blocks().front().get()));
  }
  PackageScheduleProto schedule_proto;
  XLS_RETURN_IF_ERROR(ParseTextProtoFile(schedule_path, &schedule_proto));
  XLS_ASSIGN_OR_RETURN(PackageSchedule schedules,
                       PackageSchedule::FromProto(p.get(), schedule_proto));

  absl::btree_map<std::string, std::vector<ChannelOperation>>
      channel_operations;
  absl::btree_map<std::string, std::vector<StateOperation>> state_operations;

  // Get the name to label a channel with stripping parts of the XLS name as
  // specified by `collapse_channels` and `strip_prefix`.
  auto get_channel_name = [collapse_channels,
                           strip_prefix](Channel* channel) -> std::string {
    std::string channel_name = std::string(channel->name());
    if (collapse_channels) {
      RE2::Replace(&channel_name, *kChannelSuffixRegex, "");
    }
    if (strip_prefix) {
      size_t pos = channel_name.rfind("__");
      if (pos != std::string::npos) {
        channel_name = channel_name.substr(pos + 2);
      }
    }
    return channel_name;
  };
  for (FunctionBase* f : p->GetFunctionBases()) {
    const PipelineSchedule& schedule = schedules.GetSchedule(f);
    for (Node* n : f->nodes()) {
      if (n->OpIn({Op::kSend, Op::kReceive})) {
        int64_t cycle = schedule.cycle(n);
        XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(n));
        channel_operations[get_channel_name(channel)].push_back(
            {.function_name = f->name(),
             .op_name = n->GetName(),
             .op_stage = cycle,
             .op = n->op()});
      }
      if (n->OpIn({Op::kStateRead, Op::kNext})) {
        int64_t cycle = schedule.cycle(n);
        StateElement* state =
            n->op() == Op::kStateRead
                ? n->As<StateRead>()->state_element()
                : n->As<Next>()->state_read()->As<StateRead>()->state_element();

        state_operations[absl::StrCat(
                             FunctionName(f->name(), collapse_functions,
                                          strip_prefix),
                             "\\n", state->name())]
            .push_back({.function_name = f->name(),
                        .op_name = n->GetName(),
                        .op_stage = cycle,
                        .op = n->op()});
      }
    }
  }

  DumpToDot(std::cout, schedule_proto, channel_operations, state_operations,
            collapse_functions, strip_prefix);

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  QCHECK_EQ(positional_arguments.size(), 2)
      << "Expected 2 positional arguments.";

  return xls::ExitStatus(xls::RealMain(positional_arguments[0],
                                       positional_arguments[1],
                                       absl::GetFlag(FLAGS_collapse_channels),
                                       absl::GetFlag(FLAGS_collapse_functions),
                                       absl::GetFlag(FLAGS_strip_prefix)));
}
