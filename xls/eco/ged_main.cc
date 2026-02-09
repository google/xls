#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/resource.h>

#include "absl/flags/flag.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/eco/ged.h"
#include "xls/eco/ged_cost_functions.h"
#include "xls/eco/gxl_parser.h"
#include "xls/eco/ir_patch_gen.h"
#include "xls/eco/mcs.h"

// TODO(DELETE): Stats collection for paper
struct PaperStats {
  int g1_nodes, g1_edges, g2_nodes, g2_edges;
  double mcs_runtime;
  int mcs_matched_nodes, mcs_matched_edges;
  int mcs_boundary_nodes, mcs_boundary_edges;
  int residual_g1_nodes, residual_g1_edges;
  int residual_g2_nodes, residual_g2_edges;
  double mcs_nodes_pruned_pct, mcs_edges_pruned_pct;
  double ged_runtime;
  int ged_total_cost;
  int ged_node_dels, ged_node_ins, ged_node_subs;
  int ged_edge_dels, ged_edge_ins, ged_edge_subs;
  long ged_rss_peak;
} g_stats = {};

long GetRSSBytes() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss * 1024L;
}

constexpr std::string_view kUsage = R"(ged_main - Compute graph edit distance.

Examples:
  ged_main --before_ir=a.gxl --after_ir=b.gxl
  ged_main -b a.gxl -a b.gxl -v=2 -t=30.0
  ged_main -b a.gxl -a b.gxl -p patch.bin

Alternately, two positional arguments may be provided for the GXL files.)";

ABSL_FLAG(std::string, before_ir, "",
          "Path to the first (golden) GXL graph file. Short alias: -b.");
ABSL_FLAG(std::string, after_ir, "",
          "Path to the second (revised) GXL graph file. Short alias: -a.");
ABSL_FLAG(bool, use_mcs, true,
          "Enable MCS preprocessing before GED. Short alias: -m.");
ABSL_FLAG(int, mcs_cutoff, -1,
          "Stop MCS early when remaining unmatched nodes <= this value; "
          "negative disables (runs MCS to completion). Short alias: -c.");
ABSL_FLAG(int, timeout, -1,
          "Timeout in seconds for the GED search (0 = Return the initial "
          "solution immediately). Short alias: "
          "-t.");
ABSL_FLAG(
    int, verbosity, -1,
    "Verbosity level for logging (-1 disables logging). Short alias: -v.");
ABSL_FLAG(std::string, patch, "",
          "Write a serialized IrPatchProto to this path after GED completes. "
          "Short alias: -p.");
ABSL_FLAG(bool, optimal, false,
          "Require optimal GED solution (Will override timeout and might take "
          "a long time to compute.). Short alias: -o.");

namespace {

std::vector<std::string> CanonicalizeArgs(int argc, char* argv[]) {
  const std::vector<std::pair<std::string, std::string>> kShortToLong = {
      {"-b", "--before_ir"}, {"-a", "--after_ir"},  {"-m", "--use_mcs"},
      {"-c", "--mcs_cutoff"}, {"-t", "--timeout"},   {"-v", "--verbosity"},
      {"-p", "--patch"}, {"-o", "--optimal"},
  };
  std::vector<std::string> canonical;
  canonical.reserve(argc);
  for (int i = 0; i < argc; ++i) canonical.emplace_back(argv[i]);
  for (size_t i = 1; i < canonical.size(); ++i) {
    std::string& arg = canonical[i];
    if (arg == "--no-mcs" || arg == "--no_mcs") {
      arg = "--use_mcs=false";
      continue;
    }
    for (const auto& [short_flag, long_flag] : kShortToLong) {
      if (arg == short_flag) {
        arg = long_flag;
        break;
      }
      std::string prefix = short_flag + "=";
      if (absl::StartsWith(arg, prefix)) {
        arg = absl::StrCat(long_flag, "=", arg.substr(prefix.size()));
        break;
      }
    }
  }
  return canonical;
}

void AnnotateMcsMappings(
    const std::vector<std::pair<std::string, std::string>>& boundary_pairs,
    XLSGraph& graph1, XLSGraph& graph2) {
  for (const auto& [g1_name, g2_name] : boundary_pairs) {
    auto g1_it = graph1.node_name_to_index.find(g1_name);
    auto g2_it = graph2.node_name_to_index.find(g2_name);
    if (g1_it == graph1.node_name_to_index.end() ||
        g2_it == graph2.node_name_to_index.end()) {
      VLOG(1) << "Skip MCS map annotation for pair (" << g1_name << ", "
              << g2_name << ") because one node is missing post-cut";
      continue;
    }
    graph1.nodes[g1_it->second].mcs_map_index = g2_it->second;
    graph2.nodes[g2_it->second].mcs_map_index = g1_it->second;
    VLOG(2) << "Annotated MCS pair: " << g1_name << " <-> " << g2_name;
  }
}

absl::Status RealMain(const std::vector<std::string_view>& positional_args) {
  std::string before_ir = absl::GetFlag(FLAGS_before_ir);
  std::string after_ir = absl::GetFlag(FLAGS_after_ir);
  if (before_ir.empty() && !positional_args.empty()) {
    before_ir = std::string(positional_args[0]);
  }
  if (after_ir.empty()) {
    if (positional_args.size() >= 2) {
      after_ir = std::string(positional_args[1]);
    }
  }
  if (before_ir.empty() || after_ir.empty()) {
    return absl::InvalidArgumentError(
        "Provide two GXL inputs via --before_ir/--after_ir (or aliases) or as "
        "positional arguments.");
  }

  int verbosity = absl::GetFlag(FLAGS_verbosity);
  double timeout = absl::GetFlag(FLAGS_timeout);
  bool use_mcs = absl::GetFlag(FLAGS_use_mcs);
  int mcs_cutoff = absl::GetFlag(FLAGS_mcs_cutoff);
  bool optimal = absl::GetFlag(FLAGS_optimal);
  std::string patch_path = absl::GetFlag(FLAGS_patch);

  if (verbosity >= 0) {
    absl::SetVLogLevel("*", verbosity);
    absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  }

  VLOG(0) << "Starting GED run: file1=" << before_ir << " file2=" << after_ir
          << " use_mcs=" << (use_mcs ? 1 : 0) << " mcs_cutoff=" << mcs_cutoff
          << " timeout=" << timeout << " optimal=" << optimal
          << " verbosity=" << verbosity;

  XLSGraph graph1 = parse_gxl(before_ir);
  XLSGraph graph2 = parse_gxl(after_ir);
  
  // TODO(DELETE): Store initial graph stats
  g_stats.g1_nodes = graph1.nodes.size();
  g_stats.g1_edges = graph1.edges.size();
  g_stats.g2_nodes = graph2.nodes.size();
  g_stats.g2_edges = graph2.edges.size();
  
  VLOG(1) << "Parsed graphs: G1 nodes=" << graph1.nodes.size()
          << " edges=" << graph1.edges.size()
          << " | G2 nodes=" << graph2.nodes.size()
          << " edges=" << graph2.edges.size();

  if (use_mcs) {
    VLOG(0) << "MCS preprocessing enabled";
    auto t_mcs_start = std::chrono::high_resolution_clock::now();
    mcs::MCSResult mcs = mcs::SolveMCS(graph1, graph2, mcs_cutoff);
    auto t_mcs_end = std::chrono::high_resolution_clock::now();
    g_stats.mcs_runtime = std::chrono::duration<double>(t_mcs_end - t_mcs_start).count();
    
    // TODO(DELETE): Count MCS matched nodes/edges
    g_stats.mcs_matched_nodes = mcs.mapping.size();
    int mcs_edges = 0;
    for (const auto& [u, v] : mcs.mapping) {
      for (const auto& eu : graph1.edges) {
        for (const auto& ev : graph2.edges) {
          if ((eu.endpoints.first == u || eu.endpoints.second == u) &&
              (ev.endpoints.first == v || ev.endpoints.second == v)) mcs_edges++;
        }
      }
    }
    g_stats.mcs_matched_edges = mcs_edges;

    auto boundary_nodes = GetBoundaryNodes(mcs, graph1, graph2);
    std::vector<int> boundary_g1_indices;
    std::vector<int> boundary_g2_indices;
    std::vector<std::pair<std::string, std::string>> boundary_names;
    boundary_g1_indices.reserve(boundary_nodes.size());
    boundary_g2_indices.reserve(boundary_nodes.size());
    boundary_names.reserve(boundary_nodes.size());
    for (const auto& [u, v] : boundary_nodes) {
      std::string g1_name = (u >= 0 && u < (int)graph1.nodes.size())
                                ? graph1.nodes[u].name
                                : "<invalid>";
      std::string g2_name = (v >= 0 && v < (int)graph2.nodes.size())
                                ? graph2.nodes[v].name
                                : "<invalid>";
      VLOG(2) << "Pin boundary pair: G1 idx=" << u << " name=" << g1_name
              << " G2 idx=" << v << " name=" << g2_name;
      boundary_g1_indices.push_back(u);
      boundary_g2_indices.push_back(v);
      boundary_names.emplace_back(g1_name, g2_name);
    }
    graph1.PinNodes(boundary_g1_indices);
    graph2.PinNodes(boundary_g2_indices);
    VLOG(1) << "Pinned boundary nodes: " << boundary_nodes.size();
    
    // TODO(DELETE): Store boundary stats
    g_stats.mcs_boundary_nodes = boundary_nodes.size();
    int boundary_edges = 0;
    for (const auto& e : graph1.edges) {
      for (const auto& idx : boundary_g1_indices) {
        if (e.endpoints.first == idx || e.endpoints.second == idx) {
          boundary_edges++;
          break;
        }
      }
    }
    g_stats.mcs_boundary_edges = boundary_edges;

    std::vector<int> to_cut_g1, to_cut_g2;
    for (const auto& [u, v] : mcs.mapping) {
      if (!graph1.nodes[u].pinned) to_cut_g1.push_back(u);
      if (!graph2.nodes[v].pinned) to_cut_g2.push_back(v);
    }
    VLOG(2) << "Cut candidates: G1=" << to_cut_g1.size()
            << " G2=" << to_cut_g2.size();

    graph1.Cut(to_cut_g1);
    graph2.Cut(to_cut_g2);
    graph1.ValidateEdges();
    graph2.ValidateEdges();
    AnnotateMcsMappings(boundary_names, graph1, graph2);
    VLOG(1) << "Post-MCS graph sizes: G1 nodes=" << graph1.nodes.size()
            << " edges=" << graph1.edges.size()
            << " | G2 nodes=" << graph2.nodes.size()
            << " edges=" << graph2.edges.size();
    
    // TODO(DELETE): Store residual graph stats
    g_stats.residual_g1_nodes = graph1.nodes.size();
    g_stats.residual_g1_edges = graph1.edges.size();
    g_stats.residual_g2_nodes = graph2.nodes.size();
    g_stats.residual_g2_edges = graph2.edges.size();
    g_stats.mcs_nodes_pruned_pct = 100.0 * (1.0 - (double)g_stats.residual_g1_nodes / g_stats.g1_nodes);
    g_stats.mcs_edges_pruned_pct = 100.0 * (1.0 - (double)g_stats.residual_g1_edges / g_stats.g1_edges);
  }

  ged::GEDOptions options = CreateUserCosts();
  options.timeout = timeout;
  options.optimal = optimal;

  VLOG(0) << "Running GED solver";
  auto t_ged_start = std::chrono::high_resolution_clock::now();
  ged::GEDResult result = ged::SolveGED(graph1, graph2, options);
  auto t_ged_end = std::chrono::high_resolution_clock::now();
  g_stats.ged_runtime = std::chrono::duration<double>(t_ged_end - t_ged_start).count();
  
  // TODO(DELETE): Store GED stats
  g_stats.ged_total_cost = result.total_cost;
  g_stats.ged_node_dels = result.node_deletions.size();
  g_stats.ged_node_ins = result.node_insertions.size();
  g_stats.ged_node_subs = result.node_substitutions.size();
  g_stats.ged_edge_dels = result.edge_deletions.size();
  g_stats.ged_edge_ins = result.edge_insertions.size();
  g_stats.ged_edge_subs = result.edge_substitutions.size();
  g_stats.ged_rss_peak = GetRSSBytes();
  
  VLOG(0) << "GED finished: total_cost=" << result.total_cost
          << " node_cost=" << result.node_cost
          << " edge_cost=" << result.edge_cost;
  VLOG(0) << "Node ops: "
          << "subs=" << result.node_substitutions.size()
          << " dels=" << result.node_deletions.size()
          << " ins=" << result.node_insertions.size()
          << " | Edge ops: subs=" << result.edge_substitutions.size()
          << " dels=" << result.edge_deletions.size()
          << " ins=" << result.edge_insertions.size();

  if (verbosity >= 2) {
    for (int idx : result.node_deletions) {
      const std::string name = (idx >= 0 && idx < (int)graph1.nodes.size())
                                   ? graph1.nodes[idx].name
                                   : "<invalid>";
      VLOG(2) << "Delete node G1 idx=" << idx << " name=" << name;
    }
    for (int idx : result.node_insertions) {
      const std::string name = (idx >= 0 && idx < (int)graph2.nodes.size())
                                   ? graph2.nodes[idx].name
                                   : "<invalid>";
      VLOG(2) << "Insert node G2 idx=" << idx << " name=" << name;
    }
    for (const auto& [i1, i2] : result.node_substitutions) {
      const std::string n1 = (i1 >= 0 && i1 < (int)graph1.nodes.size())
                                 ? graph1.nodes[i1].name
                                 : "<invalid>";
      const std::string n2 = (i2 >= 0 && i2 < (int)graph2.nodes.size())
                                 ? graph2.nodes[i2].name
                                 : "<invalid>";
      VLOG(2) << "Substitute node G1 idx=" << i1 << " name=" << n1
              << " -> G2 idx=" << i2 << " name=" << n2;
    }
    for (int idx : result.edge_deletions) {
      if (idx >= 0 && idx < (int)graph1.edges.size()) {
        const auto& e = graph1.edges[idx];
        VLOG(2) << "Delete edge G1 idx=" << idx
                << " from=" << graph1.nodes[e.endpoints.first].name
                << " to=" << graph1.nodes[e.endpoints.second].name
                << " attrs=" << e.cost_attributes;
      } else {
        VLOG(2) << "Delete edge G1 idx=" << idx << " attrs=<invalid>";
      }
    }
    for (int idx : result.edge_insertions) {
      if (idx >= 0 && idx < (int)graph2.edges.size()) {
        const auto& e = graph2.edges[idx];
        VLOG(2) << "Insert edge G2 idx=" << idx
                << " from=" << graph2.nodes[e.endpoints.first].name
                << " to=" << graph2.nodes[e.endpoints.second].name
                << " attrs=" << e.cost_attributes;
      } else {
        VLOG(2) << "Insert edge G2 idx=" << idx << " attrs=<invalid>";
      }
    }
    for (const auto& [e1, e2] : result.edge_substitutions) {
      if (e1 >= 0 && e1 < (int)graph1.edges.size() && e2 >= 0 &&
          e2 < (int)graph2.edges.size()) {
        const auto& a = graph1.edges[e1];
        const auto& b = graph2.edges[e2];
        VLOG(2) << "Substitute edge G1 idx=" << e1
                << " from=" << graph1.nodes[a.endpoints.first].name
                << " to=" << graph1.nodes[a.endpoints.second].name
                << " -> G2 idx=" << e2
                << " from=" << graph2.nodes[b.endpoints.first].name
                << " to=" << graph2.nodes[b.endpoints.second].name;
      } else {
        VLOG(2) << "Substitute edge G1 idx=" << e1 << " -> G2 idx=" << e2
                << " attrs=<invalid>";
      }
    }
  }

  if (!patch_path.empty()) {
    xls_eco::IrPatchProto patch_proto =
        GenerateIrPatchProto(graph1, graph2, result);
    std::string serialized;
    if (!patch_proto.SerializeToString(&serialized)) {
      return absl::InternalError("Failed to serialize patch proto.");
    }
    if (serialized.empty()) {
      return absl::InternalError(
          "GED produced an empty patch; refusing to write 0-byte proto.");
    }
    std::filesystem::path out_path(patch_path);
    if (out_path.has_parent_path()) {
      std::error_code ec;
      std::filesystem::create_directories(out_path.parent_path(), ec);
      if (ec) {
        return absl::InternalError(
            absl::StrCat("Failed to create directories for '", patch_path,
                         "': ", ec.message()));
      }
    }
    std::ofstream out(patch_path, std::ios::binary);
    if (!out) {
      return absl::InternalError(
          absl::StrCat("Failed to open '", patch_path, "' for writing."));
    }
    out.write(serialized.data(), serialized.size());
    if (!out) {
      return absl::InternalError(
          absl::StrCat("Failed to write patch proto to '", patch_path, "'."));
    }
    VLOG(0) << "Wrote patch proto: " << patch_path << " (" << serialized.size()
            << " bytes)";
  }

  // TODO(DELETE): Print final stats for paper
  std::cerr << "\n========== PAPER STATS (TODO: DELETE) ==========\n";
  std::cerr << "INITIAL GRAPHS:\n";
  std::cerr << "  G1 (Original): nodes=" << g_stats.g1_nodes << " edges=" << g_stats.g1_edges << "\n";
  std::cerr << "  G2 (Revised):  nodes=" << g_stats.g2_nodes << " edges=" << g_stats.g2_edges << "\n";
  if (use_mcs) {
    std::cerr << "MCS PREPROCESSING:\n";
    std::cerr << "  Runtime: " << g_stats.mcs_runtime << " sec\n";
    std::cerr << "  Matched: nodes=" << g_stats.mcs_matched_nodes << " edges=" << g_stats.mcs_matched_edges << "\n";
    std::cerr << "  Boundary: nodes=" << g_stats.mcs_boundary_nodes << " edges=" << g_stats.mcs_boundary_edges << "\n";
    std::cerr << "RESIDUAL GRAPHS (post-MCS):\n";
    std::cerr << "  G1: nodes=" << g_stats.residual_g1_nodes << " edges=" << g_stats.residual_g1_edges << "\n";
    std::cerr << "  G2: nodes=" << g_stats.residual_g2_nodes << " edges=" << g_stats.residual_g2_edges << "\n";
    std::cerr << "  Pruning: " << std::fixed << std::setprecision(2) << g_stats.mcs_nodes_pruned_pct << "% nodes, "
              << g_stats.mcs_edges_pruned_pct << "% edges\n";
  }
  std::cerr << "GED SOLVING:\n";
  std::cerr << "  Runtime: " << g_stats.ged_runtime << " sec\n";
  std::cerr << "  Total Cost: " << g_stats.ged_total_cost << "\n";
  std::cerr << "  Node Ops: subs=" << g_stats.ged_node_subs << " dels=" << g_stats.ged_node_dels 
            << " ins=" << g_stats.ged_node_ins << "\n";
  std::cerr << "  Edge Ops: subs=" << g_stats.ged_edge_subs << " dels=" << g_stats.ged_edge_dels 
            << " ins=" << g_stats.ged_edge_ins << "\n";
  std::cerr << "  Peak RSS: " << (g_stats.ged_rss_peak / 1048576) << " MB\n";
  std::cerr << "==========================================\n\n";

  return absl::OkStatus();
}

}  // namespace

int main(int argc, char* argv[]) {
  std::vector<std::string> canonical_args = CanonicalizeArgs(argc, argv);
  std::vector<char*> argv_ptrs;
  argv_ptrs.reserve(canonical_args.size());
  for (std::string& arg : canonical_args) {
    argv_ptrs.push_back(arg.data());
  }
  std::vector<std::string_view> positional = xls::InitXls(
      kUsage, static_cast<int>(argv_ptrs.size()), argv_ptrs.data());
  return xls::ExitStatus(RealMain(positional));
}
