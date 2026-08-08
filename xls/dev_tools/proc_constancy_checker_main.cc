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

#include <unistd.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/proc_constancy_checker.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"
#include "xls/solvers/z3_ir_translator.h"
#include "z3/src/api/z3_api.h"

enum class CheckMode { kNode, kBit };

bool AbslParseFlag(std::string_view text, CheckMode* mode, std::string* error) {
  if (text == "node") {
    *mode = CheckMode::kNode;
    return true;
  }
  if (text == "bit") {
    *mode = CheckMode::kBit;
    return true;
  }
  *error = "unknown check mode, specify 'node' or 'bit'";
  return false;
}

std::string AbslUnparseFlag(CheckMode mode) {
  return mode == CheckMode::kNode ? "node" : "bit";
}

ABSL_FLAG(std::string, ir_path, "", "Path to the XLS IR file.");
ABSL_FLAG(std::string, top_proc, "",
          "Name of the top proc. Uses top proc if empty.");
ABSL_FLAG(int64_t, unroll_count, 4,
          "Number of activations to unroll the proc.");
ABSL_FLAG(CheckMode, mode, CheckMode::kNode,
          "Constancy check mode: 'node' or 'bit'.");
ABSL_FLAG(int64_t, z3_rlimit, 0,
          "Z3 resource limit per check (0 for no limit).");
ABSL_FLAG(int64_t, z3_timeout_ms, 0,
          "Z3 timeout in milliseconds per check (0 for no timeout).");
ABSL_FLAG(
    std::string, node_filter, "",
    "Comma-separated substrings to filter target node names to check (e.g. "
    "'nor.195914,nor.195923'). If empty, checks all target nodes.");
ABSL_FLAG(
    bool, fail_on_constants, false,
    "If true, returns a non-zero exit code when any constant nodes or bits "
    "are detected.");

namespace xls {
namespace {

// Map from original proc node to its cloned Node* instances across activations.
using NodeActivationMap = absl::flat_hash_map<Node*, std::vector<Node*>>;

std::vector<Node*> FilterCandidatesByName(
    const std::vector<Node*>& candidates,
    const std::vector<std::string_view>& filter_tokens) {
  if (filter_tokens.empty()) {
    return candidates;
  }
  std::vector<Node*> filtered;
  for (Node* n : candidates) {
    bool match = false;
    for (std::string_view tok : filter_tokens) {
      if (absl::StrContains(n->GetName(), tok)) {
        match = true;
        break;
      }
    }
    if (match) {
      filtered.push_back(n);
    }
  }
  return filtered;
}

absl::StatusOr<std::vector<Node*>> FilterTargetsForChecking(
    const std::vector<Node*>& candidates,
    const NodeActivationMap& node_activations, int64_t unroll_count) {
  std::vector<Node*> target_nodes;
  for (Node* n : candidates) {
    auto it = node_activations.find(n);
    if (it == node_activations.end()) {
      continue;
    }
    if (it->second.size() != unroll_count) {
      continue;
    }
    target_nodes.push_back(n);
  }
  return target_nodes;
}

bool IsInteractiveTty(const std::ostream& os) {
  if (os.rdbuf() == std::cout.rdbuf()) {
    return isatty(fileno(stdout));
  }
  if (os.rdbuf() == std::cerr.rdbuf()) {
    return isatty(fileno(stderr));
  }
  return false;
}

// TODO: google/xls#4734 - move to gloop
class ProgressTracker {
 public:
  ProgressTracker(std::ostream& os, int64_t total_targets)
      : os_(os),
        total_targets_(total_targets),
        interactive_(IsInteractiveTty(os)) {}

  void RenderProgress(int64_t current) {
    if (!interactive_) {
      return;
    }
    float progress = total_targets_ > 0
                         ? static_cast<float>(current) / total_targets_
                         : 1.0f;
    int64_t percent = static_cast<int64_t>(progress * 100);
    int64_t bar_width = 30;
    int64_t pos = static_cast<int64_t>(bar_width * progress);

    os_ << "\r[" << std::setw(3) << percent << "%] [";
    for (int64_t i = 0; i < bar_width; ++i) {
      if (i < 10) {
        os_ << "\033[1;32m";  // Green
      } else if (i < 20) {
        os_ << "\033[1;97m";  // White
      } else {
        os_ << "\033[1;31m";  // Red
      }

      if (i < pos) {
        os_ << "=";
      } else if (i == pos) {
        os_ << ">";
      } else {
        os_ << " ";
      }
    }
    os_ << "\033[0m] " << current << "/" << total_targets_
        << " nodes (Constant: " << constant_checks_;
    if (timeout_checks_ > 0) {
      os_ << ", Timeout: " << timeout_checks_;
    }
    os_ << ")" << std::flush;
  }

  void RecordNonConstantCheck() { non_constant_checks_++; }
  void RecordConstantNode(Node* n) {
    constant_checks_++;
    if (interactive_) {
      os_ << "\r\033[K";
    }
    os_ << "UNSAT *** ALERT: CONSTANT NODE DETECTED ***: '" << n->GetName()
        << "' (" << OpToString(n->op()) << ", " << n->GetType()->ToString()
        << ")\n";
  }
  void RecordConstantBit(Node* n, int64_t b) {
    constant_checks_++;
    if (interactive_) {
      os_ << "\r\033[K";
    }
    os_ << "UNSAT *** ALERT: CONSTANT BIT DETECTED ***: '" << n->GetName()
        << "' bit [" << b << "]\n";
  }
  void RecordTimeoutNode(Node* n, std::string_view reason) {
    timeout_checks_++;
    if (interactive_) {
      os_ << "\r\033[K";
    }
    os_ << "UNKNOWN (" << (!reason.empty() ? reason : "unknown") << "): '"
        << n->GetName() << "' (" << OpToString(n->op()) << ")\n";
  }
  void RecordTimeoutBit(Node* n, int64_t b, std::string_view reason) {
    timeout_checks_++;
    if (interactive_) {
      os_ << "\r\033[K";
    }
    os_ << "UNKNOWN (" << (!reason.empty() ? reason : "unknown") << "): '"
        << n->GetName() << "' bit [" << b << "]\n";
  }

  int64_t non_constant_checks() const { return non_constant_checks_; }
  int64_t constant_checks() const { return constant_checks_; }
  int64_t timeout_checks() const { return timeout_checks_; }

 private:
  std::ostream& os_;
  int64_t total_targets_;
  bool interactive_;
  int64_t non_constant_checks_ = 0;
  int64_t constant_checks_ = 0;
  int64_t timeout_checks_ = 0;
};

absl::Status RealMain(std::ostream& os, std::string_view ir_path,
                      std::string_view top_proc_name, int64_t unroll_count,
                      CheckMode mode, int64_t rlimit, int64_t timeout_ms,
                      std::string_view node_filter_str,
                      bool fail_on_constants) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir_text));

  Proc* proc = nullptr;
  if (top_proc_name.empty()) {
    XLS_ASSIGN_OR_RETURN(proc, package->GetTopAsProc());
  } else {
    XLS_ASSIGN_OR_RETURN(proc, package->GetProc(top_proc_name));
  }

  os << "Loaded Proc '" << proc->name() << "' with " << proc->node_count()
     << " nodes.\n";

  // Strip non-synthesizable nodes before unrolling or checking.
  XLS_RETURN_IF_ERROR(StripNonSynthNodes(package.get(), proc));

  // Get candidate target nodes early so we can skip unrolling if none exist.
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> candidates,
                       GetNodesFilteringNonSynthAndTrivialConstants(proc));

  std::vector<std::string_view> filter_tokens;
  if (!node_filter_str.empty()) {
    filter_tokens = absl::StrSplit(node_filter_str, ',', absl::SkipEmpty());
  }
  candidates = FilterCandidatesByName(candidates, filter_tokens);

  if (candidates.empty()) {
    os << "Constant Checks:     0\n";
    return absl::OkStatus();
  }

  // Unroll the proc.
  os << "Unrolling proc " << unroll_count << " times...\n";
  XLS_ASSIGN_OR_RETURN((auto [unrolled_func, node_activations]),
                       UnrollProcForConstancy(proc, unroll_count));

  // Translate Unrolled Function to Z3
  os << "Translating unrolled function to Z3 SMT AST...\n";
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<solvers::z3::IrTranslator> translator,
                       solvers::z3::IrTranslator::CreateAndTranslate(
                           unrolled_func, /*allow_unsupported=*/true));

  Z3_context ctx = translator->ctx();
  Z3_solver solver = Z3_mk_solver(ctx);
  Z3_solver_inc_ref(ctx, solver);
  Z3_params params = Z3_mk_params(ctx);
  Z3_params_inc_ref(ctx, params);

  if (rlimit > 0) {
    translator->SetRlimit(rlimit);
    Z3_params_set_uint(ctx, params, Z3_mk_string_symbol(ctx, "rlimit"),
                       static_cast<unsigned>(rlimit));
  }
  if (timeout_ms > 0) {
    Z3_params_set_uint(ctx, params, Z3_mk_string_symbol(ctx, "timeout"),
                       static_cast<unsigned>(timeout_ms));
  }
  // Z3 swallows ctrl+c, interpreting it as a sign to cancel the current solve
  // rather than the whole binary.
  Z3_params_set_bool(ctx, params, Z3_mk_string_symbol(ctx, "ctrl_c"), false);

  Z3_solver_set_params(ctx, solver, params);

  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> target_nodes,
      FilterTargetsForChecking(candidates, node_activations, unroll_count));

  os << "Verifying " << target_nodes.size()
     << " non-literal, synthesizable nodes in mode '" << AbslUnparseFlag(mode)
     << "'...\n\n";

  ProgressTracker progress_tracker(os, target_nodes.size());
  progress_tracker.RenderProgress(0);

  for (int64_t idx = 0; idx < target_nodes.size(); ++idx) {
    Node* n = target_nodes[idx];
    const std::vector<Node*>& act_nodes = node_activations[n];

    if (mode == CheckMode::kNode) {
      std::vector<Z3_ast> disequalities;
      disequalities.reserve(act_nodes.size() - 1);
      Z3_ast ast_0 = translator->GetTranslation(act_nodes[0]);
      for (int64_t i = 1; i < act_nodes.size(); ++i) {
        Z3_ast ast_i = translator->GetTranslation(act_nodes[i]);
        Z3_ast eq = Z3_mk_eq(ctx, ast_0, ast_i);
        disequalities.push_back(Z3_mk_not(ctx, eq));
      }

      Z3_ast can_change =
          Z3_mk_or(ctx, disequalities.size(), disequalities.data());

      Z3_lbool check_res = Z3_solver_check_assumptions(
          ctx, solver, /*num_assumptions=*/1, /*assumptions=*/&can_change);

      if (check_res == Z3_L_TRUE) {
        progress_tracker.RecordNonConstantCheck();
      } else if (check_res == Z3_L_FALSE) {
        progress_tracker.RecordConstantNode(n);
        // Performance optimization: permanently assert proven constants into
        // the shared solver at decision level 0. Because target nodes are
        // checked in topological order, Z3's congruence-closure and bit-vector
        // engines instantly merge constant operand ASTs across activations into
        // the same equivalence class, allowing downstream checks to simplify
        // without re-exploring the proof trees of their operands.
        Z3_solver_assert(ctx, solver, Z3_mk_not(ctx, can_change));
      } else {
        const char* reason = Z3_solver_get_reason_unknown(ctx, solver);
        progress_tracker.RecordTimeoutNode(
            n, reason != nullptr ? reason : "unknown");
      }
    } else {
      // Bit mode: check each bit index b independently
      int64_t width = n->GetType()->GetFlatBitCount();

      // Collect flat bit ASTs for each activation
      std::vector<std::vector<Z3_ast>> act_flat_bits;
      act_flat_bits.reserve(act_nodes.size());
      for (Node* act_node : act_nodes) {
        Z3_ast ast = translator->GetTranslation(act_node);
        act_flat_bits.push_back(
            FlattenBitsOnly(ctx, translator.get(), n->GetType(), ast));
      }

      for (int64_t b = 0; b < width; ++b) {
        bool skip = false;
        for (int64_t i = 0; i < act_flat_bits.size(); ++i) {
          if (b >= act_flat_bits[i].size()) {
            skip = true;
            break;
          }
        }
        if (skip) {
          continue;
        }

        std::vector<Z3_ast> disequalities;
        disequalities.reserve(act_flat_bits.size() - 1);
        for (int64_t i = 1; i < act_flat_bits.size(); ++i) {
          Z3_ast eq = Z3_mk_eq(ctx, act_flat_bits[0][b], act_flat_bits[i][b]);
          disequalities.push_back(Z3_mk_not(ctx, eq));
        }

        Z3_ast can_change =
            Z3_mk_or(ctx, disequalities.size(), disequalities.data());

        Z3_lbool check_res = Z3_solver_check_assumptions(
            ctx, solver, /*num_assumptions=*/1, /*assumptions=*/&can_change);

        if (check_res == Z3_L_TRUE) {
          progress_tracker.RecordNonConstantCheck();
        } else if (check_res == Z3_L_FALSE) {
          progress_tracker.RecordConstantBit(n, b);
          // Performance optimization: permanently assert proven constant bits
          // into the shared solver at decision level 0. Downstream bit-vector
          // operations can immediately exploit these constant bit assignments
          // without re-exploring the proof trees of their operands.
          Z3_solver_assert(ctx, solver, Z3_mk_not(ctx, can_change));
        } else {
          const char* reason = Z3_solver_get_reason_unknown(ctx, solver);
          progress_tracker.RecordTimeoutBit(
              n, b, reason != nullptr ? reason : "unknown");
        }
      }
    }
    progress_tracker.RenderProgress(idx + 1);
  }

  os << "\r\033[K";  // Clear progress bar before printing summary

  os << "\n=== Verification Summary ===\n"
     << "Check Mode:          " << AbslUnparseFlag(mode) << "\n"
     << "Non-Constant Checks: " << progress_tracker.non_constant_checks()
     << "\n"
     << "Constant Checks:     " << progress_tracker.constant_checks()
     << " (Potential Static Flops)\n"
     << "Timeout Checks:      " << progress_tracker.timeout_checks() << "\n";

  Z3_params_dec_ref(ctx, params);
  Z3_solver_dec_ref(ctx, solver);

  if (fail_on_constants && progress_tracker.constant_checks() > 0) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Found %d constant nodes/bits.", progress_tracker.constant_checks()));
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  std::string ir_path = absl::GetFlag(FLAGS_ir_path);
  QCHECK(!ir_path.empty()) << "--ir_path must be specified.";

  int64_t unroll_count = absl::GetFlag(FLAGS_unroll_count);
  QCHECK_GT(unroll_count, 1)
      << "--unroll_count must be > 1 as there must be at least two activations "
         "to check for constancy.";

  std::string top_proc_name = absl::GetFlag(FLAGS_top_proc);
  CheckMode mode = absl::GetFlag(FLAGS_mode);
  int64_t rlimit = absl::GetFlag(FLAGS_z3_rlimit);
  int64_t timeout_ms = absl::GetFlag(FLAGS_z3_timeout_ms);
  std::string node_filter_str = absl::GetFlag(FLAGS_node_filter);
  bool fail_on_constants = absl::GetFlag(FLAGS_fail_on_constants);
  return xls::ExitStatus(xls::RealMain(std::cout, ir_path, top_proc_name,
                                       unroll_count, mode, rlimit, timeout_ms,
                                       node_filter_str, fail_on_constants));
}
