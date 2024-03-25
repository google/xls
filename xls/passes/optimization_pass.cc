// Copyright 2020 The XLS Authors
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

#include "xls/passes/optimization_pass.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/passes/pass_base.h"

namespace xls {

std::string_view RamKindToString(RamKind kind) {
  switch (kind) {
    case RamKind::kAbstract:
      return "abstract";
    case RamKind::k1RW:
      return "1rw";
    case RamKind::k1R1W:
      return "1r1w";
    case RamKind::k2RW:
      return "2rw";
  }
}

int64_t RamConfig::addr_width() const {
  CHECK_GE(depth, 0);
  return CeilOfLog2(static_cast<uint64_t>(depth));
}

std::optional<int64_t> RamConfig::mask_width(int64_t data_width) const {
  if (!word_partition_size.has_value()) {
    return std::nullopt;
  }

  CHECK_GT(word_partition_size.value(), 0);
  return (data_width + word_partition_size.value() - 1) /
         word_partition_size.value();
}

absl::StatusOr<RamKind> RamKindFromProto(RamKindProto proto) {
  switch (proto) {
    case RamKindProto::RAM_ABSTRACT:
      return RamKind::kAbstract;
    case RamKindProto::RAM_1RW:
      return RamKind::k1RW;
    case RamKindProto::RAM_1R1W:
      return RamKind::k1R1W;
    default:
      return absl::InvalidArgumentError("Invalid RamKind");
  }
}

/* static */ absl::StatusOr<RamConfig> RamConfig::FromProto(
    const RamConfigProto& proto) {
  XLS_ASSIGN_OR_RETURN(RamKind kind, RamKindFromProto(proto.kind()));
  return RamConfig{
      .kind = kind,
      .depth = proto.depth(),
      .word_partition_size = proto.has_word_partition_size()
                                 ? std::optional(proto.word_partition_size())
                                 : std::nullopt,
      // TODO(google/xls#861): Add support for initialization info in proto.
      .initial_value = std::nullopt,
  };
}

/* static */ absl::StatusOr<RamRewrite> RamRewrite::FromProto(
    const RamRewriteProto& proto) {
  if (proto.has_model_builder()) {
    return absl::UnimplementedError("Model builders not yet implemented.");
  }
  XLS_ASSIGN_OR_RETURN(RamConfig from_config,
                       RamConfig::FromProto(proto.from_config()));
  XLS_ASSIGN_OR_RETURN(RamConfig to_config,
                       RamConfig::FromProto(proto.to_config()));
  return RamRewrite{
      .from_config = from_config,
      .from_channels_logical_to_physical =
          absl::flat_hash_map<std::string, std::string>(
              proto.from_channels_logical_to_physical().begin(),
              proto.from_channels_logical_to_physical().end()),
      .to_config = to_config,
      .to_name_prefix = proto.to_name_prefix(),
      .model_builder = std::nullopt,
  };
}

absl::StatusOr<std::vector<RamRewrite>> RamRewritesFromProto(
    const RamRewritesProto& proto) {
  std::vector<RamRewrite> rewrites;
  rewrites.reserve(proto.rewrites_size());
  for (const auto& rewrite_proto : proto.rewrites()) {
    XLS_ASSIGN_OR_RETURN(RamRewrite rewrite,
                         RamRewrite::FromProto(rewrite_proto));
    rewrites.push_back(rewrite);
  }
  return rewrites;
}

absl::StatusOr<bool> OptimizationFunctionBasePass::RunOnFunctionBase(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  VLOG(2) << absl::StreamFormat("Running %s on function_base %s [pass #%d]",
                                long_name(), f->name(),
                                results->invocations.size());
  VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, f->DumpIr());

  XLS_ASSIGN_OR_RETURN(bool changed,
                       RunOnFunctionBaseInternal(f, options, results));

  VLOG(3) << absl::StreamFormat("After [changed = %d]:", changed);
  XLS_VLOG_LINES(3, f->DumpIr());
  return changed;
}

absl::StatusOr<bool> OptimizationFunctionBasePass::RunInternal(
    Package* p, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (FunctionBase* f : p->GetFunctionBases()) {
    XLS_ASSIGN_OR_RETURN(bool function_changed,
                         RunOnFunctionBaseInternal(f, options, results));
    changed = changed || function_changed;
  }
  return changed;
}

absl::StatusOr<bool> OptimizationFunctionBasePass::TransformNodesToFixedPoint(
    FunctionBase* f,
    std::function<absl::StatusOr<bool>(Node*)> simplify_f) const {
  // Store nodes by id to avoid running afoul of Node* pointer values being
  // reused.
  absl::flat_hash_set<int64_t> simplified_node_ids;
  bool changed = false;
  bool changed_this_time = false;
  do {
    changed_this_time = false;
    auto node_it = f->nodes().begin();
    while (node_it != f->nodes().end()) {
      // Save the next iterator because node_it may be invalidated by the call
      // to simplify_f if simpplify_f ends up deleting 'node'.
      auto next_it = std::next(node_it);
      Node* node = *node_it;
      // If the node was previously simplified and is now dead, avoid running
      // simplification on it again to avoid inf-looping while simplifying the
      // same node over and over again.
      if (!node->IsDead() || !simplified_node_ids.contains(node->id())) {
        // Grab the node ID before simplifying because the node might be
        // removed when simplifying.
        int64_t node_id = node->id();
        XLS_ASSIGN_OR_RETURN(bool node_changed, simplify_f(node));
        if (node_changed) {
          simplified_node_ids.insert(node_id);
          changed_this_time = true;
          changed = true;
        }
      }
      node_it = next_it;
    }
  } while (changed_this_time);

  return changed;
}

absl::StatusOr<bool> OptimizationProcPass::RunOnProc(
    Proc* proc, const OptimizationPassOptions& options,
    PassResults* results) const {
  VLOG(2) << absl::StreamFormat("Running %s on proc %s [pass #%d]", long_name(),
                                proc->name(), results->invocations.size());
  VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, proc->DumpIr());

  XLS_ASSIGN_OR_RETURN(bool changed, RunOnProcInternal(proc, options, results));

  VLOG(3) << absl::StreamFormat("After [changed = %d]:", changed);
  XLS_VLOG_LINES(3, proc->DumpIr());
  return changed;
}

absl::StatusOr<bool> OptimizationProcPass::RunInternal(
    Package* p, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (const auto& proc : p->procs()) {
    XLS_ASSIGN_OR_RETURN(bool proc_changed,
                         RunOnProcInternal(proc.get(), options, results));
    changed = changed || proc_changed;
  }
  return changed;
}

}  // namespace xls
