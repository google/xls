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

#ifndef XLS_PASSES_OPTIMIZATION_PASS_H_
#define XLS_PASSES_OPTIMIZATION_PASS_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/passes/pass_registry.h"
#include "xls/passes/pipeline_generator.h"

namespace xls {

inline constexpr int64_t kMaxOptLevel = 3;

// Metadata for RAMs.
// TODO(google/xls#873): Ideally this metadata should live in the IR.
//
// Kinds of RAMs.
enum class RamKind {
  kAbstract,
  k1RW,
  k1R1W,
  k2RW,
};

std::string_view RamKindToString(RamKind kind);
absl::StatusOr<RamKind> RamKindFromProto(RamKindProto proto);

// Configuration describing the behavior of a RAM.
struct RamConfig {
  RamKind kind;
  int64_t depth;
  // Determines granularity of mask.
  std::optional<int64_t> word_partition_size;
  // If nullopt, RAM has no initial value. Reading uninitialized memory is
  // undefined.
  std::optional<std::vector<Value>> initial_value = std::nullopt;

  // Computed address width: clog2(depth).
  int64_t addr_width() const;
  // Computed mask width: if word_partition_size is nullopt, 0, else
  // ceil(width/word_partition_size).
  std::optional<int64_t> mask_width(int64_t data_width) const;

  static absl::StatusOr<RamConfig> FromProto(const RamConfigProto& proto);
};

// A configuration describing a desired RAM rewrite.
struct RamRewrite {
  // Configuration of RAM we start with.
  RamConfig from_config;
  // Mapping of the starting channels, from logical (e.g. read_req) to physical,
  // (the channel's name, e.g. foo_read_req).
  absl::flat_hash_map<std::string, std::string>
      from_channels_logical_to_physical;
  // Configuration of RAM we will rewrite the "from" RAM into.
  RamConfig to_config;
  // Name prefix for the new ram model
  std::string to_name_prefix;

  // For proc-scoped channels only, this specifies which proc the channels are
  // defined in.
  std::optional<std::string> proc_name;

  static absl::StatusOr<RamRewrite> FromProto(const RamRewriteProto& proto);
};

absl::StatusOr<std::vector<RamRewrite>> RamRewritesFromProto(
    const RamRewritesProto& proto);

struct OptimizationPassOptions : public PassOptionsBase {
  OptimizationPassOptions() = default;

  // Constructor which takes a base instance. This allows construction of
  // optimization pass options from scheduling and codegen options which is
  // required for wrapped passes.
  explicit OptimizationPassOptions(const PassOptionsBase& options_base)
      : PassOptionsBase(options_base) {}

  // What opt-level was requested for this pass. This might not be the top-level
  // --opt_level flag value as compound-passes might lower this for some of
  // their segments.
  int64_t opt_level = kMaxOptLevel;

  OptimizationPassOptions WithOptLevel(int64_t opt_level) const& {
    OptimizationPassOptions opt = *this;
    opt.opt_level = opt_level;
    return opt;
  }

  OptimizationPassOptions&& WithOptLevel(int64_t opt_level) && {
    this->opt_level = opt_level;
    return std::move(*this);
  }

  // Whether narrowing is enabled in this config.
  bool narrowing_enabled() const;

  // Whether splits is enabled in this config.
  bool splits_enabled() const;

  // If this is not `std::nullopt`, convert array indexes with fewer than or
  // equal to the given number of possible indices (by range analysis) into
  // chains of selects. Otherwise, this optimization is skipped, since it can
  // sometimes reduce output quality.
  std::optional<int64_t> convert_array_index_to_select = std::nullopt;

  // If this is not `std::nullopt`, split `next_value`s that assign `sel`s to
  // state params if they have fewer than the given number of cases. Otherwise,
  // this optimization is skipped, since it can sometimes reduce output quality.
  std::optional<int64_t> split_next_value_selects = std::nullopt;

  // List of RAM rewrites, generally lowering abstract RAMs into concrete
  // variants.
  absl::Span<RamRewrite const> ram_rewrites = {};

  // Use select context during narrowing range analysis.
  bool use_context_narrowing_analysis = false;

  // Whether to eliminate no-op Next nodes; this should be disabled after
  // proc-state legalization.
  bool eliminate_noop_next = true;

  OptimizationPassOptions WithEliminateNoopNext(
      bool eliminate_noop_next) const& {
    OptimizationPassOptions opt = *this;
    opt.eliminate_noop_next = eliminate_noop_next;
    return opt;
  }

  OptimizationPassOptions&& WithEliminateNoopNext(bool eliminate_noop_next) && {
    this->eliminate_noop_next = eliminate_noop_next;
    return std::move(*this);
  }

  // Optimize for best case throughput, even at the cost of area.
  bool optimize_for_best_case_throughput = false;
};

// An object containing information about the invocation of a pass (single call
// to PassBase::Run).
// Defines the pass types for optimizations which operate strictly on XLS IR
// (i.e., xls::Package).
using OptimizationPass = PassBase<Package, OptimizationPassOptions>;
using OptimizationCompoundPass =
    CompoundPassBase<Package, OptimizationPassOptions>;
using OptimizationFixedPointCompoundPass =
    FixedPointCompoundPassBase<Package, OptimizationPassOptions>;
using OptimizationInvariantChecker = OptimizationCompoundPass::InvariantChecker;
using OptimizationPipelineGenerator =
    PipelineGeneratorBase<Package, OptimizationPassOptions>;
using OptimizationWrapperPass =
    WrapperPassBase<Package, OptimizationPassOptions>;

using OptimizationPassRegistry =
    PassRegistry<Package, OptimizationPassOptions, PassResults>;
using OptimizationPassGenerator =
    PassGenerator<Package, OptimizationPassOptions, PassResults>;

namespace internal {
// Wrapper that uses templates to force opt-level to a specific max value for
// either a specific pass or a compound pass contents.
template <typename InnerPass>
  requires(std::is_base_of_v<OptimizationPass, InnerPass>)
class DynamicCapOptLevel : public OptimizationPass {
 public:
  template <typename... Args>
  explicit DynamicCapOptLevel(int64_t level, Args... args)
      : OptimizationPass("short", "long"),
        level_(level),
        inner_(std::forward<Args>(args)...) {
    short_name_ =
        absl::StrFormat("%s(opt_level<=%d)", inner_.short_name(), level);
    long_name_ =
        absl::StrFormat("%s with opt_level <= %d", inner_.long_name(), level);
  }

  absl::StatusOr<PassPipelineProto::Element> ToProto() const override {
    XLS_ASSIGN_OR_RETURN(PassPipelineProto::Element res, inner_.ToProto());
    res.mutable_options()->set_max_opt_level(level_);
    return res;
  }

 protected:
  absl::StatusOr<bool> RunInternal(Package* ir,
                                   const OptimizationPassOptions& options,
                                   PassResults* results) const override {
    if (VLOG_IS_ON(4) && level_ < options.opt_level) {
      VLOG(4) << "Lowering opt-level of pass '" << inner_.long_name() << "' ("
              << inner_.short_name() << ") to " << level_;
    }
    return inner_.Run(
        ir, options.WithOptLevel(std::min(level_, options.opt_level)), results);
  }

 private:
  int64_t level_;
  InnerPass inner_;
};

// Wrapper that disables the pass unless the opt-level is at least kLevel.
template <typename InnerPass>
  requires(std::is_base_of_v<OptimizationPass, InnerPass>)
class DynamicIfOptLevelAtLeast : public OptimizationPass {
 public:
  template <typename... Args>
  explicit DynamicIfOptLevelAtLeast(int64_t level, Args... args)
      : OptimizationPass("short", "long"),
        level_(level),
        inner_(std::forward<Args>(args)...) {
    short_name_ =
        absl::StrFormat("%s(opt_level>=%d)", inner_.short_name(), level);
    long_name_ =
        absl::StrFormat("%s when opt_level >= %d", inner_.long_name(), level);
  }

  absl::StatusOr<PassPipelineProto::Element> ToProto() const override {
    XLS_ASSIGN_OR_RETURN(PassPipelineProto::Element res, inner_.ToProto());
    res.mutable_options()->set_min_opt_level(level_);
    return res;
  }

 protected:
  absl::StatusOr<bool> RunInternal(Package* ir,
                                   const OptimizationPassOptions& options,
                                   PassResults* results) const override {
    if (options.opt_level < level_) {
      VLOG(4) << "Skipping pass '" << inner_.long_name() << "' ("
              << inner_.short_name()
              << ") because opt-level is lower than minimum level of "
              << level_;
      return false;
    }
    return inner_.Run(ir, options, results);
  }

 private:
  int64_t level_;
  InnerPass inner_;
};
}  // namespace internal

// Wrapper that uses templates to force opt-level to a specific max value for
// either a specific pass or a compound pass contents.
template <int64_t kLevel, typename InnerPass>
  requires(std::is_base_of_v<OptimizationPass, InnerPass>)
class CapOptLevel : public internal::DynamicCapOptLevel<InnerPass> {
 public:
  template <typename... Args>
  explicit CapOptLevel(Args... args)
      : internal::DynamicCapOptLevel<InnerPass>(kLevel,
                                                std::forward<Args>(args)...) {}
};

// Wrapper that disables the pass unless the opt-level is at least kLevel.
template <int64_t kLevel, typename InnerPass>
  requires(std::is_base_of_v<OptimizationPass, InnerPass>)
class IfOptLevelAtLeast : public internal::DynamicIfOptLevelAtLeast<InnerPass> {
 public:
  template <typename... Args>
  explicit IfOptLevelAtLeast(Args... args)
      : internal::DynamicIfOptLevelAtLeast<InnerPass>(
            kLevel, std::forward<Args>(args)...) {}
};

// Wrapper that explicitly sets the opt level to a specific value.
template <int64_t kLevel, typename InnerPass>
  requires(std::is_base_of_v<OptimizationPass, InnerPass>)
class WithOptLevel : public InnerPass {
 public:
  template <typename... Args>
  explicit WithOptLevel(Args... args) : InnerPass(args...) {}

  absl::StatusOr<PassPipelineProto::Element> ToProto() const override {
    return absl::UnimplementedError("WithOptLevel not exportable to proto");
  }

 protected:
  absl::StatusOr<bool> RunInternal(Package* ir,
                                   const OptimizationPassOptions& options,
                                   PassResults* results) const override {
    return InnerPass::RunInternal(ir, options.WithOptLevel(kLevel), results);
  }
};

// Whether optimizations which split operations into multiple pieces should be
// performed at the given optimization level.
inline bool SplitsEnabled(int64_t opt_level) { return opt_level >= 3; }
inline bool NarrowingEnabled(int64_t opt_level) { return opt_level >= 2; }

inline bool OptimizationPassOptions::narrowing_enabled() const {
  return NarrowingEnabled(opt_level);
}

inline bool OptimizationPassOptions::splits_enabled() const {
  return SplitsEnabled(opt_level);
}

// Abstract base class for passes operate at function/proc scope. The derived
// class must define RunOnFunctionBaseInternal.
class OptimizationFunctionBasePass : public OptimizationPass {
 public:
  OptimizationFunctionBasePass(std::string_view short_name,
                               std::string_view long_name)
      : OptimizationPass(short_name, long_name) {}

  // Runs the pass on a single function/proc.
  absl::StatusOr<bool> RunOnFunctionBase(FunctionBase* f,
                                         const OptimizationPassOptions& options,
                                         PassResults* results) const;

 protected:
  // Iterates over each function and proc in the package calling
  // RunOnFunctionBase.
  absl::StatusOr<bool> RunInternal(Package* p,
                                   const OptimizationPassOptions& options,
                                   PassResults* results) const override;

  virtual absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results) const = 0;

  // Calls the given function for every node in the graph in a loop until no
  // further simplifications are possible.  simplify_f should return true if the
  // IR was modified. simplify_f can add or remove nodes including the node
  // passed to it.
  //
  // TransformNodesToFixedPoint returns true iff any invocations of simplify_f
  // returned true.
  absl::StatusOr<bool> TransformNodesToFixedPoint(
      FunctionBase* f,
      std::function<absl::StatusOr<bool>(Node*)> simplify_f) const;
};

// Abstract base class for passes operate on procs. The derived
// class must define RunOnProcInternal.
class OptimizationProcPass : public OptimizationPass {
 public:
  OptimizationProcPass(std::string_view short_name, std::string_view long_name)
      : OptimizationPass(short_name, long_name) {}

  // Proc the pass on a single proc.
  absl::StatusOr<bool> RunOnProc(Proc* proc,
                                 const OptimizationPassOptions& options,
                                 PassResults* results) const;

 protected:
  // Iterates over each proc in the package calling RunOnProc.
  absl::StatusOr<bool> RunInternal(Package* p,
                                   const OptimizationPassOptions& options,
                                   PassResults* results) const override;

  virtual absl::StatusOr<bool> RunOnProcInternal(
      Proc* proc, const OptimizationPassOptions& options,
      PassResults* results) const = 0;
};

}  // namespace xls

#endif  // XLS_PASSES_OPTIMIZATION_PASS_H_
