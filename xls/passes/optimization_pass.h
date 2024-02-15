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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_registry.h"
#include "xls/passes/pipeline_generator.h"

namespace xls {

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

struct RamModelBuilderResult {
  std::unique_ptr<Package> package;
  absl::flat_hash_map<std::string, std::string>
      channel_logical_name_to_physical_name;
};

using ram_model_builder_t = std::function<RamModelBuilderResult(RamConfig)>;

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
  // If populated, also add a RAM model of kind "to_kind" driving the new
  // channels using the builder function.
  std::optional<ram_model_builder_t> model_builder;

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

  // Whether to inline all procs by calling the proc inlining pass.
  // TODO(meheff): 2022/2/13 Devise a better mechanism for deciding whether or
  // not to inline procs including figuring out which procs to inline. At the
  // minimum, there should be a specialization of OptimizationPassOptions for
  // the optimization pass pipeline which holds this value.
  bool inline_procs = false;

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
  std::vector<RamRewrite> ram_rewrites;

  // Use select context during narrowing range analysis.
  bool use_context_narrowing_analysis = false;
};

// An object containing information about the invocation of a pass (single call
// to PassBase::Run).
// Defines the pass types for optimizations which operate strictly on XLS IR
// (i.e., xls::Package).
// TODO(meheff): Rename to OptimizationPass, etc.
using OptimizationPass = PassBase<Package, OptimizationPassOptions>;
using OptimizationCompoundPass =
    CompoundPassBase<Package, OptimizationPassOptions>;
using OptimizationFixedPointCompoundPass =
    FixedPointCompoundPassBase<Package, OptimizationPassOptions>;
using OptimizationInvariantChecker = OptimizationCompoundPass::InvariantChecker;
using OptimizationPipelineGenerator =
    PipelineGeneratorBase<Package, OptimizationPassOptions>;

inline constexpr int64_t kMaxOptLevel = 3;

using OptimizationPassStandardConfig = decltype(kMaxOptLevel);
using OptimizationPassRegistry =
    PassRegistry<Package, OptimizationPassOptions, PassResults,
                 OptimizationPassStandardConfig>;
using OptimizationPassGenerator =
    PassGenerator<Package, OptimizationPassOptions, PassResults,
                  OptimizationPassStandardConfig>;

// Whether optimizations which split operations into multiple pieces should be
// performed at the given optimization level.
inline bool SplitsEnabled(int64_t opt_level) { return opt_level >= 3; }
inline bool NarrowingEnabled(int64_t opt_level) { return opt_level >= 2; }

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
