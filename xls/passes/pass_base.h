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

#ifndef XLS_PASSES_PASS_BASE_H_
#define XLS_PASSES_PASS_BASE_H_

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"
#include "xls/ir/ram_rewrite.pb.h"

namespace xls {

// This file defines a set of base classes for building XLS compiler passes and
// pass pipelines. The base classes are templated allowing polymorphism of the
// data types the pass operates on.

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
  absl::flat_hash_map<std::string, int64_t> channel_logical_name_to_physical_id;
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

// Options data structure passed to each pass run invocation. This data
// structure is passed by const reference to PassBase::Run and should contain
// options which affect how passes are run.
struct PassOptions {
  // If non-empty, this is the path to the directory in which to dump
  // intermediate IR files.
  std::filesystem::path ir_dump_path;

  // If present, only passes whose short names are in this list will be run.
  std::optional<std::vector<std::string>> run_only_passes;

  // If present, passes whose short names are in this list will be skipped. If
  // both run_only_passes and skip_passes are present, then only passes which
  // are present in run_only_passes and not present in skip_passes will be run.
  std::vector<std::string> skip_passes;

  // Whether to inline all procs by calling the proc inlining pass.
  // TODO(meheff): 2022/2/13 Devise a better mechanism for deciding whether or
  // not to inline procs including figuring out which procs to inline. At the
  // minimum, there should be a specialization of PassOptions for the
  // optimization pass pipeline which holds this value.
  bool inline_procs = false;

  // If this is not `std::nullopt`, convert array indexes with fewer than or
  // equal to the given number of possible indices (by range analysis) into
  // chains of selects. Otherwise, this optimization is skipped, since it can
  // sometimes reduce output quality.
  std::optional<int64_t> convert_array_index_to_select = std::nullopt;

  // List of RAM rewrites, generally lowering abstract RAMs into concrete
  // variants.
  std::vector<RamRewrite> ram_rewrites;

  // Solver resource limit for mutual exclusion pass.
  int64_t mutual_exclusion_z3_rlimit = -1;
};

// An object containing information about the invocation of a pass (single call
// to PassBase::Run).
struct PassInvocation {
  // The name of the pass.
  std::string pass_name;

  // Whether the IR was changed by the pass.
  bool ir_changed;

  // The run duration of the pass.
  absl::Duration run_duration;
};

// A object to which metadata may be written in each pass invocation. This data
// structure is passed by mutable pointer to PassBase::Run.
struct PassResults {
  // This vector contains and entry for each invocation of each pass.
  std::vector<PassInvocation> invocations;
};

// Base class for all compiler passes. Template parameters:
//
//   IrT : The data type that the pass operates on (e.g., xls::Package). The
//     type should define 'DumpIr' and 'name' methods used for dumping and
//     logging in compound passes. A pass which strictly operate on the XLS IR
//     may use the xls::Package type as the IrT template argument. Passes which
//     operate on the IR and a schedule may be instantiated on a data structure
//     containing both an xls::Package and a schedule. Roughly, IrT should
//     contain the IR and (optionally) any metadata generated or transformed by
//     the passes which is necessary for the passes to function (e.g., not just
//     telemetry or logging info which should be held in ResultT).
//
//   OptionsT : Options type passed as an immutable object to each invocation of
//     PassBase::Run. This type should be derived from PassOptions because
//     PassOptions contains fields required by CompoundPassBase when executing
//     pass pipelines.
//
//   ResultsT : Results type passed as a mutable object to each invocation of
//     PassBase::Run. This type should be derived from PassResults because
//     PassResults contains fields required by CompoundPassBase when executing
//     pass pipelines.
template <typename IrT, typename OptionsT = PassOptions,
          typename ResultsT = PassResults>
class PassBase {
 public:
  PassBase(std::string_view short_name, std::string_view long_name)
      : short_name_(short_name), long_name_(long_name) {}

  virtual ~PassBase() = default;

  const std::string& short_name() const { return short_name_; }
  const std::string& long_name() const { return long_name_; }

  // Run the specific pass. Returns true if the graph was changed by the pass.
  // Typically the "changed" indicator is used to determine when to terminate
  // fixed point computation.
  virtual absl::StatusOr<bool> Run(IrT* ir, const OptionsT& options,
                                   ResultsT* results) const {
    XLS_VLOG(2) << absl::StreamFormat("Running %s [pass #%d]", long_name(),
                                      results->invocations.size());
    XLS_VLOG(3) << "Before:";
    XLS_VLOG_LINES(3, ir->DumpIr());

    XLS_ASSIGN_OR_RETURN(bool changed, RunInternal(ir, options, results));

    XLS_VLOG(3) << absl::StreamFormat("After [changed = %d]:", changed);
    XLS_VLOG_LINES(3, ir->DumpIr());
    return changed;
  }

  // Returns true if this is a compound pass.
  virtual bool IsCompound() const { return false; }

 protected:
  // Derived classes should override this function which is invoked from Run.
  virtual absl::StatusOr<bool> RunInternal(IrT* ir, const OptionsT& options,
                                           ResultsT* results) const = 0;

  const std::string short_name_;
  const std::string long_name_;
};

// A base class for abstractions which check invariants of the IR. These
// checkers are added to compound passes (pass pipelines) and run before and
// after each pass in the pipeline.
template <typename IrT, typename OptionsT = PassOptions,
          typename ResultsT = PassResults>
class InvariantCheckerBase {
 public:
  virtual ~InvariantCheckerBase() = default;
  virtual absl::Status Run(IrT* ir, const OptionsT& options,
                           ResultsT* results) const = 0;
};

// CompoundPass is a container for other passes. For example, the scalar
// optimizer can be a compound pass holding many passes for scalar
// optimizations.
template <typename IrT, typename OptionsT = PassOptions,
          typename ResultsT = PassResults>
class CompoundPassBase : public PassBase<IrT, OptionsT, ResultsT> {
 public:
  using Pass = PassBase<IrT, OptionsT, ResultsT>;
  using InvariantChecker = InvariantCheckerBase<IrT, OptionsT, ResultsT>;

  CompoundPassBase(std::string_view short_name, std::string_view long_name)
      : Pass(short_name, long_name) {}
  ~CompoundPassBase() override = default;

  // Add a new pass to this compound pass. Arguments to method are the arguments
  // to the pass constructor. Example usage:
  //
  //  pass->Add<FooPass>(bar, qux);
  //
  // Returns a pointer to the newly constructed pass.
  template <typename T, typename... Args>
  T* Add(Args&&... args) {
    auto* pass = new T(std::forward<Args>(args)...);
    passes_.emplace_back(pass);
    pass_ptrs_.push_back(pass);
    return pass;
  }

  absl::Span<Pass* const> passes() const { return pass_ptrs_; }
  absl::Span<Pass*> passes() { return absl::Span<Pass*>(pass_ptrs_); }

  // Adds an invariant checker to the compound pass. The invariant checker is
  // run before and after each pass contained in the compound pass. The checkers
  // are also run for nested compound passes. Arguments to method are the
  // arguments to the invariant checker constructor. Example usage:
  //
  //  pass->Add<CheckStuff>(foo);
  //
  // Returns a pointer to the newly constructed pass.
  template <typename T, typename... Args>
  T* AddInvariantChecker(Args&&... args) {
    auto* checker = new T(std::forward<Args>(args)...);
    invariant_checkers_.emplace_back(checker);
    invariant_checker_ptrs_.push_back(checker);
    return checker;
  }

  absl::StatusOr<bool> RunInternal(IrT* ir, const OptionsT& options,
                                   ResultsT* results) const override {
    if (!options.ir_dump_path.empty()) {
      // Start of the top-level pass. Dump IR.
      XLS_RETURN_IF_ERROR(DumpIr(options.ir_dump_path, ir, this->short_name(),
                                 "start",
                                 /*ordinal=*/0, /*changed=*/false));
    }
    return RunNested(ir, options, results, this->short_name(),
                     /*invariant_checkers=*/{});
  }

  // Internal implementation of Run for compound passes. Invoked when a compound
  // pass is nested within another compound pass. Enables passing of invariant
  // checkers and name of the top-level pass to nested compound passes.
  virtual absl::StatusOr<bool> RunNested(
      IrT* ir, const OptionsT& options, ResultsT* results,
      std::string_view top_level_name,
      absl::Span<const InvariantChecker* const> invariant_checkers) const;

  bool IsCompound() const override { return true; }

 protected:
  // Dump the IR to a file in the given directory. Name is determined by the
  // various arguments passed in. File names will be lexographically ordered by
  // package name and ordinal.
  absl::Status DumpIr(const std::filesystem::path& ir_dump_path, IrT* ir,
                      std::string_view top_level_name, std::string_view tag,
                      int64_t ordinal, bool changed) const {
    std::filesystem::path path =
        ir_dump_path / absl::StrFormat("%s.%s.%05d.%s.%s.ir", ir->name(),
                                       top_level_name, ordinal, tag,
                                       changed ? "changed" : "unchanged");
    return SetFileContents(path, ir->DumpIr());
  }

  std::vector<std::unique_ptr<Pass>> passes_;
  std::vector<Pass*> pass_ptrs_;

  std::vector<std::unique_ptr<InvariantChecker>> invariant_checkers_;
  std::vector<InvariantChecker*> invariant_checker_ptrs_;
};

// A compound pass which runs its set of passes to fixed point.
template <typename IrT, typename OptionsT = PassOptions,
          typename ResultsT = PassResults>
class FixedPointCompoundPassBase
    : public CompoundPassBase<IrT, OptionsT, ResultsT> {
 public:
  FixedPointCompoundPassBase(std::string_view short_name,
                             std::string_view long_name)
      : CompoundPassBase<IrT, OptionsT, ResultsT>(short_name, long_name) {}

  absl::StatusOr<bool> RunNested(
      IrT* ir, const OptionsT& options, ResultsT* results,
      std::string_view top_level_name,
      absl::Span<const typename CompoundPassBase<
          IrT, OptionsT, ResultsT>::InvariantChecker* const>
          invariant_checkers) const override {
    bool local_changed = true;
    bool global_changed = false;
    while (local_changed) {
      XLS_ASSIGN_OR_RETURN(
          local_changed,
          (CompoundPassBase<IrT, OptionsT, ResultsT>::RunNested(
              ir, options, results, top_level_name, invariant_checkers)));
      global_changed = global_changed || local_changed;
    }
    return global_changed;
  }
};

template <typename IrT, typename OptionsT, typename ResultsT>
absl::StatusOr<bool> CompoundPassBase<IrT, OptionsT, ResultsT>::RunNested(
    IrT* ir, const OptionsT& options, ResultsT* results,
    std::string_view top_level_name,
    absl::Span<const InvariantChecker* const> invariant_checkers) const {
  XLS_VLOG(1) << "Running " << this->short_name()
              << " compound pass on package " << ir->name();
  XLS_VLOG(2) << "Start of compound pass " << this->short_name() << ":";
  XLS_VLOG_LINES(5, ir->DumpIr());

  // Invariant checkers may be passed in from parent compound passes or
  // contained by this pass itself. Merge them together.
  std::vector<const InvariantChecker*> checkers(invariant_checkers.begin(),
                                                invariant_checkers.end());
  checkers.insert(checkers.end(), invariant_checker_ptrs_.begin(),
                  invariant_checker_ptrs_.end());
  auto run_invariant_checkers =
      [&](std::string_view str_context) -> absl::Status {
    for (const auto& checker : checkers) {
      absl::Status status = checker->Run(ir, options, results);
      if (!status.ok()) {
        return absl::Status(status.code(), absl::StrCat(status.message(), "; [",
                                                        str_context, "]"));
      }
    }
    return absl::OkStatus();
  };
  XLS_RETURN_IF_ERROR(run_invariant_checkers(
      absl::StrCat("start of compound pass '", this->long_name(), "'")));

  bool changed = false;
  for (const auto& pass : passes_) {
    XLS_VLOG(1) << absl::StreamFormat("Running %s (%s) pass on package %s",
                                      pass->long_name(), pass->short_name(),
                                      ir->name());

    if (!pass->IsCompound() && options.run_only_passes.has_value() &&
        std::find_if(options.run_only_passes->begin(),
                     options.run_only_passes->end(),
                     [&](const std::string& name) {
                       return pass->short_name() == name;
                     }) == options.run_only_passes->end()) {
      XLS_VLOG(1) << "Skipping pass. Not contained in run_only_passes option.";
      continue;
    }

    if (std::find_if(options.skip_passes.begin(), options.skip_passes.end(),
                     [&](const std::string& name) {
                       return pass->short_name() == name;
                     }) != options.skip_passes.end()) {
      XLS_VLOG(1) << "Skipping pass. Contained in skip_passes option.";
      continue;
    }

#ifdef DEBUG
    // Verify that the IR should change iff Run returns true. This is slow, so
    // do not check it in optimized builds.
    std::string ir_before = ir->DumpIr();
#endif
    absl::Time start = absl::Now();
    bool pass_changed;
    if (pass->IsCompound()) {
      XLS_ASSIGN_OR_RETURN(
          pass_changed,
          (down_cast<CompoundPassBase<IrT, OptionsT, ResultsT>*>(pass.get())
               ->RunNested(ir, options, results, top_level_name, checkers)));
    } else {
      XLS_ASSIGN_OR_RETURN(pass_changed, pass->Run(ir, options, results));
    }
    absl::Duration duration = absl::Now() - start;
#ifdef DEBUG
    std::string ir_after = ir->DumpIr();
    if (pass_changed) {
      if (ir_before == ir_after) {
        return absl::InternalError(absl::StrFormat(
            "Pass %s indicated IR changed, but IR is unchanged:\n\n%s",
            pass->short_name(), ir_before));
      }
    } else {
      if (ir_before != ir_after) {
        return absl::InternalError(
            absl::StrFormat("Pass %s indicated IR unchanged, but IR is "
                            "changed:\n\n[Before]\n%s  !=\n[after]\n%s",
                            pass->short_name(), ir_before, ir_after));
      }
    }
#endif
    changed |= pass_changed;
    XLS_VLOG(1) << absl::StreamFormat(
        "[elapsed %s] Pass %s %s.", FormatDuration(duration),
        pass->short_name(),
        (pass_changed ? "changed IR" : "did not change IR"));
    if (!pass->IsCompound()) {
      results->invocations.push_back(
          {pass->short_name(), pass_changed, duration});
    }
    if (!options.ir_dump_path.empty()) {
      XLS_RETURN_IF_ERROR(DumpIr(options.ir_dump_path, ir, top_level_name,
                                 absl::StrCat("after_", pass->short_name()),
                                 /*ordinal=*/results->invocations.size(),
                                 /*changed=*/pass_changed));
    }
    // Only run the verifiers if the pass changed.
    if (pass_changed) {
      absl::Time checker_start = absl::Now();
      XLS_RETURN_IF_ERROR(run_invariant_checkers(
          absl::StrFormat("after '%s' pass, dynamic pass #%d",
                          pass->long_name(), results->invocations.size() - 1)));
      XLS_VLOG(1) << absl::StreamFormat(
          "Ran invariant checkers [elapsed %s]",
          FormatDuration(absl::Now() - checker_start));
    }
    XLS_VLOG(5) << "After " << pass->long_name() << ":";
    XLS_VLOG_LINES(5, ir->DumpIr());
  }
  return changed;
}

}  // namespace xls

#endif  // XLS_PASSES_PASS_BASE_H_
