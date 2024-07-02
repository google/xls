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

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"

namespace xls {

// This file defines a set of base classes for building XLS compiler passes and
// pass pipelines. The base classes are templated allowing polymorphism of the
// data types the pass operates on.

// Options data structure passed to each pass run invocation. This data
// structure is passed by const reference to PassBase::Run and should contain
// options which affect how passes are run.
struct PassOptionsBase {
  // If non-empty, this is the path to the directory in which to dump
  // intermediate IR files.
  std::filesystem::path ir_dump_path;

  // If present, passes whose short names are in this list will be skipped. If
  // both run_only_passes and skip_passes are present, then only passes which
  // are present in run_only_passes and not present in skip_passes will be run.
  std::vector<std::string> skip_passes;

  // If present, how many passes will be allowed to run to completion. NB
  // Passes which do not cause any changes are counted in this limit. When this
  // limit is reached all subsequent passes perform no changes. NB The total
  // number of passes executed might change due to setting this field as
  // fixed-points may complete earlier.
  std::optional<int64_t> bisect_limit;
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
//
// TODO(meheff): 2024/01/18 IrT is a Package or a thin wrapper around a package
// with additional metadata. To avoid the necessity of adding methods to the
// wrapper to match the Package API, PassBase should explicitly take a Package
// and have a MetadataT template argument for any metadata.
template <typename IrT, typename OptionsT, typename ResultsT = PassResults>
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
    VLOG(2) << absl::StreamFormat("Running %s [pass #%d]", long_name(),
                                  results->invocations.size());
    VLOG(3) << "Before:";
    XLS_VLOG_LINES(3, ir->DumpIr());
    int64_t ir_count_before = ir->GetNodeCount();

    XLS_ASSIGN_OR_RETURN(bool changed, RunInternal(ir, options, results),
                         _ << "Running pass #" << results->invocations.size()
                           << ": " << long_name() << " [short: " << short_name()
                           << "]");

    VLOG(3) << absl::StreamFormat("After [changed = %d]:", changed);
    XLS_VLOG_LINES(3, ir->DumpIr());
    // Perform a fast check nothing seems to have changed if we aren't told it
    // has.
    XLS_RET_CHECK(changed || ir_count_before == ir->GetNodeCount())
        << absl::StreamFormat(
               "Pass %s indicated IR unchanged, but IR is "
               "changed: [Before] %d nodes != [after] %d nodes",
               short_name(), ir_count_before, ir->GetNodeCount());
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
template <typename IrT, typename OptionsT, typename ResultsT = PassResults>
class InvariantCheckerBase {
 public:
  virtual ~InvariantCheckerBase() = default;
  virtual absl::Status Run(IrT* ir, const OptionsT& options,
                           ResultsT* results) const = 0;
};

// Data structure holding statistics about a particular pass.
struct SinglePassResult {
  // How many times the pass was run.
  int64_t run_count = 0;
  // How many runs changed the IR.
  int64_t changed_count = 0;
  // Aggregate transformation metrics across the runs.
  TransformMetrics metrics;
  // Total duration of the running of the pass.
  absl::Duration duration;
};

// Data structure returned by contains aggregate statistics about the passes run
// in a compound pass.
class CompoundPassResult {
 public:
  // Whether the IR was changed.
  bool changed() const { return changed_; }
  void set_changed(bool value) { changed_ = value; }

  // Add the results of a single run of a pass.
  void AddSinglePassResult(std::string_view pass_name, bool changed,
                           absl::Duration duration,
                           const TransformMetrics& metrics);

  // Accumulates the statistics in `other` into this one.
  void AccumulateCompoundPassResult(const CompoundPassResult& other);

  std::string ToString() const;

 private:
  bool changed_ = false;

  // Aggregate results for each pass. Indexed by short name.
  absl::flat_hash_map<std::string, SinglePassResult> pass_results_;
};

// CompoundPass is a container for other passes. For example, the scalar
// optimizer can be a compound pass holding many passes for scalar
// optimizations.
template <typename IrT, typename OptionsT, typename ResultsT = PassResults>
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
    return AddOwned(std::make_unique<T>(std::forward<Args>(args)...));
  }

  template <typename T>
  T* AddOwned(std::unique_ptr<T> pass) {
    T* out = pass.get();
    pass_ptrs_.push_back(out);
    passes_.emplace_back(std::move(pass));
    return out;
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
    XLS_ASSIGN_OR_RETURN(CompoundPassResult compound_result,
                         RunNested(ir, options, results, this->short_name(),
                                   /*invariant_checkers=*/{}));
    return compound_result.changed();
  }

  bool IsCompound() const override { return true; }

 protected:
  // Internal implementation of Run for compound passes. Invoked when a compound
  // pass is nested within another compound pass. Enables passing of invariant
  // checkers and name of the top-level pass to nested compound passes.
  virtual absl::StatusOr<CompoundPassResult> RunNested(
      IrT* ir, const OptionsT& options, ResultsT* results,
      std::string_view top_level_name,
      absl::Span<const InvariantChecker* const> invariant_checkers) const;

  // Dump the IR to a file in the given directory. Name is determined by the
  // various arguments passed in. File names will be lexicographically ordered
  // by package name and ordinal.
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
template <typename IrT, typename OptionsT, typename ResultsT = PassResults>
class FixedPointCompoundPassBase
    : public CompoundPassBase<IrT, OptionsT, ResultsT> {
 public:
  FixedPointCompoundPassBase(std::string_view short_name,
                             std::string_view long_name)
      : CompoundPassBase<IrT, OptionsT, ResultsT>(short_name, long_name) {}

 protected:
  absl::StatusOr<CompoundPassResult> RunNested(
      IrT* ir, const OptionsT& options, ResultsT* results,
      std::string_view top_level_name,
      absl::Span<const typename CompoundPassBase<
          IrT, OptionsT, ResultsT>::InvariantChecker* const>
          invariant_checkers) const override {
    bool local_changed = true;
    int64_t iteration_count = 0;
    CompoundPassResult aggregate_result;
    while (local_changed) {
      ++iteration_count;
      XLS_ASSIGN_OR_RETURN(
          CompoundPassResult compound_result,
          (CompoundPassBase<IrT, OptionsT, ResultsT>::RunNested(
              ir, options, results, top_level_name, invariant_checkers)),
          _ << "Running pass #" << results->invocations.size() << ": "
            << this->long_name() << " [short: " << this->short_name() << "]");
      local_changed = compound_result.changed();
      aggregate_result.AccumulateCompoundPassResult(compound_result);
    }
    VLOG(1) << absl::StreamFormat(
        "Fixed point compound pass %s iterated %d times.", this->long_name(),
        iteration_count);
    XLS_VLOG_LINES(2, aggregate_result.ToString());
    return aggregate_result;
  }
};

template <typename IrT, typename OptionsT, typename ResultsT>
absl::StatusOr<CompoundPassResult>
CompoundPassBase<IrT, OptionsT, ResultsT>::RunNested(
    IrT* ir, const OptionsT& options, ResultsT* results,
    std::string_view top_level_name,
    absl::Span<const InvariantChecker* const> invariant_checkers) const {
  VLOG(1) << "Running " << this->short_name() << " compound pass on package "
          << ir->name();
  VLOG(2) << "Start of compound pass " << this->short_name() << ":";
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

  CompoundPassResult aggregate_result;
  bool changed = false;
  for (const auto& pass : passes_) {
    VLOG(1) << absl::StreamFormat("Running %s (%s, #%d) pass on package %s",
                                  pass->long_name(), pass->short_name(),
                                  results->invocations.size(), ir->name());

    TransformMetrics before_metrics;
    if (VLOG_IS_ON(1)) {
      before_metrics = ir->transform_metrics();
    }

    if (!pass->IsCompound() && options.bisect_limit &&
        results->invocations.size() >= options.bisect_limit) {
      VLOG(1) << "Skipping pass " << pass->short_name()
              << " due to hitting bisect limit.";
      continue;
    }

    if (std::find_if(options.skip_passes.begin(), options.skip_passes.end(),
                     [&](const std::string& name) {
                       return pass->short_name() == name;
                     }) != options.skip_passes.end()) {
      VLOG(1) << "Skipping pass. Contained in skip_passes option.";
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
          CompoundPassResult compound_result,
          (down_cast<CompoundPassBase<IrT, OptionsT, ResultsT>*>(pass.get())
               ->RunNested(ir, options, results, top_level_name, checkers)),
          _ << "Running pass #" << results->invocations.size() << ": "
            << pass->long_name() << " [short: " << pass->short_name() << "]");
      pass_changed = compound_result.changed();
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
    changed = changed || pass_changed;
    TransformMetrics pass_metrics = ir->transform_metrics() - before_metrics;
    VLOG(1) << absl::StreamFormat(
        "[elapsed %s] Pass %s %s.", FormatDuration(duration),
        pass->short_name(),
        (pass_changed ? "changed IR" : "did not change IR"));
    if (pass_changed) {
      VLOG(1) << absl::StrFormat("Metrics: %s", pass_metrics.ToString());
    }
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

    aggregate_result.AddSinglePassResult(pass->short_name(), pass_changed,
                                         duration, pass_metrics);

    // Only run the verifiers if the pass changed.
    if (pass_changed) {
      absl::Time checker_start = absl::Now();
      XLS_RETURN_IF_ERROR(run_invariant_checkers(
          absl::StrFormat("after '%s' pass, dynamic pass #%d",
                          pass->long_name(), results->invocations.size() - 1)));
      VLOG(1) << absl::StreamFormat(
          "Ran invariant checkers [elapsed %s]",
          FormatDuration(absl::Now() - checker_start));
    }
    VLOG(5) << "After " << pass->long_name() << ":";
    XLS_VLOG_LINES(5, ir->DumpIr());
  }

  aggregate_result.set_changed(changed);
  return aggregate_result;
}

}  // namespace xls

#endif  // XLS_PASSES_PASS_BASE_H_
