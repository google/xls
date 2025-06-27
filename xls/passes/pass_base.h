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
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/passes/pass_metrics.pb.h"
#include "xls/passes/pass_pipeline.pb.h"

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
  // The short name of the pass.
  std::string pass_name;

  // Whether the IR was changed by the pass.
  bool ir_changed = false;

  // The run duration of the pass.
  absl::Duration run_duration;

  // Number of nodes added, removed, etc.
  TransformMetrics metrics;

  // For compound passes, this holds the invocation data for the nested passes.
  std::vector<PassInvocation> nested_invocations;

  // For fixed point compound passes this is the number of iterations of the
  // pass.
  int64_t fixed_point_iterations = 0;
};

inline std::ostream& operator<<(std::ostream& os,
                                const PassInvocation& invocation) {
  os << "PassInvocation(" << invocation.pass_name << ")";
  return os;
}

// A object to which metadata may be written in each pass invocation. This data
// structure is passed by mutable pointer to PassBase::Run.
struct PassResults {
  // This vector contains and entry for each invocation of each pass.
  PassInvocation invocation;

  // The total number of leaf-level (non-compound) pass invocations including
  // nested invocations.
  int64_t total_invocations = 0;

  // Return the latest invocation (including nested invocations).
  PassInvocation& GetLatestInvocation() {
    PassInvocation* inv = &invocation;
    while (!inv->nested_invocations.empty()) {
      inv = &inv->nested_invocations.back();
    }
    return *inv;
  }

  PassPipelineMetricsProto ToProto() const;
};

// A base class for abstractions which check invariants of the IR. These
// checkers are added to compound passes (pass pipelines) and run before and
// after each pass in the pipeline.
template <typename OptionsT, typename... ContextT>
class InvariantCheckerBase {
 public:
  virtual ~InvariantCheckerBase() = default;
  virtual absl::Status Run(Package* ir, const OptionsT& options,
                           PassResults* results,
                           ContextT&... context) const = 0;
};

// Base class for all compiler passes. Template parameters:
//
//   OptionsT : Options type passed as an immutable object to each invocation of
//     PassBase::Run. This type should be derived from PassOptions because
//     PassOptions contains fields required by CompoundPassBase when executing
//     pass pipelines.
//
//   ContextT : A variable number of types (usually zero or one) which are
//     passed as mutable objects to each invocation of PassBase::Run. These
//     types are used to share non-IR information between passes (e.g., sharing
//     lazily-populated query engines across an optimization pipeline).
template <typename OptionsT, typename... ContextT>
class PassBase {
 public:
  PassBase(std::string_view short_name, std::string_view long_name)
      : short_name_(short_name), long_name_(long_name) {}

  virtual ~PassBase() = default;

  const std::string& short_name() const { return short_name_; }
  const std::string& long_name() const { return long_name_; }

  // Generate a proto that can be used to reconstitute this pass. By default is
  // 'short_name()'
  // TODO(allight): This is not very elegant. Ideally the registry could handle
  // this? Doing it there would probably be even more weird though.
  virtual absl::StatusOr<PassPipelineProto::Element> ToProto() const {
    if (IsCompound()) {
      return absl::UnimplementedError(
          "Compound pass without overriden ToProto.");
    }
    PassPipelineProto::Element res;
    *res.mutable_pass_name() = short_name_;
    return res;
  }

  // Run the specific pass. Returns true if the graph was changed by the pass.
  // Typically the "changed" indicator is used to determine when to terminate
  // fixed point computation.
  virtual absl::StatusOr<bool> Run(Package* ir, const OptionsT& options,
                                   PassResults* results,
                                   ContextT&... context) const {
    VLOG(2) << absl::StreamFormat("Running %s [pass #%d]", long_name(),
                                  results->total_invocations);
    VLOG(3) << "Before:";
    XLS_VLOG_LINES(3, ir->DumpIr());
    int64_t ir_count_before = ir->GetNodeCount();

    XLS_ASSIGN_OR_RETURN(
        bool changed, RunInternal(ir, options, results, context...),
        _ << "Running pass #" << results->total_invocations << ": "
          << long_name() << " [short: " << short_name() << "]");

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

  // Runs the passes nested within this pass. This should only be called for
  // compound passes. This method is needed (as opposed to just calling Run)
  // because additional information such as the invariant checkers to run need
  // to be passed down.
  using InvariantChecker = InvariantCheckerBase<OptionsT, ContextT...>;
  virtual absl::StatusOr<bool> RunNested(
      Package* ir, const OptionsT& options, PassResults* results,
      ContextT&... context, PassInvocation& invocation,
      absl::Span<const InvariantChecker* const> invariant_checkers) const {
    return absl::InternalError(
        absl::StrFormat("Pass `%s` is not a compound pass", short_name()));
  }

 protected:
  // Derived classes should override this function which is invoked from Run.
  virtual absl::StatusOr<bool> RunInternal(Package* ir, const OptionsT& options,
                                           PassResults* results,
                                           ContextT&... context) const = 0;

  std::string short_name_;
  std::string long_name_;
};

template <typename OptionsT, typename... ContextT>
class WrapperPassBase final : public PassBase<OptionsT, ContextT...> {
 public:
  explicit WrapperPassBase(
      std::unique_ptr<PassBase<OptionsT, ContextT...>>&& base)
      : PassBase<OptionsT, ContextT...>(base->short_name(), base->long_name()),
        base_(std::move(base)) {}

  absl::StatusOr<PassPipelineProto::Element> ToProto() const final {
    return base_->ToProto();
  }

 protected:
  absl::StatusOr<bool> RunInternal(Package* ir, const OptionsT& options,
                                   PassResults* results,
                                   ContextT&... context) const final {
    return base_->Run(ir, options, results, context...);
  }

 private:
  std::unique_ptr<PassBase<OptionsT, ContextT...>> base_;
};

// CompoundPass is a container for other passes. For example, the scalar
// optimizer can be a compound pass holding many passes for scalar
// optimizations.
template <typename OptionsT, typename... ContextT>
class CompoundPassBase : public PassBase<OptionsT, ContextT...> {
 public:
  using Pass = PassBase<OptionsT, ContextT...>;
  using InvariantChecker = PassBase<OptionsT, ContextT...>::InvariantChecker;

  CompoundPassBase(std::string_view short_name, std::string_view long_name)
      : Pass(short_name, long_name) {}
  ~CompoundPassBase() override = default;

  absl::StatusOr<PassPipelineProto::Element> ToProto() const override {
    PassPipelineProto::Element res;
    for (const auto& p : this->passes()) {
      XLS_ASSIGN_OR_RETURN(*res.mutable_pipeline()->mutable_elements()->Add(),
                           p->ToProto());
    }
    *res.mutable_pipeline()->mutable_short_name() = this->short_name();
    *res.mutable_pipeline()->mutable_long_name() = this->long_name();
    return res;
  }

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
  //  pass->AddInvariantChecker<CheckStuff>(foo);
  //
  // Returns a pointer to the newly constructed pass.
  template <typename T, typename... Args>
  T* AddInvariantChecker(Args&&... args) {
    auto* checker = new T(std::forward<Args>(args)...);
    invariant_checkers_.emplace_back(checker);
    invariant_checker_ptrs_.push_back(checker);
    return checker;
  }

  // Adds a weak invariant checker to the compound pass. The invariant checker
  // is run at the start and end of the compound pass, and not for nested
  // compound passes. Arguments to method are the arguments to the invariant
  // checker constructor. Example usage:
  //
  //  pass->AddWeakInvariantChecker<CheckStuff>(foo);
  //
  // Returns a pointer to the newly constructed pass.
  template <typename T, typename... Args>
  T* AddWeakInvariantChecker(Args&&... args) {
    auto* checker = new T(std::forward<Args>(args)...);
    weak_invariant_checkers_.emplace_back(checker);
    weak_invariant_checker_ptrs_.push_back(checker);
    return checker;
  }

  absl::StatusOr<bool> RunInternal(Package* ir, const OptionsT& options,
                                   PassResults* results,
                                   ContextT&... context) const override {
    if (!options.ir_dump_path.empty() && results->total_invocations == 0) {
      // Start of the top-level pass. Dump IR.
      XLS_RETURN_IF_ERROR(DumpIr(options.ir_dump_path, ir, this->short_name(),
                                 "start",
                                 /*ordinal=*/0, /*changed=*/false));
    }
    results->invocation.pass_name = this->short_name();
    return RunNested(ir, options, results, context..., results->invocation,
                     /*invariant_checkers=*/{});
  }

  bool IsCompound() const override { return true; }

  // Internal implementation of Run for compound passes. Invoked when a compound
  // pass is nested within another compound pass. Enables passing of invariant
  // checkers and name of the top-level pass to nested compound passes.
  absl::StatusOr<bool> RunNested(Package* ir, const OptionsT& options,
                                 PassResults* results, ContextT&... context,
                                 PassInvocation& invocation,
                                 absl::Span<const InvariantChecker* const>
                                     invariant_checkers) const override;

 protected:
  // Dump the IR to a file in the given directory. Name is determined by the
  // various arguments passed in. File names will be lexicographically ordered
  // by package name and ordinal.
  absl::Status DumpIr(const std::filesystem::path& ir_dump_path, Package* ir,
                      std::string_view top_level_name, std::string_view tag,
                      int64_t ordinal, bool changed) const {
    std::filesystem::path path =
        ir_dump_path / absl::StrFormat("%s.%05d.%s.%s.%s.ir", ir->name(),
                                       ordinal, tag, top_level_name,
                                       changed ? "changed" : "unchanged");
    return SetFileContents(path, ir->DumpIr());
  }

  std::vector<std::unique_ptr<Pass>> passes_;
  std::vector<Pass*> pass_ptrs_;

  std::vector<std::unique_ptr<InvariantChecker>> invariant_checkers_;
  std::vector<const InvariantChecker*> invariant_checker_ptrs_;

  std::vector<std::unique_ptr<InvariantChecker>> weak_invariant_checkers_;
  std::vector<const InvariantChecker*> weak_invariant_checker_ptrs_;
};

// A compound pass which runs its set of passes to fixed point.
template <typename OptionsT, typename... ContextT>
class FixedPointCompoundPassBase
    : public CompoundPassBase<OptionsT, ContextT...> {
 public:
  FixedPointCompoundPassBase(std::string_view short_name,
                             std::string_view long_name)
      : CompoundPassBase<OptionsT, ContextT...>(short_name, long_name) {}

  absl::StatusOr<PassPipelineProto::Element> ToProto() const override {
    PassPipelineProto::Element res;
    for (const auto& p : this->passes()) {
      XLS_ASSIGN_OR_RETURN(*res.mutable_fixedpoint()->mutable_elements()->Add(),
                           p->ToProto());
    }
    *res.mutable_fixedpoint()->mutable_short_name() = this->short_name();
    *res.mutable_fixedpoint()->mutable_long_name() = this->long_name();
    return res;
  }

  absl::StatusOr<bool> RunNested(
      Package* ir, const OptionsT& options, PassResults* results,
      ContextT&... context, PassInvocation& invocation,
      absl::Span<const typename PassBase<OptionsT,
                                         ContextT...>::InvariantChecker* const>
          invariant_checkers) const override {
    bool local_changed = true;
    int64_t iteration_count = 0;
    while (local_changed) {
      ++iteration_count;
      XLS_ASSIGN_OR_RETURN(local_changed,
                           (CompoundPassBase<OptionsT, ContextT...>::RunNested(
                               ir, options, results, context..., invocation,
                               invariant_checkers)),
                           _ << "Running pass #" << results->total_invocations
                             << ": " << this->long_name()
                             << " [short: " << this->short_name() << "]");
    }
    invocation.fixed_point_iterations = iteration_count;
    VLOG(1) << absl::StreamFormat(
        "Fixed point compound pass %s iterated %d times.", this->long_name(),
        iteration_count);
    return iteration_count > 1;
  }
};

template <typename OptionsT, typename... ContextT>
absl::StatusOr<bool> CompoundPassBase<OptionsT, ContextT...>::RunNested(
    Package* ir, const OptionsT& options, PassResults* results,
    ContextT&... context, PassInvocation& invocation,
    absl::Span<const InvariantChecker* const> invariant_checkers) const {
  VLOG(1) << "Running " << this->short_name() << " compound pass on package "
          << ir->name();
  VLOG(2) << "Start of compound pass " << this->short_name() << ":";
  XLS_VLOG_LINES(5, ir->DumpIr());

  auto make_invariant_checker_runner =
      [&](const std::vector<const InvariantChecker*>& checkers) {
        return [&](std::string_view str_context) -> absl::Status {
          for (const auto& checker : checkers) {
            absl::Status status =
                checker->Run(ir, options, results, context...);
            if (!status.ok()) {
              return absl::Status(
                  status.code(),
                  absl::StrFormat("%s; [%s]", status.message(), str_context));
            }
          }
          return absl::OkStatus();
        };
      };

  // Invariant checkers may be passed in from parent compound passes or
  // contained by this pass itself. Merge them together.
  std::vector<const InvariantChecker*> checkers(invariant_checkers.begin(),
                                                invariant_checkers.end());
  checkers.insert(checkers.end(), invariant_checker_ptrs_.begin(),
                  invariant_checker_ptrs_.end());

  auto run_invariant_checkers = make_invariant_checker_runner(checkers);
  auto run_weak_invariant_checkers =
      make_invariant_checker_runner(weak_invariant_checker_ptrs_);

  Stopwatch compound_stopwatch;
  TransformMetrics compound_before_metrics = ir->transform_metrics();

  Stopwatch invariant_checker_stopwatch;
  XLS_RETURN_IF_ERROR(run_weak_invariant_checkers(
      absl::StrCat("start of compound pass '", this->long_name(), "'")));
  XLS_RETURN_IF_ERROR(run_invariant_checkers(
      absl::StrCat("start of compound pass '", this->long_name(), "'")));

  bool changed = false;
  for (const auto& pass : passes_) {
    VLOG(1) << absl::StreamFormat("Running %s (%s, #%d) pass on package %s",
                                  pass->long_name(), pass->short_name(),
                                  results->total_invocations, ir->name());

    TransformMetrics before_metrics = ir->transform_metrics();

    if (!pass->IsCompound() && options.bisect_limit &&
        results->total_invocations >= options.bisect_limit) {
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
    Stopwatch pass_stopwatch;
    bool pass_changed;
    PassInvocation nested_pass_invocation{.pass_name = pass->short_name()};
    if (pass->IsCompound()) {
      XLS_ASSIGN_OR_RETURN(pass_changed,
                           pass->RunNested(ir, options, results, context...,
                                           nested_pass_invocation, checkers),
                           _ << "Running pass #" << results->total_invocations
                             << ": " << pass->long_name()
                             << " [short: " << pass->short_name() << "]");
    } else {
      XLS_ASSIGN_OR_RETURN(pass_changed,
                           pass->Run(ir, options, results, context...));
    }
    absl::Duration duration = pass_stopwatch.GetElapsedTime();
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
      if (!options.ir_dump_path.empty()) {
        XLS_RETURN_IF_ERROR(DumpIr(options.ir_dump_path, ir,
                                   invocation.pass_name,
                                   absl::StrCat("after_", pass->short_name()),
                                   /*ordinal=*/results->total_invocations,
                                   /*changed=*/pass_changed));
      }

      // Only run the verifiers if the pass changed anything. Also, only run
      // after the non-compound passes. Running after compound passes is
      // necessarily redundant because the invariant checker would have been run
      // within the compound pass.
      if (pass_changed) {
        invariant_checker_stopwatch.Reset();
        XLS_RETURN_IF_ERROR(run_invariant_checkers(absl::StrFormat(
            "after '%s' pass, dynamic pass #%d", pass->long_name(),
            results->total_invocations - 1)));
        VLOG(1) << absl::StreamFormat(
            "Ran invariant checkers [elapsed %s]",
            FormatDuration(invariant_checker_stopwatch.GetElapsedTime()));
      }
    }

    VLOG(5) << "After " << pass->long_name() << ":";
    XLS_VLOG_LINES(5, ir->DumpIr());

    nested_pass_invocation.ir_changed = pass_changed;
    nested_pass_invocation.run_duration = duration;
    nested_pass_invocation.metrics = pass_metrics;
    invocation.nested_invocations.push_back(std::move(nested_pass_invocation));

    if (!pass->IsCompound()) {
      ++results->total_invocations;
    }
  }

  if (changed) {
    XLS_RETURN_IF_ERROR(run_weak_invariant_checkers(
        absl::StrCat("end of compound pass '", this->long_name(), "'")));
  }

  invocation.ir_changed = changed;
  invocation.run_duration = compound_stopwatch.GetElapsedTime();
  invocation.metrics = ir->transform_metrics() - compound_before_metrics;

  return changed;
}

// Abstract base class for passes that operate at function/proc scope. The
// derived class must define RunOnFunctionBaseInternal.
template <typename OptionsT, typename... ContextT>
class FunctionBasePass : public PassBase<OptionsT, ContextT...> {
 public:
  using PassBase<OptionsT, ContextT...>::PassBase;

  // Runs the pass on a single function/proc.
  absl::StatusOr<bool> RunOnFunctionBase(FunctionBase* f,
                                         const OptionsT& options,
                                         PassResults* results,
                                         ContextT&... context) const {
    VLOG(2) << absl::StreamFormat("Running %s on function_base %s [pass #%d]",
                                  this->long_name(), f->name(),
                                  results->total_invocations);
    VLOG(3) << "Before:";
    XLS_VLOG_LINES(3, f->DumpIr());

    XLS_ASSIGN_OR_RETURN(bool changed, RunOnFunctionBaseInternal(
                                           f, options, results, context...));

    VLOG(3) << absl::StreamFormat("After [changed = %d]:", changed);
    XLS_VLOG_LINES(3, f->DumpIr());
    return changed;
  }

 protected:
  // Iterates over each function and proc in the package calling
  // RunOnFunctionBase.
  absl::StatusOr<bool> RunInternal(Package* p, const OptionsT& options,
                                   PassResults* results,
                                   ContextT&... context) const override {
    bool changed = false;
    for (FunctionBase* f : p->GetFunctionBases()) {
      XLS_ASSIGN_OR_RETURN(
          bool function_changed,
          RunOnFunctionBaseInternal(f, options, results, context...));
      if (function_changed) {
        GcAfterFunctionBaseChange(p, context...);
      }
      changed = changed || function_changed;
    }
    return changed;
  }

  virtual absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptionsT& options, PassResults* results,
      ContextT&... context) const = 0;

  // A subclass may optionally override this to garbage collect context data
  // after a `RunOnFunctionBaseInternal` call that makes a change.
  virtual void GcAfterFunctionBaseChange(Package* p,
                                         ContextT&... context) const {}
};

// Abstract base class for passes that operate at proc scope. The derived class
// must define RunOnProcInternal.
template <typename OptionsT, typename... ContextT>
class ProcPass : public PassBase<OptionsT, ContextT...> {
 public:
  using PassBase<OptionsT, ContextT...>::PassBase;

  // Run the pass on a single proc.
  absl::StatusOr<bool> RunOnProc(Proc* proc, const OptionsT& options,
                                 PassResults* results,
                                 ContextT&... context) const {
    VLOG(2) << absl::StreamFormat("Running %s on proc %s [pass #%d]",
                                  this->long_name(), proc->name(),
                                  results->total_invocations);
    VLOG(3) << "Before:";
    XLS_VLOG_LINES(3, proc->DumpIr());

    XLS_ASSIGN_OR_RETURN(bool changed,
                         RunOnProcInternal(proc, options, results, context...));

    VLOG(3) << absl::StreamFormat("After [changed = %d]:", changed);
    XLS_VLOG_LINES(3, proc->DumpIr());
    return changed;
  }

 protected:
  // Iterates over each proc in the package calling RunOnProc.
  absl::StatusOr<bool> RunInternal(Package* p, const OptionsT& options,
                                   PassResults* results,
                                   ContextT&... context) const override {
    bool changed = false;
    for (const auto& proc : p->procs()) {
      XLS_ASSIGN_OR_RETURN(
          bool proc_changed,
          RunOnProcInternal(proc.get(), options, results, context...));
      if (proc_changed) {
        GcAfterProcChange(p, context...);
      }
      changed = changed || proc_changed;
    }
    return changed;
  }

  virtual absl::StatusOr<bool> RunOnProcInternal(
      Proc* proc, const OptionsT& options, PassResults* results,
      ContextT&... context) const = 0;

  // A subclass may optionally override this to garbage collect context data
  // after a `RunOnProcInternal` call that makes a change.
  virtual void GcAfterProcChange(Package* p, ContextT&... context) const {}
};

}  // namespace xls

#endif  // XLS_PASSES_PASS_BASE_H_
