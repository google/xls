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

#include "absl/log/check.h"
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
#include "xls/ir/block.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/passes/pass_metrics.pb.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/passes/tools/passes_profile.h"

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

class PassResults;

// An object containing information about the invocation of a pass (single call
// to PassBase::Run).
class PassInvocation {
 public:
  explicit PassInvocation(PassInvocation* parent, std::string_view pass_name,
                          int64_t pass_number)
      : pass_name_(pass_name), pass_numbers_({pass_number}), parent_(parent) {}
  PassInvocation(const PassInvocation&) = delete;
  PassInvocation& operator=(const PassInvocation&) = delete;
  PassInvocation(PassInvocation&&) = default;
  PassInvocation& operator=(PassInvocation&&) = default;

  bool is_root() const { return parent_ == nullptr; }
  TransformMetrics& metrics() { return metrics_; }
  const TransformMetrics& metrics() const { return metrics_; }
  const absl::Duration& run_duration() const { return run_duration_; }
  absl::Duration& run_duration() { return run_duration_; }
  std::string_view pass_name() const { return pass_name_; }
  bool ir_changed() const { return ir_changed_; }
  int64_t fixed_point_iterations() const { return fixed_point_iterations_; }
  absl::Span<int64_t const> all_pass_numbers() const { return pass_numbers_; }
  // The initial pass number.
  int64_t pass_number() const { return pass_numbers_.front(); }
  void set_ir_changed(bool ir_changed) { ir_changed_ = ir_changed; }
  void IncrementFixedPointIterations() { fixed_point_iterations_++; }
  PassInvocation& parent() {
    CHECK(!is_root()) << "Attempting to get parent of root invocation.";
    return *parent_;
  }
  const PassInvocation& parent() const {
    CHECK(!is_root()) << "Attempting to get parent of root invocation.";
    return *parent_;
  }
  absl::Span<const std::unique_ptr<PassInvocation>> nested_invocations() const {
    return nested_invocations_;
  }

 private:
  // The short name of the pass.
  std::string pass_name_;

  // Whether the IR was changed by the pass.
  bool ir_changed_ = false;

  // The run duration of the pass.
  absl::Duration run_duration_;

  // Number of nodes added, removed, etc.
  TransformMetrics metrics_;

  // For compound passes, this holds the invocation data for the nested passes.
  std::vector<std::unique_ptr<PassInvocation>> nested_invocations_;

  // For fixed point compound passes this is the number of iterations of the
  // pass.
  int64_t fixed_point_iterations_ = 0;

  // How many passes ran before this pass started. Fixedpoint will have
  // multiple.
  std::vector<int64_t> pass_numbers_;

  PassInvocation* parent_ = nullptr;

  friend class PassResults;
};

inline std::ostream& operator<<(std::ostream& os,
                                const PassInvocation& invocation) {
  os << "PassInvocation(" << invocation.pass_name() << ")";
  return os;
}

// A object to which metadata may be written in each pass invocation. This data
// structure is passed by mutable pointer to PassBase::Run.
class PassResults {
 public:
  PassResults()
      : root_invocation_(std::make_unique<PassInvocation>(nullptr, "root", -1)),
        latest_(root_invocation_.get()) {};
  PassResults(const PassResults&) = delete;
  PassResults& operator=(const PassResults&) = delete;
  PassResults(PassResults&&) = default;
  PassResults& operator=(PassResults&&) = default;

  // Return the current invocation.
  PassInvocation& current_invocation() { return *latest_; }

  PassInvocation& PushInvocation(std::string_view pass_name) {
    total_invocations_++;
    latest_ = latest_->nested_invocations_
                  .emplace_back(std::make_unique<PassInvocation>(
                      latest_, pass_name, total_invocations_ - 1))
                  .get();
    return *latest_;
  }
  PassInvocation& PopInvocation() {
    CHECK(!latest_->is_root()) << "Attempting to pop the root invocation.";
    latest_ = &latest_->parent();
    finished_invocations_++;
    return *latest_;
  }
  void RestartCurrentInvocation() {
    CHECK(!latest_->is_root()) << "Attempting to restart the root invocation.";
    latest_->pass_numbers_.push_back(total_invocations_);
    latest_->fixed_point_iterations_++;
    total_invocations_++;
  }

  PassPipelineMetricsProto ToProto() const;
  const PassInvocation& root_invocation() const { return *root_invocation_; }
  int64_t total_invocations() const { return total_invocations_; }
  int64_t finished_invocations() const { return finished_invocations_; }

 private:
  // This vector contains and entry for each invocation of each pass.
  std::unique_ptr<PassInvocation> root_invocation_;
  PassInvocation* latest_;

  // The total number of leaf-level (non-compound) pass invocations including
  // nested invocations.
  int64_t total_invocations_ = 0;

  // The number of invocations which have finished. This is only tracked to
  // ensure that --ir_dump_path is numbered sequentially.
  // TODO(allight): We should probably just remove this and rethink the
  // numbering scheme of --ir_dump_path since it doesn't match with what
  // --passes_bisect_limit and other logging says the pass numbering is.
  int64_t finished_invocations_ = 0;
};

// RAII helper for holding pass invocation information. It holds and tracks pass
// results and also sets the various pprof tags for the pass.
class ScopedPassInvocation {
 public:
  ScopedPassInvocation(PassResults* results, std::string_view pass_name,
                       Package* ir)
      : results_(results),
        invocation_(results->PushInvocation(pass_name)),
        ir_(ir),
        before_metrics_(ir->transform_metrics()) {
    RecordPassEntry(pass_name);
    RecordPassAnnotation(pass_profile::kNodeCountBefore, ir->GetNodeCount());
  }
  ~ScopedPassInvocation() {
    invocation_.metrics() = (ir_->transform_metrics() - before_metrics_);
    invocation_.run_duration() = pass_stopwatch_.GetElapsedTime();
    VLOG(1) << absl::StreamFormat(
        "[elapsed %s] Pass %s %s.", FormatDuration(invocation_.run_duration()),
        invocation_.pass_name(),
        (invocation_.ir_changed() ? "changed IR" : "did not change IR"));
    if (invocation_.ir_changed()) {
      VLOG(1) << absl::StrFormat("Metrics: %s",
                                 invocation_.metrics().ToString());
    }
    RecordPassAnnotation(pass_profile::kNodeCountAfter, ir_->GetNodeCount());
    ExitPass(changed_);
    results_->PopInvocation();
  }
  PassResults* results() { return results_; }
  PassInvocation& invocation() { return invocation_; }
  const PassInvocation& invocation() const { return invocation_; }
  PassInvocation* operator->() { return &invocation_; }
  PassInvocation& operator*() { return invocation_; }
  void set_ir_changed(bool changed) {
    changed_ = changed;
    invocation_.set_ir_changed(changed);
  }
  class ChangedRef {
   public:
    ChangedRef(ScopedPassInvocation& invocation, bool& changed)
        : invocation_(invocation), changed_(changed) {}
    operator bool() const { return changed_; }
    ChangedRef& operator=(bool other) {
      changed_ = other;
      invocation_.set_ir_changed(other);
      return *this;
    }

   private:
    ScopedPassInvocation& invocation_;
    bool& changed_;
  };
  ChangedRef changed() { return ChangedRef(*this, changed_); }

 private:
  PassResults* results_;
  PassInvocation& invocation_;
  Package* ir_;
  bool changed_ = false;
  TransformMetrics before_metrics_;
  Stopwatch pass_stopwatch_;
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
template <typename OptionsT, typename... ContextT>
class WrapperPassBase;

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
  using InvariantChecker = InvariantCheckerBase<OptionsT, ContextT...>;

  PassBase(std::string_view short_name, std::string_view long_name)
      : short_name_(short_name), long_name_(long_name) {}

  virtual ~PassBase() = default;

  virtual std::string short_name() const { return short_name_; }
  // Get the short name of the underlying pass. Used to identify wrapper passes
  // and so on.
  virtual std::string_view base_short_name() const { return short_name_; }
  virtual std::string long_name() const { return long_name_; }

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
  //
  // This is the only entrypoint for a pass. It records invocation metrics and
  // the like and then calls RunInternal which must perform the actual pass.
  //
  // TODO(allight): We should change 'results' to a reference since it cannot
  // (and never really should) be null.
  absl::StatusOr<bool> Run(Package* ir, const OptionsT& options,
                           PassResults* results, ContextT&... context) const {
    XLS_RET_CHECK(results != nullptr) << "Results cannot be null";
    if (options.bisect_limit &&
        results->total_invocations() >= options.bisect_limit) {
      VLOG(1) << "Skipping pass " << short_name()
              << " due to hitting bisect limit.";
      return false;
    }

    std::string_view base_short_name = this->base_short_name();
    if (std::find_if(options.skip_passes.begin(), options.skip_passes.end(),
                     [&](const std::string& name) {
                       return base_short_name == name;
                     }) != options.skip_passes.end()) {
      VLOG(1) << "Skipping pass \'" << this->short_name()
              << "\'. Contained in skip_passes option.";
      return false;
    }

    ScopedPassInvocation invocation(results, base_short_name, ir);
    VLOG(2) << absl::StreamFormat("Running %s [pass #%d]", long_name(),
                                  invocation->pass_number());
    VLOG(3) << "Before:";
    XLS_VLOG_LINES(3, ir->DumpIr());
    int64_t ir_count_before = ir->GetNodeCount();

#ifdef DEBUG
    // Verify that the IR should change iff Run returns true. This is slow, so
    // do not check it in optimized builds.
    std::string ir_before = ir->DumpIr();
#endif
    XLS_ASSIGN_OR_RETURN(
        invocation.changed(), RunInternal(ir, options, results, context...),
        _ << "Running pass #" << invocation->pass_number() << ": "
          << long_name() << " [short: " << short_name() << "]");
    VLOG(3) << absl::StreamFormat("After [changed = %d]:",
                                  invocation.changed());
    XLS_VLOG_LINES(3, ir->DumpIr());
    if (!options.ir_dump_path.empty()) {
      XLS_RETURN_IF_ERROR(
          // NB Number this as the total invocation number to ensure they go up
          // uniformly.
          // TODO(allight): We might want to change this to add the pass number
          // as well so it can more easily be used for bisect-limit etc.
          DumpIr(options.ir_dump_path, ir, invocation->parent().pass_name(),
                 absl::StrCat("after_", short_name()),
                 /*ordinal=*/results->finished_invocations(),
                 /*changed=*/invocation.changed()));
    }
    // Perform a fast check nothing seems to have changed if we aren't told it
    // has.
    XLS_RET_CHECK(invocation.changed() || ir_count_before == ir->GetNodeCount())
        << absl::StreamFormat(
               "Pass %s indicated IR unchanged, but IR is "
               "changed: [Before] %d nodes != [after] %d nodes",
               short_name(), ir_count_before, ir->GetNodeCount());

    // Only run the verifiers if the pass changed anything. Also, only run
    // after the non-compound passes. Running after compound passes is
    // necessarily redundant because the invariant checker would have been run
    // within the compound pass.
    if (invocation.changed() && !IsCompound()) {
      Stopwatch invariant_checker_stopwatch;
      for (const auto& checker : invariant_checker_ptrs_) {
        XLS_RETURN_IF_ERROR(checker->Run(ir, options, results, context...))
            << "after '" << long_name() << "' pass, dynamic pass #"
            << results->current_invocation().pass_number();
      }
      VLOG(1) << absl::StreamFormat(
          "Ran invariant checkers [elapsed %s]",
          FormatDuration(invariant_checker_stopwatch.GetElapsedTime()));
    }
#ifdef DEBUG
    std::string ir_after = ir->DumpIr();
    if (changed) {
      if (ir_before == ir_after) {
        return absl::InternalError(absl::StrFormat(
            "Pass %s indicated IR changed, but IR is unchanged:\n\n%s",
            short_name(), ir_before));
      }
    } else {
      if (ir_before != ir_after) {
        return absl::InternalError(
            absl::StrFormat("Pass %s indicated IR unchanged, but IR is "
                            "changed:\n\n[Before]\n%s  !=\n[after]\n%s",
                            short_name(), ir_before, ir_after));
      }
    }
#endif
    return invocation.changed();
  }

  // Returns true if this is a compound pass.
  virtual bool IsCompound() const { return false; }

  // Adds an invariant checker to the pass. The invariant checker is
  // run after the pass if it changed anything. For compound passes they are
  // also run before anything is run. Arguments to method are the arguments to
  // the invariant checker constructor. Example usage:
  //
  //  pass->AddInvariantChecker<CheckStuff>(foo);
  //
  // Returns a pointer to the newly constructed pass.
  template <typename T, typename... Args>
  T* AddInvariantChecker(Args&&... args) {
    auto checker = std::make_unique<T>(std::forward<Args>(args)...);
    T* out = checker.get();
    invariant_checkers_.emplace_back(std::move(checker));
    AddInvariantCheckerPtr(out);
    return out;
  }

  virtual void AddInvariantCheckerPtr(const InvariantChecker* checker) {
    invariant_checker_ptrs_.push_back(checker);
  }

 protected:
  // Derived classes should override this function which is invoked from Run.
  virtual absl::StatusOr<bool> RunInternal(Package* ir, const OptionsT& options,
                                           PassResults* results,
                                           ContextT&... context) const = 0;

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

  std::string short_name_;
  std::string long_name_;

  std::vector<std::unique_ptr<InvariantChecker>> invariant_checkers_;
  std::vector<const InvariantChecker*> invariant_checker_ptrs_;

  friend class WrapperPassBase<OptionsT, ContextT...>;
};

template <typename OptionsT, typename... ContextT>
class WrapperPassBase : public PassBase<OptionsT, ContextT...> {
 public:
  explicit WrapperPassBase(
      std::unique_ptr<PassBase<OptionsT, ContextT...>>&& base)
      : PassBase<OptionsT, ContextT...>(base->short_name(), base->long_name()),
        base_(std::move(base)) {}

  absl::StatusOr<PassPipelineProto::Element> ToProto() const override {
    return base_->ToProto();
  }
  std::string_view base_short_name() const final {
    return base_->base_short_name();
  }

  void AddInvariantCheckerPtr(
      const PassBase<OptionsT, ContextT...>::InvariantChecker* checker)
      override {
    // Let it propagate down.
    base_->AddInvariantCheckerPtr(checker);
    PassBase<OptionsT, ContextT...>::AddInvariantCheckerPtr(checker);
  }

  PassBase<OptionsT, ContextT...>* base_pass() const { return base_.get(); }

 protected:
  absl::StatusOr<bool> RunInternal(Package* ir, const OptionsT& options,
                                   PassResults* results,
                                   ContextT&... context) const override {
    // Wrapper passes just delegate to the base pass. Since we've already got
    // the 'Run()' bits we can just call straight into RunInternal. This is a
    // special exception because it's a common use case and we don't want to
    // pollute the logs.
    return base_->RunInternal(ir, options, results, context...);
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
    for (const auto& checker : this->invariant_checker_ptrs_) {
      pass_ptrs_.back()->AddInvariantCheckerPtr(checker);
    }
    return out;
  }

  void AddInvariantCheckerPtr(const InvariantChecker* checker) override {
    for (auto& pass : passes_) {
      pass->AddInvariantCheckerPtr(checker);
    }
    Pass::AddInvariantCheckerPtr(checker);
  }

  absl::Span<Pass* const> passes() const { return pass_ptrs_; }
  absl::Span<Pass*> passes() { return absl::Span<Pass*>(pass_ptrs_); }

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
    auto checker = std::make_unique<T>(std::forward<Args>(args)...);
    T* out = checker.get();
    weak_invariant_checker_ptrs_.push_back(checker.get());
    weak_invariant_checkers_.emplace_back(std::move(checker));
    return out;
  }

  absl::StatusOr<bool> RunInternal(Package* ir, const OptionsT& options,
                                   PassResults* results,
                                   ContextT&... context) const final {
    if (!options.ir_dump_path.empty() && results->finished_invocations() == 0) {
      // Start of the top-level pass. Dump IR.
      XLS_RETURN_IF_ERROR(this->DumpIr(options.ir_dump_path, ir,
                                       this->short_name(), "start",
                                       /*ordinal=*/0, /*changed=*/false));
    }
    return RunNested(ir, options, results, context...);
  }

  bool IsCompound() const override { return true; }

 protected:
  // Internal implementation of Run for compound passes. Invoked when a compound
  // pass is nested within another compound pass. Enables passing of invariant
  // checkers and name of the top-level pass to nested compound passes.
  virtual absl::StatusOr<bool> RunNested(Package* ir, const OptionsT& options,
                                         PassResults* results,
                                         ContextT&... context) const;

  std::vector<std::unique_ptr<Pass>> passes_;
  std::vector<Pass*> pass_ptrs_;

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

  absl::StatusOr<bool> RunNested(Package* ir, const OptionsT& options,
                                 PassResults* results,
                                 ContextT&... context) const final {
    RecordPassAnnotation(pass_profile::kFixedpoint, "true");
    bool local_changed = true;
    // Mark the first fixed point iteration.
    results->current_invocation().IncrementFixedPointIterations();
    while (local_changed) {
      XLS_ASSIGN_OR_RETURN(
          local_changed,
          (CompoundPassBase<OptionsT, ContextT...>::RunNested(
              ir, options, results, context...)),
          _ << "Running fixed-point pass #"
            << results->current_invocation().pass_name()
            << " [pass #: " << results->total_invocations() << ", iteration #"
            << results->current_invocation().fixed_point_iterations() << "]: "
            << this->long_name() << " [short: " << this->short_name() << "]");
      if (options.bisect_limit &&
          results->total_invocations() >= options.bisect_limit) {
        VLOG(1) << "Skipping remaining runs of " << this->short_name()
                << " due to hitting bisect limit.";
        break;
      }
      if (local_changed) {
        if (!options.ir_dump_path.empty()) {
          XLS_RETURN_IF_ERROR(this->DumpIr(
              options.ir_dump_path, ir, this->short_name(),
              absl::StrFormat(
                  "after_%s_fixedpoint_iteration_%05d", this->short_name(),
                  results->current_invocation().fixed_point_iterations()),
              /*ordinal=*/results->finished_invocations(),
              /*changed=*/true));
        }
        results->RestartCurrentInvocation();
      }
    }
    VLOG(1) << absl::StreamFormat(
        "Fixed point compound pass %s iterated %d times.", this->long_name(),
        results->current_invocation().fixed_point_iterations());
    return results->current_invocation().fixed_point_iterations() > 1;
  }
};

template <typename OptionsT, typename... ContextT>
absl::StatusOr<bool> CompoundPassBase<OptionsT, ContextT...>::RunNested(
    Package* ir, const OptionsT& options, PassResults* results,
    ContextT&... context) const {
  RecordPassAnnotation(pass_profile::kCompound, "true");
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

  auto run_weak_invariant_checkers =
      make_invariant_checker_runner(weak_invariant_checker_ptrs_);
  {
    auto run_invariant_checkers =
        make_invariant_checker_runner(this->invariant_checker_ptrs_);

    Stopwatch invariant_checker_stopwatch;
    XLS_RETURN_IF_ERROR(run_weak_invariant_checkers(
        absl::StrCat("start of compound pass '", this->long_name(), "'")));
    XLS_RETURN_IF_ERROR(run_invariant_checkers(
        absl::StrCat("start of compound pass '", this->long_name(), "'")));
    VLOG(1) << absl::StreamFormat(
        "Ran invariant checkers [elapsed %s]",
        FormatDuration(invariant_checker_stopwatch.GetElapsedTime()));
  }

  bool changed = false;
  for (const auto& pass : passes_) {
    VLOG(1) << absl::StreamFormat(
        "Running %s (%s, #%d) pass on package %s as part of compound pass %s "
        "[pass #: %d]",
        pass->long_name(), pass->short_name(), results->total_invocations(),
        ir->name(), this->short_name(),
        results->current_invocation().pass_number());

    XLS_ASSIGN_OR_RETURN(
        bool pass_changed, pass->Run(ir, options, results, context...),
        _ << "Failed as part of compound pass " << this->short_name() << " #"
          << results->current_invocation().pass_number());
    changed = changed || pass_changed;
  }

  if (changed) {
    XLS_RETURN_IF_ERROR(run_weak_invariant_checkers(
        absl::StrCat("end of compound pass '", this->long_name(), "'")));
  }

  return changed;
}

// Abstract base class for passes that operate at function/proc/block scope. The
// derived class must define RunOnFunctionBaseInternal.
template <typename OptionsT, typename... ContextT>
class FunctionBasePass : public PassBase<OptionsT, ContextT...> {
 public:
  using PassBase<OptionsT, ContextT...>::PassBase;

  // Runs the pass on a single function/proc/block.
  //
  // This is only for testing use.
  absl::StatusOr<bool> RunOnFunctionBase(FunctionBase* f,
                                         const OptionsT& options,
                                         PassResults* results,
                                         ContextT&... context) const;

 protected:
  // Iterates over each function and proc in the package calling
  // RunOnFunctionBase.
  absl::StatusOr<bool> RunInternal(Package* p, const OptionsT& options,
                                   PassResults* results,
                                   ContextT&... context) const final {
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

template <typename OptionsT, typename... ContextT>
absl::StatusOr<bool> FunctionBasePass<OptionsT, ContextT...>::RunOnFunctionBase(
    FunctionBase* f, const OptionsT& options, PassResults* results,
    ContextT&... context) const {
  class PassRunner : public FunctionBasePass<OptionsT, ContextT...> {
   public:
    PassRunner(const FunctionBasePass<OptionsT, ContextT...>& pass,
               FunctionBase* f)
        : FunctionBasePass<OptionsT, ContextT...>(pass.short_name(),
                                                  pass.long_name()),
          pass_(pass),
          f_(f) {}
    absl::StatusOr<bool> RunOnFunctionBaseInternal(
        FunctionBase* f, const OptionsT& options, PassResults* results,
        ContextT&... context) const override {
      if (f != f_) {
        return false;
      }
      return pass_.RunOnFunctionBaseInternal(f, options, results, context...);
    }
    void GcAfterFunctionBaseChange(Package* p,
                                   ContextT&... context) const override {
      pass_.GcAfterFunctionBaseChange(p, context...);
    }

   private:
    const FunctionBasePass<OptionsT, ContextT...>& pass_;
    FunctionBase* f_;
  };
  return PassRunner(*this, f).Run(f->package(), options, results, context...);
}

// Abstract base class for passes that operate at function scope. The derived
// class must define RunOnFunctionInternal.
template <typename OptionsT, typename... ContextT>
class FunctionPass : public PassBase<OptionsT, ContextT...> {
 public:
  using PassBase<OptionsT, ContextT...>::PassBase;

  // Run the pass on a single proc.
  absl::StatusOr<bool> RunOnFunction(Function* f, const OptionsT& options,
                                     PassResults* results,
                                     ContextT&... context) const;

 protected:
  // Iterates over each function in the package calling RunOnFunction.
  absl::StatusOr<bool> RunInternal(Package* p, const OptionsT& options,
                                   PassResults* results,
                                   ContextT&... context) const override {
    bool changed = false;
    for (const auto& f : p->functions()) {
      XLS_ASSIGN_OR_RETURN(
          bool proc_changed,
          RunOnFunctionInternal(f.get(), options, results, context...));
      if (proc_changed) {
        GcAfterFunctionChange(p, context...);
      }
      changed = changed || proc_changed;
    }
    return changed;
  }

  virtual absl::StatusOr<bool> RunOnFunctionInternal(
      Function* f, const OptionsT& options, PassResults* results,
      ContextT&... context) const = 0;

  // A subclass may optionally override this to garbage collect context data
  // after a `RunOnFunctionInternal` call that makes a change.
  virtual void GcAfterFunctionChange(Package* p, ContextT&... context) const {}
};

template <typename OptionsT, typename... ContextT>
absl::StatusOr<bool> FunctionPass<OptionsT, ContextT...>::RunOnFunction(
    Function* f, const OptionsT& options, PassResults* results,
    ContextT&... context) const {
  class PassRunner : public FunctionPass<OptionsT, ContextT...> {
   public:
    PassRunner(const FunctionPass<OptionsT, ContextT...>& pass, Function* f)
        : FunctionPass<OptionsT, ContextT...>(pass.short_name(),
                                              pass.long_name()),
          pass_(pass),
          f_(f) {}
    absl::StatusOr<bool> RunOnFunctionInternal(
        Function* f, const OptionsT& options, PassResults* results,
        ContextT&... context) const override {
      if (f != f_) {
        return false;
      }
      return pass_.RunOnFunctionInternal(f, options, results, context...);
    }
    void GcAfterFunctionChange(Package* p,
                               ContextT&... context) const override {
      pass_.GcAfterFunctionChange(p, context...);
    }

   private:
    const FunctionPass<OptionsT, ContextT...>& pass_;
    Function* f_;
  };
  return PassRunner(*this, f).Run(f->package(), options, results, context...);
}

// Abstract base class for passes that operate at proc scope. The derived class
// must define RunOnProcInternal.
template <typename OptionsT, typename... ContextT>
class ProcPass : public PassBase<OptionsT, ContextT...> {
 public:
  using PassBase<OptionsT, ContextT...>::PassBase;

  // Run the pass on a single proc.
  absl::StatusOr<bool> RunOnProc(Proc* proc, const OptionsT& options,
                                 PassResults* results,
                                 ContextT&... context) const;

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

template <typename OptionsT, typename... ContextT>
absl::StatusOr<bool> ProcPass<OptionsT, ContextT...>::RunOnProc(
    Proc* proc, const OptionsT& options, PassResults* results,
    ContextT&... context) const {
  class PassRunner : public ProcPass<OptionsT, ContextT...> {
   public:
    PassRunner(const ProcPass<OptionsT, ContextT...>& pass, Proc* f)
        : ProcPass<OptionsT, ContextT...>(pass.short_name(), pass.long_name()),
          pass_(pass),
          f_(f) {}
    absl::StatusOr<bool> RunOnProcInternal(
        Proc* proc, const OptionsT& options, PassResults* results,
        ContextT&... context) const override {
      if (proc != f_) {
        return false;
      }
      return pass_.RunOnProcInternal(proc, options, results, context...);
    }
    void GcAfterProcChange(Package* p, ContextT&... context) const override {
      pass_.GcAfterProcChange(p, context...);
    }

   private:
    const ProcPass<OptionsT, ContextT...>& pass_;
    Proc* f_;
  };
  return PassRunner(*this, proc)
      .Run(proc->package(), options, results, context...);
}

// Abstract base class for passes that operate at block scope. The derived class
// must define RunOnBlockInternal.
template <typename OptionsT, typename... ContextT>
class BlockPass : public PassBase<OptionsT, ContextT...> {
 public:
  using PassBase<OptionsT, ContextT...>::PassBase;

  // Run the pass on a single block.
  absl::StatusOr<bool> RunOnBlock(Block* block, const OptionsT& options,
                                  PassResults* results,
                                  ContextT&... context) const;

 protected:
  // Iterates over each block in the package calling RunOnBlock.
  absl::StatusOr<bool> RunInternal(Package* p, const OptionsT& options,
                                   PassResults* results,
                                   ContextT&... context) const override {
    bool changed = false;
    for (const auto& block : p->blocks()) {
      XLS_ASSIGN_OR_RETURN(
          bool block_changed,
          RunOnBlockInternal(block.get(), options, results, context...));
      if (block_changed) {
        GcAfterBlockChange(p, context...);
      }
      changed = changed || block_changed;
    }
    return changed;
  }

  virtual absl::StatusOr<bool> RunOnBlockInternal(
      Block* block, const OptionsT& options, PassResults* results,
      ContextT&... context) const = 0;

  // A subclass may optionally override this to garbage collect context data
  // after a `RunOnBlockInternal` call that makes a change.
  virtual void GcAfterBlockChange(Package* p, ContextT&... context) const {}
};

template <typename OptionsT, typename... ContextT>
absl::StatusOr<bool> BlockPass<OptionsT, ContextT...>::RunOnBlock(
    Block* block, const OptionsT& options, PassResults* results,
    ContextT&... context) const {
  class PassRunner : public BlockPass<OptionsT, ContextT...> {
   public:
    PassRunner(const BlockPass<OptionsT, ContextT...>& pass, Block* f)
        : BlockPass<OptionsT, ContextT...>(pass.short_name(), pass.long_name()),
          pass_(pass),
          f_(f) {}
    absl::StatusOr<bool> RunOnBlockInternal(
        Block* block, const OptionsT& options, PassResults* results,
        ContextT&... context) const override {
      if (block != f_) {
        return false;
      }
      return pass_.RunOnBlockInternal(block, options, results, context...);
    }
    void GcAfterBlockChange(Package* p, ContextT&... context) const override {
      pass_.GcAfterBlockChange(p, context...);
    }

   private:
    const BlockPass<OptionsT, ContextT...>& pass_;
    Block* f_;
  };
  return PassRunner(*this, block)
      .Run(block->package(), options, results, context...);
}
}  // namespace xls

#endif  // XLS_PASSES_PASS_BASE_H_
