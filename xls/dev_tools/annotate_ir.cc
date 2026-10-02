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

#include "xls/dev_tools/annotate_ir.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/analyze_critical_path.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/post_dominator_analysis.h"
#include "xls/passes/visibility_analysis.h"

namespace xls {

absl::StatusOr<AnnotatorKind> ParseAnnotatorKind(
    std::string_view annotator_str) {
  static absl::NoDestructor<
      absl::flat_hash_map<std::string_view, AnnotatorKind>>
      annotators({
          {"none", AnnotatorKind::kNone},
          {"visibility", AnnotatorKind::kVisibility},
          {"delay", AnnotatorKind::kDelay},
          {"critical_path", AnnotatorKind::kCriticalPath},
      });
  if (auto it = annotators->find(annotator_str); it != annotators->end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Unknown annotator '%s'. Expected one of: none, visibility, delay, "
      "critical_path.",
      annotator_str));
}

namespace {

Annotation FormatAnnotationAsCommentSuffix(const Annotation& note) {
  auto clean = [](const std::optional<std::string>& str) -> std::string_view {
    if (!str.has_value()) {
      return "";
    }
    return absl::StripLeadingAsciiWhitespace(
        absl::StripPrefix(absl::StripLeadingAsciiWhitespace(*str), "//"));
  };
  std::string_view prefix = clean(note.prefix);
  std::string_view suffix = clean(note.suffix);
  std::string combined = !prefix.empty() && !suffix.empty()
                             ? absl::StrCat(prefix, " ", suffix)
                             : std::string(prefix.empty() ? suffix : prefix);
  return Annotation{
      .filter = note.filter,
      .suffix = combined.empty()
                    ? std::nullopt
                    : std::make_optional(absl::StrCat("// ", combined)),
  };
}

struct CommentSuffixAnnotator : public IrAnnotatorRef<IrAnnotator> {
  using IrAnnotatorRef::IrAnnotatorRef;
  Annotation FunctionAnnotation(const Function* function) const override {
    return IrAnnotator::FunctionAnnotation(function);
  }
  Annotation NodeAnnotation(Node* node) const override {
    Annotation note =
        FormatAnnotationAsCommentSuffix(IrAnnotatorRef::NodeAnnotation(node));
    // This is to ensure that param annotations are emitted as comments above
    // the function definition:
    if (node->Is<Param>() && note.suffix.has_value()) {
      note.prefix = "//";
    }
    return note;
  }
};

using AnnotatorFn = absl::StatusOr<std::unique_ptr<IrAnnotator>> (*)(
    FunctionBase* fb, std::string_view delay_model);

// IrAnnotator wrapper that caches an annotator per function of the package
class PackageAnnotator : public IrAnnotator {
 public:
  static absl::StatusOr<PackageAnnotator> Create(Package* package,
                                                 AnnotatorFn make_annotator,
                                                 std::string_view delay_model) {
    absl::flat_hash_map<const FunctionBase*, std::unique_ptr<IrAnnotator>>
        annotators;
    for (FunctionBase* fb : package->GetFunctionBases()) {
      XLS_ASSIGN_OR_RETURN(annotators[fb], make_annotator(fb, delay_model));
    }
    return PackageAnnotator(std::move(annotators));
  }

  Annotation NodeAnnotation(Node* node) const override {
    if (auto it = annotators_.find(node->function_base());
        it != annotators_.end()) {
      return it->second->NodeAnnotation(node);
    }
    return {};
  }

 private:
  explicit PackageAnnotator(
      absl::flat_hash_map<const FunctionBase*, std::unique_ptr<IrAnnotator>>
          annotators)
      : annotators_(std::move(annotators)) {}

  absl::flat_hash_map<const FunctionBase*, std::unique_ptr<IrAnnotator>>
      annotators_;
};

absl::StatusOr<std::unique_ptr<IrAnnotator>> AnnotateNone(
    FunctionBase* /*fb*/, std::string_view /*delay_model*/) {
  return std::make_unique<IrAnnotator>();
}

class FunctionVisibilityAnnotator : public IrAnnotator {
 public:
  static absl::StatusOr<std::unique_ptr<IrAnnotator>> Create(FunctionBase* fb) {
    auto annotator = std::make_unique<FunctionVisibilityAnnotator>();
    XLS_RETURN_IF_ERROR(annotator->nda_.Attach(fb).status());
    XLS_RETURN_IF_ERROR(annotator->post_dom_.Attach(fb).status());
    annotator->bdd_engine_ = BddQueryEngine::MakeDefault();
    XLS_RETURN_IF_ERROR(annotator->bdd_engine_->Populate(fb).status());
    XLS_ASSIGN_OR_RETURN(OperandVisibilityAnalysis operand_visibility,
                         OperandVisibilityAnalysis::Create(
                             &annotator->nda_, annotator->bdd_engine_.get()));
    annotator->operand_visibility_.emplace(std::move(operand_visibility));
    XLS_ASSIGN_OR_RETURN(
        annotator->visibility_,
        VisibilityAnalysis::Create(&*annotator->operand_visibility_,
                                   annotator->bdd_engine_.get(),
                                   &annotator->post_dom_));
    return annotator;
  }

  Annotation NodeAnnotation(Node* node) const override {
    return visibility_->annotator().NodeAnnotation(node);
  }

 private:
  NodeForwardDependencyAnalysis nda_;
  LazyPostDominatorAnalysis post_dom_;
  std::unique_ptr<BddQueryEngine> bdd_engine_;
  std::optional<OperandVisibilityAnalysis> operand_visibility_;
  std::unique_ptr<VisibilityAnalysis> visibility_;
};

absl::StatusOr<std::unique_ptr<IrAnnotator>> AnnotateVisibility(
    FunctionBase* fb, std::string_view /*delay_model*/) {
  return FunctionVisibilityAnnotator::Create(fb);
}

absl::StatusOr<std::unique_ptr<IrAnnotator>> AnnotateDelay(
    FunctionBase* fb, std::string_view delay_model) {
  XLS_ASSIGN_OR_RETURN(DelayEstimator * estimator,
                       GetDelayEstimator(delay_model));
  XLS_ASSIGN_OR_RETURN(DelayAnnotator delay_annotator,
                       DelayAnnotator::Create(fb, *estimator));
  return std::make_unique<DelayAnnotator>(std::move(delay_annotator));
}

absl::StatusOr<std::unique_ptr<IrAnnotator>> AnnotateCriticalPath(
    FunctionBase* fb, std::string_view delay_model) {
  XLS_ASSIGN_OR_RETURN(DelayEstimator * estimator,
                       GetDelayEstimator(delay_model));
  XLS_ASSIGN_OR_RETURN(CriticalPathAnnotator cp_annotator,
                       CriticalPathAnnotator::Create(
                           fb, /*clock_period_ps=*/std::nullopt, *estimator));
  return std::make_unique<CriticalPathAnnotator>(std::move(cp_annotator));
}

const absl::flat_hash_map<AnnotatorKind, AnnotatorFn>&
GetAnnotatorDispatchMap() {
  static const absl::NoDestructor<
      absl::flat_hash_map<AnnotatorKind, AnnotatorFn>>
      map({
          {AnnotatorKind::kNone, &AnnotateNone},
          {AnnotatorKind::kVisibility, &AnnotateVisibility},
          {AnnotatorKind::kDelay, &AnnotateDelay},
          {AnnotatorKind::kCriticalPath, &AnnotateCriticalPath},
      });
  return *map;
}

}  // namespace

absl::StatusOr<std::string> AnnotateIr(Package* package, AnnotatorKind kind,
                                       std::string_view delay_model) {
  const auto& map = GetAnnotatorDispatchMap();
  auto it = map.find(kind);
  if (it == map.end()) {
    return absl::InternalError("Unhandled AnnotatorKind");
  }
  if (kind == AnnotatorKind::kNone || package->GetFunctionBases().empty()) {
    return package->DumpIr();
  }
  XLS_ASSIGN_OR_RETURN(
      PackageAnnotator annotator,
      PackageAnnotator::Create(package, it->second, delay_model));
  return package->DumpIr(CommentSuffixAnnotator{annotator});
}

}  // namespace xls
