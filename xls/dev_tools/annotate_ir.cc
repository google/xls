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
#include <type_traits>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/analyze_critical_path.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

std::string CombineAnnotation(const Annotation& note) {
  auto clean = [](const std::optional<std::string>& str) -> std::string_view {
    if (!str.has_value()) {
      return "";
    }
    return absl::StripAsciiWhitespace(*str);
  };
  std::string_view prefix = clean(note.prefix);
  std::string_view suffix = clean(note.suffix);
  return !prefix.empty() && !suffix.empty()
             ? absl::StrCat(prefix, " ", suffix)
             : std::string(prefix.empty() ? suffix : prefix);
}

Annotation FormatAnnotationAsCommentSuffix(const Annotation& note) {
  std::string combined = CombineAnnotation(note);
  return Annotation{
      .filter = note.filter,
      .suffix = combined.empty()
                    ? std::nullopt
                    : std::make_optional(absl::StrCat("// ", combined)),
  };
}

class CommentAnnotator : public IrAnnotatorRef<IrAnnotator> {
 public:
  using IrAnnotatorRef::IrAnnotatorRef;

  Annotation FunctionAnnotation(const FunctionBase* fb) const override {
    std::string func_prefix = "";
    for (Param* p : fb->params()) {
      Annotation note = IrAnnotatorRef::NodeAnnotation(p);
      std::string full_annotation = CombineAnnotation(note);
      if (!full_annotation.empty()) {
        absl::StrAppend(&func_prefix, "// ", p->ToString(), " ",
                        full_annotation, "\n");
      }
    }
    if (func_prefix.empty()) {
      return {};
    }
    return {.prefix = func_prefix};
  }

  Annotation NodeAnnotation(Node* node) const override {
    if (node->Is<Param>()) {
      return {};
    }
    return FormatAnnotationAsCommentSuffix(
        IrAnnotatorRef::NodeAnnotation(node));
  }
};

// IrAnnotator wrapper that caches an annotator per function of the package
class PackageAnnotator : public IrAnnotator {
 public:
  template <typename AnnotatorFn>
    requires(std::is_invocable_r_v<absl::StatusOr<std::unique_ptr<IrAnnotator>>,
                                   AnnotatorFn, FunctionBase*,
                                   std::string_view /*delay_model*/>)
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

using AnnotatorFn = absl::StatusOr<std::unique_ptr<IrAnnotator>> (*)(
    FunctionBase* fb, std::string_view delay_model);

const absl::flat_hash_map<AnnotatorKind, AnnotatorFn>&
GetAnnotatorDispatchMap() {
  static const absl::NoDestructor<
      absl::flat_hash_map<AnnotatorKind, AnnotatorFn>>
      kAnnotatorMap({
          {AnnotatorKind::kNone, &AnnotateNone},
          {AnnotatorKind::kDelay, &AnnotateDelay},
          {AnnotatorKind::kCriticalPath, &AnnotateCriticalPath},
      });
  return *kAnnotatorMap;
}

}  // namespace

absl::StatusOr<std::string> AnnotateIr(AnnotatorKind kind, Package* package,
                                       std::string_view delay_model) {
  const absl::flat_hash_map<AnnotatorKind, AnnotatorFn>& map =
      GetAnnotatorDispatchMap();
  auto it = map.find(kind);
  if (it == map.end()) {
    return absl::InternalError("Unhandled AnnotatorKind");
  }
  XLS_ASSIGN_OR_RETURN(
      PackageAnnotator annotator,
      PackageAnnotator::Create(package, it->second, delay_model));
  return package->DumpIr(CommentAnnotator{std::move(annotator)});
}

absl::StatusOr<std::string> AnnotateIr(const IrAnnotator& annotator,
                                       Package* package) {
  return package->DumpIr(CommentAnnotator{annotator});
}

}  // namespace xls
