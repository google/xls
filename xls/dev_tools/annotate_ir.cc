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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/strip.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/analyze_critical_path.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/call_graph.h"
#include "xls/ir/channel.h"
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
namespace {

Annotation FormatAnnotationAsCommentSuffix(Annotation note) {
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

// Wraps an IrAnnotator so that:
// 1. Parameter nodes are not annotated inline in the function header (they are
//    emitted as comment lines above the function instead).
// 2. Non-parameter node annotations are formatted as trailing `// ...` comments
//    so the dumped IR remains parseable by the XLS IR parser.
class ParseableIrAnnotator : public IrAnnotator {
 public:
  explicit ParseableIrAnnotator(const IrAnnotator& underlying)
      : underlying_(underlying) {}

  std::optional<std::vector<Node*>> NodeOrder(FunctionBase* fb) const override {
    return underlying_.NodeOrder(fb);
  }

  Annotation NodeAnnotation(Node* node) const override {
    if (node->Is<Param>()) {
      return {};
    }
    return FormatAnnotationAsCommentSuffix(underlying_.NodeAnnotation(node));
  }

 private:
  const IrAnnotator& underlying_;
};

absl::StatusOr<std::string> DumpPackageWithFormatter(
    Package* package,
    const std::function<absl::StatusOr<std::string>(FunctionBase*)>&
        format_fb) {
  Package header_pkg(package->name());
  for (const auto& [fileno, filename] : package->fileno_to_name()) {
    header_pkg.SetFileno(fileno, filename);
  }

  // Each section (`header_pkg.DumpIr()`, the channel block, and each
  // `FunctionBase::DumpIr()`) ends with a single '\n', so joining sections with
  // "\n" separates them by a blank line and leaves a single trailing '\n'.
  std::vector<std::string> sections = {header_pkg.DumpIr()};
  if (!package->channels().empty()) {
    sections.push_back(
        absl::StrCat(absl::StrJoin(package->channels(), "\n",
                                   [](std::string* out, const Channel* ch) {
                                     absl::StrAppend(out, ch->ToString());
                                   }),
                     "\n"));
  }

  for (FunctionBase* fb : FunctionsInPostOrder(package)) {
    XLS_ASSIGN_OR_RETURN(std::string fb_ir, format_fb(fb));
    sections.push_back(std::move(fb_ir));
  }

  return absl::StrJoin(sections, "\n");
}

}  // namespace

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

std::string AnnotateFunctionBaseIr(FunctionBase* fb,
                                   const IrAnnotator& annotator) {
  std::string out;
  for (Param* param : fb->params()) {
    Annotation note =
        FormatAnnotationAsCommentSuffix(annotator.NodeAnnotation(param));
    if (!note.filter && note.suffix.has_value()) {
      absl::StrAppendFormat(&out, "// %s\n", note.Decorate(param->ToString()));
    }
  }
  ParseableIrAnnotator parseable_annotator(annotator);
  absl::StrAppend(&out, fb->DumpIr(parseable_annotator));
  return out;
}

std::string AnnotateIr(Package* package, const IrAnnotator& annotator) {
  absl::StatusOr<std::string> res = DumpPackageWithFormatter(
      package, [&](FunctionBase* fb) -> absl::StatusOr<std::string> {
        return AnnotateFunctionBaseIr(fb, annotator);
      });
  CHECK_OK(res.status());
  return *res;
}

namespace {

absl::StatusOr<std::string> AnnotateNone(FunctionBase* fb,
                                         std::string_view /*delay_model*/) {
  return AnnotateFunctionBaseIr(fb, IrAnnotator{});
}

absl::StatusOr<std::string> AnnotateVisibility(
    FunctionBase* fb, std::string_view /*delay_model*/) {
  NodeForwardDependencyAnalysis nda;
  XLS_RETURN_IF_ERROR(nda.Attach(fb).status());
  LazyPostDominatorAnalysis post_dom;
  XLS_RETURN_IF_ERROR(post_dom.Attach(fb).status());
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_RETURN_IF_ERROR(bdd_engine->Populate(fb).status());
  XLS_ASSIGN_OR_RETURN(
      OperandVisibilityAnalysis operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<VisibilityAnalysis> visibility,
                       VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  return AnnotateFunctionBaseIr(fb, visibility->annotator());
}

absl::StatusOr<std::string> AnnotateDelay(FunctionBase* fb,
                                          std::string_view delay_model) {
  XLS_ASSIGN_OR_RETURN(DelayEstimator * estimator,
                       GetDelayEstimator(delay_model));
  XLS_ASSIGN_OR_RETURN(DelayAnnotator delay_annotator,
                       DelayAnnotator::Create(fb, *estimator));
  return AnnotateFunctionBaseIr(fb, delay_annotator);
}

absl::StatusOr<std::string> AnnotateCriticalPath(FunctionBase* fb,
                                                 std::string_view delay_model) {
  XLS_ASSIGN_OR_RETURN(DelayEstimator * estimator,
                       GetDelayEstimator(delay_model));
  XLS_ASSIGN_OR_RETURN(CriticalPathAnnotator cp_annotator,
                       CriticalPathAnnotator::Create(
                           fb, /*clock_period_ps=*/std::nullopt, *estimator));
  return AnnotateFunctionBaseIr(fb, cp_annotator);
}

using AnnotatorFn = absl::StatusOr<std::string> (*)(
    FunctionBase* fb, std::string_view delay_model);

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
  AnnotatorFn handler = it->second;
  return DumpPackageWithFormatter(
      package, [&](FunctionBase* fb) { return handler(fb, delay_model); });
}

}  // namespace xls
