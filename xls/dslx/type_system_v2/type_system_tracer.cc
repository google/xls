// Copyright 2025 The XLS Authors
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

#include "xls/dslx/type_system_v2/type_system_tracer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xls/common/indent.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_annotation_filter.h"

namespace xls::dslx {

// Kinds of tasks that a trace can indicate.
enum class TraceKind : uint8_t {
  kRoot,
  kUnify,
  kResolve,
  kConvertActualArgument,
  kConvertNode,
  kConvertInvocation,
  kInferImplicitParametrics,
  kEvaluate,
  kCollectConstants,
  kConcretize,
  kUnroll,
  kFilter
};

struct TypeSystemTraceImpl {
  TypeSystemTraceImpl* parent = nullptr;
  TraceKind kind;
  int level = 0;
  std::optional<absl::Time> start_time;
  std::optional<const void*> address;
  std::optional<const AstNode*> node;
  std::optional<const TypeAnnotation*> annotation;
  std::optional<const NameRef*> inference_variable;
  std::optional<const ParametricContext*> parametric_context;
  std::optional<TypeAnnotationFilter> filter;
  std::optional<std::vector<const TypeAnnotation*>> annotations;
  std::optional<absl::flat_hash_set<const ParametricBinding*>> bindings;
  std::optional<const TypeAnnotation*> result_annotation;
  std::optional<InterpValue> result_value;
  std::optional<bool> used_cache;
  std::optional<bool> populated_cache;
  std::optional<bool> convert_for_type_variable_unification;
  absl::Duration duration;
};

std::string TraceKindToString(TraceKind kind) {
  switch (kind) {
    case TraceKind::kRoot:
      return "Root";
    case TraceKind::kUnify:
      return "Unify";
    case TraceKind::kResolve:
      return "Resolve";
    case TraceKind::kConvertActualArgument:
      return "ConvertActualArg";
    case TraceKind::kConvertNode:
      return "ConvertNode";
    case TraceKind::kConvertInvocation:
      return "ConvertInvocation";
    case TraceKind::kInferImplicitParametrics:
      return "InferImplicitParametrics";
    case TraceKind::kEvaluate:
      return "Evaluate";
    case TraceKind::kCollectConstants:
      return "CollectConstants";
    case TraceKind::kConcretize:
      return "Concretize";
    case TraceKind::kUnroll:
      return "Unroll";
    case TraceKind::kFilter:
      return "Filter";
  }
}

std::ostream& operator<<(std::ostream& out, TraceKind kind) {
  out << TraceKindToString(kind);
  return out;
}

std::string TraceImplToString(const TypeSystemTraceImpl& impl) {
  std::vector<std::string> pieces;
  if (impl.address.has_value()) {
    pieces.push_back(absl::StrFormat("%p", *impl.address));
  }
  if (impl.node.has_value()) {
    pieces.push_back(absl::StrCat("node: ", (*impl.node)->ToString()));
  }
  if (impl.inference_variable.has_value()) {
    pieces.push_back(
        absl::StrCat("var: ", (*impl.inference_variable)->ToString()));
  }
  if (impl.filter.has_value()) {
    pieces.push_back(absl::StrCat("filter: ", impl.filter->ToString()));
  }
  if (impl.annotation.has_value()) {
    pieces.push_back(
        absl::StrCat("annotation: ", (*impl.annotation)->ToString()));
  }
  if (impl.annotations.has_value()) {
    std::vector<std::string> annotation_strings;
    for (const TypeAnnotation* annotation : *impl.annotations) {
      annotation_strings.push_back(annotation->ToString());
    }
    pieces.push_back(
        absl::StrCat("annotations: ", absl::StrJoin(annotation_strings, ", ")));
  }
  if (impl.bindings.has_value()) {
    std::vector<std::string> annotation_strings;
    for (const ParametricBinding* binding : *impl.bindings) {
      annotation_strings.push_back(binding->identifier());
    }
    pieces.push_back(
        absl::StrCat("bindings: ", absl::StrJoin(annotation_strings, ", ")));
  }
  if (impl.convert_for_type_variable_unification.has_value()) {
    pieces.push_back(absl::StrCat("for_var_unification: ",
                                  *impl.convert_for_type_variable_unification));
  }
  if (impl.parametric_context.has_value()) {
    pieces.push_back(
        absl::StrCat("context: ", (*impl.parametric_context)->ToString()));
  }
  if (impl.result_annotation.has_value()) {
    pieces.push_back(
        absl::StrCat("result: ", (*impl.result_annotation)->ToString()));
  }
  if (impl.result_value.has_value()) {
    pieces.push_back(absl::StrCat("value: ", (*impl.result_value).ToString()));
  }
  if (impl.used_cache.has_value()) {
    pieces.push_back(absl::StrCat("used_global_cache: ", *impl.used_cache));
  }
  if (impl.populated_cache.has_value()) {
    pieces.push_back(
        absl::StrCat("populated_global_cache: ", *impl.populated_cache));
  }
  return absl::StrJoin(pieces, ", ");
}

class TypeSystemTracerImpl : public TypeSystemTracer {
 public:
  TypeSystemTracerImpl(bool time_every_action)
      : root_trace_(&traces_.emplace_back(
                        TypeSystemTraceImpl{.kind = TraceKind::kRoot}),
                    /*cleanup=*/[] {}),
        time_every_action_(time_every_action) {
    stack_.push(&*traces_.begin());
  }

  TypeSystemTrace TraceUnify(const AstNode* node) final {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kUnify,
                                     .address = node,
                                     .node = node});
  }

  TypeSystemTrace TraceUnify(const NameRef* type_variable) final {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kUnify,
                                     .inference_variable = type_variable});
  }

  TypeSystemTrace TraceUnify(
      const std::vector<const TypeAnnotation*>& annotations) final {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kUnify,
                                     .annotations = annotations});
  }

  TypeSystemTrace TraceFilter(
      TypeAnnotationFilter filter,
      const std::vector<const TypeAnnotation*>& annotations) final {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kFilter,
                                     .filter = filter,
                                     .annotations = annotations});
  }

  TypeSystemTrace TraceResolve(
      const TypeAnnotation* annotation,
      std::optional<const ParametricContext*> parametric_context) final {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kResolve,
                                     .address = annotation,
                                     .annotation = annotation,
                                     .parametric_context = parametric_context});
  }

  TypeSystemTrace TraceConvertNode(const AstNode* node) final {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kConvertNode,
                                     .address = node,
                                     .node = node});
  }

  TypeSystemTrace TraceConvertActualArgument(const AstNode* node) final {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kConvertActualArgument,
                                     .address = node,
                                     .node = node});
  }

  TypeSystemTrace TraceConvertInvocation(
      const Invocation* invocation,
      std::optional<const ParametricContext*> caller_context,
      std::optional<bool> convert_for_type_variable_unification) final {
    return Trace(
        TypeSystemTraceImpl{.parent = stack_.top(),
                            .kind = TraceKind::kConvertInvocation,
                            .address = invocation,
                            .node = invocation,
                            .parametric_context = caller_context,
                            .convert_for_type_variable_unification =
                                convert_for_type_variable_unification});
  }

  TypeSystemTrace TraceInferImplicitParametrics(
      const absl::flat_hash_set<const ParametricBinding*>& bindings) final {
    return Trace(
        TypeSystemTraceImpl{.parent = stack_.top(),
                            .kind = TraceKind::kInferImplicitParametrics,
                            .bindings = bindings});
  }

  TypeSystemTrace TraceEvaluate(
      std::optional<const ParametricContext*> parametric_context,
      const Expr* expr) final {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kEvaluate,
                                     .address = expr,
                                     .node = expr,
                                     .parametric_context = parametric_context});
  }

  TypeSystemTrace TraceCollectConstants(
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node) final {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kCollectConstants,
                                     .address = node,
                                     .node = node,
                                     .parametric_context = parametric_context});
  }

  TypeSystemTrace TraceConcretize(const TypeAnnotation* annotation) final {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kConcretize,
                                     .address = annotation,
                                     .annotation = annotation});
  }

  TypeSystemTrace TraceUnroll(const AstNode* node) final {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kUnroll,
                                     .address = node,
                                     .node = node});
  }

  std::string ConvertTracesToString() const override {
    std::string result;
    for (const TypeSystemTraceImpl& trace : traces_) {
      if (trace.kind == TraceKind::kRoot) {
        continue;
      }
      absl::StrAppend(
          &result,
          Indent(absl::StrFormat("%s (%s)%s\n", TraceKindToString(trace.kind),
                                 TraceImplToString(trace),
                                 time_every_action_
                                     ? absl::StrCat(" ", absl::FormatDuration(
                                                             trace.duration))
                                     : ""),
                 (trace.level - 1) * 3));
    }
    return result;
  }

  std::string ConvertStatsToString() const override {
    std::string result;

    // Sort a local copy of `stats_` in descending order by total processing
    // time.
    std::vector<std::pair<const AstNode*, NodeStats>> stats;
    for (const auto& [node, node_stats] : stats_) {
      stats.push_back(std::make_pair(node, node_stats));
    }
    absl::c_sort(stats, [](const auto& x, const auto& y) {
      return x.second.total_processing_time > y.second.total_processing_time;
    });

    absl::StrAppendFormat(
        &result, "Type variable unifications: %d (%d used global cache).\n\n",
        variable_unification_count_, cached_variable_unification_count_);

    for (auto& [node, node_stats] : stats) {
      std::string node_string = node->ToString();
      if (const auto* fn = dynamic_cast<const Function*>(node)) {
        node_string = fn->identifier();
      } else if (const auto* proc = dynamic_cast<const Proc*>(node)) {
        node_string = proc->identifier();
      } else if (const auto* constant_def =
                     dynamic_cast<const ConstantDef*>(node)) {
        node_string = constant_def->identifier();
      }
      absl::StrAppendFormat(&result, "Node: `%s`\n", node_string);
      absl::StrAppendFormat(&result, "Kind: `%s`\n",
                            AstNodeKindToString(node->kind()));
      absl::StrAppendFormat(
          &result, "Span: %s\n",
          node->GetSpan().has_value()
              ? node->GetSpan()->ToString(*node->owner()->file_table())
              : "<unknown>");
      absl::StrAppendFormat(&result, "Times converted: %d\n",
                            node_stats.conversion_count);
      absl::StrAppendFormat(
          &result, "Total processing duration: %s\n\n",
          absl::FormatDuration(node_stats.total_processing_time));
    }

    return result;
  }

 private:
  void Cleanup(TypeSystemTraceImpl* impl) {
    CHECK(stack_.top() == impl);
    if (impl->start_time.has_value() &&
        (impl->kind == TraceKind::kConvertNode || time_every_action_)) {
      impl->duration = absl::Now() - *impl->start_time;
      if (impl->node && impl->kind == TraceKind::kConvertNode) {
        NodeStats& node_stats = stats_[*impl->node];
        node_stats.total_processing_time += impl->duration;
        ++node_stats.conversion_count;
      }
    }
    if (impl->kind == TraceKind::kUnify &&
        impl->inference_variable.has_value()) {
      ++variable_unification_count_;
      if (impl->used_cache.has_value() && *impl->used_cache) {
        ++cached_variable_unification_count_;
      }
    }
    stack_.pop();
  }

  // Helper for inserting anything but the root trace into the data structure.
  TypeSystemTrace Trace(TypeSystemTraceImpl&& impl) {
    CHECK_NE(impl.parent, nullptr);
    CHECK_NE(impl.kind, TraceKind::kRoot);

    if (impl.kind == TraceKind::kConvertNode || time_every_action_) {
      impl.start_time = absl::Now();
    }

    TypeSystemTraceImpl* impl_ptr = &traces_.emplace_back(std::move(impl));
    impl_ptr->level = impl_ptr->parent->level + 1;
    stack_.push(impl_ptr);
    return TypeSystemTrace(impl_ptr, [this, impl_ptr] { Cleanup(impl_ptr); });
  }

  // Performance stats at the node level.
  struct NodeStats {
    int conversion_count = 0;
    absl::Duration total_processing_time;
  };

  std::list<TypeSystemTraceImpl> traces_;
  std::stack<TypeSystemTraceImpl*> stack_;
  TypeSystemTrace root_trace_;
  absl::flat_hash_map<const AstNode*, NodeStats> stats_;
  int variable_unification_count_ = 0;
  int cached_variable_unification_count_ = 0;
  bool time_every_action_;
};

// Implements the tracer interface with negligible overhead, for when tracing is
// not requested.
class NoopTracer final : public TypeSystemTracer {
 public:
  TypeSystemTrace TraceUnify(const AstNode* node) final { return Noop(); }

  TypeSystemTrace TraceUnify(const NameRef* type_variable) final {
    return Noop();
  }

  TypeSystemTrace TraceUnify(
      const std::vector<const TypeAnnotation*>& annotations) final {
    return Noop();
  }

  TypeSystemTrace TraceFilter(
      TypeAnnotationFilter filter,
      const std::vector<const TypeAnnotation*>& annotations) final {
    return Noop();
  }

  TypeSystemTrace TraceResolve(
      const TypeAnnotation* annotation,
      std::optional<const ParametricContext*> parametric_context) final {
    return Noop();
  }

  TypeSystemTrace TraceConvertActualArgument(const AstNode* node) final {
    return Noop();
  }

  TypeSystemTrace TraceConvertNode(const AstNode* node) final { return Noop(); }

  TypeSystemTrace TraceConvertInvocation(
      const Invocation* invocation,
      std::optional<const ParametricContext*> caller_context,
      std::optional<bool> convert_for_type_variable_unification) final {
    return Noop();
  }

  TypeSystemTrace TraceInferImplicitParametrics(
      const absl::flat_hash_set<const ParametricBinding*>& bindings) final {
    return Noop();
  }

  TypeSystemTrace TraceEvaluate(std::optional<const ParametricContext*> context,
                                const Expr* expr) final {
    return Noop();
  }

  TypeSystemTrace TraceCollectConstants(
      std::optional<const ParametricContext*> context, const AstNode*) final {
    return Noop();
  }

  TypeSystemTrace TraceConcretize(const TypeAnnotation* annotation) final {
    return Noop();
  }

  TypeSystemTrace TraceUnroll(const AstNode* node) final { return Noop(); }

  std::string ConvertTracesToString() const final { return ""; }
  std::string ConvertStatsToString() const final { return ""; }

  static void NoopCleanup() {}

 private:
  TypeSystemTrace Noop() { return TypeSystemTrace(&impl_, &NoopCleanup); }

  TypeSystemTraceImpl impl_;
};

std::unique_ptr<TypeSystemTracer> TypeSystemTracer::Create(
    bool active, bool time_every_action) {
  if (active) {
    return std::make_unique<TypeSystemTracerImpl>(time_every_action);
  }
  return std::make_unique<NoopTracer>();
}

void TypeSystemTrace::SetResult(const TypeAnnotation* annotation) {
  impl_->result_annotation = annotation;
}

void TypeSystemTrace::SetResult(const InterpValue& value) {
  impl_->result_value = value;
}

void TypeSystemTrace::SetUsedCache(bool value) { impl_->used_cache = value; }

void TypeSystemTrace::SetPopulatedCache(bool value) {
  impl_->populated_cache = value;
}

}  // namespace xls::dslx
