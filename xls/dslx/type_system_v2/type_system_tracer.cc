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

#include <cstdint>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/indent.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
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
  kConcretize,
  kUnroll,
  kFilter
};

struct TypeSystemTraceImpl {
  TypeSystemTraceImpl* parent = nullptr;
  TraceKind kind;
  int level = 0;
  std::optional<const AstNode*> node;
  std::optional<const TypeAnnotation*> annotation;
  std::optional<const NameRef*> inference_variable;
  std::optional<const ParametricContext*> parametric_context;
  std::optional<TypeAnnotationFilter> filter;
  std::optional<std::vector<const TypeAnnotation*>> annotations;
  std::optional<absl::flat_hash_set<const ParametricBinding*>> bindings;
  std::optional<const TypeAnnotation*> result_annotation;
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
  if (impl.parametric_context.has_value()) {
    pieces.push_back(
        absl::StrCat("context: ", (*impl.parametric_context)->ToString()));
  }
  if (impl.result_annotation.has_value()) {
    pieces.push_back(
        absl::StrCat("result: ", (*impl.result_annotation)->ToString()));
  }
  return absl::StrJoin(pieces, ", ");
}

class TypeSystemTracerImpl : public TypeSystemTracer {
 public:
  TypeSystemTracerImpl()
      : root_trace_(&traces_.emplace_back(
                        TypeSystemTraceImpl{.kind = TraceKind::kRoot}),
                    /*cleanup=*/[] {}) {
    stack_.push(&*traces_.begin());
  }

  TypeSystemTrace TraceUnify(const AstNode* node) override {
    return Trace(TypeSystemTraceImpl{
        .parent = stack_.top(), .kind = TraceKind::kUnify, .node = node});
  }

  TypeSystemTrace TraceUnify(const NameRef* type_variable) override {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kUnify,
                                     .inference_variable = type_variable});
  }

  TypeSystemTrace TraceUnify(
      const std::vector<const TypeAnnotation*>& annotations) override {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kUnify,
                                     .annotations = annotations});
  }

  TypeSystemTrace TraceFilter(
      TypeAnnotationFilter filter,
      const std::vector<const TypeAnnotation*>& annotations) override {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kFilter,
                                     .filter = filter,
                                     .annotations = annotations});
  }

  TypeSystemTrace TraceResolve(
      const TypeAnnotation* annotation,
      std::optional<const ParametricContext*> parametric_context) override {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kResolve,
                                     .annotation = annotation,
                                     .parametric_context = parametric_context});
  }

  TypeSystemTrace TraceConvertNode(const AstNode* node) override {
    return Trace(TypeSystemTraceImpl{
        .parent = stack_.top(), .kind = TraceKind::kConvertNode, .node = node});
  }

  TypeSystemTrace TraceConvertActualArgument(const AstNode* node) override {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kConvertActualArgument,
                                     .node = node});
  }

  TypeSystemTrace TraceConvertInvocation(
      const Invocation* invocation,
      std::optional<const ParametricContext*> caller_context) override {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kConvertInvocation,
                                     .node = invocation,
                                     .parametric_context = caller_context});
  }

  TypeSystemTrace TraceInferImplicitParametrics(
      const absl::flat_hash_set<const ParametricBinding*>& bindings) override {
    return Trace(
        TypeSystemTraceImpl{.parent = stack_.top(),
                            .kind = TraceKind::kInferImplicitParametrics,
                            .bindings = bindings});
  }

  TypeSystemTrace TraceEvaluate(
      std::optional<const ParametricContext*> parametric_context,
      const Expr* expr) override {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kEvaluate,
                                     .node = expr,
                                     .parametric_context = parametric_context});
  }

  TypeSystemTrace TraceConcretize(const TypeAnnotation* annotation) override {
    return Trace(TypeSystemTraceImpl{.parent = stack_.top(),
                                     .kind = TraceKind::kConcretize,
                                     .annotation = annotation});
  }

  TypeSystemTrace TraceUnroll(const AstNode* node) override {
    return Trace(TypeSystemTraceImpl{
        .parent = stack_.top(), .kind = TraceKind::kUnroll, .node = node});
  }

  std::string ConvertTracesToString() override {
    std::string result;
    for (const TypeSystemTraceImpl& trace : traces_) {
      if (trace.kind == TraceKind::kRoot) {
        continue;
      }
      absl::StrAppend(
          &result,
          Indent(absl::StrFormat("%s (%s)\n", TraceKindToString(trace.kind),
                                 TraceImplToString(trace)),
                 (trace.level - 1) * 3));
    }
    return result;
  }

 private:
  void Cleanup(TypeSystemTraceImpl* impl) {
    CHECK(stack_.top() == impl);
    stack_.pop();
  }

  // Helper for inserting anything but the root trace into the data structure.
  TypeSystemTrace Trace(TypeSystemTraceImpl&& impl) {
    CHECK_NE(impl.parent, nullptr);
    CHECK_NE(impl.kind, TraceKind::kRoot);

    TypeSystemTraceImpl* impl_ptr = &traces_.emplace_back(std::move(impl));
    impl_ptr->level = impl_ptr->parent->level + 1;
    stack_.push(impl_ptr);
    return TypeSystemTrace(impl_ptr, [this, impl_ptr] { Cleanup(impl_ptr); });
  }

  std::list<TypeSystemTraceImpl> traces_;
  std::stack<TypeSystemTraceImpl*> stack_;
  TypeSystemTrace root_trace_;
};

std::unique_ptr<TypeSystemTracer> TypeSystemTracer::Create() {
  return std::make_unique<TypeSystemTracerImpl>();
}

void TypeSystemTrace::SetResult(const TypeAnnotation* annotation) {
  impl_->result_annotation = annotation;
}

}  // namespace xls::dslx
