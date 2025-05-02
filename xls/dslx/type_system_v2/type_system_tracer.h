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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_TYPE_SYSTEM_TRACER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_TYPE_SYSTEM_TRACER_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_annotation_filter.h"

namespace xls::dslx {

class TypeSystemTracerImpl;
class TypeSystemTraceImpl;

// An object that represents the scope of a trace record. These objects are
// dealt out by a `TypeSystemTracer`. When a trace object gets deleted, that
// indicates we are done with the work it refers to.
class TypeSystemTrace {
  friend class TypeSystemTracerImpl;

 public:
  TypeSystemTrace(TypeSystemTrace&& other)
      : impl_(other.impl_), cleanup_(std::move(other.cleanup_)) {
    other.cleanup_ = nullptr;
    other.impl_ = nullptr;
  }

  TypeSystemTrace& operator=(TypeSystemTrace&& other) {
    impl_ = other.impl_;
    cleanup_ = std::move(other.cleanup_);
    other.impl_ = nullptr;
    other.cleanup_ = nullptr;
    return *this;
  }

  TypeSystemTrace(TypeSystemTrace&) = delete;
  TypeSystemTrace& operator=(TypeSystemTrace&) = delete;

  void SetResult(const TypeAnnotation* annotation);

  ~TypeSystemTrace() {
    if (cleanup_) {
      std::move(cleanup_)();
    }
  }

 private:
  TypeSystemTrace(TypeSystemTraceImpl* impl, absl::AnyInvocable<void()> cleanup)
      : impl_(impl), cleanup_(std::move(cleanup)) {}

  TypeSystemTraceImpl* impl_;
  absl::AnyInvocable<void()> cleanup_;
};

// An object used to record the tree of logical tasks done during type
// inference. When the type system starts doing operation X, it should call
// TraceX and keep the resulting `TypeSystemTrace` around until that is done.
class TypeSystemTracer {
 public:
  static std::unique_ptr<TypeSystemTracer> Create();

  virtual ~TypeSystemTracer() = default;

  virtual TypeSystemTrace TraceUnify(const AstNode* node) = 0;
  virtual TypeSystemTrace TraceUnify(const NameRef* type_variable) = 0;
  virtual TypeSystemTrace TraceUnify(
      const std::vector<const TypeAnnotation*>& annotations) = 0;
  virtual TypeSystemTrace TraceFilter(
      TypeAnnotationFilter filter,
      const std::vector<const TypeAnnotation*>& annotations) = 0;
  virtual TypeSystemTrace TraceResolve(
      const TypeAnnotation* annotation,
      std::optional<const ParametricContext*> parametric_context) = 0;
  virtual TypeSystemTrace TraceConvertActualArgument(const AstNode* node) = 0;
  virtual TypeSystemTrace TraceConvertNode(const AstNode* node) = 0;
  virtual TypeSystemTrace TraceConvertInvocation(
      const Invocation* invocation,
      std::optional<const ParametricContext*> caller_context) = 0;
  virtual TypeSystemTrace TraceInferImplicitParametrics(
      const absl::flat_hash_set<const ParametricBinding*>& bindings) = 0;
  virtual TypeSystemTrace TraceEvaluate(
      std::optional<const ParametricContext*> context, const Expr* expr) = 0;
  virtual TypeSystemTrace TraceConcretize(const TypeAnnotation* annotation) = 0;
  virtual TypeSystemTrace TraceUnroll(const AstNode* node) = 0;

  virtual std::string ConvertTracesToString() = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TYPE_SYSTEM_TRACER_H_
