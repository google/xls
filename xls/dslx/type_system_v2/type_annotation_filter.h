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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_FILTER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_FILTER_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"

namespace xls::dslx {

// A filter for a set of `TypeAnnotation` objects, which encapsulates both the
// filtering logic and the ability to explain what it is filtering for tracing
// purposes. Filters only contain a small amount of data, which is const, and
// they can be treated as value objects.
class TypeAnnotationFilter {
 public:
  // Creates a filter that excludes nothing.
  static TypeAnnotationFilter None();

  // Creates a filter that excludes only single-any type annotations.
  static TypeAnnotationFilter FilterSingleAny();

  // Creates a filter that excludes only multi-any type annotations.
  static TypeAnnotationFilter FilterMultiAny();

  // Creates a filter that excludes any `ParamTypeAnnotation`.
  static TypeAnnotationFilter FilterParamTypes();

  // Creates a filter that excludes any annotation containing a reference (e.g.
  // `NameRef`) with no type info in `ti`.
  static TypeAnnotationFilter FilterRefsWithMissingTypeInfo(const TypeInfo* ti);

  // Creates a filter that excludes any annotation containing a reference to the
  // given struct.
  static TypeAnnotationFilter FilterReferencesToStruct(
      const InferenceTable* table,
      std::optional<const ParametricContext*> parametric_context,
      const StructDef* struct_def, const ImportData* import_data);

  // Creates a filter that excludes the given specific annotation, with the goal
  // of preventing recursion on an analysis of it.
  static TypeAnnotationFilter BlockRecursion(const TypeAnnotation* annotation);

  // Treat like a value object and allow copy and move.
  TypeAnnotationFilter(const TypeAnnotationFilter&);
  TypeAnnotationFilter(TypeAnnotationFilter&& other);
  TypeAnnotationFilter& operator=(const TypeAnnotationFilter&);
  TypeAnnotationFilter& operator=(TypeAnnotationFilter&&);

  // Returns true if this filter doesn't ever filter anything.
  bool IsNone() const;

  // Converts this filter to a string for tracing purposes.
  std::string ToString() const;

  // Chains this filter to `other`, with the resulting filter applying this
  // one's logic first and the other one's second.
  TypeAnnotationFilter Chain(TypeAnnotationFilter other);

  // Returns whether the given annotation should be excluded from consideration.
  bool Filter(const TypeAnnotation* annotation);

  ~TypeAnnotationFilter();

 private:
  class Impl;

  explicit TypeAnnotationFilter(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

// Removes any annotations in the given vector which are filtered by the given
// `filter`.
void FilterAnnotations(std::vector<const TypeAnnotation*>& annotations,
                       TypeAnnotationFilter filter);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_FILTER_H_
