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

#include "xls/dslx/type_system_v2/type_annotation_filter.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/expand_variables.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"

namespace xls::dslx {
namespace {

using FilterClosure = std::function<bool(const TypeAnnotation*)>;

enum class TypeAnnotationFilterKind : uint8_t {
  kNone,
  kParamType,
  kSingleAny,
  kMultiAny,
  kBlockRecursion,
  kCaptureVariables,
  kCaptureRejectCount,
  kMissingTypeInfo,
  kStructRef
};

std::string KindToString(TypeAnnotationFilterKind kind) {
  switch (kind) {
    case TypeAnnotationFilterKind::kNone:
      return "none";
    case TypeAnnotationFilterKind::kParamType:
      return "param";
    case TypeAnnotationFilterKind::kSingleAny:
      return "single-any";
    case TypeAnnotationFilterKind::kMultiAny:
      return "multi-any";
    case TypeAnnotationFilterKind::kBlockRecursion:
      return "block-recursion";
    case TypeAnnotationFilterKind::kCaptureVariables:
      return "capture-vars";
    case TypeAnnotationFilterKind::kCaptureRejectCount:
      return "capture-reject-count";
    case TypeAnnotationFilterKind::kMissingTypeInfo:
      return "refs-with-missing-type-info";
    case TypeAnnotationFilterKind::kStructRef:
      return "refs-to-struct";
  }
}

// Returns true if `annotation` contains any `NameRef` whose type info has not
// (yet) been generated. The effective `TypeInfo` is either `default_ti`; or,
// for invocation-scoped annotations, the `TypeInfo` for the relevant
// parametric invocation.
bool HasAnyReferencesWithMissingTypeInfo(const TypeInfo* ti,
                                         const TypeAnnotation* annotation) {
  FreeVariables vars =
      GetFreeVariablesByLambda(annotation, [&](const NameRef& ref) {
        if (!std::holds_alternative<const NameDef*>(ref.name_def())) {
          return false;
        }
        const NameDef* name_def = std::get<const NameDef*>(ref.name_def());
        bool result = name_def->parent() != nullptr &&
                      name_def->parent()->kind() != AstNodeKind::kFunction &&
                      !ti->GetItem(name_def).has_value() &&
                      !ti->IsKnownConstExpr(name_def);
        if (result) {
          VLOG(6) << "Found free var " << name_def->ToString()
                  << " in annotation " << annotation->ToString();
        }
        return result;
      });
  return vars.GetFreeVariableCount() > 0;
}

// Determines whether the given type annotation has any reference to the given
// `struct_def`, taking into consideration the expansions of any variable or
// indirect type annotations in the annotation tree.
absl::StatusOr<bool> RefersToStruct(
    const InferenceTable& table,
    std::optional<const ParametricContext*> parametric_context,
    const TypeAnnotation* annotation, const StructDef& struct_def,
    const ImportData& import_data) {
  if (annotation->IsAnnotation<ElementTypeAnnotation>()) {
    annotation =
        annotation->AsAnnotation<ElementTypeAnnotation>()->container_type();
  }
  if (annotation->IsAnnotation<MemberTypeAnnotation>()) {
    const std::vector<const TypeAnnotation*> annotations =
        ExpandVariables(annotation, table, parametric_context);
    for (const TypeAnnotation* annotation : annotations) {
      XLS_ASSIGN_OR_RETURN(std::optional<const StructDefBase*> def,
                           GetStructOrProcDef(annotation, import_data));
      if (def.has_value() && *def == &struct_def) {
        return true;
      }
    }
    return false;
  }
  XLS_ASSIGN_OR_RETURN(std::optional<const StructDefBase*> def,
                       GetStructOrProcDef(annotation, import_data));
  return def.has_value() && *def == &struct_def;
}

class FilterElement {
 public:
  FilterElement(TypeAnnotationFilterKind kind, FilterClosure filter,
                std::string_view custom_string = "")
      : kind_(kind),
        filter_(std::move(filter)),
        count_(nullptr),
        custom_string_(custom_string) {}

  FilterElement(TypeAnnotationFilterKind kind, int* count,
                std::string_view custom_string = "")
      : kind_(kind),
        filter_([](const TypeAnnotation*) { return false; }),
        count_(count),
        custom_string_(custom_string) {}

  TypeAnnotationFilterKind kind() const { return kind_; }

  void IncrementCount() { ++*count_; }

  bool Filter(const TypeAnnotation* annotation) const {
    return filter_(annotation);
  }

  std::string ToString() const {
    return custom_string_.empty()
               ? KindToString(kind_)
               : absl::Substitute("$0 ($1)", KindToString(kind_),
                                  custom_string_);
  }

 private:
  const TypeAnnotationFilterKind kind_;
  FilterClosure filter_;
  int* const count_;
  std::string custom_string_;
};

}  // namespace

class TypeAnnotationFilter::Impl {
 public:
  explicit Impl(FilterElement element) : chain_{std::move(element)} {}
  explicit Impl(std::list<FilterElement> chain) : chain_(std::move(chain)) {}

  std::list<FilterElement>& chain() { return chain_; }

 private:
  std::list<FilterElement> chain_;
};

TypeAnnotationFilter::TypeAnnotationFilter(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TypeAnnotationFilter::TypeAnnotationFilter(const TypeAnnotationFilter& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

TypeAnnotationFilter::TypeAnnotationFilter(TypeAnnotationFilter&& other) =
    default;

TypeAnnotationFilter& TypeAnnotationFilter::operator=(
    const TypeAnnotationFilter& other) {
  impl_ = std::make_unique<Impl>(*other.impl_);
  return *this;
}

TypeAnnotationFilter& TypeAnnotationFilter::operator=(TypeAnnotationFilter&&) =
    default;

TypeAnnotationFilter::~TypeAnnotationFilter() {}

TypeAnnotationFilter TypeAnnotationFilter::None() {
  return TypeAnnotationFilter(std::make_unique<Impl>(
      FilterElement(TypeAnnotationFilterKind::kNone,
                    [](const TypeAnnotation*) { return false; })));
}

TypeAnnotationFilter TypeAnnotationFilter::FilterSingleAny() {
  return TypeAnnotationFilter(std::make_unique<Impl>(FilterElement(
      TypeAnnotationFilterKind::kSingleAny,
      [](const TypeAnnotation* annotation) {
        return annotation->IsAnnotation<AnyTypeAnnotation>() &&
               !annotation->AsAnnotation<AnyTypeAnnotation>()->multiple();
      })));
}

TypeAnnotationFilter TypeAnnotationFilter::FilterMultiAny() {
  return TypeAnnotationFilter(std::make_unique<Impl>(FilterElement(
      TypeAnnotationFilterKind::kMultiAny,
      [](const TypeAnnotation* annotation) {
        return annotation->IsAnnotation<AnyTypeAnnotation>() &&
               annotation->AsAnnotation<AnyTypeAnnotation>()->multiple();
      })));
}

TypeAnnotationFilter TypeAnnotationFilter::FilterParamTypes() {
  return TypeAnnotationFilter(std::make_unique<Impl>(
      FilterElement(TypeAnnotationFilterKind::kMultiAny,
                    [](const TypeAnnotation* annotation) {
                      return annotation->IsAnnotation<ParamTypeAnnotation>();
                    })));
}

TypeAnnotationFilter TypeAnnotationFilter::FilterRefsWithMissingTypeInfo(
    const TypeInfo* ti) {
  return TypeAnnotationFilter(std::make_unique<Impl>(
      FilterElement(TypeAnnotationFilterKind::kMissingTypeInfo,
                    [ti](const TypeAnnotation* candidate) {
                      return HasAnyReferencesWithMissingTypeInfo(ti, candidate);
                    })));
}

TypeAnnotationFilter TypeAnnotationFilter::FilterReferencesToStruct(
    const InferenceTable* table,
    std::optional<const ParametricContext*> parametric_context,
    const StructDef* struct_def, const ImportData* import_data) {
  return TypeAnnotationFilter(std::make_unique<Impl>(FilterElement(
      TypeAnnotationFilterKind::kStructRef,
      [=](const TypeAnnotation* candidate) {
        absl::StatusOr<bool> result = RefersToStruct(
            *table, parametric_context, candidate, *struct_def, *import_data);
        CHECK(result.ok());
        return *result;
      },
      /*custom_string=*/struct_def->identifier())));
}

TypeAnnotationFilter TypeAnnotationFilter::BlockRecursion(
    const TypeAnnotation* annotation) {
  return TypeAnnotationFilter(std::make_unique<Impl>(FilterElement(
      TypeAnnotationFilterKind::kBlockRecursion,
      [annotation](const TypeAnnotation* candidate) {
        return candidate == annotation;
      },
      /*custom_string=*/annotation->ToString())));
}

TypeAnnotationFilter TypeAnnotationFilter::CaptureVariables(
    absl::flat_hash_set<const NameRef*>* container) {
  return TypeAnnotationFilter(std::make_unique<Impl>(FilterElement(
      TypeAnnotationFilterKind::kCaptureVariables,
      [container](const TypeAnnotation* candidate) {
        if (candidate->IsAnnotation<TypeVariableTypeAnnotation>()) {
          container->insert(
              candidate->AsAnnotation<TypeVariableTypeAnnotation>()
                  ->type_variable());
        }
        return false;
      })));
}

TypeAnnotationFilter TypeAnnotationFilter::CaptureRejectCount(int* count) {
  *count = 0;
  return TypeAnnotationFilter(std::make_unique<Impl>(
      FilterElement(TypeAnnotationFilterKind::kCaptureRejectCount, count)));
}

bool TypeAnnotationFilter::IsNone() const {
  return absl::c_all_of(impl_->chain(), [](const FilterElement& element) {
    return element.kind() == TypeAnnotationFilterKind::kNone;
  });
}

TypeAnnotationFilter TypeAnnotationFilter::Chain(TypeAnnotationFilter other) {
  std::list<FilterElement> chain = impl_->chain();
  chain.insert(chain.end(), other.impl_->chain().begin(),
               other.impl_->chain().end());
  return TypeAnnotationFilter(std::make_unique<Impl>(std::move(chain)));
}

bool TypeAnnotationFilter::Filter(const TypeAnnotation* annotation) {
  std::list<FilterElement*> rejection_counters;
  for (FilterElement& element : impl_->chain()) {
    if (element.kind() == TypeAnnotationFilterKind::kCaptureRejectCount) {
      rejection_counters.push_back(&element);
    }
    bool filtered = element.Filter(annotation);
    if (filtered) {
      for (FilterElement* rejection_counter : rejection_counters) {
        rejection_counter->IncrementCount();
      }
      return true;
    }
  }
  return false;
}

std::string TypeAnnotationFilter::ToString() const {
  if (impl_->chain().size() == 1) {
    return impl_->chain().begin()->ToString();
  }
  std::vector<std::string> chain_strings;
  for (const FilterElement& element : impl_->chain()) {
    chain_strings.push_back(element.ToString());
  }
  return absl::Substitute("chain: [$0]", absl::StrJoin(chain_strings, ", "));
}

void FilterAnnotations(std::vector<const TypeAnnotation*>& annotations,
                       TypeAnnotationFilter filter) {
  annotations.erase(std::remove_if(annotations.begin(), annotations.end(),
                                   [&](const TypeAnnotation* annotation) {
                                     return filter.Filter(annotation);
                                   }),
                    annotations.end());
}

}  // namespace xls::dslx
