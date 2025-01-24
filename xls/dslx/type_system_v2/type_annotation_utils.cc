// Copyright 2024 The XLS Authors
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

#include "xls/dslx/type_system_v2/type_annotation_utils.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/ir/bits.h"
#include "xls/ir/number_parser.h"

namespace xls::dslx {
namespace {

std::optional<StructOrProcDef> GetStructOrProcDef(
    const TypeAnnotation* annotation) {
  const auto* type_ref_annotation =
      dynamic_cast<const TypeRefTypeAnnotation*>(annotation);
  if (type_ref_annotation == nullptr) {
    return std::nullopt;
  }
  const TypeDefinition& def =
      type_ref_annotation->type_ref()->type_definition();
  return absl::visit(
      Visitor{[](TypeAlias* alias) -> std::optional<StructOrProcDef> {
                return GetStructOrProcDef(&alias->type_annotation());
              },
              [](StructDef* struct_def) -> std::optional<StructOrProcDef> {
                return struct_def;
              },
              [](ProcDef* proc_def) -> std::optional<StructOrProcDef> {
                return proc_def;
              },
              [](ColonRef* colon_ref) -> std::optional<StructOrProcDef> {
                if (std::holds_alternative<TypeRefTypeAnnotation*>(
                        colon_ref->subject())) {
                  return GetStructOrProcDef(
                      std::get<TypeRefTypeAnnotation*>(colon_ref->subject()));
                }
                return std::nullopt;
              },
              [](EnumDef*) -> std::optional<StructOrProcDef> {
                return std::nullopt;
              }},
      def);
}

}  // namespace

TypeAnnotation* CreateUnOrSnAnnotation(Module& module, const Span& span,
                                       bool is_signed, int64_t bit_count) {
  return module.Make<ArrayTypeAnnotation>(
      span,
      module.Make<BuiltinTypeAnnotation>(
          span, is_signed ? BuiltinType::kSN : BuiltinType::kUN,
          module.GetOrCreateBuiltinNameDef(is_signed ? "sN" : "uN")),
      module.Make<Number>(span, absl::StrCat(bit_count), NumberKind::kOther,
                          /*type_annotation=*/nullptr));
}

TypeAnnotation* CreateBoolAnnotation(Module& module, const Span& span) {
  return module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kBool, module.GetOrCreateBuiltinNameDef("bool"));
}

TypeAnnotation* CreateU32Annotation(Module& module, const Span& span) {
  return module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU32, module.GetOrCreateBuiltinNameDef("u32"));
}

TypeAnnotation* CreateStructAnnotation(
    Module& module, StructDef* def, std::vector<ExprOrType> parametrics,
    std::optional<const StructInstance*> instantiator) {
  return module.Make<TypeRefTypeAnnotation>(
      def->span(), module.Make<TypeRef>(def->span(), def),
      std::move(parametrics), instantiator);
}

absl::StatusOr<SignednessAndBitCountResult> GetSignednessAndBitCount(
    const TypeAnnotation* annotation) {
  if (const auto* builtin_annotation =
          dynamic_cast<const BuiltinTypeAnnotation*>(annotation)) {
    // Handle things like `s32` and `u32`, which have an implied signedness and
    // bit count.
    XLS_ASSIGN_OR_RETURN(bool signedness, builtin_annotation->GetSignedness());
    return SignednessAndBitCountResult(signedness,
                                       builtin_annotation->GetBitCount());
  }
  if (const auto* array_annotation =
          dynamic_cast<const ArrayTypeAnnotation*>(annotation)) {
    SignednessAndBitCountResult result;
    bool multi_dimensional = false;
    if (const auto* inner_array_annotation =
            dynamic_cast<const ArrayTypeAnnotation*>(
                array_annotation->element_type())) {
      // If the array has 2 dimensions, let's work with the hypothesis that it's
      // an `xN[S][N]` kind of annotation. We retain the bit count, which is the
      // outer dim, and unwrap the inner array to be processed below. If it
      // turns out to be some other multi-dimensional array type that does not
      // have a signedness and bit count, we will fail below.
      result.bit_count = array_annotation->dim();
      array_annotation = inner_array_annotation;
      multi_dimensional = true;
    }
    // If the element type has a zero bit count, that means the bit count is
    // captured by a wrapping array dim. If it has a nonzero bit count, then
    // it's an array of multiple integers with an implied bit count (e.g.
    // `s32[N]`). This function isn't applicable to the latter, and will error
    // below.
    if (const auto* builtin_element_annotation =
            dynamic_cast<const BuiltinTypeAnnotation*>(
                array_annotation->element_type());
        builtin_element_annotation != nullptr &&
        builtin_element_annotation->GetBitCount() == 0) {
      if (builtin_element_annotation->builtin_type() == BuiltinType::kXN) {
        // `xN` has an expression for the signedness, which appears as the inner
        // array dim.
        result.signedness = array_annotation->dim();
      } else if (multi_dimensional) {
        // This is something like uN[32][2].
        return absl::InvalidArgumentError(absl::Substitute(
            "Type annotation $0 does not have a signedness and bit count.",
            annotation->ToString()));
      } else {
        // All other types, e.g. `uN`, `sN`, and `bits`, have an implied
        // signedness that we can just get as a bool.
        result.bit_count = array_annotation->dim();
        XLS_ASSIGN_OR_RETURN(result.signedness,
                             builtin_element_annotation->GetSignedness());
      }
      return result;
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Cannot extract signedness and bit count from annotation: ",
                   annotation->ToString()));
}

absl::StatusOr<TypeAnnotation*> CreateAnnotationSizedToFit(
    Module& module, const Number& number) {
  switch (number.number_kind()) {
    case NumberKind::kCharacter:
      return module.Make<BuiltinTypeAnnotation>(
          number.span(), BuiltinType::kU8,
          module.GetOrCreateBuiltinNameDef("u8"));
    case NumberKind::kBool:
      return module.Make<BuiltinTypeAnnotation>(
          number.span(), BuiltinType::kBool,
          module.GetOrCreateBuiltinNameDef("bool"));
    case NumberKind::kOther:
      std::pair<bool, Bits> sign_magnitude;
      XLS_ASSIGN_OR_RETURN(sign_magnitude, GetSignAndMagnitude(number.text()));
      const auto& [sign, magnitude] = sign_magnitude;
      return CreateUnOrSnAnnotation(module, number.span(), sign,
                                    magnitude.bit_count() + (sign ? 1 : 0));
  }
}

TypeAnnotation* CreateUnitTupleAnnotation(Module& module, const Span& span) {
  return module.Make<TupleTypeAnnotation>(
      span, /*members=*/std::vector<TypeAnnotation*>{});
}

const ArrayTypeAnnotation* CastToNonBitsArrayTypeAnnotation(
    const TypeAnnotation* annotation) {
  const auto* array_annotation =
      dynamic_cast<const ArrayTypeAnnotation*>(annotation);
  if (array_annotation == nullptr) {
    return nullptr;
  }
  // If the signedness and bit count can be retrieved, then it's some flavor of
  // xN, uN, sN, etc. and not what this function is looking for.
  absl::StatusOr<SignednessAndBitCountResult> signedness_and_bit_count =
      GetSignednessAndBitCount(annotation);
  return !signedness_and_bit_count.ok() ? array_annotation : nullptr;
}

std::optional<StructOrProcRef> GetStructOrProcRef(
    const TypeAnnotation* annotation) {
  const auto* type_ref_annotation =
      dynamic_cast<const TypeRefTypeAnnotation*>(annotation);
  if (type_ref_annotation == nullptr) {
    return std::nullopt;
  }
  std::optional<StructOrProcDef> def = GetStructOrProcDef(type_ref_annotation);
  if (!def.has_value()) {
    return std::nullopt;
  }
  return StructOrProcRef{.def = *def,
                         .parametrics = type_ref_annotation->parametrics(),
                         .instantiator = type_ref_annotation->instantiator()};
}

}  // namespace xls::dslx
