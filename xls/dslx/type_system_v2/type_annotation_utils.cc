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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/ir/bits.h"
#include "xls/ir/number_parser.h"

namespace xls::dslx {
namespace {

Expr* CreateElementCountInvocation(Module& module, TypeAnnotation* annotation) {
  NameRef* element_count =
      module.Make<NameRef>(annotation->span(), "element_count",
                           module.GetOrCreateBuiltinNameDef("element_count"));
  return module.Make<Invocation>(annotation->span(), element_count,
                                 std::vector<Expr*>{},
                                 std::vector<ExprOrType>{annotation});
}

}  // namespace

Number* CreateUntypedZero(Module& module, const Span& span) {
  return module.Make<Number>(span, "0", NumberKind::kOther,
                             /*type_annotation=*/nullptr);
}

TypeAnnotation* CreateUnOrSnAnnotation(Module& module, const Span& span,
                                       bool is_signed, int64_t bit_count) {
  return CreateUnOrSnAnnotation(
      module, span, is_signed,
      module.Make<Number>(span, absl::StrCat(bit_count), NumberKind::kOther,
                          /*type_annotation=*/nullptr));
}

TypeAnnotation* CreateUnOrSnAnnotation(Module& module, const Span& span,
                                       bool is_signed, Expr* bit_count) {
  return module.Make<ArrayTypeAnnotation>(
      span, CreateUnOrSnElementAnnotation(module, span, is_signed), bit_count);
}

TypeAnnotation* CreateUnOrSnElementAnnotation(Module& module, const Span& span,
                                              bool is_signed) {
  return module.Make<BuiltinTypeAnnotation>(
      span, is_signed ? BuiltinType::kSN : BuiltinType::kUN,
      module.GetOrCreateBuiltinNameDef(is_signed ? "sN" : "uN"));
}

TypeAnnotation* CreateBoolAnnotation(Module& module, const Span& span) {
  return module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kBool, module.GetOrCreateBuiltinNameDef("bool"));
}

TypeAnnotation* CreateU32Annotation(Module& module, const Span& span) {
  return module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU32, module.GetOrCreateBuiltinNameDef("u32"));
}

TypeAnnotation* CreateU8Annotation(Module& module, const Span& span) {
  return module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU8, module.GetOrCreateBuiltinNameDef("u8"));
}

TypeAnnotation* CreateS32Annotation(Module& module, const Span& span) {
  return module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kS32, module.GetOrCreateBuiltinNameDef("s32"));
}

TypeAnnotation* CreateBuiltinTypeAnnotation(Module& module,
                                            BuiltinNameDef* name_def,
                                            const Span& span) {
  BuiltinType builtin_type = *BuiltinTypeFromString(name_def->ToString());
  return module.Make<BuiltinTypeAnnotation>(span, builtin_type, name_def);
}

TypeAnnotation* CreateStructAnnotation(
    Module& module, StructDef* def, std::vector<ExprOrType> parametrics,
    std::optional<const StructInstanceBase*> instantiator) {
  return module.Make<TypeRefTypeAnnotation>(
      def->span(), module.Make<TypeRef>(def->span(), def),
      std::move(parametrics), instantiator);
}

TypeAnnotation* CreateStructAnnotation(Module& module,
                                       const StructOrProcRef& ref) {
  CHECK(ref.def->kind() == AstNodeKind::kStructDef);
  return CreateStructAnnotation(
      module, dynamic_cast<StructDef*>(const_cast<StructDefBase*>(ref.def)),
      ref.parametrics, std::nullopt);
}

ChannelTypeAnnotation* GetChannelArrayElementType(
    Module& module, const ChannelTypeAnnotation* channel_array_type) {
  std::optional<std::vector<Expr*>> rest_of_dims;
  if (channel_array_type->dims().has_value() &&
      channel_array_type->dims()->size() > 1) {
    rest_of_dims.emplace(channel_array_type->dims()->begin(),
                         channel_array_type->dims()->end() - 1);
  }
  return module.Make<ChannelTypeAnnotation>(
      channel_array_type->span(), channel_array_type->direction(),
      channel_array_type->payload(), rest_of_dims);
}

absl::StatusOr<SignednessAndBitCountResult> GetSignednessAndBitCount(
    const TypeAnnotation* annotation, bool ignore_missing_dimensions) {
  if (const auto* builtin_annotation =
          dynamic_cast<const BuiltinTypeAnnotation*>(annotation)) {
    if (ignore_missing_dimensions) {
      absl::StatusOr<bool> signedness = builtin_annotation->GetSignedness();
      return SignednessAndBitCountResult(signedness.value_or(false),
                                         builtin_annotation->GetBitCount());
    }
    switch (builtin_annotation->builtin_type()) {
      case BuiltinType::kXN:
        return absl::InvalidArgumentError(
            "`xN` requires a specified signedness.");
      case BuiltinType::kUN:
      case BuiltinType::kSN:
        return absl::InvalidArgumentError(absl::Substitute(
            "`$0` requires a specified bit count.", annotation->ToString()));
      case BuiltinType::kToken:
      case BuiltinType::kChannelIn:
      case BuiltinType::kChannelOut:
        return absl::NotFoundError(absl::StrCat("Annotation is not bits-like: ",
                                                annotation->ToString()));
      default:
        // Handle things like `s32` and `u32`, which have an implied signedness
        // and bit count. This logic also handles `bits`, which stores its
        // array-style bit count inside the builtin annotation.
        XLS_ASSIGN_OR_RETURN(bool signedness,
                             builtin_annotation->GetSignedness());
        return SignednessAndBitCountResult(signedness,
                                           builtin_annotation->GetBitCount());
    }
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
        if (!multi_dimensional && !ignore_missing_dimensions) {
          return absl::InvalidArgumentError(
              "`xN` requires a specified bit count.");
        }
        result.signedness = array_annotation->dim();
      } else if (multi_dimensional) {
        // This is something like uN[32][2].
        return absl::NotFoundError(absl::Substitute(
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
  return absl::NotFoundError(
      absl::StrCat("Cannot extract signedness and bit count from annotation: ",
                   annotation->ToString()));
}

absl::StatusOr<SignednessAndBitCountResult>
GetSignednessAndBitCountWithUserFacingError(
    const TypeAnnotation* annotation, const FileTable& file_table,
    absl::AnyInvocable<absl::Status()> default_error_factory) {
  absl::StatusOr<SignednessAndBitCountResult> result =
      GetSignednessAndBitCount(annotation);
  if (result.ok()) {
    return result;
  }
  if (absl::IsNotFound(result.status())) {
    return default_error_factory();
  }
  return TypeInferenceErrorStatus(annotation->span(), /*type=*/nullptr,
                                  result.status().message(), file_table);
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
      XLS_ASSIGN_OR_RETURN((auto [sign, magnitude]),
                           GetSignAndMagnitude(number.text()));
      XLS_ASSIGN_OR_RETURN(Bits raw_bits, ParseNumber(number.text()));
      const bool is_negative = sign == Sign::kNegative;
      return CreateUnOrSnAnnotation(module, number.span(), is_negative,
                                    raw_bits.bit_count());
  }
}

TypeAnnotation* CreateUnitTupleAnnotation(Module& module, const Span& span) {
  return module.Make<TupleTypeAnnotation>(
      span, /*members=*/std::vector<TypeAnnotation*>{});
}

FunctionTypeAnnotation* CreateFunctionTypeAnnotation(Module& module,
                                                     const Function& function) {
  std::vector<const TypeAnnotation*> param_types;
  param_types.reserve(function.params().size());
  for (const Param* param : function.params()) {
    param_types.push_back(param->type_annotation());
  }
  return module.Make<FunctionTypeAnnotation>(
      param_types,
      const_cast<TypeAnnotation*>(GetReturnType(module, function)));
}

const TypeAnnotation* GetReturnType(Module& module, const Function& fn) {
  return fn.return_type() != nullptr
             ? fn.return_type()
             : CreateUnitTupleAnnotation(module, fn.span());
}

const ArrayTypeAnnotation* CastToNonBitsArrayTypeAnnotation(
    const TypeAnnotation* annotation) {
  const auto* array_annotation =
      dynamic_cast<const ArrayTypeAnnotation*>(annotation);
  if (array_annotation == nullptr) {
    return nullptr;
  }
  // If the signedness and bit count can be retrieved, then it's some flavor of
  // xN, uN, sN, etc. and not what this function is looking for. If that yields
  // an invalid argument error, then it's still one of those types but
  // malformed, and the malformedness doesn't mean it's an array.
  absl::StatusOr<SignednessAndBitCountResult> signedness_and_bit_count =
      GetSignednessAndBitCount(annotation);
  return !signedness_and_bit_count.ok() &&
                 !absl::IsInvalidArgument(signedness_and_bit_count.status())
             ? array_annotation
             : nullptr;
}

absl::Status VerifyAllParametricsSatisfied(
    const std::vector<ParametricBinding*>& bindings,
    const std::vector<ExprOrType>& actual_parametrics,
    std::string_view binding_owner_identifier, const Span& error_span,
    const FileTable& file_table) {
  std::vector<std::string> missing_parametric_names;
  for (int i = actual_parametrics.size(); i < bindings.size(); i++) {
    const ParametricBinding* binding = bindings[i];
    if (binding->expr() == nullptr) {
      missing_parametric_names.push_back(binding->identifier());
    }
  }
  if (missing_parametric_names.empty()) {
    return absl::OkStatus();
  }
  return TypeInferenceErrorStatus(
      error_span, /*type=*/nullptr,
      absl::Substitute("Use of `$0` with missing parametric(s): $1",
                       binding_owner_identifier,
                       absl::StrJoin(missing_parametric_names, ", ")),
      file_table);
}

CloneReplacer NameRefMapper(
    const absl::flat_hash_map<const NameDef*, ExprOrType>& map) {
  return [&](const AstNode* node) -> absl::StatusOr<std::optional<AstNode*>> {
    if (const auto* ref = dynamic_cast<const NameRef*>(node);
        ref != nullptr &&
        std::holds_alternative<const NameDef*>(ref->name_def())) {
      const auto it = map.find(std::get<const NameDef*>(ref->name_def()));
      if (it != map.end()) {
        return ToAstNode(it->second);
      }
    }
    return std::nullopt;
  };
}

Expr* CreateElementCountSum(Module& module, TypeAnnotation* lhs,
                            TypeAnnotation* rhs) {
  return module.Make<Binop>(
      lhs->span(), BinopKind::kAdd, CreateElementCountInvocation(module, lhs),
      CreateElementCountInvocation(module, rhs), Span::None());
}

Expr* CreateElementCountOffset(Module& module, TypeAnnotation* lhs,
                               int64_t offset) {
  if (offset == 0) {
    return CreateElementCountInvocation(module, lhs);
  }
  return module.Make<Binop>(
      lhs->span(), BinopKind::kSub, CreateElementCountInvocation(module, lhs),
      module.Make<Number>(lhs->span(), absl::StrCat(offset), NumberKind::kOther,
                          /*type_annotation=*/nullptr),
      Span::None());
}

Expr* CreateRangeElementCount(Module& module, const Range* range) {
  const Span& span = range->span();
  // Cast start and end to s32 since array size type is assumed to be U32, this
  // ensures arithmetic correctness for types smaller than 32 bit, or types
  // greater than 32 bits as long as the difference fits in a U32.
  // If the difference does not fit in a U32, for example,
  // 0xFFFF,FFFF,FFFF,FFFF..0x0000111100001111, it is silently truncated to U32,
  // and this needed to be checked at const evaluation of the range.
  Expr* start = module.Make<Cast>(span, range->start(),
                                  CreateS32Annotation(module, span));
  Expr* end =
      module.Make<Cast>(span, range->end(), CreateS32Annotation(module, span));
  Expr* result = module.Make<Binop>(span, BinopKind::kSub, end, start, span);
  if (range->inclusive_end()) {
    Expr* one = module.Make<Number>(span, "1", NumberKind::kOther, nullptr);
    result = module.Make<Binop>(span, BinopKind::kAdd, result, one, span);
  }
  return result;
}

absl::StatusOr<InterpValueWithTypeAnnotation> GetBuiltinMember(
    Module& module, bool is_signed, uint32_t bit_count,
    std::string_view member_name, const Span& span,
    std::string_view object_type_for_error, const FileTable& file_table) {
  const TypeAnnotation* result_annotation =
      CreateUnOrSnAnnotation(module, span, is_signed, bit_count);
  if (member_name == "ZERO") {
    return InterpValueWithTypeAnnotation{
        .type_annotation = result_annotation,
        .value = InterpValue::MakeZeroValue(is_signed, bit_count)};
  }
  if (member_name == "MAX") {
    return InterpValueWithTypeAnnotation{
        .type_annotation = result_annotation,
        .value = InterpValue::MakeMaxValue(is_signed, bit_count)};
  }
  if (member_name == "MIN") {
    return InterpValueWithTypeAnnotation{
        .type_annotation = result_annotation,
        .value = InterpValue::MakeMinValue(is_signed, bit_count)};
  }
  return TypeInferenceErrorStatus(
      span, nullptr,
      absl::Substitute("Builtin type '$0' does not have attribute '$1'.",
                       object_type_for_error, member_name),
      file_table);
}

bool IsToken(const TypeAnnotation* annotation) {
  if (const auto* built_in =
          dynamic_cast<const BuiltinTypeAnnotation*>(annotation)) {
    return built_in->builtin_type() == BuiltinType::kToken;
  }
  return false;
}

const BuiltinTypeAnnotation* CastToTokenType(const TypeAnnotation* annotation) {
  if (!IsToken(annotation)) {
    return nullptr;
  }
  return dynamic_cast<const BuiltinTypeAnnotation*>(annotation);
}

const FunctionTypeAnnotation* ExpandVarargs(
    Module& module, const FunctionTypeAnnotation* signature, int count) {
  CHECK_GT(signature->param_types().size(), 0);

  const TypeAnnotation* last_param_type = signature->param_types().back();
  std::vector<const TypeAnnotation*> param_types(
      signature->param_types().begin(), signature->param_types().end() - 1);
  for (int i = 0; i < count; ++i) {
    param_types.push_back(last_param_type);
  }

  return module.Make<FunctionTypeAnnotation>(param_types,
                                             signature->return_type());
}

bool IsBitsLikeFragment(const TypeAnnotation* annotation) {
  if (const auto* builtin =
          dynamic_cast<const BuiltinTypeAnnotation*>(annotation)) {
    return builtin->builtin_type() == BuiltinType::kXN ||
           builtin->builtin_type() == BuiltinType::kUN ||
           builtin->builtin_type() == BuiltinType::kSN ||
           builtin->builtin_type() == BuiltinType::kBits;
  }
  if (const auto* array =
          dynamic_cast<const ArrayTypeAnnotation*>(annotation)) {
    // The immediate element type of a complete xN annotation would be a nested
    // array type and not xN.
    const auto* builtin_element =
        dynamic_cast<const BuiltinTypeAnnotation*>(array->element_type());
    return builtin_element != nullptr &&
           builtin_element->builtin_type() == BuiltinType::kXN;
  }
  return false;
}

bool IsRangeInMatchArm(const Range* range) {
  if (const NameDefTree* namedef =
          dynamic_cast<const NameDefTree*>(range->parent())) {
    if (const MatchArm* matcharm =
            dynamic_cast<const MatchArm*>(namedef->parent())) {
      auto& patterns = matcharm->patterns();
      if (std::find(patterns.begin(), patterns.end(), namedef) !=
          patterns.end()) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace xls::dslx
