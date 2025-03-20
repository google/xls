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
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
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

Expr* CreateS32Zero(Module& module, const Span& span) {
  return module.Make<Number>(span, "0", NumberKind::kOther,
                             CreateS32Annotation(module, span));
}

// Expands the `Expr` for a slice bound, which may be `nullptr`, into its
// longhand form. For example, in `x[-3:]`, the `-3` actually means
// `element_count<typeof(x)>() - 3`. The null `Expr` for the RHS bound means
// `element_count<typeof(x)>()`. Since it's not in a position to decide whether
// a non-null expr is negative, this utility will produce an `Expr` like `if
// bound < 0 { element_count<typeof(x)> + bound } else { bound }` and ultimately
// `ConstExprEvaluator`, given the right context, can evaluate that. Note that
// in v2 we do not silently clamp slice bounds as in v1; this is an intentional
// design change.
absl::StatusOr<Expr*> ExpandSliceBoundExpr(Module& module,
                                           TypeAnnotation* source_array_type,
                                           Expr* bound, bool start) {
  if (bound == nullptr) {
    return start ? CreateS32Zero(module, source_array_type->span())
                 : module.Make<Cast>(
                       source_array_type->span(),
                       CreateElementCountInvocation(module, source_array_type),
                       CreateS32Annotation(module, source_array_type->span()));
  }
  // The following is the AST we generate here.
  //
  // if bound < 0 {
  //   (element_count<source_array_type>() as s32) + bound
  // } else {
  //   bound
  // }
  //
  // TODO - https://github.com/google/xls/issues/193: it should be using
  // `widening_cast<s32>(bound)` for each occurrence of `bound`, but currently
  // there is an issue invoking builtins other than `element_count` from the
  // context of a fabricated expression.

  XLS_ASSIGN_OR_RETURN(AstNode * bound_copy1, CloneAst(bound));
  XLS_ASSIGN_OR_RETURN(AstNode * bound_copy2, CloneAst(bound));
  XLS_ASSIGN_OR_RETURN(AstNode * bound_copy3, CloneAst(bound));
  Expr* bound_less_than_zero = module.Make<Binop>(
      bound->span(), BinopKind::kLt, down_cast<Expr*>(bound_copy1),
      CreateS32Zero(module, bound->span()), Span::None());
  Statement::Wrapped wrapped_add = *Statement::NodeToWrapped(module.Make<Binop>(
      bound->span(), BinopKind::kAdd,
      module.Make<Cast>(bound->span(),
                        CreateElementCountInvocation(module, source_array_type),
                        CreateS32Annotation(module, bound->span())),
      down_cast<Expr*>(bound_copy2), Span::None()));
  Statement::Wrapped wrapped_bound =
      *Statement::NodeToWrapped(down_cast<Expr*>(bound_copy3));
  return module.Make<Conditional>(
      bound->span(), bound_less_than_zero,
      module.Make<StatementBlock>(
          bound->span(),
          std::vector<Statement*>{module.Make<Statement>(wrapped_add)},
          /*trailing_semi=*/false),
      module.Make<StatementBlock>(
          bound->span(),
          std::vector<Statement*>{module.Make<Statement>(wrapped_bound)},
          /*trailing_semi=*/false));
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
  std::vector<TypeAnnotation*> param_types;
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
  // xN, uN, sN, etc. and not what this function is looking for.
  absl::StatusOr<SignednessAndBitCountResult> signedness_and_bit_count =
      GetSignednessAndBitCount(annotation);
  return !signedness_and_bit_count.ok() ? array_annotation : nullptr;
}

absl::StatusOr<std::optional<StructOrProcRef>> GetStructOrProcRef(
    const TypeAnnotation* annotation, const FileTable& file_table) {
  const auto* type_ref_annotation =
      dynamic_cast<const TypeRefTypeAnnotation*>(annotation);
  if (type_ref_annotation == nullptr) {
    return std::nullopt;
  }

  // Collect parametrics and instantiator by walking through any type
  // aliases before getting the struct or proc definition.
  std::vector<ExprOrType> parametrics = type_ref_annotation->parametrics();
  std::optional<const StructInstanceBase*> instantiator =
      type_ref_annotation->instantiator();
  TypeDefinition maybe_alias =
      type_ref_annotation->type_ref()->type_definition();

  while (std::holds_alternative<TypeAlias*>(maybe_alias) &&
         dynamic_cast<TypeRefTypeAnnotation*>(
             &std::get<TypeAlias*>(maybe_alias)->type_annotation())) {
    type_ref_annotation = dynamic_cast<TypeRefTypeAnnotation*>(
        &std::get<TypeAlias*>(maybe_alias)->type_annotation());
    if (!parametrics.empty() && !type_ref_annotation->parametrics().empty()) {
      return TypeInferenceErrorStatus(
          annotation->span(), /* type= */ nullptr,
          absl::StrFormat(
              "Parametric values defined multiple times for annotation: `%s`",
              annotation->ToString()),
          file_table);
    }

    parametrics =
        parametrics.empty() ? type_ref_annotation->parametrics() : parametrics;
    instantiator = instantiator.has_value()
                       ? instantiator
                       : type_ref_annotation->instantiator();
    maybe_alias = type_ref_annotation->type_ref()->type_definition();
  }

  std::optional<const StructDefBase*> def =
      GetStructOrProcDef(type_ref_annotation);
  if (!def.has_value()) {
    return std::nullopt;
  }
  return StructOrProcRef{
      .def = *def, .parametrics = parametrics, .instantiator = instantiator};
}

std::optional<const StructDefBase*> GetStructOrProcDef(
    const TypeAnnotation* annotation) {
  const auto* type_ref_annotation =
      dynamic_cast<const TypeRefTypeAnnotation*>(annotation);
  if (type_ref_annotation == nullptr) {
    return std::nullopt;
  }
  const TypeDefinition& def =
      type_ref_annotation->type_ref()->type_definition();
  return absl::visit(
      Visitor{[](TypeAlias* alias) -> std::optional<const StructDefBase*> {
                return GetStructOrProcDef(&alias->type_annotation());
              },
              [](StructDef* struct_def) -> std::optional<const StructDefBase*> {
                return struct_def;
              },
              [](ProcDef* proc_def) -> std::optional<const StructDefBase*> {
                return proc_def;
              },
              [](ColonRef* colon_ref) -> std::optional<const StructDefBase*> {
                if (std::holds_alternative<TypeRefTypeAnnotation*>(
                        colon_ref->subject())) {
                  return GetStructOrProcDef(
                      std::get<TypeRefTypeAnnotation*>(colon_ref->subject()));
                }
                return std::nullopt;
              },
              [](EnumDef*) -> std::optional<const StructDefBase*> {
                return std::nullopt;
              },
              [](UseTreeEntry*) -> std::optional<const StructDefBase*> {
                // TODO(https://github.com/google/xls/issues/352): 2025-01-23
                // Resolve possible Struct or Proc definition through the extern
                // UseTreeEntry.
                return std::nullopt;
              }},
      def);
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

absl::StatusOr<StartAndWidthExprs> CreateSliceStartAndWidthExprs(
    Module& module, TypeAnnotation* source_array_type, const AstNode* slice) {
  CHECK(slice->kind() == AstNodeKind::kSlice ||
        slice->kind() == AstNodeKind::kWidthSlice);
  if (const auto* width_slice = dynamic_cast<const WidthSlice*>(slice)) {
    // In a width slice, the LHS is the start index and the RHS is a type
    // annotation whose element count is the number of elements the slice is
    // asking to extract from the source array.
    return StartAndWidthExprs{
        .start = width_slice->start(),
        .width = CreateElementCountInvocation(module, width_slice->width())};
  }
  const auto* bounded_slice = dynamic_cast<const Slice*>(slice);
  XLS_ASSIGN_OR_RETURN(
      Expr * limit,
      ExpandSliceBoundExpr(module, source_array_type, bounded_slice->limit(),
                           /*start=*/false));
  XLS_ASSIGN_OR_RETURN(
      Expr * start,
      ExpandSliceBoundExpr(module, source_array_type, bounded_slice->start(),
                           /*start=*/true));
  return StartAndWidthExprs{
      .start = start,
      .width = module.Make<Binop>(*slice->GetSpan(), BinopKind::kSub, limit,
                                  start, Span::None())};
}

void FilterAnnotations(
    std::vector<const TypeAnnotation*>& annotations,
    absl::FunctionRef<bool(const TypeAnnotation*)> accept_predicate) {
  annotations.erase(std::remove_if(annotations.begin(), annotations.end(),
                                   [&](const TypeAnnotation* annotation) {
                                     return !accept_predicate(annotation);
                                   }),
                    annotations.end());
}

Expr* CreateRangeElementCount(Module& module, const Range* range) {
  const Span& span = range->span();
  // Cast start and end to s32 since array size type is assumed to be U32, this
  // ensures arithmetic correctness for types smaller than 32 bit, or types
  // greater than 32 bits as long as the difference fits in a U32.
  // If the difference does not fit in a U32, for example,
  // 0xFFFF,FFFF,FFFF,FFFF..0x0000111100001111, it is silently truncated to U32,
  // and this needed to be checked at validate_concrete_type.
  Expr* start = module.Make<Cast>(span, range->start(),
                                  CreateS32Annotation(module, span));
  Expr* end =
      module.Make<Cast>(span, range->end(), CreateS32Annotation(module, span));
  return module.Make<Binop>(span, BinopKind::kSub, end, start, span);
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

}  // namespace xls::dslx
