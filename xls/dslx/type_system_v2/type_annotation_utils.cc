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
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
#include "xls/ir/number_parser.h"

namespace xls::dslx {
namespace {

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

Expr* CreateElementCountInvocation(Module& module, TypeAnnotation* annotation) {
  NameRef* bit_count =
      module.Make<NameRef>(annotation->span(), "element_count",
                           module.GetOrCreateBuiltinNameDef("element_count"));
  return module.Make<Invocation>(annotation->span(), bit_count,
                                 std::vector<Expr*>{},
                                 std::vector<ExprOrType>{annotation});
}

Number* CreateS32Zero(Module& module, const Span& span) {
  return module.Make<Number>(span, "0", NumberKind::kOther,
                             CreateS32Annotation(module, span));
}

// Expands the `Expr` for a slice bound, which may be `nullptr`, into its
// longhand form. For example, in `x[-3:]`, the `-3` actually means
// `element_count<typeof(x)>() - 3`. The null `Expr` for the RHS bound means
// `element_count<typeof(x)>()`. Since it's not in a position to decide whether
// a non-null expr is negative, this utility will produce an `Expr` like `if
// bound < 0 { element_count<typeof(x)> + bound } else { bound }` and ultimately
// `ConstExprEvaluator`, given the right context, can evaluate that.
// TODO - https://github.com/google/xls/issues/193: The bounds are supposed to
// be clamped such that the expanded start is in [0, element_count<typeof(x)>())
// and the expanded limit is in [start, element_count<typeof(x)>()). We should
// add an integer clamping builtin and do this in a follow-up, because it would
// be quite messy without that.
absl::StatusOr<Expr*> ExpandSliceBoundExpr(Module& module,
                                           TypeAnnotation* source_array_type,
                                           Expr* bound, bool start) {
  if (bound == nullptr) {
    return start ? CreateS32Zero(module, source_array_type->span())
                 : CreateElementCountInvocation(module, source_array_type);
  }
  // The following is the AST we generate here:
  // if bound < 0 { element_count<source_array_type>() + bound } else { bound }
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

TypeAnnotation* CreateS32Annotation(Module& module, const Span& span) {
  return module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kS32, module.GetOrCreateBuiltinNameDef("s32"));
}

TypeAnnotation* CreateStructAnnotation(
    Module& module, StructDef* def, std::vector<ExprOrType> parametrics,
    std::optional<const StructInstance*> instantiator) {
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
      const bool is_negative = sign == Sign::kNegative;
      return CreateUnOrSnAnnotation(
          module, number.span(), is_negative,
          magnitude.bit_count() + (is_negative ? 1 : 0));
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

std::optional<StructOrProcRef> GetStructOrProcRef(
    const TypeAnnotation* annotation) {
  const auto* type_ref_annotation =
      dynamic_cast<const TypeRefTypeAnnotation*>(annotation);
  if (type_ref_annotation == nullptr) {
    return std::nullopt;
  }
  std::optional<const StructDefBase*> def =
      GetStructOrProcDef(type_ref_annotation);
  if (!def.has_value()) {
    return std::nullopt;
  }
  return StructOrProcRef{.def = *def,
                         .parametrics = type_ref_annotation->parametrics(),
                         .instantiator = type_ref_annotation->instantiator()};
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
    XLS_ASSIGN_OR_RETURN(
        Expr * start,
        ExpandSliceBoundExpr(module, source_array_type, width_slice->start(),
                             /*start=*/true));
    // In a width slice, the LHS is the start index and the RHS is a type
    // annotation whose element count is the number of elements the slice is
    // asking to extract from the source array.
    return StartAndWidthExprs{
        .start = start,
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

}  // namespace xls::dslx
