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

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {
namespace {

bool IsAbstractStructOrProcRef(const StructOrProcRef& ref) {
  return GetRequiredParametricBindings(ref.def->parametric_bindings()).size() >
         ref.parametrics.size();
}

}  // namespace

absl::StatusOr<Number*> MakeTypeCheckedNumber(
    Module& module, InferenceTable& table, const Span& span,
    const InterpValue& value, const TypeAnnotation* type_annotation) {
  // Invariant: nodes created into `module` should either have a "no-file" span
  // (for internally-fabricated nodes) or a span that points at `module`'s own
  // source file. Violating this makes downstream consumers that resolve nodes
  // by (kind, span) fragile and can lead to confusing "could not find node"
  // errors.
  //
  // Note: not all modules have a filesystem path (e.g. in-memory modules); we
  // only enforce this when `fs_path()` is known.
  XLS_RET_CHECK(module.file_table() != nullptr);
  if (span.HasFile() && module.fs_path().has_value()) {
    std::string_view span_filename = span.GetFilename(*module.file_table());
    XLS_RET_CHECK_EQ(span_filename, module.fs_path()->generic_string())
        << "MakeTypeCheckedNumber span filename must match module fs_path; "
        << "module name: `" << module.name() << "`; "
        << "span: `" << span.ToString(*module.file_table()) << "`; "
        << "module fs_path: `" << module.fs_path()->generic_string() << "`";
  }

  VLOG(5) << "Creating type-checked number: " << value.ToString()
          << " of type: " << type_annotation->ToString();
  Number* number = module.Make<Number>(span, value.ToString(/*humanize=*/true),
                                       NumberKind::kOther, nullptr);
  XLS_RETURN_IF_ERROR(table.SetTypeAnnotation(number, type_annotation));
  return number;
}

absl::StatusOr<Number*> MakeTypeCheckedNumber(
    Module& module, InferenceTable& table, const Span& span, int64_t value,
    const TypeAnnotation* type_annotation) {
  return MakeTypeCheckedNumber(module, table, span, InterpValue::MakeS64(value),
                               type_annotation);
}

bool IsColonRefWithTypeTarget(const InferenceTable& table, const Expr* expr) {
  if (expr->kind() != AstNodeKind::kColonRef) {
    return false;
  }
  std::optional<const AstNode*> colon_ref_target =
      table.GetColonRefTarget(absl::down_cast<const ColonRef*>(expr));
  return colon_ref_target.has_value() &&
         ((*colon_ref_target)->kind() == AstNodeKind::kTypeAlias ||
          (*colon_ref_target)->kind() == AstNodeKind::kEnumDef ||
          (*colon_ref_target)->kind() == AstNodeKind::kTypeAnnotation);
}

CloneReplacer NameRefMapper(
    InferenceTable& table,
    const absl::flat_hash_map<const NameDef*, ExprOrType>& map,
    std::optional<Module*> target_module,
    bool add_parametric_binding_type_annotation) {
  return [table = &table, map = &map, target_module,
          add_parametric_binding_type_annotation](
             const AstNode* node, Module* new_module,
             const absl::flat_hash_map<const AstNode*, AstNode*>&)
             -> absl::StatusOr<std::optional<AstNode*>> {
    if (node->kind() != AstNodeKind::kNameRef) {
      return std::nullopt;
    }
    const auto* ref = absl::down_cast<const NameRef*>(node);
    if (!std::holds_alternative<const NameDef*>(ref->name_def())) {
      return std::nullopt;
    }
    const NameDef* name_def = std::get<const NameDef*>(ref->name_def());
    const auto it = map->find(name_def);
    if (it == map->end()) {
      return std::nullopt;
    }
    Module* module_for_clone = target_module ? *target_module : new_module;
    XLS_ASSIGN_OR_RETURN(
        AstNode * clone,
        table->Clone(ToAstNode(it->second), &PreserveTypeDefinitionsReplacer,
                     module_for_clone));

    if (!add_parametric_binding_type_annotation ||
        name_def->parent()->kind() != AstNodeKind::kParametricBinding) {
      return clone;
    }

    // Note that within direct children of type annotations or indices, we
    // generally do not need the `add_parametric_binding_type_annotation`
    // behavior in order to infer the correct type for the literal. Since adding
    // it hurts error message readability, we filter out those cases here, even
    // though adding the annotation would technically be equally correct. For
    // example, we filter out `uN[5]` from the behavior but not `uN[5 + X]`.
    if (clone->kind() == AstNodeKind::kNumber &&
        (ref->parent() == nullptr ||
         (ref->parent()->kind() != AstNodeKind::kIndex &&
          ref->parent()->kind() != AstNodeKind::kTypeAnnotation))) {
      absl::down_cast<Number*>(clone)->SetTypeAnnotation(
          absl::down_cast<TypeAnnotation*>(
              down_cast<ParametricBinding*>(name_def->parent())
                  ->type_annotation()),
          /*update_span=*/false);
    }
    return clone;
  };
}

absl::StatusOr<bool> IsReferenceToAbstractType(const AstNode* node,
                                               const ImportData& import_data,
                                               const InferenceTable& table) {
  std::optional<StructOrProcRef> ref;
  if (node->kind() == AstNodeKind::kColonRef &&
      IsColonRefWithTypeTarget(table, absl::down_cast<const ColonRef*>(node))) {
    XLS_ASSIGN_OR_RETURN(
        ref, GetStructOrProcRef(absl::down_cast<const ColonRef*>(node),
                                import_data));
  } else if (node->kind() == AstNodeKind::kTypeAlias ||
             (node->kind() == AstNodeKind::kNameDef &&
              node->parent() != nullptr &&
              node->parent()->kind() == AstNodeKind::kTypeAlias)) {
    const TypeAlias* alias =
        node->kind() == AstNodeKind::kTypeAlias
            ? absl::down_cast<const TypeAlias*>(node)
            : absl::down_cast<const TypeAlias*>(node->parent());
    XLS_ASSIGN_OR_RETURN(
        ref, GetStructOrProcRef(&alias->type_annotation(), import_data));
  }
  return ref.has_value() && IsAbstractStructOrProcRef(*ref);
}

absl::StatusOr<ColonRef*> ConvertGenericColonRefToDirect(
    const InferenceTable& table, const ImportData& import_data,
    std::optional<const ParametricContext*> parametric_context,
    const ColonRef* colon_ref) {
  const auto* tvta =
      std::get<TypeVariableTypeAnnotation*>(colon_ref->subject());
  const auto* name_def =
      std::get<const NameDef*>(tvta->type_variable()->name_def());
  XLS_ASSIGN_OR_RETURN(TypeAnnotation * actual_type,
                       table.GetGenericType(parametric_context, name_def));
  XLS_ASSIGN_OR_RETURN(std::optional<const EnumDef*> enum_def,
                       GetEnumDef(actual_type, import_data));
  if (enum_def.has_value()) {
    return name_def->owner()->Make<ColonRef>(
        Span::None(),
        name_def->owner()->Make<NameRef>(
            Span::None(), (*enum_def)->name_def()->identifier(), name_def),
        colon_ref->attr());
  }

  XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                       GetStructOrProcRef(actual_type, import_data));
  if (struct_or_proc_ref.has_value()) {
    return name_def->owner()->Make<ColonRef>(
        Span::None(),
        name_def->owner()->Make<TypeRefTypeAnnotation>(
            Span::None(),
            name_def->owner()->Make<TypeRef>(
                Span::None(),
                const_cast<StructDef*>(absl::down_cast<const StructDef*>(
                    struct_or_proc_ref->def))),
            struct_or_proc_ref->parametrics),
        colon_ref->attr());
  }

  return TypeInferenceErrorStatus(
      colon_ref->span(), /*type=*/nullptr,
      absl::Substitute("Cannot resolve generic member reference "
                       "`$0` to a member of a real type.",
                       colon_ref->ToString()),
      *colon_ref->owner()->file_table());
}

bool VariableHasAnyExplicitTypeAnnotations(
    const InferenceTable& table,
    std::optional<const ParametricContext*> parametric_context,
    const NameRef* type_variable) {
  absl::StatusOr<std::vector<const TypeAnnotation*>> annotations =
      table.GetTypeAnnotationsForTypeVariable(parametric_context,
                                              type_variable);
  return annotations.ok() &&
         absl::c_any_of(
             *annotations, [&table](const TypeAnnotation* annotation) {
               TypeInferenceFlag flag = table.GetAnnotationFlag(annotation);
               return !flag.HasNonExplicitTypeSemantics();
             });
}

}  // namespace xls::dslx
