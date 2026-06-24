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

#include "xls/dslx/type_system_v2/inference_table_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
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
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {
namespace {

bool IsAbstractStructOrProcRef(const StructOrProcRef& ref) {
  return GetRequiredParametricBindings(ref.def->parametric_bindings()).size() >
         ref.parametrics.size();
}

bool IsAbstractSumRef(const SumRef& ref) {
  return GetRequiredParametricBindings(ref.def->parametric_bindings()).size() >
         ref.parametrics.size();
}

absl::StatusOr<const TypeAnnotation*> GetTypeArgumentAnnotation(
    ExprOrType argument, const InferenceTable& table,
    const FileTable& file_table) {
  if (std::holds_alternative<TypeAnnotation*>(argument)) {
    return std::get<TypeAnnotation*>(argument);
  } else {
    const Expr* expr = std::get<Expr*>(argument);
    if (IsColonRefWithTypeTarget(table, expr)) {
      std::optional<const TypeAnnotation*> annotation =
          table.GetTypeAnnotation(expr);
      if (annotation.has_value()) {
        return *annotation;
      }
    }
    return TypeInferenceErrorStatus(
        expr->span(), nullptr,
        absl::Substitute("Expected parametric type, saw `$0`",
                         expr->ToString()),
        file_table);
  }
}

}  // namespace

absl::StatusOr<ExprOrType> NormalizeParametricArgument(
    const ParametricBinding& binding, ExprOrType argument,
    const InferenceTable& table, const FileTable& file_table) {
  if (binding.type_annotation()->IsAnnotation<GenericTypeAnnotation>()) {
    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* type,
        GetTypeArgumentAnnotation(argument, table, file_table));
    return ExprOrType(const_cast<TypeAnnotation*>(type));
  } else if (std::holds_alternative<TypeAnnotation*>(argument) ||
             IsColonRefWithTypeTarget(table, std::get<Expr*>(argument))) {
    const AstNode* node = ToAstNode(argument);
    return TypeInferenceErrorStatus(
        *node->GetSpan(), nullptr,
        absl::Substitute("Expected parametric value, saw `$0`",
                         node->ToString()),
        file_table);
  } else {
    return argument;
  }
}

absl::StatusOr<const TypeAnnotation*> SubstituteTypeParametrics(
    const TypeAnnotation* type,
    const absl::flat_hash_map<const NameDef*, ExprOrType>& actual_values,
    InferenceTable& table, std::optional<const TypeAnnotation*> real_self_type,
    bool clone_if_no_parametrics) {
  // Whole-type replacements do not necessarily have traversable NameRefs.
  // Use the same lookup when checking for substitutions and applying them.
  auto get_type_replacement =
      [&](const AstNode* node) -> absl::StatusOr<const AstNode*> {
    const AstNode* replacement = nullptr;
    if (node->kind() == AstNodeKind::kTypeAnnotation) {
      const auto* annotation = absl::down_cast<const TypeAnnotation*>(node);
      if (real_self_type.has_value() &&
          annotation->IsAnnotation<SelfTypeAnnotation>()) {
        replacement = *real_self_type;
      } else {
        const NameDef* name_def = nullptr;
        if (annotation->IsAnnotation<TypeVariableTypeAnnotation>()) {
          name_def = std::get<const NameDef*>(
              annotation->AsAnnotation<TypeVariableTypeAnnotation>()
                  ->type_variable()
                  ->name_def());
        } else if (annotation->IsAnnotation<TypeRefTypeAnnotation>()) {
          // Type aliases in impls use TypeRefTypeAnnotations, not TVTAs.
          TypeDefinition definition =
              annotation->AsAnnotation<TypeRefTypeAnnotation>()
                  ->type_ref()
                  ->type_definition();
          if (std::holds_alternative<TypeAlias*>(definition)) {
            name_def = &std::get<TypeAlias*>(definition)->name_def();
          }
        }
        if (name_def != nullptr) {
          const auto it = actual_values.find(name_def);
          if (it != actual_values.end()) {
            XLS_ASSIGN_OR_RETURN(
                replacement,
                GetTypeArgumentAnnotation(it->second, table,
                                          *type->owner()->file_table()));
          }
        }
      }
    }
    return replacement;
  };

  if (!clone_if_no_parametrics) {
    XLS_ASSIGN_OR_RETURN(std::vector<const AstNode*> nodes,
                         CollectUnder(type, /*want_types=*/true));
    bool needs_substitution = false;
    for (const AstNode* node : nodes) {
      XLS_ASSIGN_OR_RETURN(const AstNode* replacement,
                           get_type_replacement(node));
      if (replacement != nullptr) {
        needs_substitution = true;
      } else if (node->kind() == AstNodeKind::kNameRef) {
        const auto* ref = absl::down_cast<const NameRef*>(node);
        needs_substitution =
            std::holds_alternative<const NameDef*>(ref->name_def()) &&
            actual_values.contains(std::get<const NameDef*>(ref->name_def()));
      }
      if (needs_substitution) {
        break;
      }
    }
    if (!needs_substitution) {
      return type;
    }
  }

  CloneReplacer replacer = ChainCloneReplacers(
      &PreserveTypeDefinitionsReplacer,
      ChainCloneReplacers(
          NameRefMapper(table, actual_values, type->owner(),
                        /*add_parametric_binding_type_annotation=*/true),
          [&](const AstNode* node, Module*,
              const absl::flat_hash_map<const AstNode*, AstNode*>&)
              -> absl::StatusOr<std::optional<AstNode*>> {
            // Leave attrs in place; they never need parametric replacement
            // here.
            if (node->kind() == AstNodeKind::kAttr) {
              return const_cast<AstNode*>(node);
            }
            return std::nullopt;
          }));

  replacer = ChainCloneReplacers(
      std::move(replacer),
      [&](const AstNode* node, Module* module,
          const absl::flat_hash_map<const AstNode*, AstNode*>&)
          -> absl::StatusOr<std::optional<AstNode*>> {
        if (const auto* colon_ref = dynamic_cast<const ColonRef*>(node)) {
          XLS_ASSIGN_OR_RETURN(
              const AstNode* subject_replacement,
              get_type_replacement(ToAstNode(colon_ref->subject())));
          if (subject_replacement != nullptr) {
            const auto* subject =
                absl::down_cast<const TypeAnnotation*>(subject_replacement);
            const auto* type_ref_annotation =
                dynamic_cast<const TypeRefTypeAnnotation*>(subject);
            const TypeAlias* subject_alias = nullptr;
            if (type_ref_annotation != nullptr &&
                std::holds_alternative<TypeAlias*>(
                    type_ref_annotation->type_ref()->type_definition())) {
              subject_alias = std::get<TypeAlias*>(
                  type_ref_annotation->type_ref()->type_definition());
            }
            if (subject->IsAnnotation<BuiltinTypeAnnotation>()) {
              // A primitive annotation cannot itself be a ColonRef subject.
              // Keep the ordinary builtin reference and defer member semantics
              // to the same population and evaluation path as u32::ZERO.
              const auto* builtin =
                  subject->AsAnnotation<BuiltinTypeAnnotation>();
              const auto& name = builtin->builtin_name_def()->identifier();
              auto* name_ref = module->Make<NameRef>(
                  colon_ref->span(), name,
                  module->GetOrCreateBuiltinNameDef(name));
              return module->Make<ColonRef>(colon_ref->span(), name_ref,
                                            colon_ref->attr(),
                                            colon_ref->in_parens());
            } else if (subject_alias != nullptr &&
                       type_ref_annotation->parametrics().empty()) {
              // Use the ordinary alias member form so later alias resolution
              // cannot replace a ColonRef subject with a primitive annotation.
              auto* name_ref = module->Make<NameRef>(
                  colon_ref->span(), subject_alias->name_def().identifier(),
                  &subject_alias->name_def());
              return module->Make<ColonRef>(colon_ref->span(), name_ref,
                                            colon_ref->attr(),
                                            colon_ref->in_parens());
            } else if ((subject->IsAnnotation<ArrayTypeAnnotation>() &&
                        GetSignednessAndBitCount(subject).ok()) ||
                       subject_alias != nullptr) {
              // Keep dimension expressions and alias arguments reachable for
              // subsequent substitution, population, and evaluation. A
              // detached alias would hide them behind the NameRef's borrowed
              // definition.
              XLS_ASSIGN_OR_RETURN(
                  subject, SubstituteTypeParametrics(
                               subject, actual_values, table, real_self_type,
                               /*clone_if_no_parametrics=*/false));
              XLS_ASSIGN_OR_RETURN(
                  (absl::flat_hash_map<const AstNode*, AstNode*>
                       subject_clones),
                  CloneAstAndGetAllPairs(subject, module,
                                         &PreserveTypeDefinitionsReplacer));
              auto* name_def = module->Make<NameDef>(
                  colon_ref->span(), "Subject", /*definer=*/nullptr);
              auto* alias = module->Make<TypeAlias>(
                  colon_ref->span(), *name_def,
                  *absl::down_cast<TypeAnnotation*>(subject_clones.at(subject)),
                  /*is_public=*/false);
              name_def->set_definer(alias);
              auto* member = module->Make<ColonRef>(
                  colon_ref->span(),
                  module->Make<NameRef>(colon_ref->span(),
                                        name_def->identifier(), name_def),
                  colon_ref->attr());
              auto* block = module->Make<StatementBlock>(
                  colon_ref->span(),
                  std::vector<Statement*>{module->Make<Statement>(alias),
                                          module->Make<Statement>(member)},
                  /*trailing_semi=*/false);
              block->set_in_parens(colon_ref->in_parens());
              return block;
            } else if (!subject->IsAnnotation<TypeRefTypeAnnotation>() &&
                       !subject->IsAnnotation<TypeVariableTypeAnnotation>() &&
                       !subject->IsAnnotation<SelfTypeAnnotation>()) {
              return TypeInferenceErrorStatusForAnnotation(
                  colon_ref->span(), subject,
                  absl::Substitute("Type `$0` has no member `$1`.",
                                   subject->ToString(), colon_ref->attr()),
                  *module->file_table());
            }
          }
        }
        XLS_ASSIGN_OR_RETURN(const AstNode* replacement,
                             get_type_replacement(node));
        if (replacement == nullptr) {
          return std::nullopt;
        } else if (absl::down_cast<const TypeAnnotation*>(node)
                       ->IsAnnotation<SelfTypeAnnotation>()) {
          return table.Clone(replacement, &NoopCloneReplacer, type->owner());
        } else {
          return const_cast<AstNode*>(replacement);
        }
      });

  XLS_ASSIGN_OR_RETURN(
      (absl::flat_hash_map<const AstNode*, AstNode*> clones),
      CloneAstAndGetAllPairs(type, type->owner(), std::move(replacer)));
  AstNode* result = clones.at(type);
  return absl::down_cast<const TypeAnnotation*>(result);
}

absl::StatusOr<Number*> MakeTypeCheckedNumber(
    Module& module, InferenceTable& table, const Span& span,
    const InterpValue& value, const TypeAnnotation* type_annotation) {
  // Invariant: nodes created into `module` should either have a "no-file" span
  // (for internally-fabricated nodes), a recognized internal synthetic file,
  // or a span that points at `module`'s own source file. Violating this makes
  // downstream consumers that resolve nodes by (kind, span) fragile and can
  // lead to confusing "could not find node" errors.
  //
  // Note: not all modules have a filesystem path (e.g. in-memory modules); we
  // only enforce this when `fs_path()` is known.
  XLS_RET_CHECK(module.file_table() != nullptr);
  if (span.HasFile() && module.fs_path().has_value()) {
    std::string_view span_filename = span.GetFilename(*module.file_table());
    const bool is_specialization_span =
        absl::StartsWith(span_filename, "<specialization:");
    XLS_RET_CHECK(is_specialization_span ||
                  span_filename == module.fs_path()->generic_string())
        << "MakeTypeCheckedNumber span filename must match module fs_path or "
           "use a recognized specialization pseudo-file; "
        << "module name: `" << module.name() << "`; "
        << "span: `" << span.ToString(*module.file_table()) << "`; "
        << "module fs_path: `" << module.fs_path()->generic_string() << "`";
  }

  VLOG(5) << "Creating type-checked number: " << value.ToString()
          << " of type: " << type_annotation->ToString();
  Number* number = module.Make<Number>(
      span, value.ToString(/*humanize=*/true), NumberKind::kOther,
      const_cast<TypeAnnotation*>(type_annotation));
  XLS_RETURN_IF_ERROR(table.SetTypeAnnotation(number, type_annotation));
  return number;
}

absl::StatusOr<Number*> MakeTypeCheckedNumber(
    Module& module, InferenceTable& table, const Span& span, int64_t value,
    const TypeAnnotation* type_annotation) {
  return MakeTypeCheckedNumber(module, table, span, InterpValue::MakeS64(value),
                               type_annotation);
}

absl::StatusOr<Expr*> MakeTypeCheckedNumberOrEnumValue(
    Module& module, InferenceTable& table, const Span& span,
    const InterpValue& value, const TypeAnnotation* type_annotation,
    const Type& type) {
  if (!value.IsEnum() || !type.IsEnum()) {
    return MakeTypeCheckedNumber(module, table, span, value, type_annotation);
  }

  const auto& enum_type = type.AsEnum();
  const EnumDef& enum_def = enum_type.nominal_type();
  std::optional<std::string> member_name;
  for (int i = 0; i < enum_def.values().size(); ++i) {
    if (enum_type.members().at(i) == value) {
      member_name = enum_def.GetMemberName(i);
      break;
    }
  }
  XLS_RET_CHECK(member_name.has_value())
      << "Could not find enum member matching value " << value.ToString()
      << " in enum " << enum_def.identifier();

  std::optional<ColonRef::Subject> subject;
  const NameDef* target_name_def = nullptr;

  if (type_annotation->IsAnnotation<TypeRefTypeAnnotation>()) {
    const auto* trta = type_annotation->AsAnnotation<TypeRefTypeAnnotation>();
    const TypeDefinition& type_def = trta->type_ref()->type_definition();
    if (std::holds_alternative<ColonRef*>(type_def)) {
      subject = std::get<ColonRef*>(type_def);
    } else if (std::holds_alternative<EnumDef*>(type_def)) {
      EnumDef* def = std::get<EnumDef*>(type_def);
      subject = module.Make<NameRef>(span, def->identifier(), def->name_def());
      target_name_def = def->name_def();
    } else if (std::holds_alternative<UseTreeEntry*>(type_def)) {
      return absl::UnimplementedError("`use` syntax is not yet supported.");
    } else if (std::holds_alternative<TypeAlias*>(type_def)) {
      TypeAlias* alias = std::get<TypeAlias*>(type_def);
      subject =
          module.Make<NameRef>(span, alias->identifier(), &alias->name_def());
      target_name_def = &alias->name_def();
    }
  }

  XLS_RET_CHECK(subject.has_value());
  ColonRef* colon_ref = module.Make<ColonRef>(span, *subject, *member_name);
  if (target_name_def != nullptr) {
    table.SetColonRefTarget(colon_ref, target_name_def);
  }
  XLS_RETURN_IF_ERROR(table.SetTypeAnnotation(colon_ref, type_annotation));
  return colon_ref;
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
          (*colon_ref_target)->kind() == AstNodeKind::kSumDef ||
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

    if (clone->kind() == AstNodeKind::kNumber) {
      auto* number = absl::down_cast<Number*>(clone);
      auto* binding =
          name_def->parent()->kind() == AstNodeKind::kParametricBinding
              ? absl::down_cast<ParametricBinding*>(name_def->parent())
              : nullptr;
      const bool add_binding_type =
          add_parametric_binding_type_annotation && binding != nullptr &&
          (ref->parent() == nullptr ||
           (ref->parent()->kind() != AstNodeKind::kIndex &&
            ref->parent()->kind() != AstNodeKind::kTypeAnnotation));
      auto* binding_type =
          add_binding_type ? binding->type_annotation() : nullptr;
      if (binding_type != nullptr &&
          binding_type->IsAnnotation<BuiltinTypeAnnotation>() &&
          binding_type->AsAnnotation<BuiltinTypeAnnotation>()->GetBitCount() >
              0) {
        // A concrete declared width takes precedence over a cloned default's
        // incidental minimal width. Dependent types retain the actual's type.
        number->SetTypeAnnotation(binding_type, /*update_span=*/false);
      } else if (number->type_annotation() == nullptr) {
        std::optional<const TypeAnnotation*> known_type =
            table->GetTypeAnnotation(number);
        if (known_type.has_value()) {
          // MakeTypeCheckedNumber may record its concrete type only in the
          // table. Preserve it in the AST through later cloning/population,
          // rather than reattaching an unresolved formal binding type.
          number->SetTypeAnnotation(const_cast<TypeAnnotation*>(*known_type),
                                    /*update_span=*/false);
        } else if (binding_type != nullptr) {
          // Untyped values still need the binding's width in expressions like
          // `uN[5 + X]`. Direct dimensions such as `uN[5]` do not need the
          // prefix.
          number->SetTypeAnnotation(binding_type, /*update_span=*/false);
        }
      }
    }
    return clone;
  };
}

absl::StatusOr<bool> IsReferenceToAbstractType(const AstNode* node,
                                               const ImportData& import_data,
                                               const InferenceTable& table) {
  std::optional<StructOrProcRef> struct_or_proc_ref;
  std::optional<SumRef> sum_ref;
  if (node->kind() == AstNodeKind::kColonRef &&
      IsColonRefWithTypeTarget(table, absl::down_cast<const ColonRef*>(node))) {
    XLS_ASSIGN_OR_RETURN(
        struct_or_proc_ref,
        GetStructOrProcRef(absl::down_cast<const ColonRef*>(node),
                           import_data));
    XLS_ASSIGN_OR_RETURN(
        sum_ref,
        GetSumRef(absl::down_cast<const ColonRef*>(node), import_data));
  } else if (node->kind() == AstNodeKind::kTypeAlias ||
             (node->kind() == AstNodeKind::kNameDef &&
              node->parent() != nullptr &&
              node->parent()->kind() == AstNodeKind::kTypeAlias)) {
    const TypeAlias* alias =
        node->kind() == AstNodeKind::kTypeAlias
            ? absl::down_cast<const TypeAlias*>(node)
            : absl::down_cast<const TypeAlias*>(node->parent());
    XLS_ASSIGN_OR_RETURN(
        struct_or_proc_ref,
        GetStructOrProcRef(&alias->type_annotation(), import_data));
    XLS_ASSIGN_OR_RETURN(sum_ref,
                         GetSumRef(&alias->type_annotation(), import_data));
  }
  return (struct_or_proc_ref.has_value() &&
          IsAbstractStructOrProcRef(*struct_or_proc_ref)) ||
         (sum_ref.has_value() && IsAbstractSumRef(*sum_ref));
}

absl::StatusOr<std::optional<ColonRef*>> ConvertGenericColonRefToDirect(
    const InferenceTable& table, const ImportData& import_data,
    std::optional<const ParametricContext*> parametric_context,
    const ColonRef* colon_ref) {
  XLS_ASSIGN_OR_RETURN(
      std::optional<const TypeVariableTypeAnnotation*> tvta,
      GetTypeVariableTypeAnnotationForSubject(colon_ref, import_data));
  if (!tvta.has_value()) {
    return std::nullopt;
  }
  const auto* name_def =
      std::get<const NameDef*>((*tvta)->type_variable()->name_def());
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

  XLS_ASSIGN_OR_RETURN(std::optional<SumRef> sum_ref,
                       GetSumRef(actual_type, import_data));
  if (sum_ref.has_value()) {
    return name_def->owner()->Make<ColonRef>(
        Span::None(),
        name_def->owner()->Make<TypeRefTypeAnnotation>(
            Span::None(),
            name_def->owner()->Make<TypeRef>(Span::None(),
                                             const_cast<SumDef*>(sum_ref->def)),
            sum_ref->parametrics),
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

absl::StatusOr<std::optional<const TypeAnnotation*>> GetExplicitTypeAnnotation(
    const InferenceTable& table,
    std::optional<const ParametricContext*> parametric_context,
    const AstNode* node) {
  std::optional<const NameRef*> type_variable = table.GetTypeVariable(node);
  if (!type_variable.has_value()) {
    return std::nullopt;
  }
  XLS_ASSIGN_OR_RETURN(std::vector<const TypeAnnotation*> annotations,
                       table.GetTypeAnnotationsForTypeVariable(
                           parametric_context, *type_variable));
  for (const TypeAnnotation* annotation : annotations) {
    if (!table.GetAnnotationFlag(annotation).HasNonExplicitTypeSemantics()) {
      return annotation;
    }
  }
  return std::nullopt;
}

}  // namespace xls::dslx
