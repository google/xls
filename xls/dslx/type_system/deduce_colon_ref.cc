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

#include "xls/dslx/type_system/deduce_colon_ref.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/deduce_enum_def.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_expression.h"
#include "xls/dslx/type_system/scoped_fn_stack_entry.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"

namespace xls::dslx {

// Deduces a colon-ref in the particular case when the subject is known to be an
// import.
static absl::StatusOr<std::unique_ptr<Type>> DeduceColonRefToModule(
    const ColonRef* node, Module* module, DeduceCtx* ctx) {
  VLOG(5) << "DeduceColonRefToModule: " << node->ToString();

  XLS_VLOG_LINES(5, ctx->GetFnStackDebugString());

  std::optional<ModuleMember*> elem = module->FindMemberWithName(node->attr());
  if (!elem.has_value()) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Attempted to refer to module %s member '%s' "
                        "which does not exist.",
                        module->name(), node->attr()),
        ctx->file_table());
  }
  if (!IsPublic(*elem.value())) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Attempted to refer to module member %s that "
                        "is not public.",
                        ToAstNode(*elem.value())->ToString()),
        ctx->file_table());
  }

  XLS_ASSIGN_OR_RETURN(TypeInfo * imported_type_info,
                       ctx->import_data()->GetRootTypeInfo(module));
  if (std::holds_alternative<Function*>(*elem.value())) {
    auto* f_ptr = std::get<Function*>(*elem.value());
    XLS_RET_CHECK(f_ptr != nullptr);
    auto& f = *f_ptr;

    if (!imported_type_info->Contains(f.name_def())) {
      VLOG(2) << "Function name not in imported_type_info; indicates it is "
                 "parametric.";
      XLS_RET_CHECK(f.IsParametric());
      // We don't type check parametric functions until invocations.
      // Let's typecheck this imported parametric function with respect to its
      // module (this will only get the type signature, the body gets
      // typechecked after parametric instantiation).
      std::unique_ptr<DeduceCtx> imported_ctx =
          ctx->MakeCtx(imported_type_info, module);
      const FnStackEntry& peek_entry = ctx->fn_stack().back();
      imported_ctx->AddFnStackEntry(peek_entry);
      XLS_RETURN_IF_ERROR(ctx->typecheck_function()(f, imported_ctx.get()));
      imported_type_info = imported_ctx->type_info();
    }
  }

  AstNode* member_node = ToAstNode(*elem.value());
  std::optional<Type*> type = imported_type_info->GetItem(member_node);
  XLS_RET_CHECK(type.has_value()) << member_node->ToString();
  return type.value()->CloneToUnique();
}

static absl::StatusOr<std::unique_ptr<Type>> DeduceColonRefToBuiltinNameDef(
    BuiltinNameDef* builtin_name_def, const ColonRef* node) {
  VLOG(5) << "DeduceColonRefToBuiltinNameDef: " << node->ToString();

  const FileTable& file_table = *builtin_name_def->owner()->file_table();
  const auto& sized_type_keywords = GetSizedTypeKeywordsMetadata();
  if (auto it = sized_type_keywords.find(builtin_name_def->identifier());
      it != sized_type_keywords.end()) {
    auto [is_signed, size] = it->second;
    if (IsBuiltinBitsTypeAttr(node->attr())) {
      return std::make_unique<BitsType>(is_signed, size);
    }
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Builtin type '%s' does not have attribute '%s'.",
                        builtin_name_def->identifier(), node->attr()),
        file_table);
  }
  return TypeInferenceErrorStatus(
      node->span(), nullptr,
      absl::StrFormat("Builtin '%s' has no attributes.",
                      builtin_name_def->identifier()),
      file_table);
}

static absl::StatusOr<std::unique_ptr<Type>> DeduceColonRefToArrayType(
    ArrayTypeAnnotation* array_type, const ColonRef* node, DeduceCtx* ctx) {
  VLOG(5) << "DeduceColonRefToArrayType: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> resolved, ctx->Deduce(array_type));
  XLS_ASSIGN_OR_RETURN(resolved,
                       UnwrapMetaType(std::move(resolved), array_type->span(),
                                      "array type", ctx->file_table()));
  if (!IsBitsLike(*resolved)) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Cannot use '::' on type %s -- only bits types support "
                        "'::' attributes",
                        resolved->ToString()),
        ctx->file_table());
  }
  if (IsBuiltinBitsTypeAttr(node->attr())) {
    VLOG(5) << "DeduceColonRefToArrayType result: " << resolved->ToString();
    return resolved;
  }
  return TypeInferenceErrorStatus(
      node->span(), nullptr,
      absl::StrFormat("Type '%s' does not have attribute '%s'.",
                      array_type->ToString(), node->attr()),
      ctx->file_table());
}

static std::optional<Type*> TryNoteColonRefForConstant(
    const ConstantDef* constant, const ColonRef* colonref,
    const TypeInfo* impl_ti, TypeInfo* colonref_ti) {
  std::optional<Type*> type = impl_ti->GetItem(constant);
  if (!type.has_value()) {
    return std::nullopt;
  }
  absl::StatusOr<InterpValue> value = impl_ti->GetConstExpr(constant);
  if (!value.ok()) {
    return std::nullopt;
  }
  colonref_ti->NoteConstExpr(colonref, value.value());
  return type;
}

static absl::StatusOr<std::unique_ptr<Type>> DeduceColonRefToImpl(
    Impl* impl, const ColonRef* node, DeduceCtx* impl_ctx,
    TypeInfo* colonref_ti) {
  VLOG(5) << "DeduceColonRefToImpl: " << node->ToString();
  std::optional<ConstantDef*> constant = impl->GetConstant(node->attr());
  if (!constant.has_value()) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Name '%s' is not defined by the impl for struct '%s'.",
                        node->attr(), impl->struct_ref()->ToString()),
        impl_ctx->file_table());
  }
  // It's possible the constant has already been deduced.
  std::optional<Type*> type = TryNoteColonRefForConstant(
      constant.value(), node, impl_ctx->type_info(), colonref_ti);
  if (type.has_value()) {
    return type.value()->CloneToUnique();
  }

  // If not, deduce constants in impl and try again.
  for (const auto& con : impl->constants()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> _,
                         impl_ctx->Deduce(ToAstNode(con)));
  }

  type = TryNoteColonRefForConstant(constant.value(), node,
                                    impl_ctx->type_info(), colonref_ti);
  XLS_RET_CHECK(type.has_value());
  return type.value()->CloneToUnique();
}

static absl::StatusOr<std::unique_ptr<Type>> DeduceColonRefToStructType(
    StructDef* struct_def, const std::optional<Type*> type,
    const ColonRef* node, DeduceCtx* ctx) {
  VLOG(5) << "DeduceColonRefToStructType " << node->ToString();
  if (!struct_def->impl().has_value()) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Struct '%s' has no impl defining '%s'",
                        struct_def->identifier(), node->attr()),
        ctx->file_table());
  }
  Impl* impl = struct_def->impl().value();

  // Use the ctx for the impl when trying to read/deduce the value, but will
  // note the result to the colonref ctx for retrieval later.
  XLS_ASSIGN_OR_RETURN(TypeInfo * impl_ti,
                       ctx->import_data()->GetRootTypeInfo(impl->owner()));
  std::unique_ptr<DeduceCtx> impl_ctx = ctx->MakeCtx(impl_ti, impl->owner());
  ScopedFnStackEntry top =
      ScopedFnStackEntry::MakeForTop(impl_ctx.get(), impl->owner());
  if (!type.has_value()) {
    return DeduceColonRefToImpl(impl, node, impl_ctx.get(), ctx->type_info());
  }

  XLS_RET_CHECK(type.value()->IsMeta());
  const MetaType& meta_type = type.value()->AsMeta();
  const Type* wrapped = meta_type.wrapped().get();
  const StructType& struct_type = wrapped->AsStruct();

  const absl::flat_hash_map<std::string, TypeDim>& dims =
      struct_type.nominal_type_dims_by_identifier();

  // If there are no parametrics to handle, this is the basic impl case.
  if (dims.empty()) {
    return DeduceColonRefToImpl(impl, node, impl_ctx.get(), ctx->type_info());
  }

  // Process any resolved parametrics associated with the type.
  TypeInfo* derived_ti = impl_ctx->AddDerivedTypeInfo();
  for (const auto& binding : struct_def->parametric_bindings()) {
    auto it = dims.find(binding->identifier());
    if (it == dims.end()) {
      continue;
    }
    TypeDim instance_val = it->second;
    if (std::holds_alternative<InterpValue>(instance_val.value())) {
      impl_ctx->type_info()->NoteConstExpr(
          binding->name_def(), std::get<InterpValue>(instance_val.value()));
    } else {
      auto& owned_param =
          std::get<TypeDim::OwnedParametric>(instance_val.value());
      ParametricExpression::Evaluated evaluated = owned_param->Evaluate(
          ToParametricEnv(ctx->GetCurrentParametricEnv()));
      if (std::holds_alternative<InterpValue>(evaluated)) {
        impl_ctx->type_info()->NoteConstExpr(binding->name_def(),
                                             std::get<InterpValue>(evaluated));
      }
    }
  }

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Type> colon_ref_type,
      DeduceColonRefToImpl(impl, node, impl_ctx.get(), ctx->type_info()));
  XLS_RETURN_IF_ERROR(impl_ctx->PopDerivedTypeInfo(derived_ti));
  top.Finish();

  XLS_ASSIGN_OR_RETURN(InterpValue colon_ref_value,
                       ctx->type_info()->GetConstExpr(node));
  ctx->type_info()->NoteConstExpr(node, colon_ref_value);
  return colon_ref_type;
}

absl::StatusOr<std::unique_ptr<Type>> DeduceColonRef(const ColonRef* node,
                                                     DeduceCtx* ctx) {
  VLOG(5) << "DeduceColonRef: " << node->ToString() << " @ "
          << node->span().ToString(ctx->file_table());

  XLS_VLOG_LINES(5, ctx->GetFnStackDebugString());

  ImportData* import_data = ctx->import_data();
  XLS_ASSIGN_OR_RETURN(auto subject, ResolveColonRefSubjectForTypeChecking(
                                         import_data, ctx->type_info(), node));

  // We get the root type information for the referred-to entity's module (the
  // subject of the colon-ref) and create a fresh deduce context for its top
  // level.
  using ReturnT = absl::StatusOr<std::unique_ptr<Type>>;
  Module* subject_module = ToAstNode(subject)->owner();
  XLS_ASSIGN_OR_RETURN(TypeInfo * subject_type_info,
                       import_data->GetRootTypeInfo(subject_module));
  auto subject_ctx = ctx->MakeCtx(subject_type_info, subject_module);

  ScopedFnStackEntry top =
      ScopedFnStackEntry::MakeForTop(subject_ctx.get(), subject_module);

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Type> result,
      absl::visit(
          Visitor{
              [&](Module* module) -> ReturnT {
                return DeduceColonRefToModule(node, module, subject_ctx.get());
              },
              [&](EnumDef* enum_def) -> ReturnT {
                if (!enum_def->HasValue(node->attr())) {
                  return TypeInferenceErrorStatus(
                      node->span(), nullptr,
                      absl::StrFormat(
                          "Name '%s' is not defined by the enum %s.",
                          node->attr(), enum_def->identifier()),
                      ctx->file_table());
                }
                XLS_ASSIGN_OR_RETURN(
                    auto enum_type, DeduceEnumDef(enum_def, subject_ctx.get()));
                return UnwrapMetaType(std::move(enum_type), node->span(),
                                      "enum type", ctx->file_table());
              },
              [&](BuiltinNameDef* builtin_name_def) -> ReturnT {
                return DeduceColonRefToBuiltinNameDef(builtin_name_def, node);
              },
              [&](ArrayTypeAnnotation* type) -> ReturnT {
                return DeduceColonRefToArrayType(type, node, subject_ctx.get());
              },

              // Possible subjects for impl colon ref. In these cases, use the
              // colon-ref `DeduceCtx`. The `impl` ctx will be accessed within
              // `DeduceColonRefToStructType`.
              [&](StructDef* struct_def) -> ReturnT {
                return DeduceColonRefToStructType(struct_def, std::nullopt,
                                                  node, ctx);
              },
              [&](TypeRefTypeAnnotation* struct_ref) -> ReturnT {
                XLS_ASSIGN_OR_RETURN(
                    StructDef * struct_def,
                    DerefToStruct(node->span(), struct_ref->ToString(),
                                  *struct_ref, ctx->type_info()));
                return DeduceColonRefToStructType(
                    struct_def, ctx->type_info()->GetItem(struct_ref), node,
                    ctx);
              },
              [&](ColonRef* colon_ref) -> ReturnT {
                // Note: this should be unreachable, as it's a colon-reference
                // that refers *directly* to another colon-ref. Generally you
                // need an intervening construct, like a type alias.
                return absl::InternalError(
                    "Colon-reference subject was another colon-reference.");
              },
          },
          subject));
  top.Finish();

  VLOG(5) << "DeduceColonRef result: " << result->ToString();
  return result;
}

}  // namespace xls::dslx
