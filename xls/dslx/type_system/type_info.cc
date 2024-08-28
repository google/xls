// Copyright 2020 The XLS Authors
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

#include "xls/dslx/type_system/type_info.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

InvocationData::InvocationData(
    const Invocation* node, const Function* caller,
    absl::flat_hash_map<ParametricEnv, InvocationCalleeData> env_to_callee_data)
    : node_(node),
      caller_(caller),
      env_to_callee_data_(std::move(env_to_callee_data)) {
  // If we have a caller, check that the invocation node is contained within the
  // confines of the caller.
  //
  // Since we have the caller and the invocation here, it's an ideal place to do
  // this integrity check.
  //
  // Special case: functions present in parametric procs refer to the
  // proc-level parametric AST nodes, so they have invocation expressions that
  // can be outside of the function body. That is:
  //
  //  proc MyProc<N: u32 = {clog(MY_CONSTANT)}> {
  //    next(...) { ... }
  //  --^~~^ implicitly refers to the clog() call in the parametric above
  //  }
  auto is_fn_in_parametric_proc = [caller]() -> bool {
    return caller->IsInProc() && caller->proc().value()->IsParametric();
  };
  if (caller != nullptr && !is_fn_in_parametric_proc() &&
      !ContainedWithinFunction(*node, *caller)) {
    LOG(FATAL) << "Invocation node: `" << node->ToString() << "` @ "
               << node->span()
               << " is not contained within caller: " << caller->identifier()
               << " @ " << caller->span();
  }

  for (const auto& [env, _] : env_to_callee_data_) {
    CHECK_OK(ValidateEnvForCaller(env));
  }
}

std::string InvocationData::ToString() const {
  return absl::StrCat(
      "{",
      absl::StrJoin(env_to_callee_data_, ", ",
                    [](std::string* out, const auto& item) {
                      const InvocationCalleeData& callee_data = item.second;
                      absl::StrAppendFormat(
                          out, "%s: [%s, %p]", item.first.ToString(),
                          callee_data.callee_bindings.ToString(),
                          callee_data.derived_type_info);
                    }),
      "}");
}

absl::Status InvocationData::Add(ParametricEnv caller_env,
                                 InvocationCalleeData callee_data) {
  XLS_RETURN_IF_ERROR(ValidateEnvForCaller(caller_env));

  // TODO(leary): 2024-02-13 It'd be nice to be able to fail if the invocation
  // data is already present -- right now we do redundant work when parametric
  // functions are instantiated, if we eliminate that we should be able to make
  // a "this is populated exactly once" check.
  env_to_callee_data_.insert_or_assign(std::move(caller_env),
                                       std::move(callee_data));
  return absl::OkStatus();
}

absl::Status InvocationData::ValidateEnvForCaller(
    const ParametricEnv& env) const {
  for (const auto& k : env.GetKeySet()) {
    if (!caller_->parametric_keys().contains(k)) {
      return absl::InternalError(
          absl::StrFormat("caller `%s` given env with key `%s` not present in "
                          "parametric keys: {%s}",
                          caller_->identifier(), k,
                          absl::StrJoin(caller_->parametric_keys(), ",")));
    }
  }
  return absl::OkStatus();
}

// -- class TypeInfoOwner

absl::StatusOr<TypeInfo*> TypeInfoOwner::New(Module* module, TypeInfo* parent) {
  // Note: private constructor so not using make_unique.
  type_infos_.push_back(absl::WrapUnique(new TypeInfo(module, parent)));
  TypeInfo* result = type_infos_.back().get();
  if (parent == nullptr) {
    // Check we only have a single nullptr-parent TypeInfo for a given module.
    XLS_RET_CHECK(!module_to_root_.contains(module))
        << "module " << module->name() << " already has a root TypeInfo";
    VLOG(5) << "Making root type info for module: " << module->name() << " @ "
            << result;
    module_to_root_[module] = result;
  } else {
    // Check that we don't have parent links that traverse modules.
    XLS_RET_CHECK_EQ(parent->module(), module);
    VLOG(5) << "Making derived type info for module: " << module->name()
            << " parent: " << parent << " @ " << result;
  }
  return result;
}

absl::StatusOr<TypeInfo*> TypeInfoOwner::GetRootTypeInfo(const Module* module) {
  auto it = module_to_root_.find(module);
  if (it == module_to_root_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Could not find (root) type information for module: ", module->name()));
  }
  return it->second;
}

// -- class TypeInfo

void TypeInfo::NoteConstExpr(const AstNode* const_expr, InterpValue value) {
  VLOG(5) << absl::StreamFormat(
      "noting node: `%s` (%p) has constexpr value: `%s`",
      const_expr->ToString(), const_expr, value.ToString());
  const_exprs_.insert_or_assign(const_expr, std::move(value));
}

absl::StatusOr<InterpValue> TypeInfo::GetConstExpr(
    const AstNode* const_expr) const {
  CHECK_EQ(const_expr->owner(), module_)
      << const_expr->owner()->name() << " vs " << module_->name()
      << " node: " << const_expr->ToString();

  if (auto it = const_exprs_.find(const_expr); it != const_exprs_.end()) {
    return it->second.value();
  }

  if (parent_ != nullptr) {
    // In a case where a [child] type info is for a parametric function
    // specialization, it won't contain top-level constants, but its parent
    // [transitively] will.
    return parent_->GetConstExpr(const_expr);
  }

  return absl::NotFoundError(
      absl::StrFormat("No constexpr value found for node `%s` (%s) @ %s",
                      const_expr->ToString(), const_expr->GetNodeTypeName(),
                      SpanToString(const_expr->GetSpan())));
}

std::optional<InterpValue> TypeInfo::GetConstExprOption(
    const AstNode* const_expr) const {
  CHECK_EQ(const_expr->owner(), module_)
      << const_expr->owner()->name() << " vs " << module_->name()
      << " node: " << const_expr->ToString();

  if (auto it = const_exprs_.find(const_expr); it != const_exprs_.end()) {
    return it->second;
  }

  if (parent_ != nullptr) {
    // In a case where a [child] type info is for a parametric function
    // specialization, it won't contain top-level constants, but its parent
    // [transitively] will.
    return parent_->GetConstExprOption(const_expr);
  }

  return std::nullopt;
}

bool TypeInfo::IsKnownConstExpr(const AstNode* node) const {
  if (const_exprs_.contains(node)) {
    return const_exprs_.at(node).has_value();
  }

  if (parent_ != nullptr) {
    return parent_->IsKnownConstExpr(node);
  }

  return false;
}

bool TypeInfo::IsKnownNonConstExpr(const AstNode* node) const {
  if (const_exprs_.contains(node)) {
    return !const_exprs_.at(node).has_value();
  }

  if (parent_ != nullptr) {
    return parent_->IsKnownNonConstExpr(node);
  }

  return false;
}

void TypeInfo::NoteUnrolledLoop(const UnrollFor* loop, const ParametricEnv& env,
                                Expr* unrolled_expr) {
  VLOG(4) << "Converted unroll_for! at " << loop->span().ToString()
          << " with bindings: " << env.ToString()
          << " to: " << unrolled_expr->ToString();
  unrolled_loops_[loop][env] = unrolled_expr;
}

std::optional<Expr*> TypeInfo::GetUnrolledLoop(const UnrollFor* loop,
                                               const ParametricEnv& env) const {
  const auto exprs_it = unrolled_loops_.find(loop);
  if (exprs_it != unrolled_loops_.end()) {
    const auto it = exprs_it->second.find(env);
    if (it != exprs_it->second.end()) {
      return it->second;
    }
  }
  if (parent_ != nullptr) {
    return parent_->GetUnrolledLoop(loop, env);
  }
  VLOG(4) << "Loop at " << loop->span().ToString()
          << " has not been unrolled for " << env.ToString();
  return std::nullopt;
}

absl::StatusOr<TypeInfo::TypeSource> TypeInfo::ResolveTypeDefinition(
    TypeDefinition source) {
  return absl::visit(
      Visitor{
          [this](StructDef* sd) -> absl::StatusOr<TypeInfo::TypeSource> {
            return TypeInfo::TypeSource{.type_info = this, .definition = sd};
          },
          [this](TypeAlias* sd) -> absl::StatusOr<TypeInfo::TypeSource> {
            return TypeInfo::TypeSource{.type_info = this, .definition = sd};
          },
          [this](EnumDef* sd) -> absl::StatusOr<TypeInfo::TypeSource> {
            return TypeInfo::TypeSource{.type_info = this, .definition = sd};
          },
          [this](ColonRef* sd) -> absl::StatusOr<TypeInfo::TypeSource> {
            return ResolveTypeDefinition(sd);
          },
      },
      source);
}
absl::StatusOr<TypeInfo::TypeSource> TypeInfo::ResolveTypeDefinition(
    ColonRef* source) {
  // Resolve the colon-ref to the import it comes from.
  std::optional<Import*> import = source->ResolveImportSubject();
  XLS_RET_CHECK(import) << "Invalid colonref in type position";
  XLS_ASSIGN_OR_RETURN(const ImportedInfo* imported,
                       GetImportedOrError(*import));
  XLS_ASSIGN_OR_RETURN(TypeDefinition imported_def,
                       imported->module->GetTypeDefinition(source->attr()));
  return imported->type_info->ResolveTypeDefinition(imported_def);
}
absl::StatusOr<std::optional<std::string>> TypeInfo::FindSvType(
    TypeAnnotation* source) {
  auto* ref_type = dynamic_cast<TypeRefTypeAnnotation*>(source);
  // builtin/explicit tuples etc can't have an sv type.
  if (ref_type == nullptr) {
    return std::nullopt;
  }
  XLS_ASSIGN_OR_RETURN(
      TypeInfo::TypeSource src,
      ResolveTypeDefinition(ref_type->type_ref()->type_definition()));
  using Res = absl::StatusOr<std::optional<std::string>>;
  return absl::visit(
      Visitor{
          // Base cases
          [](StructDef* sd) -> Res { return sd->extern_type_name(); },
          [](EnumDef* sd) -> Res { return sd->extern_type_name(); },
          [src](TypeAlias* ta) -> Res {
            std::optional<std::string> sv_type = ta->extern_type_name();
            if (sv_type) {
              // Found one.
              return sv_type;
            }
            // Not annotated. Try the source type.
            return src.type_info->FindSvType(&ta->type_annotation());
          },
      },
      src.definition);
}

bool TypeInfo::Contains(AstNode* key) const {
  CHECK_EQ(key->owner(), module_);
  return dict_.contains(key) || (parent_ != nullptr && parent_->Contains(key));
}

std::string TypeInfo::GetImportsDebugString() const {
  return absl::StrFormat(
      "module %s imports:\n  %s", module()->name(),
      absl::StrJoin(imports_, "\n  ", [](std::string* out, const auto& item) {
        absl::StrAppend(out, item.second.module->name());
      }));
}

std::string TypeInfo::GetTypeInfoTreeString() const {
  const TypeInfo* top = GetRoot();
  CHECK(top != nullptr);
  std::vector<std::string> pieces = {absl::StrFormat("root %p:", top)};
  for (const auto& [invocation, invocation_data] : top->invocations_) {
    CHECK(invocation != nullptr);
    CHECK_EQ(invocation, invocation_data.node());

    pieces.push_back(
        absl::StrFormat("  `%s` @ %v", invocation_data.node()->ToString(),
                        SpanToString(invocation_data.node()->span())));
    for (const auto& [env, callee_data] :
         invocation_data.env_to_callee_data()) {
      pieces.push_back(absl::StrFormat(
          "    caller: %s => callee: %s type_info: %p", env.ToString(),
          callee_data.callee_bindings.ToString(),
          callee_data.derived_type_info));
    }
  }
  return absl::StrJoin(pieces, "\n");
}

std::optional<Type*> TypeInfo::GetItem(const AstNode* key) const {
  CHECK_EQ(key->owner(), module_)
      << key->owner()->name() << " vs " << module_->name()
      << " key: " << key->ToString();
  auto it = dict_.find(key);
  if (it != dict_.end()) {
    return it->second.get();
  }
  if (parent_ != nullptr) {
    return parent_->GetItem(key);
  }
  return std::nullopt;
}

absl::StatusOr<Type*> TypeInfo::GetItemOrError(const AstNode* key) const {
  auto maybe_type = GetItem(key);
  if (maybe_type.has_value()) {
    return *maybe_type;
  }

  return absl::NotFoundError(
      absl::StrCat("Could not find concrete type for node: ", key->ToString()));
}

absl::Status TypeInfo::AddInvocationTypeInfo(const Invocation& invocation,
                                             const Function* caller,
                                             const ParametricEnv& caller_env,
                                             const ParametricEnv& callee_env,
                                             TypeInfo* derived_type_info) {
  CHECK_EQ(invocation.owner(), module_);

  // We keep all instantiation info on the top-level type info. The "context
  // stack" doesn't matter so it creates a more understandable tree to flatten
  // it all against the top level.
  TypeInfo* top = GetRoot();

  VLOG(3) << "Type info " << top
          << " adding instantiation call bindings for invocation: `"
          << invocation.ToString() << "` @ " << invocation.span()
          << " caller_env: " << caller_env.ToString()
          << " callee_env: " << callee_env.ToString();
  auto it = top->invocations_.find(&invocation);
  if (it == top->invocations_.end()) {
    // No data for this invocation yet.
    absl::flat_hash_map<ParametricEnv, InvocationCalleeData> env_to_callee_data;
    env_to_callee_data[caller_env] =
        InvocationCalleeData{callee_env, derived_type_info};

    top->invocations_.emplace(
        &invocation,
        InvocationData{&invocation, caller, std::move(env_to_callee_data)});
    return absl::OkStatus();
  }
  VLOG(3) << "Adding to existing invocation data.";
  InvocationData& invocation_data = it->second;
  return invocation_data.Add(
      caller_env, InvocationCalleeData{callee_env, derived_type_info});
}

std::optional<bool> TypeInfo::GetRequiresImplicitToken(
    const Function& f) const {
  CHECK_EQ(f.owner(), module_) << "function owner: " << f.owner()->name()
                               << " module: " << module_->name();
  const TypeInfo* root = GetRoot();
  const absl::flat_hash_map<const Function*, bool>& map =
      root->requires_implicit_token_;
  auto it = map.find(&f);
  if (it == map.end()) {
    return std::nullopt;
  }
  bool result = it->second;
  VLOG(6) << absl::StreamFormat("GetRequiresImplicitToken %p %s::%s => %s",
                                root, f.owner()->name(), f.identifier(),
                                (result ? "true" : "false"));
  return result;
}

void TypeInfo::NoteRequiresImplicitToken(const Function& f, bool is_required) {
  TypeInfo* root = GetRoot();
  VLOG(6) << absl::StreamFormat("NoteRequiresImplicitToken %p: %s::%s => %s",
                                root, f.owner()->name(), f.identifier(),
                                is_required ? "true" : "false");
  root->requires_implicit_token_.emplace(&f, is_required);
}

std::optional<TypeInfo*> TypeInfo::GetInvocationTypeInfo(
    const Invocation* invocation, const ParametricEnv& caller) const {
  CHECK_EQ(invocation->owner(), module_)
      << invocation->owner()->name() << " vs " << module_->name();
  const TypeInfo* top = GetRoot();

  // Find the data for this invocation node.
  auto it = top->invocations_.find(invocation);
  if (it == top->invocations_.end()) {
    VLOG(5) << "Could not find instantiation for invocation: "
            << invocation->ToString();
    return std::nullopt;
  }

  // Find the callee data given the caller's parametric environment.
  const InvocationData& invocation_data = it->second;
  VLOG(5) << "Invocation " << invocation->ToString()
          << " caller bindings: " << caller
          << " invocation data: " << invocation_data.ToString();
  auto it2 = invocation_data.env_to_callee_data().find(caller);
  if (it2 == invocation_data.env_to_callee_data().end()) {
    return std::nullopt;
  }
  return it2->second.derived_type_info;
}

absl::StatusOr<TypeInfo*> TypeInfo::GetInvocationTypeInfoOrError(
    const Invocation* invocation, const ParametricEnv& caller) const {
  std::optional<TypeInfo*> maybe_ti = GetInvocationTypeInfo(invocation, caller);
  if (maybe_ti.has_value() && maybe_ti.value() != nullptr) {
    return maybe_ti.value();
  }

  return absl::NotFoundError(
      absl::StrFormat("TypeInfo could not find information for invocation `%s` "
                      "with caller environment: %s",
                      invocation->ToString(), caller.ToString()));
}

absl::Status TypeInfo::SetTopLevelProcTypeInfo(const Proc* p, TypeInfo* ti) {
  if (parent_ != nullptr) {
    return absl::InvalidArgumentError(
        "SetTopLevelTypeInfo may only be called on a module top-level type "
        "info.");
  }
  XLS_RET_CHECK_EQ(p->owner(), module_);
  top_level_proc_type_info_[p] = ti;
  return absl::OkStatus();
}

absl::StatusOr<TypeInfo*> TypeInfo::GetTopLevelProcTypeInfo(const Proc* p) {
  if (!top_level_proc_type_info_.contains(p)) {
    return absl::NotFoundError(absl::StrCat(
        "Top-level type info not found for proc \"", p->identifier(), "\"."));
  }
  return top_level_proc_type_info_.at(p);
}

std::optional<const ParametricEnv*> TypeInfo::GetInvocationCalleeBindings(
    const Invocation* invocation, const ParametricEnv& caller) const {
  CHECK_EQ(invocation->owner(), module_)
      << "attempting to get callee bindings for invocation `"
      << invocation->ToString()
      << "` which is owned by module: " << invocation->owner()->name()
      << " but this type information relates to module: " << module_->name();

  const TypeInfo* top = GetRoot();
  VLOG(3) << absl::StreamFormat(
      "TypeInfo %p getting instantiation symbolic bindings: %p %s @ %s %s", top,
      invocation, invocation->ToString(), invocation->span().ToString(),
      caller.ToString());
  auto it = top->invocations().find(invocation);
  if (it == top->invocations().end()) {
    VLOG(3) << "Could not find instantiation " << invocation
            << " in top-level type info: " << top;
    return std::nullopt;
  }
  const InvocationData& invocation_data = it->second;
  auto it2 = invocation_data.env_to_callee_data().find(caller);
  if (it2 == invocation_data.env_to_callee_data().end()) {
    VLOG(3) << "Could not find caller symbolic bindings in instantiation data: "
            << caller.ToString() << " " << invocation->ToString() << " @ "
            << invocation->span();
    return std::nullopt;
  }
  const ParametricEnv* result = &it2->second.callee_bindings;
  VLOG(3) << "Resolved instantiation symbolic bindings for "
          << invocation->ToString() << ": " << result->ToString();
  return result;
}

void TypeInfo::AddSliceStartAndWidth(Slice* node,
                                     const ParametricEnv& parametric_env,
                                     StartAndWidth start_width) {
  CHECK_EQ(node->owner(), module_);
  TypeInfo* top = GetRoot();
  auto it = top->slices_.find(node);
  if (it == top->slices_.end()) {
    top->slices_[node] = SliceData{node, {{parametric_env, start_width}}};
  } else {
    top->slices_[node].bindings_to_start_width.emplace(parametric_env,
                                                       start_width);
  }
}

std::optional<StartAndWidth> TypeInfo::GetSliceStartAndWidth(
    Slice* node, const ParametricEnv& parametric_env) const {
  CHECK_EQ(node->owner(), module_);
  const TypeInfo* top = GetRoot();
  auto it = top->slices_.find(node);
  if (it == top->slices_.end()) {
    return std::nullopt;
  }
  const SliceData& data = it->second;
  auto it2 = data.bindings_to_start_width.find(parametric_env);
  if (it2 == data.bindings_to_start_width.end()) {
    return std::nullopt;
  }
  return it2->second;
}

void TypeInfo::AddImport(Import* import, Module* module, TypeInfo* type_info) {
  CHECK_EQ(import->owner(), module_);
  GetRoot()->imports_[import] = ImportedInfo{module, type_info};
}

std::optional<const ImportedInfo*> TypeInfo::GetImported(Import* import) const {
  CHECK_EQ(import->owner(), module_) << absl::StreamFormat(
      "Import node is owned by: `%s` vs this TypeInfo is for `%s`",
      import->owner()->name(), module_->name());
  auto* self = GetRoot();
  auto it = self->imports_.find(import);
  if (it == self->imports_.end()) {
    return std::nullopt;
  }
  return &it->second;
}

absl::StatusOr<const ImportedInfo*> TypeInfo::GetImportedOrError(
    Import* import) const {
  auto maybe_imported = GetImported(import);
  if (maybe_imported.has_value()) {
    return maybe_imported.value();
  }

  return absl::NotFoundError(
      absl::StrCat("Could not find import for \"", import->ToString(), "\"."));
}

std::optional<TypeInfo*> TypeInfo::GetImportedTypeInfo(Module* m) {
  TypeInfo* root = GetRoot();
  if (root != this) {
    return root->GetImportedTypeInfo(m);
  }
  if (m == module()) {
    return this;
  }
  for (auto& [import, info] : imports_) {
    if (info.module == m) {
      return info.type_info;
    }
  }
  return std::nullopt;
}

TypeInfo::TypeInfo(Module* module, TypeInfo* parent)
    : module_(module), parent_(parent) {
  VLOG(6) << "Created type info for module \"" << module_->name() << "\" @ "
          << this << " parent " << parent << " root " << GetRoot();
}

TypeInfo::~TypeInfo() {
  // Only the root type information contains certain data.
  if (!IsRoot()) {
    CHECK(imports_.empty());
    CHECK(invocations_.empty());
    CHECK(slices_.empty());
    CHECK(imports_.empty());
    CHECK(requires_implicit_token_.empty());
    CHECK(top_level_proc_type_info_.empty());
  }
}

}  // namespace xls::dslx
