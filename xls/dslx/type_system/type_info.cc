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
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "xls/common/status/ret_check.h"

namespace xls::dslx {

std::string InvocationData::ToString() const {
  return absl::StrCat("[",
                      absl::StrJoin(parametric_env_map, ", ",
                                    [](std::string* out, const auto& item) {
                                      absl::StrAppendFormat(
                                          out, "%s: %s", item.first.ToString(),
                                          item.second.ToString());
                                    }),
                      "]");
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
    XLS_VLOG(5) << "Making root type info for module: " << module->name()
                << " @ " << result;
    module_to_root_[module] = result;
  } else {
    // Check that we don't have parent links that traverse modules.
    XLS_RET_CHECK_EQ(parent->module(), module);
    XLS_VLOG(5) << "Making derived type info for module: " << module->name()
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
  const_exprs_.insert({const_expr, value});
}

absl::StatusOr<InterpValue> TypeInfo::GetConstExpr(
    const AstNode* const_expr) const {
  XLS_CHECK_EQ(const_expr->owner(), module_)
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

  return absl::NotFoundError(absl::StrFormat(
      "No constexpr value found for node `%s` @ %s", const_expr->ToString(),
      SpanToString(const_expr->GetSpan())));
}

bool TypeInfo::IsKnownConstExpr(const AstNode* node) {
  if (const_exprs_.contains(node)) {
    return const_exprs_.at(node).has_value();
  }

  if (parent_ != nullptr) {
    return parent_->IsKnownConstExpr(node);
  }

  return false;
}

bool TypeInfo::IsKnownNonConstExpr(const AstNode* node) {
  if (const_exprs_.contains(node)) {
    return !const_exprs_.at(node).has_value();
  }

  if (parent_ != nullptr) {
    return parent_->IsKnownNonConstExpr(node);
  }

  return false;
}

bool TypeInfo::Contains(AstNode* key) const {
  XLS_CHECK_EQ(key->owner(), module_);
  return dict_.contains(key) || (parent_ != nullptr && parent_->Contains(key));
}

std::string TypeInfo::GetImportsDebugString() const {
  return absl::StrFormat(
      "module %s imports:\n  %s", module()->name(),
      absl::StrJoin(imports_, "\n  ", [](std::string* out, const auto& item) {
        absl::StrAppend(out, item.second.module->name());
      }));
}

std::optional<ConcreteType*> TypeInfo::GetItem(const AstNode* key) const {
  XLS_CHECK_EQ(key->owner(), module_)
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

absl::StatusOr<ConcreteType*> TypeInfo::GetItemOrError(
    const AstNode* key) const {
  auto maybe_type = GetItem(key);
  if (maybe_type.has_value()) {
    return *maybe_type;
  }

  return absl::NotFoundError(
      absl::StrCat("Could not find concrete type for node: ", key->ToString()));
}

void TypeInfo::AddInvocationCallBindings(const Invocation* call,
                                         ParametricEnv caller,
                                         ParametricEnv callee) {
  XLS_CHECK_EQ(call->owner(), module_);
  TypeInfo* top = GetRoot();
  XLS_VLOG(3) << "Type info " << top
              << " adding instantiation call bindings for invocation: "
              << call->ToString() << " @ " << call->span()
              << " caller: " << caller.ToString()
              << " callee: " << callee.ToString();
  auto it = top->invocations_.find(call);
  if (it == top->invocations_.end()) {
    absl::flat_hash_map<ParametricEnv, ParametricEnv> symbind_map;
    symbind_map.emplace(std::move(caller), std::move(callee));
    top->invocations_[call] =
        InvocationData{call, std::move(symbind_map)};
    return;
  }
  XLS_VLOG(3) << "Adding to existing invocation data.";
  InvocationData& data = it->second;
  data.parametric_env_map.emplace(std::move(caller), std::move(callee));
}

std::optional<bool> TypeInfo::GetRequiresImplicitToken(
    const Function* f) const {
  XLS_CHECK_EQ(f->owner(), module_) << "function owner: " << f->owner()->name()
                                    << " module: " << module_->name();
  const TypeInfo* root = GetRoot();
  const absl::flat_hash_map<const Function*, bool>& map =
      root->requires_implicit_token_;
  auto it = map.find(f);
  if (it == map.end()) {
    return std::nullopt;
  }
  bool result = it->second;
  XLS_VLOG(6) << absl::StreamFormat("GetRequiresImplicitToken %p %s::%s => %s",
                                    root, f->owner()->name(), f->identifier(),
                                    (result ? "true" : "false"));
  return result;
}

void TypeInfo::NoteRequiresImplicitToken(const Function* f, bool is_required) {
  TypeInfo* root = GetRoot();
  XLS_VLOG(6) << absl::StreamFormat(
      "NoteRequiresImplicitToken %p: %s::%s => %s", root, f->owner()->name(),
      f->identifier(), is_required ? "true" : "false");
  root->requires_implicit_token_.emplace(f, is_required);
}

std::optional<TypeInfo*> TypeInfo::GetInvocationTypeInfo(
    const Invocation* invocation, const ParametricEnv& caller) const {
  XLS_CHECK_EQ(invocation->owner(), module_)
      << invocation->owner()->name() << " vs " << module_->name();
  const TypeInfo* top = GetRoot();
  auto it = top->invocations_.find(invocation);
  if (it == top->invocations_.end()) {
    XLS_VLOG(5) << "Could not find instantiation for invocation: "
                << invocation->ToString();
    return std::nullopt;
  }
  const InvocationData& data = it->second;
  XLS_VLOG(5) << "Invocation " << invocation->ToString()
              << " caller bindings: " << caller
              << " invocation data: " << data.ToString();
  auto it2 = data.instantiations.find(caller);
  if (it2 == data.instantiations.end()) {
    return std::nullopt;
  }
  return it2->second;
}

absl::StatusOr<TypeInfo*> TypeInfo::GetInvocationTypeInfoOrError(
    const Invocation* invocation, const ParametricEnv& caller) const {
  auto maybe_ti = GetInvocationTypeInfo(invocation, caller);
  if (maybe_ti.has_value()) {
    return *maybe_ti;
  }

  return absl::NotFoundError(
      absl::StrCat("Could not find child type info with caller bindngs: ",
                   caller.ToString()));
}

void TypeInfo::SetInvocationTypeInfo(const Invocation* invocation,
                                     ParametricEnv caller,
                                     TypeInfo* type_info) {
  XLS_CHECK_EQ(invocation->owner(), module_);
  TypeInfo* top = GetRoot();
  InvocationData& data = top->invocations_[invocation];
  data.instantiations[caller] = type_info;
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
  XLS_CHECK_EQ(invocation->owner(), module_)
      << invocation->owner()->name() << " vs " << module_->name();
  const TypeInfo* top = GetRoot();
  XLS_VLOG(3) << absl::StreamFormat(
      "TypeInfo %p getting instantiation symbolic bindings: %p %s @ %s %s", top,
      invocation, invocation->ToString(),
      invocation->span().ToString(), caller.ToString());
  auto it = top->invocations().find(invocation);
  if (it == top->invocations().end()) {
    XLS_VLOG(3) << "Could not find instantiation " << invocation
                << " in top-level type info: " << top;
    return std::nullopt;
  }
  const InvocationData& data = it->second;
  auto it2 = data.parametric_env_map.find(caller);
  if (it2 == data.parametric_env_map.end()) {
    XLS_VLOG(3)
        << "Could not find caller symbolic bindings in instantiation data: "
        << caller.ToString() << " " << invocation->ToString() << " @ "
        << invocation->span();
    return std::nullopt;
  }
  const ParametricEnv* result = &it2->second;
  XLS_VLOG(3) << "Resolved instantiation symbolic bindings for "
              << invocation->ToString() << ": " << result->ToString();
  return result;
}

void TypeInfo::AddSliceStartAndWidth(Slice* node,
                                     const ParametricEnv& parametric_env,
                                     StartAndWidth start_width) {
  XLS_CHECK_EQ(node->owner(), module_);
  TypeInfo* top = GetRoot();
  auto it = top->slices_.find(node);
  if (it == top->slices_.end()) {
    top->slices_[node] =
        SliceData{node, {{parametric_env, std::move(start_width)}}};
  } else {
    top->slices_[node].bindings_to_start_width.emplace(parametric_env,
                                                       start_width);
  }
}

std::optional<StartAndWidth> TypeInfo::GetSliceStartAndWidth(
    Slice* node, const ParametricEnv& parametric_env) const {
  XLS_CHECK_EQ(node->owner(), module_);
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
  XLS_CHECK_EQ(import->owner(), module_);
  GetRoot()->imports_[import] = ImportedInfo{module, type_info};
}

std::optional<const ImportedInfo*> TypeInfo::GetImported(
    Import* import) const {
  XLS_CHECK_EQ(import->owner(), module_)
      << "Import node from: " << import->owner()->name() << " vs TypeInfo for "
      << module_->name();
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
  XLS_VLOG(6) << "Created type info for module \"" << module_->name() << "\" @ "
              << this << " parent " << parent << " root " << GetRoot();
}

}  // namespace xls::dslx
