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

#include "xls/dslx/type_info.h"

#include "xls/common/status/ret_check.h"

namespace xls::dslx {

std::string InstantiationData::ToString() const {
  return absl::StrCat("[",
                      absl::StrJoin(symbolic_bindings_map, ", ",
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

absl::StatusOr<TypeInfo*> TypeInfoOwner::GetRootTypeInfo(Module* module) {
  auto it = module_to_root_.find(module);
  if (it == module_to_root_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Could not find (root) type information for module: ", module->name()));
  }
  return it->second;
}

// -- class TypeInfo

void TypeInfo::NoteConstExpr(Expr* const_expr, InterpValue value) {
  const_exprs_.insert({const_expr, value});
}

absl::optional<InterpValue> TypeInfo::GetConstExpr(Expr* const_expr) {
  if (auto it = const_exprs_.find(const_expr); it != const_exprs_.end()) {
    return it->second;
  }
  return absl::nullopt;
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

absl::optional<ConcreteType*> TypeInfo::GetItem(AstNode* key) const {
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
  return absl::nullopt;
}

void TypeInfo::AddInstantiationCallBindings(Invocation* invocation,
                                            SymbolicBindings caller,
                                            SymbolicBindings callee) {
  XLS_CHECK_EQ(invocation->owner(), module_);
  TypeInfo* top = GetRoot();
  XLS_VLOG(3) << "Type info " << top
              << " adding instantiation call bindings for invocation: "
              << invocation << " " << invocation->ToString() << " @ "
              << invocation->span() << " caller: " << caller.ToString()
              << " callee: " << callee.ToString();
  auto it = top->instantiations_.find(invocation);
  if (it == top->instantiations_.end()) {
    absl::flat_hash_map<SymbolicBindings, SymbolicBindings> symbind_map;
    symbind_map.emplace(std::move(caller), std::move(callee));
    top->instantiations_[invocation] =
        InstantiationData{invocation, std::move(symbind_map)};
    return;
  }
  XLS_VLOG(3) << "Adding to existing invocation data.";
  InstantiationData& data = it->second;
  data.symbolic_bindings_map.emplace(std::move(caller), std::move(callee));
}

bool TypeInfo::HasInstantiation(Invocation* invocation,
                                const SymbolicBindings& caller) const {
  XLS_CHECK_EQ(invocation->owner(), module_)
      << invocation->owner()->name() << " vs " << module_->name() << " @ "
      << invocation->span();
  return GetInstantiationTypeInfo(invocation, caller).has_value();
}

absl::optional<bool> TypeInfo::GetRequiresImplicitToken(Function* f) const {
  XLS_CHECK_EQ(f->owner(), module_) << "function owner: " << f->owner()->name()
                                    << " module: " << module_->name();
  const TypeInfo* root = GetRoot();
  const absl::flat_hash_map<Function*, bool>& map =
      root->requires_implicit_token_;
  auto it = map.find(f);
  if (it == map.end()) {
    return absl::nullopt;
  }
  bool result = it->second;
  XLS_VLOG(6) << absl::StreamFormat("GetRequiresImplicitToken %p %s::%s => %s",
                                    root, f->owner()->name(), f->identifier(),
                                    (result ? "true" : "false"));
  return result;
}

void TypeInfo::NoteRequiresImplicitToken(Function* f, bool is_required) {
  TypeInfo* root = GetRoot();
  XLS_VLOG(6) << absl::StreamFormat(
      "NoteRequiresImplicitToken %p: %s::%s => %s", root, f->owner()->name(),
      f->identifier(), is_required ? "true" : "false");
  root->requires_implicit_token_.emplace(f, is_required);
}

absl::optional<TypeInfo*> TypeInfo::GetInstantiationTypeInfo(
    Invocation* invocation, const SymbolicBindings& caller) const {
  XLS_CHECK_EQ(invocation->owner(), module_);
  const TypeInfo* top = GetRoot();
  auto it = top->instantiations_.find(invocation);
  if (it == top->instantiations_.end()) {
    XLS_VLOG(5) << "Could not find instantiation for invocation: "
                << invocation->ToString();
    return absl::nullopt;
  }
  const InstantiationData& data = it->second;
  XLS_VLOG(5) << "Invocation " << invocation->ToString()
              << " caller bindings: " << caller
              << " invocation data: " << data.ToString();
  auto it2 = data.instantiations.find(caller);
  if (it2 == data.instantiations.end()) {
    return absl::nullopt;
  }
  return it2->second;
}

void TypeInfo::SetInstantiationTypeInfo(Invocation* invocation,
                                        SymbolicBindings caller,
                                        TypeInfo* type_info) {
  XLS_CHECK_EQ(invocation->owner(), module_);
  TypeInfo* top = GetRoot();
  InstantiationData& data = top->instantiations_[invocation];
  data.instantiations[caller] = type_info;
}

absl::optional<const SymbolicBindings*>
TypeInfo::GetInstantiationCalleeBindings(Invocation* invocation,
                                         const SymbolicBindings& caller) const {
  XLS_CHECK_EQ(invocation->owner(), module_)
      << invocation->owner()->name() << " vs " << module_->name();
  const TypeInfo* top = GetRoot();
  XLS_VLOG(3) << absl::StreamFormat(
      "TypeInfo %p getting invocation symbolic bindings: %p %s @ %s %s", top,
      invocation, invocation->ToString(), invocation->span().ToString(),
      caller.ToString());
  auto it = top->instantiations_.find(invocation);
  if (it == top->instantiations_.end()) {
    XLS_VLOG(3) << "Could not find invocation " << invocation
                << " in top-level type info: " << top;
    return absl::nullopt;
  }
  const InstantiationData& data = it->second;
  auto it2 = data.symbolic_bindings_map.find(caller);
  if (it2 == data.symbolic_bindings_map.end()) {
    XLS_VLOG(3)
        << "Could not find caller symbolic bindings in invocation data: "
        << caller.ToString() << " " << invocation->ToString() << " @ "
        << invocation->span();
    return absl::nullopt;
  }
  const SymbolicBindings* result = &it2->second;
  XLS_VLOG(3) << "Resolved invocation symbolic bindings for "
              << invocation->ToString() << ": " << result->ToString();
  return result;
}

void TypeInfo::AddSliceStartAndWidth(Slice* node,
                                     const SymbolicBindings& symbolic_bindings,
                                     StartAndWidth start_width) {
  XLS_CHECK_EQ(node->owner(), module_);
  TypeInfo* top = GetRoot();
  auto it = top->slices_.find(node);
  if (it == top->slices_.end()) {
    top->slices_[node] =
        SliceData{node, {{symbolic_bindings, std::move(start_width)}}};
  } else {
    top->slices_[node].bindings_to_start_width.emplace(symbolic_bindings,
                                                       std::move(start_width));
  }
}

absl::optional<StartAndWidth> TypeInfo::GetSliceStartAndWidth(
    Slice* node, const SymbolicBindings& symbolic_bindings) const {
  XLS_CHECK_EQ(node->owner(), module_);
  const TypeInfo* top = GetRoot();
  auto it = top->slices_.find(node);
  if (it == top->slices_.end()) {
    return absl::nullopt;
  }
  const SliceData& data = it->second;
  auto it2 = data.bindings_to_start_width.find(symbolic_bindings);
  if (it2 == data.bindings_to_start_width.end()) {
    return absl::nullopt;
  }
  return it2->second;
}

void TypeInfo::AddImport(Import* import, Module* module, TypeInfo* type_info) {
  XLS_CHECK_EQ(import->owner(), module_);
  GetRoot()->imports_[import] = ImportedInfo{module, type_info};
}

absl::optional<const ImportedInfo*> TypeInfo::GetImported(
    Import* import) const {
  XLS_CHECK_EQ(import->owner(), module_)
      << "Import node from: " << import->owner()->name() << " vs TypeInfo for "
      << module_->name();
  auto* self = GetRoot();
  auto it = self->imports_.find(import);
  if (it == self->imports_.end()) {
    return absl::nullopt;
  }
  return &it->second;
}

absl::optional<TypeInfo*> TypeInfo::GetImportedTypeInfo(Module* m) {
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
  return absl::nullopt;
}

absl::optional<Expr*> TypeInfo::GetConstant(NameDef* name_def) const {
  XLS_CHECK_EQ(name_def->owner(), module_);
  auto it = name_to_const_.find(name_def);
  if (it == name_to_const_.end()) {
    if (parent_ != nullptr) {
      return parent_->GetConstant(name_def);
    }
    return absl::nullopt;
  }
  return it->second->value();
}

TypeInfo::TypeInfo(Module* module, TypeInfo* parent)
    : module_(module), parent_(parent) {
  XLS_VLOG(6) << "Created type info for module \"" << module_->name() << "\" @ "
              << this << " parent " << parent << " root " << GetRoot();
}

}  // namespace xls::dslx
