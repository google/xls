// Copyright 2020 Google LLC
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

namespace xls::dslx {

bool SymbolicBindings::operator==(const SymbolicBindings& other) const {
  if (bindings_.size() != other.bindings_.size()) {
    return false;
  }
  for (int64 i = 0; i < bindings_.size(); ++i) {
    if (bindings_[i] != other.bindings_[i]) {
      return false;
    }
  }
  return true;
}
bool SymbolicBindings::operator!=(const SymbolicBindings& other) const {
  return !(*this == other);
}

std::string SymbolicBindings::ToString() const {
  return absl::StrFormat(
      "(%s)", absl::StrJoin(bindings_, ", ",
                            [](std::string* out, const SymbolicBinding& sb) {
                              absl::StrAppendFormat(out, "%s: %d",
                                                    sb.identifier, sb.value);
                            }));
}

void TypeInfo::Update(const TypeInfo& other) {
  for (const auto& [node, type] : other.dict_) {
    dict_[node] = type->CloneToUnique();
  }
  for (const auto& item : other.imports_) {
    imports_.insert(item);
  }
  // Merge in all the invocation information.
  for (const auto& [node, other_data] : other.invocations_) {
    auto it = invocations_.find(node);
    if (it == invocations_.end()) {
      invocations_[node] = other_data;
    } else {
      InvocationData& data = it->second;
      data.Update(other_data);
    }
  }
  // Merge in all the slice information.
  for (const auto& [node, other_data] : other.slices_) {
    auto it = slices_.find(node);
    if (it == slices_.end()) {
      slices_[node] = other_data;
    } else {
      SliceData& data = it->second;
      data.Update(other_data);
    }
  }
}

bool TypeInfo::Contains(AstNode* key) const {
  return dict_.contains(key) || (parent_ != nullptr && parent_->Contains(key));
}

absl::optional<ConcreteType*> TypeInfo::GetItem(AstNode* key) const {
  auto it = dict_.find(key);
  if (it != dict_.end()) {
    return it->second.get();
  }
  if (parent_ != nullptr) {
    return parent_->GetItem(key);
  }
  return absl::nullopt;
}

void TypeInfo::AddInvocationSymbolicBindings(Invocation* invocation,
                                             SymbolicBindings caller,
                                             SymbolicBindings callee) {
  TypeInfo* top = GetTop();
  XLS_VLOG(3) << "Type info " << top
              << " adding symbolic bindings for invocation: " << invocation
              << " " << invocation->ToString() << " @ " << invocation->span()
              << " caller: " << caller.ToString()
              << " callee: " << callee.ToString();
  auto it = top->invocations_.find(invocation);
  if (it == top->invocations_.end()) {
    absl::flat_hash_map<SymbolicBindings, SymbolicBindings> symbind_map;
    symbind_map.emplace(std::move(caller), std::move(callee));
    top->invocations_[invocation] =
        InvocationData{invocation, std::move(symbind_map)};
    return;
  }
  XLS_VLOG(3) << "Adding to existing invocation data.";
  InvocationData& data = it->second;
  data.symbolic_bindings_map.emplace(std::move(caller), std::move(callee));
}

bool TypeInfo::HasInstantiation(Invocation* invocation,
                                const SymbolicBindings& caller) const {
  return GetInstantiation(invocation, caller).has_value();
}

absl::optional<std::shared_ptr<TypeInfo>> TypeInfo::GetInstantiation(
    Invocation* invocation, const SymbolicBindings& caller) const {
  const TypeInfo* top = GetTop();
  auto it = top->invocations_.find(invocation);
  if (it == top->invocations_.end()) {
    return absl::nullopt;
  }
  const InvocationData& data = it->second;
  auto it2 = data.instantiations.find(caller);
  if (it2 == data.instantiations.end()) {
    return absl::nullopt;
  }
  return it2->second;
}

void TypeInfo::AddInstantiation(Invocation* invocation, SymbolicBindings caller,
                                const std::shared_ptr<TypeInfo>& type_info) {
  TypeInfo* top = GetTop();
  InvocationData& data = top->invocations_[invocation];
  data.instantiations[caller] = type_info;
}

absl::optional<const SymbolicBindings*> TypeInfo::GetInvocationSymbolicBindings(
    Invocation* invocation, const SymbolicBindings& caller) const {
  const TypeInfo* top = GetTop();
  XLS_VLOG(3) << absl::StreamFormat(
      "TypeInfo %p getting invocation symbolic bindings: %p %s @ %s %s", top,
      invocation, invocation->ToString(), invocation->span().ToString(),
      caller.ToString());
  auto it = top->invocations_.find(invocation);
  if (it == top->invocations_.end()) {
    XLS_VLOG(3) << "Could not find invocation " << invocation
                << " in top-level type info: " << top;
    return absl::nullopt;
  }
  const InvocationData& data = it->second;
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

void TypeInfo::AddSliceStartWidth(Slice* node,
                                  const SymbolicBindings& symbolic_bindings,
                                  SliceData::StartWidth start_width) {
  TypeInfo* top = GetTop();
  auto it = top->slices_.find(node);
  if (it == top->slices_.end()) {
    top->slices_[node] =
        SliceData{node, {{symbolic_bindings, std::move(start_width)}}};
  } else {
    top->slices_[node].bindings_to_start_width.emplace(symbolic_bindings,
                                                       std::move(start_width));
  }
}

absl::optional<SliceData::StartWidth> TypeInfo::GetSliceStartWidth(
    Slice* node, const SymbolicBindings& symbolic_bindings) const {
  const TypeInfo* top = GetTop();
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

void TypeInfo::AddImport(Import* import, const std::shared_ptr<Module>& module,
                         const std::shared_ptr<TypeInfo>& type_info) {
  imports_[import] = ImportedInfo{module, type_info};
  Update(*type_info);
}

absl::optional<const ImportedInfo*> TypeInfo::GetImported(
    Import* import) const {
  auto it = imports_.find(import);
  if (it == imports_.end()) {
    if (parent_ != nullptr) {
      return parent_->GetImported(import);
    }
    return absl::nullopt;
  }
  return &it->second;
}

absl::optional<Expr*> TypeInfo::GetConstInt(NameDef* name_def) const {
  auto it = name_to_const_.find(name_def);
  if (it == name_to_const_.end()) {
    if (parent_ != nullptr) {
      return parent_->GetConstInt(name_def);
    }
    return absl::nullopt;
  }
  return it->second->value();
}

}  // namespace xls::dslx
