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

#include "xls/dslx/interp_bindings.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

/* static */ InterpBindings InterpBindings::CloneWith(
    InterpBindings* parent, NameDefTree* name_def_tree, InterpValue value) {
  InterpBindings new_bindings(/*parent=*/parent);
  new_bindings.AddValueTree(name_def_tree, value);
  new_bindings.set_fn_ctx(parent->fn_ctx_);
  return new_bindings;
}

/* static */ std::string_view InterpBindings::VariantAsString(const Entry& e) {
  if (std::holds_alternative<InterpValue>(e)) {
    return "Value";
  }
  if (std::holds_alternative<TypeAlias*>(e)) {
    return "TypeAlias";
  }
  if (std::holds_alternative<EnumDef*>(e)) {
    return "EnumDef";
  }
  if (std::holds_alternative<StructDef*>(e)) {
    return "StructDef";
  }
  if (std::holds_alternative<Module*>(e)) {
    return "Module";
  }
  LOG(FATAL) << "Unhandled binding entry variant.";
}

InterpBindings::InterpBindings(const InterpBindings* parent) : parent_(parent) {
  if (parent_ != nullptr) {
    fn_ctx_ = parent_->fn_ctx();
  }
}

void InterpBindings::AddValueTree(NameDefTree* name_def_tree,
                                  InterpValue value) {
  if (name_def_tree->is_leaf()) {
    NameDefTree::Leaf leaf = name_def_tree->leaf();
    if (std::holds_alternative<NameDef*>(leaf)) {
      NameDef* name_def = std::get<NameDef*>(leaf);
      AddValue(name_def->identifier(), value);
    }
    return;
  }
  for (int64_t i = 0; i < value.GetLength().value(); ++i) {
    NameDefTree* sub_tree = name_def_tree->nodes()[i];
    const InterpValue& sub_value = value.GetValuesOrDie()[i];
    AddValueTree(sub_tree, sub_value);
  }
}

absl::StatusOr<InterpValue> InterpBindings::ResolveValueFromIdentifier(
    std::string_view identifier, const Span* ref_span) const {
  std::optional<Entry> entry = ResolveEntry(identifier);
  if (!entry.has_value()) {
    std::string span_str;
    if (ref_span != nullptr) {
      span_str = " @ " + ref_span->ToString();
    }
    return absl::NotFoundError(absl::StrFormat(
        "InterpBindings could not find bindings entry for identifier: \"%s\"%s",
        identifier, span_str));
  }
  InterpValue* value = absl::get_if<InterpValue>(&entry.value());
  if (value == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Attempted to resolve a value but identifier \"%s\" "
                        "was not bound to a value; got %s",
                        identifier, VariantAsString(entry.value())));
  }
  return *value;
}

absl::StatusOr<Module*> InterpBindings::ResolveModule(
    std::string_view identifier) const {
  std::optional<Entry> entry = ResolveEntry(identifier);
  if (!entry.has_value()) {
    return absl::NotFoundError(
        absl::StrFormat("No binding for identifier \"%s\"", identifier));
  }
  if (std::holds_alternative<Module*>(entry.value())) {
    return std::get<Module*>(entry.value());
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Attempted to resolve a module but identifier \"%s\" was bound to a %s",
      identifier, VariantAsString(entry.value())));
}

absl::StatusOr<TypeAnnotation*> InterpBindings::ResolveTypeAnnotation(
    std::string_view identifier) const {
  std::optional<Entry> entry = ResolveEntry(identifier);
  if (!entry.has_value()) {
    return absl::NotFoundError(
        absl::StrFormat("No binding for identifier \"%s\"", identifier));
  }
  if (std::holds_alternative<EnumDef*>(entry.value())) {
    return std::get<EnumDef*>(entry.value())->type_annotation();
  }
  if (std::holds_alternative<TypeAlias*>(entry.value())) {
    return &std::get<TypeAlias*>(entry.value())->type_annotation();
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Attempted to resolve a type but identifier \"%s\" was bound to a %s",
      identifier, VariantAsString(entry.value())));
}

absl::StatusOr<std::variant<TypeAnnotation*, EnumDef*, StructDef*>>
InterpBindings::ResolveTypeDefinition(std::string_view identifier) const {
  std::optional<Entry> entry = ResolveEntry(identifier);
  if (!entry.has_value()) {
    return absl::NotFoundError(absl::StrFormat(
        "Could not resolve type definition for identifier: \"%s\"",
        identifier));
  }
  if (std::holds_alternative<TypeAlias*>(entry.value())) {
    return &std::get<TypeAlias*>(entry.value())->type_annotation();
  }
  if (std::holds_alternative<EnumDef*>(entry.value())) {
    return std::get<EnumDef*>(entry.value());
  }
  if (std::holds_alternative<StructDef*>(entry.value())) {
    return std::get<StructDef*>(entry.value());
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Attempted to resolve a type definition but identifier "
                      "\"%s\" was bound to a %s",
                      identifier, VariantAsString(entry.value())));
}

absl::flat_hash_set<std::string> InterpBindings::GetKeys() const {
  absl::flat_hash_set<std::string> result;
  for (const auto& item : map_) {
    result.insert(item.first);
  }
  if (parent_ != nullptr) {
    absl::flat_hash_set<std::string> parent_keys = parent_->GetKeys();
    result.insert(parent_keys.begin(), parent_keys.end());
  }
  return result;
}

std::optional<InterpBindings::Entry> InterpBindings::ResolveEntry(
    std::string_view identifier) const {
  auto it = map_.find(identifier);
  if (it != map_.end()) {
    return it->second;
  }

  if (parent_ != nullptr) {
    return parent_->ResolveEntry(identifier);
  }

  return std::nullopt;
}

}  // namespace xls::dslx
