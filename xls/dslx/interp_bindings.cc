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

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

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
