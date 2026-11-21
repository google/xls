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

#ifndef XLS_DSLX_TYPE_SYSTEM_TYPE_SPECIALIZATION_H_
#define XLS_DSLX_TYPE_SYSTEM_TYPE_SPECIALIZATION_H_

#include <cstdint>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"

namespace xls::dslx {

class AstNode;
class InterpValue;
class Module;
class Type;

namespace type_system_internal {

enum class SpecializationKind {
  kUnsignedBits,
  kSignedBits,
  kEnum,
  kProc,
  kStruct,
  kSum,
  kUnitVariant,
  kTupleVariant,
  kStructVariant,
  kTuple,
  kArray,
  kFunction,
  kChannelIn,
  kChannelOut,
  kMeta,
  kModule,
  kToken,
};

enum class BindingKind { kType, kBits, kTuple, kArray, kUnsupported };
BindingKind GetBindingKind(const InterpValue& value);

// For a module type, only module is populated. For a nominal type, equality
// uses the declaration's identity and mangling uses its qualified name.
struct NominalPart {
  const AstNode* declaration;
  const Module* module;
  std::optional<std::string_view> identifier;

  bool operator==(const NominalPart& other) const {
    return declaration == other.declaration && module == other.module;
  }
};

using SpecializationPart =
    std::variant<SpecializationKind, int64_t, std::string_view, NominalPart,
                 const Type*, const InterpValue*>;

// Describes one level of a checked type for both specialization equality and
// mangling. The kind comes first, followed by its distinguishing properties.
// Numbers are resolved dimensions or counts, strings are binding names, and
// pointers refer to child types or binding values. The result borrows from
// type and its declarations.
absl::StatusOr<std::vector<SpecializationPart>> DescribeSpecialization(
    const Type& type);

}  // namespace type_system_internal
}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_TYPE_SPECIALIZATION_H_
