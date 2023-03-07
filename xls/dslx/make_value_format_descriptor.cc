// Copyright 2023 The XLS Authors
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

#include "xls/dslx/make_value_format_descriptor.h"

#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls::dslx {
namespace {

absl::StatusOr<std::unique_ptr<StructFormatDescriptor>>
MakeStructFormatDescriptor(const StructType& struct_type,
                           FormatPreference field_preference) {
  std::vector<StructFormatDescriptor::Element> elements;
  for (size_t i = 0; i < struct_type.size(); ++i) {
    const ConcreteType& member_type = struct_type.GetMemberType(i);
    std::string_view name = struct_type.GetMemberName(i);
    XLS_ASSIGN_OR_RETURN(
        auto desc, MakeValueFormatDescriptor(member_type, field_preference));
    elements.push_back(
        StructFormatDescriptor::Element{std::string(name), std::move(desc)});
  }
  return std::make_unique<StructFormatDescriptor>(
      struct_type.nominal_type().identifier(), std::move(elements));
}

absl::StatusOr<std::unique_ptr<TupleFormatDescriptor>>
MakeTupleFormatDescriptor(const TupleType& tuple_type,
                          FormatPreference field_preference) {
  std::vector<std::unique_ptr<ValueFormatDescriptor>> elements;
  for (size_t i = 0; i < tuple_type.size(); ++i) {
    const ConcreteType& member_type = tuple_type.GetMemberType(i);
    XLS_ASSIGN_OR_RETURN(
        auto vfd, MakeValueFormatDescriptor(member_type, field_preference));
    elements.push_back(std::move(vfd));
  }
  return std::make_unique<TupleFormatDescriptor>(std::move(elements));
}

absl::StatusOr<std::unique_ptr<ArrayFormatDescriptor>>
MakeArrayFormatDescriptor(const ArrayType& type,
                          FormatPreference field_preference) {
  XLS_ASSIGN_OR_RETURN(int64_t size, type.size().GetAsInt64());
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ValueFormatDescriptor> element_type_descriptor,
      MakeValueFormatDescriptor(type.element_type(), field_preference));
  return std::make_unique<ArrayFormatDescriptor>(
      std::move(element_type_descriptor), size);
}

absl::StatusOr<std::unique_ptr<EnumFormatDescriptor>> MakeEnumFormatDescriptor(
    const EnumType& type, FormatPreference field_preference) {
  absl::flat_hash_map<Bits, std::string> value_to_name;
  const EnumDef& enum_def = type.nominal_type();
  for (size_t i = 0; i < enum_def.values().size(); ++i) {
    const std::string& s = enum_def.GetMemberName(i);
    const InterpValue& v = type.members().at(i);
    XLS_RET_CHECK(v.IsBits());
    value_to_name[v.GetBitsOrDie()] = s;
  }
  return std::make_unique<EnumFormatDescriptor>(enum_def.identifier(),
                                                std::move(value_to_name));
}

}  // namespace

absl::StatusOr<std::unique_ptr<ValueFormatDescriptor>>
MakeValueFormatDescriptor(const ConcreteType& type,
                          FormatPreference field_preference) {
  if (auto* a = dynamic_cast<const ArrayType*>(&type)) {
    return MakeArrayFormatDescriptor(*a, field_preference);
  }
  if (auto* s = dynamic_cast<const StructType*>(&type)) {
    return MakeStructFormatDescriptor(*s, field_preference);
  }
  if (auto* t = dynamic_cast<const TupleType*>(&type)) {
    return MakeTupleFormatDescriptor(*t, field_preference);
  }
  if (auto* e = dynamic_cast<const EnumType*>(&type)) {
    return MakeEnumFormatDescriptor(*e, field_preference);
  }
  if (auto* t = dynamic_cast<const BitsType*>(&type)) {
    return std::make_unique<LeafValueFormatDescriptor>(field_preference);
  }
  return absl::InvalidArgumentError(
      "Cannot make a ValueFormatDescriptor for type: " +
      type.GetDebugTypeName());
}

}  // namespace xls::dslx
