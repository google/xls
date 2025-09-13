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

#include "xls/dslx/value_format_descriptor.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

ValueFormatDescriptor ValueFormatDescriptor::MakeLeafValue(
    FormatPreference format) {
  ValueFormatDescriptor vfd(ValueFormatDescriptorKind::kLeafValue);
  vfd.format_ = format;
  return vfd;
}

ValueFormatDescriptor ValueFormatDescriptor::MakeEnum(
    std::string_view enum_name,
    absl::flat_hash_map<Bits, std::string> value_to_name) {
  ValueFormatDescriptor vfd(ValueFormatDescriptorKind::kEnum);
  vfd.enum_name_ = enum_name;
  vfd.value_to_name_ = std::move(value_to_name);
  return vfd;
}

ValueFormatDescriptor ValueFormatDescriptor::MakeArray(
    const ValueFormatDescriptor& element_format, size_t size) {
  ValueFormatDescriptor vfd(ValueFormatDescriptorKind::kArray);
  vfd.children_ = {element_format};
  vfd.size_ = size;
  return vfd;
}

ValueFormatDescriptor ValueFormatDescriptor::MakeTuple(
    absl::Span<const ValueFormatDescriptor> elements) {
  ValueFormatDescriptor vfd(ValueFormatDescriptorKind::kTuple);
  vfd.children_ =
      std::vector<ValueFormatDescriptor>(elements.begin(), elements.end());
  vfd.size_ = elements.size();
  return vfd;
}

ValueFormatDescriptor ValueFormatDescriptor::MakeStruct(
    std::string_view struct_name, absl::Span<const std::string> field_names,
    absl::Span<const ValueFormatDescriptor> field_formats) {
  CHECK_EQ(field_names.size(), field_formats.size());
  ValueFormatDescriptor vfd(ValueFormatDescriptorKind::kStruct);
  vfd.struct_name_ = struct_name;
  vfd.children_ = std::vector<ValueFormatDescriptor>(field_formats.begin(),
                                                     field_formats.end());
  vfd.size_ = field_names.size();
  vfd.struct_field_names_ =
      std::vector<std::string>(field_names.begin(), field_names.end());
  return vfd;
}

absl::Status ValueFormatDescriptor::Accept(ValueFormatVisitor& v) const {
  switch (kind()) {
    case ValueFormatDescriptorKind::kLeafValue:
      return v.HandleLeafValue(*this);
    case ValueFormatDescriptorKind::kEnum:
      return v.HandleEnum(*this);
    case ValueFormatDescriptorKind::kArray:
      return v.HandleArray(*this);
    case ValueFormatDescriptorKind::kTuple:
      return v.HandleTuple(*this);
    case ValueFormatDescriptorKind::kStruct:
      return v.HandleStruct(*this);
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Out of bounds ValueFormatDescriptorKind: %d", static_cast<int>(kind())));
}

}  // namespace xls::dslx
