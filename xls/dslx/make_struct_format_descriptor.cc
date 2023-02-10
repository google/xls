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

#include "xls/dslx/make_struct_format_descriptor.h"

namespace xls::dslx {

std::unique_ptr<StructFormatDescriptor> MakeStructFormatDescriptor(
    const StructType& struct_type) {
  std::vector<StructFormatDescriptor::Element> elements;
  for (size_t i = 0; i < struct_type.size(); ++i) {
    const ConcreteType& member_type = struct_type.GetMemberType(i);
    std::string_view name = struct_type.GetMemberName(i);
    if (member_type.IsStruct()) {
      elements.push_back(StructFormatDescriptor::Element{
          std::string(name),
          MakeStructFormatDescriptor(member_type.AsStruct())});
    } else {
      elements.push_back(StructFormatDescriptor::Element{
          std::string(name), StructFormatFieldDescriptor{}});
    }
  }
  return std::make_unique<StructFormatDescriptor>(
      struct_type.nominal_type().identifier(), std::move(elements));
}

}  // namespace xls::dslx
