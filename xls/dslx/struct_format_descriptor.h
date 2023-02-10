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

#ifndef XLS_DSLX_STRUCT_FORMAT_DESCRIPTOR_H_
#define XLS_DSLX_STRUCT_FORMAT_DESCRIPTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "xls/ir/format_preference.h"

namespace xls {

// Describes how a leaf field / value of a struct should be formatted.
struct StructFormatFieldDescriptor {
  FormatPreference format = FormatPreference::kDefault;
};

// Describes how a struct should be formatted.
//
// (Note: recursive type, as this is also used for sub-structs under the top
// level struct.)
class StructFormatDescriptor {
 public:
  // A given element has a field name and either describes a leaf of formatting
  // (a value in a field) or a sub-struct via a boxed StructFormatDescriptor.
  struct Element {
    std::string field_name;
    std::variant<std::unique_ptr<StructFormatDescriptor>,
                 StructFormatFieldDescriptor>
        fmt;
  };

  StructFormatDescriptor(std::string struct_name, std::vector<Element> elements)
      : struct_name_(std::move(struct_name)), elements_(std::move(elements)) {}

  const std::string& struct_name() const { return struct_name_; }
  const std::vector<Element>& elements() const { return elements_; }

 private:
  std::string struct_name_;
  std::vector<Element> elements_;
};

}  // namespace xls

#endif  // XLS_DSLX_STRUCT_FORMAT_DESCRIPTOR_H_
