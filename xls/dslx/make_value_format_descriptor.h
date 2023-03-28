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

#ifndef XLS_DSLX_MAKE_STRUCT_FORMAT_DESCRIPTOR_H_
#define XLS_DSLX_MAKE_STRUCT_FORMAT_DESCRIPTOR_H_

#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/value_format_descriptor.h"

namespace xls::dslx {

absl::StatusOr<std::unique_ptr<ValueFormatDescriptor>>
MakeValueFormatDescriptor(const ConcreteType& type,
                          FormatPreference field_preference);

}  // namespace xls::dslx

#endif  // XLS_DSLX_MAKE_STRUCT_FORMAT_DESCRIPTOR_H_
