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

#ifndef XLS_DSLX_TYPE_SYSTEM_UNWRAP_METATYPE_H_
#define XLS_DSLX_TYPE_SYSTEM_UNWRAP_METATYPE_H_

#include <memory>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

// Unwraps a metatype (i.e. type-type) and returns the wrapped type.
//
// This is useful when we want to check that a type has been deduced in a given
// position, but then we want to unwrap that type because ultimately we care
// about the expression that yields that particular type.
//
// Returns a TypeInferenceError if t is not a metatype, and the error message
// will say: "Expected a type in ${context}".
absl::StatusOr<std::unique_ptr<Type>> UnwrapMetaType(std::unique_ptr<Type> t,
                                                     const Span& span,
                                                     std::string_view context);

absl::StatusOr<const Type*> UnwrapMetaType(const Type& t);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_UNWRAP_METATYPE_H_
