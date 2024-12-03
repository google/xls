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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_TYPE_SYSTEM_TEST_UTILS_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_TYPE_SYSTEM_TEST_UTILS_H_

#include <string>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Converts the given `TypeInfo` into a human-readable string. If the root
// `TypeInfo` is passed in, this does not convert the invocation `TypeInfo`
// objects for parametric function instantiations.
absl::StatusOr<std::string> TypeInfoToString(const TypeInfo& ti,
                                             const FileTable& file_table);

// Variant of `TypeInfoToString` that converts the root `TypeInfo` of the given
// `module`.
absl::StatusOr<std::string> TypeInfoToString(const TypecheckedModule& module);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TYPE_SYSTEM_TEST_UTILS_H_
