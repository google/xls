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

#ifndef XLS_DSLX_INTERP_VALUE_FROM_STRING_H_
#define XLS_DSLX_INTERP_VALUE_FROM_STRING_H_

#include <filesystem>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

// A powerful-but-heavyweight from-string function.
//
// This uses the full power of the parser and constexpr evaluator and can even
// have expressions that reference the stdlib.
absl::StatusOr<InterpValue> InterpValueFromString(
    std::string_view s, const std::filesystem::path& dslx_stdlib_path);

}  // namespace xls::dslx

#endif  // XLS_DSLX_INTERP_VALUE_FROM_STRING_H_
