// Copyright 2025 The XLS Authors
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

#ifndef XLS_DSLX_FRONTEND_BUILTIN_STUBS_UTILS_H_
#define XLS_DSLX_FRONTEND_BUILTIN_STUBS_UTILS_H_

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

inline constexpr std::string_view kBuiltinStubsModuleName = "<builtin_stubs>";

absl::StatusOr<std::filesystem::path> BuiltinStubsPath();

// Load the (empty) functions in the builtin_stubs.x file into a Module.
absl::StatusOr<std::unique_ptr<Module>> LoadBuiltinStubs(FileTable& file_table);

// Returns true if the given function is a builtin.
bool IsBuiltin(const Function* node);

// Returns whether the given span is the builtin stubs module.
bool IsSpanInBuiltinStubs(const Span& span, const FileTable& file_table);

// Returns the DSLX content of the builtin stubs module.
std::string_view GetBuiltinStubsContent();

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_BUILTIN_STUBS_UTILS_H_
