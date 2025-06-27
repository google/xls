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

#ifndef XLS_COMMON_UNDECLARED_OUTPUTS_H_
#define XLS_COMMON_UNDECLARED_OUTPUTS_H_

#include <filesystem>
#include <optional>

namespace xls {

// Returns the directory in which persistent test artifacts can be written which
// are not declared by the build system. Returns std::nullopt if the program is
// not being run in a test environment which declares the required environment
// variable.
std::optional<std::filesystem::path> GetUndeclaredOutputDirectory();

}  // namespace xls

#endif  // XLS_COMMON_UNDECLARED_OUTPUTS_H_
