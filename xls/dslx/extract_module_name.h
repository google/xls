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

#ifndef XLS_DSLX_EXTRACT_MODULE_NAME_H_
#define XLS_DSLX_EXTRACT_MODULE_NAME_H_

#include <filesystem>
#include <string>

#include "absl/status/statusor.h"

namespace xls::dslx {

// Extracts an (implied) DSLX module name from a filesystem path, or returns an
// error if it is not possible to do so.
absl::StatusOr<std::string> ExtractModuleName(
    const std::filesystem::path& path);

}  // namespace xls::dslx

#endif  // XLS_DSLX_EXTRACT_MODULE_NAME_H_
