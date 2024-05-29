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

#include "xls/public/c_api.h"

#include "xls/public/runtime_build_actions.h"

namespace {

std::vector<std::filesystem::path> ToCpp(const char* additional_search_paths[],
                                         size_t additional_search_paths_count) {
  std::vector<std::filesystem::path> additional_search_paths_cpp;
  additional_search_paths_cpp.reserve(additional_search_paths_count);
  for (size_t i = 0; i < additional_search_paths_count; ++i) {
    const char* additional_search_path = additional_search_paths[i];
    CHECK(additional_search_path != nullptr);
    additional_search_paths_cpp.push_back(additional_search_path);
  }
  return additional_search_paths_cpp;
}

char* ToOwnedCString(const std::string& s) { return strdup(s.c_str()); }

}  // namespace

extern "C" {

bool xls_convert_dslx_to_ir(const char* dslx, const char* path,
                            const char* module_name,
                            const char* dslx_stdlib_path,
                            const char* additional_search_paths[],
                            size_t additional_search_paths_count,
                            char** error_out, char** ir_out) {
  CHECK(dslx != nullptr);
  CHECK(path != nullptr);
  CHECK(dslx_stdlib_path != nullptr);

  std::vector<std::filesystem::path> additional_search_paths_cpp =
      ToCpp(additional_search_paths, additional_search_paths_count);

  absl::StatusOr<std::string> result = xls::ConvertDslxToIr(
      dslx, path, module_name, dslx_stdlib_path, additional_search_paths_cpp);
  if (result.ok()) {
    *ir_out = ToOwnedCString(result.value());
    return true;
  }

  *ir_out = nullptr;
  *error_out = ToOwnedCString(result.status().ToString());
  return false;
}

bool xls_convert_dslx_path_to_ir(const char* path, const char* dslx_stdlib_path,
                                 const char* additional_search_paths[],
                                 size_t additional_search_paths_count,
                                 char** error_out, char** ir_out) {
  CHECK(path != nullptr);
  CHECK(dslx_stdlib_path != nullptr);

  std::vector<std::filesystem::path> additional_search_paths_cpp =
      ToCpp(additional_search_paths, additional_search_paths_count);

  absl::StatusOr<std::string> result = xls::ConvertDslxPathToIr(
      path, dslx_stdlib_path, additional_search_paths_cpp);
  if (result.ok()) {
    *ir_out = ToOwnedCString(result.value());
    return true;
  }

  *ir_out = nullptr;
  *error_out = ToOwnedCString(result.status().ToString());
  return false;
}

}  // extern "C"
