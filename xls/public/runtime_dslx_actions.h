// Copyright 2021 The XLS Authors
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

#ifndef XLS_PUBLIC_RUNTIME_DSLX_ACTIONS_H_
#define XLS_PUBLIC_RUNTIME_DSLX_ACTIONS_H_

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace xls {

std::string_view GetDefaultDslxStdlibPath();

struct ConvertDslxToIrOptions {
  std::string_view dslx_stdlib_path;
  absl::Span<const std::filesystem::path> additional_search_paths;
  absl::Span<const std::string_view> enable_warnings;
  absl::Span<const std::string_view> disable_warnings;
  bool warnings_as_errors = true;
  std::vector<std::string>* warnings_out = nullptr;
  bool force_implicit_token_calling_convention = false;
  bool lower_to_proc_scoped_channels = false;
};

absl::StatusOr<std::string> ConvertDslxToIr(
    std::string_view dslx, std::string_view path, std::string_view module_name,
    const ConvertDslxToIrOptions& options);

absl::StatusOr<std::string> ConvertDslxPathToIr(
    const std::filesystem::path& path, const ConvertDslxToIrOptions& options);

absl::StatusOr<std::string> MangleDslxName(std::string_view module_name,
                                           std::string_view function_name);

absl::StatusOr<std::string> ProtoToDslx(std::string_view proto_def,
                                        std::string_view message_name,
                                        std::string_view text_proto,
                                        std::string_view binding_name);

}  // namespace xls

#endif  // XLS_PUBLIC_RUNTIME_DSLX_ACTIONS_H_
