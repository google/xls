// Copyright 2026 The XLS Authors
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

#ifndef XLS_DEV_TOOLS_ANNOTATE_TYPES_H_
#define XLS_DEV_TOOLS_ANNOTATE_TYPES_H_

#include <filesystem>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {

struct AnnotateTypesOptions {
  std::string_view dslx_stdlib_path = "";
  absl::Span<const std::filesystem::path> dslx_paths = {};
  bool type_inference_v2 = false;
  bool warnings_as_errors = false;
  WarningKindSet warnings = kDefaultWarningsSet;
};

// Parses and typechecks `dslx_code` and returns a modified string where all
// unannotated `let` bindings have explicit type annotations inserted (e.g.
// `let x = ...` -> `let x: u32 = ...`).
//
// Comments, newlines, and surrounding indentation are preserved.
absl::StatusOr<std::string> AnnotateTypes(
    std::string_view dslx_code, std::string_view module_name,
    std::string_view path = "input.x",
    const AnnotateTypesOptions& options = AnnotateTypesOptions{});

}  // namespace xls::dslx

#endif  // XLS_DEV_TOOLS_ANNOTATE_TYPES_H_
