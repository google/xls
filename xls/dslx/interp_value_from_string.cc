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

#include "xls/dslx/interp_value_from_string.h"

#include <filesystem>  // NOLINT
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {

absl::StatusOr<InterpValue> InterpValueFromString(
    std::string_view s, const std::filesystem::path& dslx_stdlib_path) {
  std::string program = absl::StrFormat("import std; const C = (%s);", s);
  ImportData import_data = CreateImportData(
      dslx_stdlib_path,
      /*additional_search_paths=*/absl::Span<const std::filesystem::path>{},
      kDefaultWarningsSet, std::make_unique<RealFilesystem>());
  XLS_ASSIGN_OR_RETURN(TypecheckedModule tm,
                       ParseAndTypecheck(program, "cmdline_constant.x",
                                         "cmdline_constant", &import_data));
  XLS_ASSIGN_OR_RETURN(ConstantDef * constant_node,
                       tm.module->GetMemberOrError<ConstantDef>("C"));
  return tm.type_info->GetConstExpr(constant_node);
}

}  // namespace xls::dslx
