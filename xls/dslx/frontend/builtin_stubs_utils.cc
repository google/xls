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
#include "xls/dslx/frontend/builtin_stubs_utils.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"

namespace xls::dslx {

absl::StatusOr<std::unique_ptr<Module>> LoadBuiltinStubs() {
  const std::string path = "xls/dslx/frontend/builtin_stubs.x";
  XLS_ASSIGN_OR_RETURN(const std::filesystem::path full_path,
                       GetXlsRunfilePath(path));
  VLOG(5) << "Loading built-in stubs from: " << full_path;
  XLS_ASSIGN_OR_RETURN(std::string text, GetFileContents(full_path));
  FileTable file_table;
  Fileno fileno = file_table.GetOrCreate(path);
  Scanner s{file_table, fileno, text};
  Parser parser{"<builtin_stubs>", &s, true};
  return parser.ParseModule();
}

}  // namespace xls::dslx
