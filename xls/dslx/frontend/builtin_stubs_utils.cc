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
#include <string_view>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/builtin_stubs.h"  // generated
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"

namespace xls::dslx {

constexpr std::string_view kBuiltinStubsPath =
    "xls/dslx/frontend/builtin_stubs.x";

absl::StatusOr<std::filesystem::path> BuiltinStubsPath() {
  return std::filesystem::path(kBuiltinStubsPath);
}

absl::StatusOr<std::unique_ptr<Module>> LoadBuiltinStubs() {
  FileTable file_table;
  Fileno fileno = file_table.GetOrCreate(kBuiltinStubsPath);
  Scanner s = {file_table, fileno, std::string(kBuiltinStubs)};
  Parser parser = {std::string(kBuiltinStubsModuleName), &s, true};
  return parser.ParseModule();
}

bool IsBuiltin(const Function* node) {
  return node->owner()->name() == kBuiltinStubsModuleName;
}

}  // namespace xls::dslx
