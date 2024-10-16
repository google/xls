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

#include <string>
#include <string_view>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/frontend/scanner.h"

namespace xls::dslx {

std::pair<Module, Proc*> CreateEmptyProc(FileTable& file_table,
                                         std::string_view name) {
  const std::string_view code_template = R"(proc %s {
    config() { () }
    init { () }
    next(state: ()) { () }
})";
  Scanner s(file_table, Fileno(0), absl::StrFormat(code_template, name));
  Parser parser{"test", &s};
  Bindings bindings;
  absl::StatusOr<Proc*> proc = parser.ParseProc(/*is_public=*/false, bindings);
  CHECK(proc.ok());
  return {std::move(parser.module()), *proc};
}

}  // namespace xls::dslx
