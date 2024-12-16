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

#include "xls/dslx/lsp/document_symbols.h"

#include <string_view>

#include "gtest/gtest.h"
#include "verible/common/lsp/lsp-protocol-enums.h"
#include "verible/common/lsp/lsp-protocol.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls {

TEST(DocumentSymbolsTest, TestUseConstruct) {
  const std::string_view kModule = R"(#![feature(use_syntax)]
use foo::{bar, baz::{qux, quux}};
)";
  dslx::FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<dslx::Module> module,
      dslx::ParseModule(kModule, "/path/to/sample.x", /*module_name=*/"sample",
                        file_table));
  std::vector<verible::lsp::DocumentSymbol> symbols =
      dslx::ToDocumentSymbols(*module);
  EXPECT_EQ(symbols.size(), 3);
  EXPECT_EQ(symbols[0].name, "bar");
  EXPECT_EQ(symbols[1].name, "qux");
  EXPECT_EQ(symbols[2].name, "quux");

  for (const auto& symbol : symbols) {
    EXPECT_EQ(symbol.kind, verible::lsp::SymbolKind::kModule);
  }
}

}  // namespace xls