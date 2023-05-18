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

#ifndef XLS_DSLX_LSP_LANGAUGE_SERVER_ADAPTER_H_
#define XLS_DSLX_LSP_LANGAUGE_SERVER_ADAPTER_H_

#include <filesystem>  // NOLINT
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "external/verible/common/lsp/lsp-protocol.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {

// While stdin and stdout are to communicate with the json rpc dispatcher,
// stderr is our log stream that the editor typically makes available
// somewhere.
inline std::ostream& LspLog() { return std::cerr; }

class LanguageServerAdapter {
 public:
  LanguageServerAdapter(std::string_view stdlib,
                        const std::vector<std::filesystem::path>& dslx_paths);

  // Note: this is parsing is triggered for every keystroke. Fine for now.
  void Update(std::string_view file_uri, std::string_view dslx_code);

  // Generate LSP diagnostics for the last file update.
  std::vector<verible::lsp::Diagnostic> GenerateParseDiagnostics(
      std::string_view uri) const;

  std::vector<verible::lsp::DocumentSymbol> GenerateDocumentSymbols(
      std::string_view uri) const;

  // Note: the return type is slightly unintuitive, but the latest LSP protocol
  // supports multiple defining locations for a single reference.
  std::vector<verible::lsp::Location> FindDefinitions(
      std::string_view uri, const verible::lsp::Position& position) const;

 private:
  const std::string stdlib_;
  const std::vector<std::filesystem::path> dslx_paths_;

  std::optional<ImportData> last_import_data_;
  std::string last_dslx_code_;
  absl::StatusOr<TypecheckedModule> last_parse_result_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_LSP_LANGAUGE_SERVER_ADAPTER_H_
