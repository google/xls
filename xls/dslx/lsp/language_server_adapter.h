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

#ifndef XLS_DSLX_LSP_LANGUAGE_SERVER_ADAPTER_H_
#define XLS_DSLX_LSP_LANGUAGE_SERVER_ADAPTER_H_

#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "external/verible/common/lsp/lsp-protocol.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {

// While stdin and stdout are to communicate with the json rpc dispatcher,
// stderr is our log stream that the editor typically makes available
// somewhere.
inline std::ostream& LspLog() { return std::cerr; }

// Note: this is a thread-compatible implementation, but not thread safe (e.g.
// we assume the language server request handler acts as a concurrency
// serializing entity).
class LanguageServerAdapter {
 public:
  LanguageServerAdapter(std::string_view stdlib,
                        const std::vector<std::filesystem::path>& dslx_paths);

  // Note: this is parsing is triggered for every keystroke. Fine for now.
  // Successful and unsuccessful parses are memoized so that their status
  // and can be queried.
  // Implementation note: since we currently do not react to buffer closed
  // events in the buffer change listener, we keep track of every file ever
  // opened and never delete.
  absl::Status Update(std::string_view file_uri, std::string_view dslx_code);

  // Generate LSP diagnostics for the file with given uri
  std::vector<verible::lsp::Diagnostic> GenerateParseDiagnostics(
      std::string_view uri) const;

  std::vector<verible::lsp::DocumentSymbol> GenerateDocumentSymbols(
      std::string_view uri) const;

  // Note: the return type is slightly unintuitive, but the latest LSP protocol
  // supports multiple defining locations for a single reference.
  std::vector<verible::lsp::Location> FindDefinitions(
      std::string_view uri, const verible::lsp::Position& position) const;

  // Implements the functionality for:
  // https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_rangeFormatting
  absl::StatusOr<std::vector<verible::lsp::TextEdit>> FormatRange(
      std::string_view uri, const verible::lsp::Range& range) const;

  // Present links to imports to directly open the relevant file.
  std::vector<verible::lsp::DocumentLink> ProvideImportLinks(
      std::string_view uri) const;

 private:
  struct ParseData;
  const std::string stdlib_;
  const std::vector<std::filesystem::path> dslx_paths_;

  // Find parse result of opened file with given URI or nullptr, if not opened.
  const ParseData* FindParsedForUri(std::string_view uri) const;

  // Everything relevant for a parsed editor buffer.
  // Note, each buffer independently currently keeps track of its import data.
  // This could maybe be considered to be put in a single place.
  struct ParseData {
    ImportData import_data;
    absl::StatusOr<TypecheckedModule> typechecked_module;

    bool ok() const { return typechecked_module.ok(); }
    absl::Status status() const { return typechecked_module.status(); }

    const Module& module() const { return *typechecked_module->module; }
  };

  absl::flat_hash_map<std::string, std::unique_ptr<ParseData>> uri_parse_data_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_LSP_LANGUAGE_SERVER_ADAPTER_H_
