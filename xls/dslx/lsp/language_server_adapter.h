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
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "external/verible/common/lsp/lsp-protocol.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/lsp/import_sensitivity.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type_info.h"

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

  // Takes note that `dslx_code` is the current file contents for `file_uri`
  // and performs a parse-and-typecheck using that file/contents as the entry
  // point.
  //
  // `dslx_code` can be nullopt when we're re-evaluating the previous contents
  // again; i.e. because we think a dependency may have been corrected.
  //
  // Note: this is parsing is triggered for every keystroke. Fine for now.
  // Successful and unsuccessful parses are memoized so that their status
  // and can be queried.
  //
  // Implementation note: since we currently do not react to buffer closed
  // events in the buffer change listener, we keep track of every file ever
  // opened and never delete.
  absl::Status Update(std::string_view file_uri,
                      std::optional<std::string_view> dslx_code);

  // Generate LSP diagnostics for the file with given `uri`.
  //
  // Note that this only finds the existing parse-and-typecheck result, it does
  // not trigger any new parsing activity, for that we need to `Update`.
  std::vector<verible::lsp::Diagnostic> GenerateParseDiagnostics(
      std::string_view uri) const;

  std::vector<verible::lsp::DocumentSymbol> GenerateDocumentSymbols(
      std::string_view uri) const;

  // Note: the return type is slightly unintuitive, but the latest LSP protocol
  // supports multiple defining locations for a single reference.
  absl::StatusOr<std::vector<verible::lsp::Location>> FindDefinitions(
      std::string_view uri, const verible::lsp::Position& position) const;

  // Implements the functionality for full document formatting:
  // https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_formatting
  //
  // TODO(cdleary): 2024-03-10 We may want to consider adding formatting
  // options.
  absl::StatusOr<std::vector<verible::lsp::TextEdit>> FormatDocument(
      std::string_view uri) const;

  // Present links to imports to directly open the relevant file.
  std::vector<verible::lsp::DocumentLink> ProvideImportLinks(
      std::string_view uri) const;

  absl::StatusOr<std::vector<verible::lsp::InlayHint>> InlayHint(
      std::string_view uri, const verible::lsp::Range& range) const;

  absl::StatusOr<std::optional<verible::lsp::Range>> PrepareRename(
      std::string_view uri, const verible::lsp::Position& position) const;

  absl::StatusOr<std::optional<verible::lsp::WorkspaceEdit>> Rename(
      std::string_view uri, const verible::lsp::Position& position,
      std::string_view new_name) const;

  // See
  // https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_documentHighlight
  absl::StatusOr<std::vector<verible::lsp::DocumentHighlight>>
  DocumentHighlight(std::string_view uri,
                    const verible::lsp::Position& position) const;

  ImportSensitivity& import_sensitivity() { return import_sensitivity_; }

  const absl::flat_hash_map<std::string, std::string>& vfs_contents() const {
    return vfs_contents_;
  }

 private:
  class ParseData;

  // Find parse result of opened file with given URI or nullptr, if not opened.
  ParseData* FindParsedForUri(std::string_view uri) const;

  struct TypecheckedModuleWithComments {
    TypecheckedModule tm;
    Comments comments;
  };

  // Everything relevant for a parsed editor buffer.
  // Note, each buffer independently currently keeps track of its import data.
  // This could maybe be considered to be put in a single place.
  class ParseData {
   public:
    ParseData(ImportData&& import_data,
              absl::StatusOr<TypecheckedModuleWithComments> tmc)
        : import_data_(std::move(import_data)), tmc_(std::move(tmc)) {}

    bool ok() const { return tmc_.ok(); }
    absl::Status status() const { return tmc_.status(); }

    ImportData& import_data() { return import_data_; }
    FileTable& file_table() { return import_data_.file_table(); }
    const Module& module() const {
      CHECK_OK(tmc_.status());
      return *tmc_->tm.module;
    }
    const TypeInfo& type_info() const {
      CHECK_OK(tmc_.status());
      return *tmc_->tm.type_info;
    }
    const Comments& comments() const {
      CHECK_OK(tmc_.status());
      return tmc_->comments;
    }
    const TypecheckedModule& typechecked_module() const {
      CHECK_OK(tmc_.status());
      return tmc_->tm;
    }

   private:
    ImportData import_data_;
    absl::StatusOr<TypecheckedModuleWithComments> tmc_;
  };

  const std::string stdlib_;
  const std::vector<std::filesystem::path> dslx_paths_;
  absl::flat_hash_map<std::string, std::unique_ptr<ParseData>> uri_parse_data_;

  // The language server, in effect, needs to maintain a virtual filesystem
  // layer, that's interwoven with the true filesystem; i.e. if a file is
  // present on disk but not opened in the LSP workspace, we resolve it on
  // disk.
  absl::flat_hash_map<std::string, std::string> vfs_contents_;

  ImportSensitivity import_sensitivity_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_LSP_LANGUAGE_SERVER_ADAPTER_H_
