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

#include "xls/dslx/lsp/language_server_adapter.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "external/verible/common/lsp/lsp-file-utils.h"
#include "external/verible/common/lsp/lsp-protocol-enums.h"
#include "external/verible/common/lsp/lsp-protocol.h"
#include "xls/common/indent.h"
#include "xls/common/logging/logging.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/extract_module_name.h"
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/lsp/document_symbols.h"
#include "xls/dslx/lsp/find_definition.h"
#include "xls/dslx/lsp/lsp_type_utils.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

static const char kSource[] = "DSLX";

// Convert error included in status message to LSP Diagnostic
void AppendDiagnosticFromStatus(
    const absl::Status& status,
    std::vector<verible::lsp::Diagnostic>* diagnostic_sink) {
  absl::StatusOr<PositionalErrorData> extracted_error_or =
      GetPositionalErrorData(status, std::nullopt);
  if (!extracted_error_or.ok()) {
    LspLog() << extracted_error_or.status() << "\n";
    return;  // best effort. Ignore.
  }
  const PositionalErrorData& err = *extracted_error_or;
  diagnostic_sink->push_back(verible::lsp::Diagnostic{
      .range = ConvertSpanToLspRange(err.span),
      .severity = verible::lsp::DiagnosticSeverity::kError,
      .has_severity = true,
      .source = kSource,
      .message = err.message});
}

void AppendDiagnosticFromTypecheck(
    const TypecheckedModule& module,
    std::vector<verible::lsp::Diagnostic>* diagnostic_sink) {
  for (const WarningCollector::Entry& warning : module.warnings.warnings()) {
    diagnostic_sink->push_back(verible::lsp::Diagnostic{
        .range = ConvertSpanToLspRange(warning.span),
        .severity = verible::lsp::DiagnosticSeverity::kWarning,
        .has_severity = true,
        .source = kSource,
        .message = warning.message});
  }
}

}  // namespace

LanguageServerAdapter::LanguageServerAdapter(
    std::string_view stdlib,
    const std::vector<std::filesystem::path>& dslx_paths)
    : stdlib_(stdlib), dslx_paths_(dslx_paths) {}

const LanguageServerAdapter::ParseData* LanguageServerAdapter::FindParsedForUri(
    std::string_view uri) const {
  if (auto found = uri_parse_data_.find(uri); found != uri_parse_data_.end()) {
    return found->second.get();
  }
  return nullptr;
}

absl::Status LanguageServerAdapter::Update(std::string_view file_uri,
                                           std::string_view dslx_code) {
  const absl::Time start = absl::Now();
  absl::StatusOr<std::string> module_name_or = ExtractModuleName(file_uri);
  if (!module_name_or.ok()) {
    LspLog() << "Could not determine module name from file URI: " << file_uri
             << " status: " << module_name_or.status() << "\n";
    return absl::OkStatus();
  }

  auto inserted = uri_parse_data_.emplace(file_uri, nullptr);
  std::unique_ptr<ParseData>& insert_value = inserted.first->second;

  ImportData import_data =
      CreateImportData(stdlib_, dslx_paths_, kAllWarningsSet);
  const std::string& module_name = module_name_or.value();

  std::vector<CommentData> comments;
  absl::StatusOr<TypecheckedModule> typechecked_module =
      ParseAndTypecheck(dslx_code, /*path=*/file_uri,
                        /*module_name=*/module_name, &import_data, &comments);

  if (typechecked_module.ok()) {
    insert_value.reset(new ParseData{
        std::move(import_data), TypecheckedModuleWithComments{
                                    .tm = std::move(typechecked_module).value(),
                                    .comments = Comments::Create(comments),
                                }});
  } else {
    insert_value.reset(
        new ParseData{std::move(import_data), typechecked_module.status()});
  }

  const absl::Duration duration = absl::Now() - start;
  if (duration > absl::Milliseconds(200)) {
    LspLog() << "Parsing " << file_uri << " took " << duration << "\n";
  }

  return insert_value->status();
}

std::vector<verible::lsp::Diagnostic>
LanguageServerAdapter::GenerateParseDiagnostics(std::string_view uri) const {
  std::vector<verible::lsp::Diagnostic> result;
  if (const ParseData* parsed = FindParsedForUri(uri)) {
    if (parsed->ok()) {
      const TypecheckedModule& tm = parsed->typechecked_module();
      AppendDiagnosticFromTypecheck(tm, &result);
    } else {
      AppendDiagnosticFromStatus(parsed->status(), &result);
    }
  }
  return result;
}

std::vector<verible::lsp::DocumentSymbol>
LanguageServerAdapter::GenerateDocumentSymbols(std::string_view uri) const {
  VLOG(1) << "GenerateDocumentSymbols; uri: " << uri;
  if (const ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    return ToDocumentSymbols(parsed->module());
  }
  return {};
}

std::vector<verible::lsp::Location> LanguageServerAdapter::FindDefinitions(
    std::string_view uri, const verible::lsp::Position& position) const {
  const Pos pos = ConvertLspPositionToPos(uri, position);
  VLOG(1) << "FindDefinition; uri: " << uri << " pos: " << pos;
  if (const ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    std::optional<Span> maybe_definition_span =
        xls::dslx::FindDefinition(parsed->module(), pos);
    if (maybe_definition_span.has_value()) {
      verible::lsp::Location location =
          ConvertSpanToLspLocation(maybe_definition_span.value());
      location.uri = uri;
      return {location};
    }
  }
  return {};
}

absl::StatusOr<std::vector<verible::lsp::TextEdit>>
LanguageServerAdapter::FormatDocument(std::string_view uri) const {
  using ResultT = std::vector<verible::lsp::TextEdit>;
  if (const ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    const Module& module = parsed->module();
    std::string new_contents = AutoFmt(module, parsed->comments());
    return ResultT{
        verible::lsp::TextEdit{.range = ConvertSpanToLspRange(module.span()),
                               .newText = new_contents}};
  }
  return ResultT{};
}

std::vector<verible::lsp::DocumentLink>
LanguageServerAdapter::ProvideImportLinks(std::string_view uri) const {
  std::vector<verible::lsp::DocumentLink> result;
  if (const ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    const Module& module = parsed->module();
    for (const auto& [_, import_node] : module.GetImportByName()) {
      const ImportTokens tok(import_node->subject());
      absl::StatusOr<ModuleInfo*> info = parsed->import_data.Get(tok);
      if (!info.ok()) {
        continue;
      }
      verible::lsp::DocumentLink link = {
          .range = ConvertSpanToLspRange(import_node->name_def().span()),
          .target = verible::lsp::PathToLSPUri(info.value()->path().string()),
          .has_target = true,
      };
      result.emplace_back(link);
    }
  }
  return result;
}

}  // namespace xls::dslx
