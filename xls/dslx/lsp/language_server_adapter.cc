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

#include "xls/dslx/create_import_data.h"
#include "xls/dslx/extract_module_name.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/lsp/document_symbols.h"
#include "xls/dslx/lsp/find_definition.h"
#include "xls/dslx/lsp/lsp_type_utils.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/warning_collector.h"

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
    LspLog() << extracted_error_or.status() << "\n" << std::flush;
    return;  // best effort. Ignore.
  }
  const PositionalErrorData& err = *extracted_error_or;
  diagnostic_sink->push_back(
      verible::lsp::Diagnostic{.range = ConvertSpanToLspRange(err.span),
                               .source = kSource,
                               .message = err.message});
}

void AppendDiagnosticFromTypecheck(
    const TypecheckedModule& module,
    std::vector<verible::lsp::Diagnostic>* diagnostic_sink) {
  for (const WarningCollector::Entry& warning : module.warnings.warnings()) {
    diagnostic_sink->push_back(
        verible::lsp::Diagnostic{.range = ConvertSpanToLspRange(warning.span),
                                 .source = kSource,
                                 .message = warning.message});
  }
}

}  // namespace

LanguageServerAdapter::LanguageServerAdapter(
    std::string_view stdlib,
    const std::vector<std::filesystem::path>& dslx_paths)
    : stdlib_(stdlib),
      dslx_paths_(dslx_paths),
      last_parse_data_(absl::FailedPreconditionError(
          "No DSLX file has been parsed yet by the Language Server.")) {}

absl::Status LanguageServerAdapter::Update(std::string_view file_uri,
                                           std::string_view dslx_code) {
  // TODO(hzeller): remember per file_uri for more sophisticated features.
  ImportData import_data = CreateImportData(stdlib_, dslx_paths_);
  const absl::Time start = absl::Now();
  std::string contents{dslx_code};

  absl::StatusOr<std::string> module_name_or = ExtractModuleName(file_uri);
  if (!module_name_or.ok()) {
    LspLog() << "Could not determine module name from file URI: " << file_uri
             << " status: " << module_name_or.status() << "\n"
             << std::flush;
    return absl::OkStatus();
  }

  const std::string& module_name = module_name_or.value();
  absl::StatusOr<TypecheckedModule> typechecked_module_or = ParseAndTypecheck(
      contents, /*path=*/file_uri, /*module_name=*/module_name, &import_data);
  const absl::Duration duration = absl::Now() - start;
  if (duration > absl::Milliseconds(200)) {
    LspLog() << "Parsing " << file_uri << " took " << duration << "\n"
             << std::flush;
  }

  if (typechecked_module_or.ok()) {
    last_parse_data_.emplace(LastParseData{
        std::move(import_data), std::move(typechecked_module_or).value(),
        std::filesystem::path{file_uri}, std::move(contents)});
  } else {
    last_parse_data_ = typechecked_module_or.status();
  }
  return last_parse_data_.status();
}

std::vector<verible::lsp::Diagnostic>
LanguageServerAdapter::GenerateParseDiagnostics(std::string_view uri) const {
  std::vector<verible::lsp::Diagnostic> result;
  if (last_parse_data_.ok()) {
    const TypecheckedModule& tm = last_parse_data_->typechecked_module;
    AppendDiagnosticFromTypecheck(tm, &result);
  } else {
    AppendDiagnosticFromStatus(last_parse_data_.status(), &result);
  }
  return result;
}

std::vector<verible::lsp::DocumentSymbol>
LanguageServerAdapter::GenerateDocumentSymbols(std::string_view uri) const {
  XLS_VLOG(1) << "GenerateDocumentSymbols; uri: " << uri;
  if (last_parse_data_.ok()) {
    const Module& module = last_parse_data_->module();
    return ToDocumentSymbols(module);
  }
  return {};
}

std::vector<verible::lsp::Location> LanguageServerAdapter::FindDefinitions(
    std::string_view uri, const verible::lsp::Position& position) const {
  const Pos pos = ConvertLspPositionToPos(uri, position);
  XLS_VLOG(1) << "FindDefinition; uri: " << uri << " pos: " << pos;
  if (last_parse_data_.ok()) {
    const TypecheckedModule& tm = last_parse_data_->typechecked_module;
    const Module& m = *tm.module;
    std::optional<Span> maybe_definition_span =
        xls::dslx::FindDefinition(m, pos);
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
LanguageServerAdapter::FormatRange(std::string_view uri,
                                   const verible::lsp::Range& range) const {
  // TODO(cdleary): 2023-05-25 We start simple, formatting only when the
  // requested range exactly intercepts a block.
  const Span target = ConvertLspRangeToSpan(uri, range);
  if (last_parse_data_.ok()) {
    const Module& module = last_parse_data_->module();
    const AstNode* intercepting_block =
        module.FindNode(AstNodeKind::kBlock, target);
    if (intercepting_block == nullptr) {
      if (XLS_VLOG_IS_ON(5)) {
        std::vector<const AstNode*> intercepting_start =
            module.FindIntercepting(target.start());
        for (const AstNode* node : intercepting_start) {
          XLS_VLOG(5) << node->GetSpan().value() << " :: " << node->ToString();
        }
      }
      return absl::NotFoundError(
          "Could not find a formattable AST node with the target range: " +
          target.ToString());
    }
    return std::vector<verible::lsp::TextEdit>{verible::lsp::TextEdit{
        .range = range, .newText = intercepting_block->ToString()}};
  }
  return absl::FailedPreconditionError(
      "Language server did not have a successful prior parse to format.");
}

}  // namespace xls::dslx
