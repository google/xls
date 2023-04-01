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
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/lsp/document_symbols.h"
#include "xls/dslx/lsp/lsp_type_utils.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

// Convert error included in status message to LSP Diagnostic
void AppendDiagnosticFromStatus(
    const absl::Status& status,
    std::vector<verible::lsp::Diagnostic>* diagnostic_sink) {
  absl::StatusOr<PositionalErrorData> extracted_error_or =
      GetPositionalErrorData(status, std::nullopt);
  if (!extracted_error_or.ok()) {
    LspLog() << extracted_error_or.status() << std::endl;
    return;  // best effort. Ignore.
  }
  const PositionalErrorData& err = *extracted_error_or;
  diagnostic_sink->push_back(
      verible::lsp::Diagnostic{.range = ConvertSpanToRange(err.span),
                               .source = "XLS",
                               .message = err.message});
}

void AppendDiagnosticFromTypecheck(
    const TypecheckedModule& module,
    std::vector<verible::lsp::Diagnostic>* diagnostic_sink) {
  for (const WarningCollector::Entry& warning : module.warnings.warnings()) {
    diagnostic_sink->push_back(
        verible::lsp::Diagnostic{.range = ConvertSpanToRange(warning.span),
                                 .source = "XLS",
                                 .message = warning.message});
  }
}

}  // namespace

LanguageServerAdapter::LanguageServerAdapter(
    std::string_view stdlib,
    const std::vector<std::filesystem::path>& dslx_paths)
    : stdlib_(stdlib),
      dslx_paths_(dslx_paths),
      last_parse_result_(absl::NotFoundError("not parsed yet")) {}

void LanguageServerAdapter::Update(std::string_view file_uri,
                                   std::string_view dslx_code) {
  // TODO(hzeller): remember per file_uri for more sophisticated features.
  last_import_data_.emplace(CreateImportData(stdlib_, dslx_paths_));
  const absl::Time start = absl::Now();
  last_dslx_code_ = dslx_code;
  last_parse_result_ =
      ParseAndTypecheck(last_dslx_code_, /*path=*/"", /*module_name=*/"foo",
                        &last_import_data_.value());
  const absl::Duration duration = absl::Now() - start;
  if (duration > absl::Milliseconds(200)) {
    LspLog() << "Parsing " << file_uri << " took " << duration << std::endl;
  }
}

std::vector<verible::lsp::Diagnostic>
LanguageServerAdapter::GenerateParseDiagnostics(std::string_view uri) const {
  std::vector<verible::lsp::Diagnostic> result;
  if (last_parse_result_.ok()) {
    AppendDiagnosticFromTypecheck(*last_parse_result_, &result);
  } else {
    AppendDiagnosticFromStatus(last_parse_result_.status(), &result);
  }
  return result;
}

std::vector<verible::lsp::DocumentSymbol>
LanguageServerAdapter::GenerateDocumentSymbols(std::string_view uri) const {
  XLS_VLOG(1) << "GenerateDocumentSymbols; uri: " << uri;
  if (last_parse_result_.ok()) {
    XLS_CHECK(last_parse_result_->module != nullptr);
    return ToDocumentSymbols(*last_parse_result_->module);
  }
  return {};
}

}  // namespace xls::dslx
