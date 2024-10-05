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
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "external/verible/common/lsp/lsp-file-utils.h"
#include "external/verible/common/lsp/lsp-protocol-enums.h"
#include "external/verible/common/lsp/lsp-protocol.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
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
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

static const char kSource[] = "DSLX";

// Convert error included in status message to LSP Diagnostic
void AppendDiagnosticFromStatus(
    const absl::Status& status,
    std::vector<verible::lsp::Diagnostic>* diagnostic_sink,
    FileTable& file_table) {
  absl::StatusOr<PositionalErrorData> extracted_error_or =
      GetPositionalErrorData(status, std::nullopt, file_table);
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

// If the span's filename is a relative path, we map it into a URI by finding
// its path on disk.
//
// Note: there may be a more holistic way to do this long term -- the LSP
// effectively requires us to create a filesystem overlay that mixes real files
// and in-memory files with modifications. For now, this at least enables
// things like go-to-definition in files that have not yet been loaded into the
// language server.
absl::StatusOr<std::string> MaybeRelpathToUri(
    std::string_view path_or_uri,
    absl::Span<const std::filesystem::path> dslx_paths) {
  if (absl::StartsWith(path_or_uri, "file://") ||
      absl::StartsWith(path_or_uri, "memfile://")) {
    return std::string{path_or_uri};
  }

  std::vector<std::string> results;
  for (std::filesystem::path dirpath : dslx_paths) {
    if (dirpath.empty()) {
      dirpath = std::filesystem::current_path();
    }
    std::filesystem::path full = dirpath / path_or_uri;
    if (std::filesystem::exists(full)) {
      results.push_back(absl::StrCat("file://", full.c_str()));
    }
  }

  if (results.empty()) {
    return absl::NotFoundError(absl::StrFormat(
        "Could not find path to convert to URI: `%s`", path_or_uri));
  }

  if (results.size() > 1) {
    LspLog() << "Found more than one URI for path: " << path_or_uri
             << " results: " << absl::StrJoin(results, " :: ");
  }

  return results.at(0);
}

}  // namespace

LanguageServerAdapter::LanguageServerAdapter(
    std::string_view stdlib,
    const std::vector<std::filesystem::path>& dslx_paths)
    : stdlib_(stdlib), dslx_paths_(dslx_paths) {}

LanguageServerAdapter::ParseData* LanguageServerAdapter::FindParsedForUri(
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
  if (ParseData* parsed = FindParsedForUri(uri)) {
    FileTable& file_table = parsed->import_data.file_table();
    if (parsed->ok()) {
      const TypecheckedModule& tm = parsed->typechecked_module();
      AppendDiagnosticFromTypecheck(tm, &result);
    } else {
      AppendDiagnosticFromStatus(parsed->status(), &result, file_table);
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

absl::StatusOr<std::vector<verible::lsp::Location>>
LanguageServerAdapter::FindDefinitions(
    std::string_view uri, const verible::lsp::Position& position) const {
  if (ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    FileTable& file_table = parsed->import_data.file_table();
    const Pos pos = ConvertLspPositionToPos(uri, position, file_table);
    VLOG(1) << "FindDefinition; uri: " << uri << " pos: " << pos;
    std::optional<const NameDef*> maybe_definition = xls::dslx::FindDefinition(
        parsed->module(), pos, parsed->type_info(), parsed->import_data);
    if (maybe_definition.has_value()) {
      const Span& definition_span = maybe_definition.value()->span();
      VLOG(1) << "FindDefinition; span: "
              << definition_span.ToString(file_table);
      verible::lsp::Location location =
          ConvertSpanToLspLocation(definition_span);
      XLS_ASSIGN_OR_RETURN(
          location.uri, MaybeRelpathToUri(definition_span.GetFilename(
                                              parsed->import_data.file_table()),
                                          dslx_paths_));
      return std::vector<verible::lsp::Location>{location};
    }
  }
  return std::vector<verible::lsp::Location>{};
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

absl::StatusOr<std::vector<verible::lsp::InlayHint>>
LanguageServerAdapter::InlayHint(std::string_view uri,
                                 const verible::lsp::Range& range) const {
  std::vector<verible::lsp::InlayHint> results;
  if (ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    FileTable& file_table = parsed->import_data.file_table();
    const Span want_span = ConvertLspRangeToSpan(uri, range, file_table);
    const Module& module = parsed->module();
    const TypeInfo& type_info = parsed->type_info();
    // Get let bindings in the AST that fall in the given range.
    const std::vector<const AstNode*> contained =
        module.FindContained(want_span);
    for (const AstNode* node : contained) {
      if (node->kind() == AstNodeKind::kLet) {
        const auto* let = down_cast<const Let*>(node);
        if (let->type_annotation() != nullptr) {
          // Already has a type annotated, no need for inlay.
          continue;
        }
        const auto* name_def_tree = let->name_def_tree();
        std::optional<Type*> maybe_type = type_info.GetItem(name_def_tree);
        if (!maybe_type.has_value()) {
          // Should not happen, but if it does somehow we don't want to crash
          // the language server.
          LspLog() << "No type information available for: "
                   << name_def_tree->ToString() << " @ "
                   << name_def_tree->span().ToString(file_table);
          continue;
        }
        const Type& type = *maybe_type.value();
        results.push_back(verible::lsp::InlayHint{
            .position = ConvertPosToLspPosition(name_def_tree->span().limit()),
            .label = absl::StrCat(": ", type.ToInlayHintString()),
            .kind = verible::lsp::InlayHintKind::kType,
            .paddingRight = true,
        });
      }
    }
  }
  return results;
}

absl::StatusOr<std::optional<verible::lsp::Range>>
LanguageServerAdapter::PrepareRename(
    std::string_view uri, const verible::lsp::Position& position) const {
  if (ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    FileTable& file_table = parsed->import_data.file_table();

    const Pos pos = ConvertLspPositionToPos(uri, position, file_table);
    VLOG(1) << "FindDefinition; uri: " << uri << " pos: " << pos;
    std::optional<const NameDef*> maybe_definition = xls::dslx::FindDefinition(
        parsed->module(), pos, parsed->type_info(), parsed->import_data);
    if (maybe_definition.has_value()) {
      return ConvertSpanToLspRange(maybe_definition.value()->span());
    }
  }
  return std::nullopt;
}

// Generic function that renames all `NameRefs` that point at `name_def`, as
// well as `name_def` itself, under `container`.
//
// Implementation note: since `name_def` does not currently have "use" links
// maintained, this is linear in the number of nodes in `container`.
static absl::Status RenameInGeneric(
    const AstNode& container, const NameDef& name_def,
    std::string_view new_name, std::vector<verible::lsp::TextEdit>& edits) {
  // Get all the references to the name def and rename them all.
  XLS_ASSIGN_OR_RETURN(std::vector<const NameRef*> name_refs,
                       CollectNameRefsUnder(&container, &name_def));
  for (const NameRef* name_ref : name_refs) {
    edits.push_back(verible::lsp::TextEdit{
        .range = ConvertSpanToLspRange(name_ref->span()),
        .newText = std::string{new_name},
    });
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<verible::lsp::WorkspaceEdit>>
LanguageServerAdapter::Rename(std::string_view uri,
                              const verible::lsp::Position& position,
                              std::string_view new_name) const {
  std::vector<verible::lsp::TextEdit> edits;
  if (ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    FileTable& file_table = parsed->import_data.file_table();

    const Pos pos = ConvertLspPositionToPos(uri, position, file_table);
    VLOG(1) << "FindDefinition; uri: " << uri << " pos: " << pos;
    std::optional<const NameDef*> maybe_name_def = xls::dslx::FindDefinition(parsed->module(), pos, parsed->type_info(),
                              parsed->import_data);
    if (!maybe_name_def.has_value()) {
      VLOG(1) << "No definition found for attempted rename to: `" << new_name
              << "`";
      return std::nullopt;
    }

    const NameDef* name_def = maybe_name_def.value();

    // We always want to edit the original name definition to the new name.
    edits.push_back(verible::lsp::TextEdit{
        .range = ConvertSpanToLspRange(name_def->span()),
        .newText = std::string{new_name},
    });

    const auto* module = name_def->owner();
    if (AstNode* definer = name_def->definer();
        definer != nullptr && !module->IsPublicMember(*definer)) {
      // For non-public module members we can rename -- public may require
      // cross-file edits.
      XLS_RETURN_IF_ERROR(RenameInGeneric(*module, *name_def, new_name, edits));
    } else {
      // Traverse up parent links until we find a container node of interest;
      // i.e. function/module.
      const AstNode* node = name_def;
      while (true) {
        node = node->parent();
        VLOG(3) << absl::StreamFormat("Traversed to parent AST node: `%s`",
                                      node->ToString());
        if (node == nullptr) {
          return std::nullopt;
        }
        if (node->kind() == AstNodeKind::kFunction) {
          const auto* function = down_cast<const Function*>(node);
          XLS_RETURN_IF_ERROR(
              RenameInGeneric(*function, *name_def, new_name, edits));
          break;
        }
      }
    }

    nlohmann::json edits_json;
    for (const auto& edit : edits) {
      nlohmann::json o;
      verible::lsp::to_json(o, edit);
      edits_json.push_back(o);
    }

    return verible::lsp::WorkspaceEdit{
        .changes = nlohmann::json::object({
            {std::string{uri}, edits_json},
        }),
    };
  }
  return std::nullopt;
}

}  // namespace xls::dslx
