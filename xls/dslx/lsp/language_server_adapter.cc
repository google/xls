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

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "nlohmann/json.hpp"
#include "verible/common/lsp/lsp-file-utils.h"
#include "verible/common/lsp/lsp-protocol-enums.h"
#include "verible/common/lsp/lsp-protocol.h"
#include "xls/common/casts.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/extract_module_name.h"
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/lsp/document_symbols.h"
#include "xls/dslx/lsp/find_definition.h"
#include "xls/dslx/lsp/lsp_type_utils.h"
#include "xls/dslx/lsp/lsp_uri.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/virtualizable_file_system.h"
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
  absl::StatusOr<PositionalErrorData> extracted_error =
      GetPositionalErrorData(status, std::nullopt, file_table);
  if (!extracted_error.ok()) {
    LspLog() << extracted_error.status() << "\n";
    return;  // best effort. Ignore.
  }
  const PositionalErrorData& err = *extracted_error;
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

// Implements an overlay on top of the underlying filesystem that prefers the
// language server's versions when they are present.
//
// TODO(cdleary): 2024-10-20 Note that this is not currently hooked into
// workspace file creation/deletion events explicitly, so everything comes via
// textual updates.
class LanguageServerFilesystem : public VirtualizableFilesystem {
 public:
  explicit LanguageServerFilesystem(LanguageServerAdapter& parent)
      : parent_(parent) {}

  absl::Status FileExists(const std::filesystem::path& path) final {
    LspUri uri(verible::lsp::PathToLSPUri(path.c_str()));
    auto it = parent_.vfs_contents().find(uri);
    if (it == parent_.vfs_contents().end()) {
      return xls::FileExists(path);
    }

    return absl::OkStatus();
  }

  absl::StatusOr<std::string> GetFileContents(
      const std::filesystem::path& path) final {
    // First we check if it exists in the virtual layer.
    LspUri uri(verible::lsp::PathToLSPUri(path.c_str()));
    auto it = parent_.vfs_contents().find(uri);
    if (it == parent_.vfs_contents().end()) {
      return xls::GetFileContents(path);
    }

    return it->second;
  }

  absl::StatusOr<std::filesystem::path> GetCurrentDirectory() final {
    XLS_ASSIGN_OR_RETURN(std::filesystem::path current,
                         xls::GetCurrentDirectory());
    return verible::lsp::PathToLSPUri(current.c_str());
  }

 private:
  LanguageServerAdapter& parent_;
};

LanguageServerAdapter::LanguageServerAdapter(
    LspUri stdlib, const std::vector<LspUri>& dslx_paths)
    : stdlib_(stdlib), dslx_paths_(dslx_paths) {}

LanguageServerAdapter::ParseData* LanguageServerAdapter::FindParsedForUri(
    LspUri uri) const {
  if (auto found = uri_parse_data_.find(uri); found != uri_parse_data_.end()) {
    return found->second.get();
  }
  return nullptr;
}

std::vector<std::filesystem::path>
LanguageServerAdapter::GetDslxPathsAsFilesystemPaths() const {
  std::vector<std::filesystem::path> result;
  result.reserve(dslx_paths_.size());
  for (const LspUri& path : dslx_paths_) {
    result.push_back(path.GetFilesystemPath());
  }
  return result;
}

absl::Status LanguageServerAdapter::Update(
    LspUri file_uri, std::optional<std::string_view> dslx_code) {
  // Either update or get the last contents from the virtual filesystem map.
  if (dslx_code.has_value()) {
    vfs_contents_[file_uri] = std::string{dslx_code.value()};
  } else {
    auto it = vfs_contents_.find(file_uri);
    if (it == vfs_contents_.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Could not find previous contents for file URI: ", file_uri));
    }
    dslx_code = it->second;
  }
  XLS_RET_CHECK(dslx_code.has_value());

  const absl::Time start = absl::Now();
  absl::StatusOr<std::string> module_name =
      ExtractModuleName(file_uri.GetFilesystemPath());
  if (!module_name.ok()) {
    LspLog() << "Could not determine module name from file URI: " << file_uri
             << " status: " << module_name.status() << "\n";
    return absl::OkStatus();
  }

  auto inserted = uri_parse_data_.emplace(file_uri, nullptr);
  std::unique_ptr<ParseData>& insert_value = inserted.first->second;

  std::vector<std::filesystem::path> dslx_paths_as_filesystem_paths =
      GetDslxPathsAsFilesystemPaths();

  ImportData import_data = CreateImportData(
      stdlib_.GetFilesystemPath(), dslx_paths_as_filesystem_paths,
      kAllWarningsSet, std::make_unique<LanguageServerFilesystem>(*this));

  import_data.SetImporterStackObserver(
      [&](const Span& importer_span, const std::filesystem::path& imported) {
        // Here we check that the filename as reported by the span is a valid
        // URI. When we are using the LSP we expect /all/ files in the file
        // table to be in URI form.
        std::string_view importer_filename =
            importer_span.GetFilename(import_data.file_table());
        CHECK(!absl::StartsWith(importer_filename, "file://"))
            << "importer_filename: " << importer_filename
            << " imported: " << imported;
        const auto importer_uri = LspUri::FromFilesystemPath(importer_filename);

        const LspUri imported_uri(verible::lsp::PathToLSPUri(imported.c_str()));
        import_sensitivity_.NoteImportAttempt(importer_uri, imported_uri);
      });

  std::vector<CommentData> comments;
  absl::StatusOr<TypecheckedModule> typechecked_module = ParseAndTypecheck(
      dslx_code.value(), /*path=*/file_uri.GetFilesystemPath().c_str(),
      /*module_name=*/*module_name, &import_data, &comments);

  if (typechecked_module.ok()) {
    insert_value = std::make_unique<ParseData>(
        std::move(import_data), TypecheckedModuleWithComments{
                                    .tm = std::move(typechecked_module).value(),
                                    .comments = Comments::Create(comments),
                                    .contents = std::string(*dslx_code),
                                });
  } else {
    insert_value = std::make_unique<ParseData>(std::move(import_data),
                                               typechecked_module.status());
  }

  const absl::Duration duration = absl::Now() - start;
  if (duration > absl::Milliseconds(200)) {
    LspLog() << "Parsing " << file_uri << " took " << duration << "\n";
  }

  return insert_value->status();
}

std::vector<verible::lsp::Diagnostic>
LanguageServerAdapter::GenerateParseDiagnostics(LspUri uri) const {
  std::vector<verible::lsp::Diagnostic> result;
  if (ParseData* parsed = FindParsedForUri(uri)) {
    FileTable& file_table = parsed->file_table();
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
LanguageServerAdapter::GenerateDocumentSymbols(LspUri uri) const {
  VLOG(1) << "GenerateDocumentSymbols; uri: " << uri;
  if (const ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    return ToDocumentSymbols(parsed->module());
  }
  return {};
}

absl::StatusOr<std::vector<verible::lsp::Location>>
LanguageServerAdapter::FindDefinitions(
    LspUri uri, const verible::lsp::Position& position) const {
  if (ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    FileTable& file_table = parsed->file_table();
    const Pos pos = ConvertLspPositionToPos(uri, position, file_table);
    VLOG(1) << "FindDefinition; uri: " << uri << " pos: " << pos;

    std::optional<const NameDef*> maybe_definition = xls::dslx::FindDefinition(
        parsed->module(), pos, parsed->type_info(), parsed->import_data());
    if (maybe_definition.has_value()) {
      // We've found a definition for the entity at the target `position` -- it
      // has a span we want to return as an LSP location.
      const Span& definition_span = maybe_definition.value()->span();
      VLOG(1) << "FindDefinition; span: "
              << definition_span.ToString(file_table);

      verible::lsp::Location location =
          ConvertSpanToLspLocation(definition_span, file_table);
      return std::vector<verible::lsp::Location>{location};
    }
  }
  return std::vector<verible::lsp::Location>{};
}

absl::StatusOr<std::vector<verible::lsp::TextEdit>>
LanguageServerAdapter::FormatDocument(LspUri uri) const {
  using ResultT = std::vector<verible::lsp::TextEdit>;
  if (ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    const Module& module = parsed->module();
    const std::string& dslx_code = parsed->contents();
    XLS_ASSIGN_OR_RETURN(std::string new_contents,
                         AutoFmt(parsed->import_data().vfs(), module,
                                 parsed->comments(), dslx_code));
    return ResultT{
        verible::lsp::TextEdit{.range = ConvertSpanToLspRange(module.span()),
                               .newText = new_contents}};
  }
  return ResultT{};
}

std::vector<verible::lsp::DocumentLink>
LanguageServerAdapter::ProvideImportLinks(LspUri uri) const {
  std::vector<verible::lsp::DocumentLink> result;
  if (ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    const Module& module = parsed->module();
    for (const auto& [_, import_node] : module.GetImportByName()) {
      const ImportTokens tok(import_node->subject());
      absl::StatusOr<ModuleInfo*> info = parsed->import_data().Get(tok);
      if (!info.ok()) {
        continue;
      }
      verible::lsp::DocumentLink link = {
          .range = ConvertSpanToLspRange(import_node->name_def().span()),
          .target = verible::lsp::PathToLSPUri(info.value()->path().c_str()),
          .has_target = true,
      };
      result.emplace_back(link);
    }
  }
  return result;
}

absl::StatusOr<std::vector<verible::lsp::InlayHint>>
LanguageServerAdapter::InlayHint(LspUri uri,
                                 const verible::lsp::Range& range) const {
  std::vector<verible::lsp::InlayHint> results;
  if (ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    FileTable& file_table = parsed->file_table();
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
          // This can happen because we have a parametric function -- we don't
          // have concrete types because it could be instantiated in different
          // ways from different invocations.
          VLOG(5) << "InlayHint; no type information available for: "
                  << name_def_tree->ToString() << " @ "
                  << name_def_tree->span().ToString(file_table) << " within `"
                  << let->ToString() << "`";
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
    LspUri uri, const verible::lsp::Position& position) const {
  if (ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    FileTable& file_table = parsed->file_table();

    const Pos pos = ConvertLspPositionToPos(uri, position, file_table);
    VLOG(1) << "FindDefinition; uri: " << uri << " pos: " << pos;
    std::optional<const NameDef*> maybe_definition = xls::dslx::FindDefinition(
        parsed->module(), pos, parsed->type_info(), parsed->import_data());
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
LanguageServerAdapter::Rename(LspUri uri,
                              const verible::lsp::Position& position,
                              std::string_view new_name) const {
  std::vector<verible::lsp::TextEdit> edits;
  if (ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    FileTable& file_table = parsed->file_table();

    const Pos pos = ConvertLspPositionToPos(uri, position, file_table);
    VLOG(1) << "FindDefinition; uri: " << uri << " pos: " << pos;
    std::optional<const NameDef*> maybe_name_def = xls::dslx::FindDefinition(
        parsed->module(), pos, parsed->type_info(), parsed->import_data());
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
            {std::string{uri.GetStringView()}, edits_json},
        }),
    };
  }
  return std::nullopt;
}

absl::StatusOr<std::vector<verible::lsp::DocumentHighlight>>
LanguageServerAdapter::DocumentHighlight(
    LspUri uri, const verible::lsp::Position& position) const {
  if (ParseData* parsed = FindParsedForUri(uri); parsed && parsed->ok()) {
    FileTable& file_table = parsed->file_table();
    const Pos pos = ConvertLspPositionToPos(uri, position, file_table);
    VLOG(1) << "FindDefinition; uri: " << uri << " pos: " << pos;
    const Module& module = parsed->module();
    std::optional<const NameDef*> maybe_definition = xls::dslx::FindDefinition(
        module, pos, parsed->type_info(), parsed->import_data());
    if (maybe_definition.has_value()) {
      const NameDef* name_def = maybe_definition.value();
      const Span& definition_span = name_def->span();
      std::vector<verible::lsp::DocumentHighlight> highlights = {
          verible::lsp::DocumentHighlight{
              .range = ConvertSpanToLspRange(definition_span),
          },
      };
      XLS_ASSIGN_OR_RETURN(std::vector<const NameRef*> refs,
                           CollectNameRefsUnder(&module, name_def));
      for (const NameRef* ref : refs) {
        highlights.push_back(verible::lsp::DocumentHighlight{
            .range = ConvertSpanToLspRange(ref->span()),
        });
      }
      return highlights;
    }
  }
  return std::vector<verible::lsp::DocumentHighlight>{};
}

}  // namespace xls::dslx
