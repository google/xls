// Copyright 2021 The XLS Authors
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

#include "xls/dslx/import_data.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/errors.h"

namespace xls::dslx {

/* static */ absl::StatusOr<ImportTokens> ImportTokens::FromString(
    std::string_view module_name) {
  return ImportTokens(absl::StrSplit(module_name, '.'));
}

absl::StatusOr<ModuleInfo*> ImportData::Get(const ImportTokens& subject) {
  auto it = modules_.find(subject);
  if (it == modules_.end()) {
    return absl::NotFoundError("Module information was not found for import " +
                               subject.ToString());
  }
  return it->second.get();
}

absl::StatusOr<ModuleInfo*> ImportData::Put(
    const ImportTokens& subject, std::unique_ptr<ModuleInfo> module_info) {
  auto* pmodule_info = module_info.get();
  auto [it, inserted] = modules_.emplace(subject, std::move(module_info));
  if (!inserted) {
    return absl::InvalidArgumentError(
        "Module is already loaded for import of " + subject.ToString());
  }
  path_to_module_info_[std::string(pmodule_info->path())] = pmodule_info;
  return pmodule_info;
}

absl::StatusOr<TypeInfo*> ImportData::GetRootTypeInfoForNode(
    const AstNode* node) {
  XLS_RET_CHECK(node != nullptr);
  return type_info_owner().GetRootTypeInfo(node->owner());
}

absl::StatusOr<const TypeInfo*> ImportData::GetRootTypeInfoForNode(
    const AstNode* node) const {
  XLS_ASSIGN_OR_RETURN(
      TypeInfo * ti,
      const_cast<ImportData*>(this)->GetRootTypeInfoForNode(node));
  return ti;
}

absl::StatusOr<TypeInfo*> ImportData::GetRootTypeInfo(const Module* module) {
  return type_info_owner().GetRootTypeInfo(module);
}

InterpBindings& ImportData::GetOrCreateTopLevelBindings(Module* module) {
  auto it = top_level_bindings_.find(module);
  if (it == top_level_bindings_.end()) {
    it = top_level_bindings_
             .emplace(module,
                      std::make_unique<InterpBindings>(/*parent=*/nullptr))
             .first;
  }
  return *it->second;
}

void ImportData::SetTopLevelBindings(Module* module,
                                     std::unique_ptr<InterpBindings> tlb) {
  auto it = top_level_bindings_.emplace(module, std::move(tlb));
  XLS_CHECK(it.second) << "Module already had top level bindings: "
                       << module->name();
}

void ImportData::SetBytecodeCache(
    std::unique_ptr<BytecodeCacheInterface> bytecode_cache) {
  bytecode_cache_ = std::move(bytecode_cache);
}

BytecodeCacheInterface* ImportData::bytecode_cache() {
  return bytecode_cache_.get();
}

absl::StatusOr<const EnumDef*> ImportData::FindEnumDef(const Span& span) const {
  XLS_ASSIGN_OR_RETURN(const Module* module, FindModule(span));
  const EnumDef* enum_def = module->FindEnumDef(span);
  if (enum_def == nullptr) {
    return absl::NotFoundError(
        absl::StrFormat("Could not find enum def @ %s within module %s",
                        span.ToString(), module->name()));
  }
  return enum_def;
}

absl::StatusOr<const StructDef*> ImportData::FindStructDef(
    const Span& span) const {
  XLS_ASSIGN_OR_RETURN(const Module* module, FindModule(span));
  const StructDef* struct_def = module->FindStructDef(span);
  if (struct_def == nullptr) {
    return absl::NotFoundError(
        absl::StrFormat("Could not find struct def @ %s within module %s",
                        span.ToString(), module->name()));
  }
  return struct_def;
}

absl::StatusOr<const Module*> ImportData::FindModule(const Span& span) const {
  auto it = path_to_module_info_.find(span.filename());
  if (it == path_to_module_info_.end()) {
    std::vector<std::string> paths;
    for (const auto& [path, module_info] : path_to_module_info_) {
      paths.push_back(std::string(path));
    }
    return absl::NotFoundError(
        absl::StrCat("Could not find module: ", span.filename(),
                     "; have: ", absl::StrJoin(paths, ", ")));
  }
  return &it->second->module();
}

absl::StatusOr<const AstNode*> ImportData::FindNode(AstNodeKind kind,
                                                    const Span& span) const {
  XLS_ASSIGN_OR_RETURN(const Module* module, FindModule(span));
  const AstNode* node = module->FindNode(kind, span);
  if (node == nullptr) {
    return absl::NotFoundError(absl::StrFormat(
        "Could not find node with kind %s @ %s within module %s",
        AstNodeKindToString(kind), span.ToString(), module->name()));
  }
  return node;
}

absl::Status ImportData::AddToImporterStack(const Span& import_span,
                                            std::filesystem::path imported) {
  XLS_VLOG(3) << "Checking import span: " << import_span;

#if 0
  // Try to get a realpath to help properly unique-ify filesystem entities --
  // some filesystems may not support this, so we attempt and fall back to
  // whatever path was discovered if we can't obtain a realpath.
  absl::StatusOr<std::filesystem::path> imported_realpath_or =
      GetRealPath(std::string(imported));
  if (imported_realpath_or.ok()) {
    imported = imported_realpath_or.value();
  }
#endif

  ImportRecord new_import_record{imported, import_span};

  // Note: linear scan over importers for simplicity, this will likely need to
  // improve as we scale.
  for (size_t i = 0; i < importer_stack_.size(); ++i) {
    const ImportRecord& existing = importer_stack_.at(i);
    if (imported == existing.imported) {
      std::vector<ImportRecord> cycle(importer_stack_.begin() + i,
                                      importer_stack_.end());
      cycle.push_back(new_import_record);
      return RecursiveImportErrorStatus(import_span, existing.imported_from,
                                        cycle);
    }
  }

  XLS_VLOG(3) << "Adding import span to stack: " << import_span;
  importer_stack_.push_back(new_import_record);
  return absl::OkStatus();
}

absl::Status ImportData::PopFromImporterStack(const Span& import_span) {
  XLS_RET_CHECK(!importer_stack_.empty());
  XLS_RET_CHECK_EQ(import_span, importer_stack_.back().imported_from);
  XLS_VLOG(3) << "Popping import span from stack: "
              << importer_stack_.back().imported_from;
  importer_stack_.pop_back();
  return absl::OkStatus();
}

}  // namespace xls::dslx
