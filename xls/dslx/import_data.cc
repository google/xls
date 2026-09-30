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

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode_cache_interface.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_record.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"

namespace xls::dslx {

/* static */ absl::StatusOr<ImportTokens> ImportTokens::FromString(
    std::string_view module_name) {
  return ImportTokens(absl::StrSplit(module_name, '.'));
}

absl::StatusOr<ModuleInfo*> ImportData::Get(const ImportTokens& subject) const {
  auto it = modules_.find(subject);
  if (it == modules_.end()) {
    return absl::NotFoundError("Module information was not found for import " +
                               subject.ToString());
  }
  return it->second.get();
}

absl::StatusOr<InferenceTableConverter*> ImportData::GetInferenceTableConverter(
    Module* module) {
  const auto it = module_to_inference_table_converter_.find(module);
  if (it == module_to_inference_table_converter_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No converter exists for module: ", module->name()));
  }
  return it->second;
}

absl::StatusOr<Module*> ImportData::GetBuiltinStubsModule() const {
  if (builtin_stubs_module_ == nullptr) {
    return absl::NotFoundError("Builtin stubs module not loaded yet.");
  }
  return builtin_stubs_module_;
}

absl::StatusOr<InferenceTableConverter*> ImportData::GetInferenceTableConverter(
    std::string_view module_name) {
  XLS_ASSIGN_OR_RETURN(ImportTokens import_tokens,
                       ImportTokens::FromString(module_name));
  XLS_ASSIGN_OR_RETURN(ModuleInfo * info, Get(import_tokens));
  return GetInferenceTableConverter(&info->module());
}

void ImportData::KeepAlive(std::unique_ptr<ModuleInfo> module_info) {
  discarded_modules_.push_back(std::move(module_info));
}

absl::StatusOr<ModuleInfo*> ImportData::Put(
    const ImportTokens& subject, std::unique_ptr<ModuleInfo> module_info) {
  auto* pmodule_info = module_info.get();
  auto [it, inserted] = modules_.emplace(subject, std::move(module_info));
  if (!inserted) {
    return absl::InvalidArgumentError(
        "Module is already loaded for import of " + subject.ToString());
  }
  if (pmodule_info->inference_table_converter() != nullptr) {
    SetInferenceTableConverter(&pmodule_info->module(),
                               pmodule_info->inference_table_converter());
  }
  path_to_module_info_[std::string{pmodule_info->path()}] = pmodule_info;
  if (pmodule_info->builtin_stubs()) {
    builtin_stubs_module_ = &pmodule_info->module();
  }
  return pmodule_info;
}

absl::StatusOr<TypeInfo*> ImportData::GetRootTypeInfoForNode(
    const AstNode* node) {
  XLS_RET_CHECK(node != nullptr);
  return type_info_owner().GetRootTypeInfo();
}

absl::StatusOr<const TypeInfo*> ImportData::GetRootTypeInfoForNode(
    const AstNode* node) const {
  XLS_ASSIGN_OR_RETURN(
      TypeInfo * ti,
      const_cast<ImportData*>(this)->GetRootTypeInfoForNode(node));
  return ti;
}

absl::StatusOr<TypeInfo*> ImportData::GetRootTypeInfo() {
  return type_info_owner().GetRootTypeInfo();
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
  CHECK(it.second) << "Module already had top level bindings: "
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
                        span.ToString(file_table_), module->name()));
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
                        span.ToString(file_table_), module->name()));
  }
  return struct_def;
}

absl::StatusOr<const ProcDef*> ImportData::FindProcDef(const Span& span) const {
  XLS_ASSIGN_OR_RETURN(const Module* module, FindModule(span));
  const ProcDef* proc_def = module->FindProcDef(span);
  if (proc_def == nullptr) {
    return absl::NotFoundError(
        absl::Substitute("Could not find proc def @ $0 within module $1",
                         span.ToString(file_table_), module->name()));
  }
  return proc_def;
}

absl::StatusOr<const SumDef*> ImportData::FindSumDef(const Span& span) const {
  XLS_ASSIGN_OR_RETURN(const Module* module, FindModule(span));
  const SumDef* sum_def = module->FindSumDef(span);
  if (sum_def == nullptr) {
    return absl::NotFoundError(
        absl::Substitute("Could not find sum def @ $0 within module $1",
                         span.ToString(file_table_), module->name()));
  }
  return sum_def;
}

absl::StatusOr<const Module*> ImportData::FindModule(const Span& span) const {
  auto it = path_to_module_info_.find(span.GetFilename(file_table_));
  if (it == path_to_module_info_.end()) {
    std::vector<std::string> paths;
    for (const auto& [path, module_info] : path_to_module_info_) {
      paths.push_back(std::string(path));
    }
    return absl::NotFoundError(
        absl::StrCat("Could not find module: ", span.GetFilename(file_table_),
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
        AstNodeKindToString(kind), span.ToString(file_table_), module->name()));
  }
  return node;
}

absl::Status ImportData::AddToImporterStack(
    const Span& importer_span, const std::filesystem::path& imported) {
  VLOG(3) << "Checking import span: " << importer_span.ToString(file_table());

  ImportRecord new_import_record{imported, importer_span};

  // Note: linear scan over importers for simplicity, this will likely need to
  // improve as we scale.
  for (size_t i = 0; i < importer_stack_.size(); ++i) {
    const ImportRecord& existing = importer_stack_.at(i);
    if (imported == existing.imported) {
      std::vector<ImportRecord> cycle(importer_stack_.begin() + i,
                                      importer_stack_.end());
      cycle.push_back(new_import_record);
      return RecursiveImportErrorStatus(importer_span, existing.imported_from,
                                        cycle, file_table());
    }
  }

  if (importer_stack_observer_ != nullptr) {
    importer_stack_observer_(importer_span, imported);
  }

  VLOG(3) << "Adding import span to stack: "
          << importer_span.ToString(file_table());
  importer_stack_.push_back(new_import_record);
  return absl::OkStatus();
}

absl::Status ImportData::PopFromImporterStack(const Span& import_span) {
  XLS_RET_CHECK(!importer_stack_.empty());
  XLS_RET_CHECK(import_span == importer_stack_.back().imported_from);
  VLOG(3) << "Popping import span from stack: "
          << importer_stack_.back().imported_from.ToString(file_table());
  importer_stack_.pop_back();
  return absl::OkStatus();
}

namespace {

std::string PathToDottedModuleId(std::string_view raw) {
  std::string_view s = absl::StripSuffix(raw, ".x");
  while (absl::StartsWith(s, "./")) {
    s.remove_prefix(2);
  }
  while (absl::StartsWith(s, "/")) {
    s.remove_prefix(1);
  }
  return absl::StrReplaceAll(s, {{"/", "."}});
}

std::optional<std::string_view> StripPathPrefix(std::string_view full_path,
                                                std::string_view prefix) {
  while (absl::StartsWith(full_path, "./")) {
    full_path.remove_prefix(2);
  }
  while (absl::StartsWith(prefix, "./")) {
    prefix.remove_prefix(2);
  }
  while (absl::EndsWith(prefix, "/")) {
    prefix.remove_suffix(1);
  }
  if (prefix.empty()) {
    return std::nullopt;
  }
  if (absl::StartsWith(full_path, prefix) && full_path.size() > prefix.size() &&
      full_path[prefix.size()] == '/') {
    std::string_view remainder = full_path.substr(prefix.size() + 1);
    while (absl::StartsWith(remainder, "/")) {
      remainder.remove_prefix(1);
    }
    return remainder;
  }
  return std::nullopt;
}

}  // namespace

bool ImportData::ModuleMatchesScope(
    std::string_view scope_id, std::string_view module_name,
    const std::filesystem::path& module_path) const {
  std::string norm_scope = PathToDottedModuleId(scope_id);
  if (norm_scope.empty()) {
    return false;
  }
  if (PathToDottedModuleId(module_name) == norm_scope) {
    return true;
  }
  if (!module_path.empty()) {
    std::string path_str = module_path.generic_string();
    if (PathToDottedModuleId(path_str) == norm_scope) {
      return true;
    }
    for (const std::filesystem::path& search_path : additional_search_paths_) {
      if (std::optional<std::string_view> rel =
              StripPathPrefix(path_str, search_path.generic_string());
          rel.has_value() && PathToDottedModuleId(*rel) == norm_scope) {
        return true;
      }
    }
    if (!stdlib_path_.empty()) {
      if (std::optional<std::string_view> rel =
              StripPathPrefix(path_str, stdlib_path_.generic_string());
          rel.has_value() && PathToDottedModuleId(*rel) == norm_scope) {
        return true;
      }
    }
    if (vfs_ != nullptr) {
      if (absl::StatusOr<std::filesystem::path> cwd =
              vfs_->GetCurrentDirectory();
          cwd.ok() && !cwd->empty()) {
        if (std::optional<std::string_view> rel =
                StripPathPrefix(path_str, cwd->generic_string());
            rel.has_value() && PathToDottedModuleId(*rel) == norm_scope) {
          return true;
        }
      }
    }
  }
  return false;
}

absl::Status ImportData::RegisterConfiguredValues(
    absl::Span<const std::string> configured_values) {
  for (const std::string& item : configured_values) {
    std::vector<std::string> key_value =
        absl::StrSplit(item, absl::MaxSplits(':', 1));
    if (key_value.size() != 2) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Configured value '%s' is not in the form 'key:value'.", item));
    }
    std::string_view lhs = key_value[0];
    const std::string& rhs = key_value[1];

    ConfiguredValueGroup new_group;
    new_group.value = rhs;
    if (absl::StrContains(lhs, '@')) {
      std::vector<std::string_view> key_scope =
          absl::StrSplit(lhs, absl::MaxSplits('@', 1));
      new_group.key = std::string(key_scope[0]);
      for (std::string_view s : absl::StrSplit(key_scope[1], '+')) {
        if (!s.empty()) {
          new_group.scope_modules.push_back(std::string(s));
        }
      }
      if (new_group.scope_modules.empty()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Configured value '%s' has empty module scope after '@'.", item));
      }
    } else {
      new_group.key = std::string(lhs);
    }
    if (new_group.key.empty()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Configured value '%s' has an empty key.", item));
    }

    bool duplicate = false;
    for (const auto& existing : configured_value_groups_) {
      if (existing.key != new_group.key) {
        continue;
      }
      if (existing.scope_modules.empty() && new_group.scope_modules.empty()) {
        if (existing.value == new_group.value) {
          duplicate = true;
          break;
        }
        return absl::InvalidArgumentError(
            absl::StrFormat("Conflicting unscoped configured values for key "
                            "'%s': '%s' vs '%s'.",
                            new_group.key, existing.value, new_group.value));
      }
      if (!existing.scope_modules.empty() && !new_group.scope_modules.empty()) {
        if (existing.scope_modules == new_group.scope_modules &&
            existing.value == new_group.value) {
          duplicate = true;
          break;
        }
        for (const std::string& s1 : existing.scope_modules) {
          for (const std::string& s2 : new_group.scope_modules) {
            if (PathToDottedModuleId(s1) == PathToDottedModuleId(s2) &&
                existing.value != new_group.value) {
              return absl::InvalidArgumentError(absl::StrFormat(
                  "Conflicting configured values for key '%s' on module '%s': "
                  "'%s' vs '%s'.",
                  new_group.key, s1, existing.value, new_group.value));
            }
          }
        }
      }
    }
    if (!duplicate) {
      configured_value_groups_.push_back(std::move(new_group));
    }
  }
  return absl::OkStatus();
}

absl::flat_hash_map<std::string, std::string>
ImportData::ResolveConfiguredValuesForModule(
    std::string_view module_name, const std::filesystem::path& module_path,
    bool is_entry_module) {
  absl::flat_hash_map<std::string, std::string> result;
  for (const auto& group : configured_value_groups_) {
    if (!group.scope_modules.empty()) {
      for (const std::string& scope_id : group.scope_modules) {
        if (ModuleMatchesScope(scope_id, module_name, module_path)) {
          result[group.key] = group.value;
          break;
        }
      }
    }
  }
  if (is_entry_module) {
    for (const auto& group : configured_value_groups_) {
      if (group.scope_modules.empty()) {
        result[group.key] = group.value;
      }
    }
  }
  return result;
}

}  // namespace xls::dslx
