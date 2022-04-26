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

#include "absl/strings/str_split.h"
#include "xls/common/status/ret_check.h"

namespace xls::dslx {

/* static */ absl::StatusOr<ImportTokens> ImportTokens::FromString(
    absl::string_view module_name) {
  return ImportTokens(absl::StrSplit(module_name, '.'));
}

absl::StatusOr<const ModuleInfo*> ImportData::Get(
    const ImportTokens& subject) const {
  auto it = cache_.find(subject);
  if (it == cache_.end()) {
    return absl::NotFoundError("Module information was not found for import " +
                               subject.ToString());
  }
  return &it->second;
}

absl::StatusOr<const ModuleInfo*> ImportData::Put(const ImportTokens& subject,
                                                  ModuleInfo module_info) {
  auto it = cache_.insert({subject, std::move(module_info)});
  if (!it.second) {
    return absl::InvalidArgumentError(
        "Module is already loaded for import of " + subject.ToString());
  }
  return &it.first->second;
}

absl::StatusOr<TypeInfo*> ImportData::GetRootTypeInfoForNode(
    const AstNode* node) {
  XLS_RET_CHECK(node != nullptr);
  return type_info_owner().GetRootTypeInfo(node->owner());
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

}  // namespace xls::dslx
