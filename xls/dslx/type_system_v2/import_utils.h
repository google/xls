// Copyright 2024 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_IMPORT_UTILS_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_IMPORT_UTILS_H_

#include <optional>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {

// Resolves the definition and parametrics for the struct or proc type referred
// to by `annotation`.
absl::StatusOr<std::optional<StructOrProcRef>> GetStructOrProcRef(
    const TypeAnnotation* annotation, const ImportData& import_data);

// Variant that takes a `ColonRef`. This will only yield a struct ref if the
// `ColonRef` itself refers to an actual struct or alias of one. It will yield
// `nullopt` for a `ColonRef` to an impl member. This makes it easier for the
// caller to distinguish the two scenarios.
absl::StatusOr<std::optional<StructOrProcRef>> GetStructOrProcRef(
    const ColonRef* colon_ref, const ImportData& import_data);

// Resolves the definition and parametrics for the old style Proc type referred
// to by `node`.
absl::StatusOr<std::optional<OldStyleProcRef>> GetOldStyleProcRef(
    const AstNode* node, const ImportData& import_data);

// Variant that operates on the subject of a `ColonRef`, where the whole thing
// refers to a struct member.
absl::StatusOr<std::optional<StructOrProcRef>> GetStructOrProcRefForSubject(
    const ColonRef* colon_ref, const ImportData& import_data);

// Resolves the struct base definition for the struct or proc type referred to
// by `annotation`.
absl::StatusOr<std::optional<const StructDefBase*>> GetStructOrProcDef(
    const TypeAnnotation* annotation, const ImportData& import_data);

// Returns whether the given node is a reference to a parametric struct or proc
// without sufficient parametrics specified (i.e. abstract and non-concretizable
// in the parametric sense). A node meeting the criteria can be a `TypeAlias`,
// the `NameDef` of a `TypeAlias`, or a `ColonRef`.
absl::StatusOr<bool> IsReferenceToAbstractType(const AstNode* node,
                                               const ImportData& import_data,
                                               const InferenceTable& table);

// Finds and returns a public module member for the given `ColonRef`. Returns
// an error if it doesn't exist or isn't public.
absl::StatusOr<ModuleMember> GetPublicModuleMember(const Module& module,
                                                   const ColonRef* node,
                                                   const FileTable& file_table);

// Retrieves the `ModuleInfo` for the given `ColonRef`.
absl::StatusOr<std::optional<ModuleInfo*>> GetImportedModuleInfo(
    const ColonRef* colon_ref, const ImportData& import_data);

// Gets the enum definition for the enum type referred to by `annotation`.
absl::StatusOr<std::optional<const EnumDef*>> GetEnumDef(
    const TypeAnnotation* annotation, const ImportData& import_data);

// Returns whether `colon_ref` is imported from a different module.
bool IsImport(const ColonRef* colon_ref);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_IMPORT_UTILS_H_
