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
#include <vector>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
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

// Variant that operates on the subject of a `ColonRef`, where the whole thing
// refers to a struct member.
absl::StatusOr<std::optional<StructOrProcRef>> GetStructOrProcRefForSubject(
    const ColonRef* colon_ref, const ImportData& import_data);

// Resolves the struct base definition for the struct or proc type referred to
// by `annotation`.
absl::StatusOr<std::optional<const StructDefBase*>> GetStructOrProcDef(
    const TypeAnnotation* annotation, const ImportData& import_data);

// If `f` is in an impl, returns the struct or proc def that the impl belongs
// to; otherwise returns `nullopt`.
absl::StatusOr<std::optional<const StructDefBase*>> GetStructOrProcDef(
    const Function* f, const ImportData& import_data);

// Returns the definition of the struct or proc whose impl contains `node`, if
// any.
absl::StatusOr<std::optional<const StructDefBase*>>
GetContainingStructOrProcDef(const AstNode* node,
                             const ImportData& import_data);

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

// Returns whether `f` is a `next` function in an impl-style proc.
absl::StatusOr<bool> IsProcDefNextFunction(const Function* f,
                                           const ImportData& import_data);

// Returns whether `f` is an auto-generated `spawn` function in an impl-style
// proc.
absl::StatusOr<bool> IsProcDefSpawnFunction(const Function* f);

// Returns whether `colon_ref` is imported from a different module.
bool IsImport(const ColonRef* colon_ref);

// Returns whether the given `type` of a member of a `ProcDef` indicates that
// the member is a state element. All state members of a `ProcDef` have the
// inferred type `State<T>` where `T` is the type written by the programmer, so
// this amounts to checking if `type` is a "State-wrapped" type.
bool IsProcDefStateType(const Type& type, const ImportData& import_data);

// Returns all the members of `proc_def` that are state elements.
absl::StatusOr<std::vector<StructMemberNode*>> GetProcDefStateMembers(
    const ProcDef* proc_def, const ImportData& import_data,
    const TypeInfo& type_info);

// Returns all functions in `proc` that are constructors by signature, i.e.
// static functions returning `Self`.
absl::StatusOr<std::optional<const ProcDef*>> GetProcConstructedByFunction(
    const Function* f, const TypeInfo* ti);

// Returns whether the given function is a constructor for the proc indicated by
// `proc_def`, assuming the function is a member of the proc.
bool IsProcConstructor(const Function* function, const ProcDef* proc_def,
                       const FunctionType& function_type);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_IMPORT_UTILS_H_
