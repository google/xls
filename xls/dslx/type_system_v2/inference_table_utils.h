// Copyright 2026 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_UTILS_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_UTILS_H_

#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system_v2/inference_table.h"

namespace xls::dslx {

// Validates the formal type/value kind and returns a type annotation for a
// type argument, including a ColonRef to an imported type. Expression arguments
// must already be populated in `table`; returned type annotations are borrowed.
absl::StatusOr<ExprOrType> NormalizeParametricArgument(
    const ParametricBinding& binding, ExprOrType argument,
    const InferenceTable& table, const FileTable& file_table);

// Substitutes mapped references and optional Self annotations without
// populating the resulting subtree. Cloned nodes belong to type->owner();
// mapped whole-type replacements are borrowed from their original Modules.
// When cloning is optional and no substitution is found, returns `type`.
// The caller must populate the result before resolving indirect annotations.
absl::StatusOr<const TypeAnnotation*> SubstituteTypeParametrics(
    const TypeAnnotation* type,
    const absl::flat_hash_map<const NameDef*, ExprOrType>& actual_values,
    InferenceTable& table,
    std::optional<const TypeAnnotation*> real_self_type = std::nullopt,
    bool clone_if_no_parametrics = true);

// Fabricates a `Number` node and sets the given type annotation for it in the
// inference table.
absl::StatusOr<Number*> MakeTypeCheckedNumber(
    Module& module, InferenceTable& table, const Span& span,
    const InterpValue& value, const TypeAnnotation* type_annotation);

// Variant that takes a raw `int64_t` value for the number.
absl::StatusOr<Number*> MakeTypeCheckedNumber(
    Module& module, InferenceTable& table, const Span& span, int64_t value,
    const TypeAnnotation* type_annotation);

// Variant of MakeTypeCheckedNumber that preserves enum values as ColonRefs.
absl::StatusOr<Expr*> MakeTypeCheckedNumberOrEnumValue(
    Module& module, InferenceTable& table, const Span& span,
    const InterpValue& value, const TypeAnnotation* type_annotation,
    const Type& type);

// Returns whether the given `expr` is a `ColonRef` to a type as opposed to a
// value. The determination is based on table data for `expr`; this function
// will not actually resolve and analyze the `ColonRef` itself.
bool IsColonRefWithTypeTarget(const InferenceTable& table, const Expr* expr);

// Creates a CloneReplacer that replaces any `NameRef` to a `NameDef` in `map`
// with the corresponding `ExprOrType`. This is used for replacement of
// parametric variables with values. Each time the returned replacer uses a node
// that is value in `map`, it clones it via `table.Clone()`.

// A mapped Number retains its syntax type annotation, or materializes its
// inference-table annotation when it has no syntax annotation. If neither is
// present and `add_parametric_binding_type_annotation` is true, use the formal
// parametric binding's type where needed to preserve the literal's width.
// Otherwise subsequent inference can presume the minimum width that fits the
// value. Existing concrete annotations take precedence over a formal type that
// may still contain parametric references.
CloneReplacer NameRefMapper(
    InferenceTable& table,
    const absl::flat_hash_map<const NameDef*, ExprOrType>& map,
    std::optional<Module*> target_module = std::nullopt,
    bool add_parametric_binding_type_annotation = false);

// Returns whether the given node is a reference to a parametric struct, proc,
// or sum without sufficient parametrics specified (i.e. abstract and
// non-concretizable in the parametric sense). A node meeting the criteria can
// be a `TypeAlias`, the `NameDef` of a `TypeAlias`, or a `ColonRef`.
absl::StatusOr<bool> IsReferenceToAbstractType(const AstNode* node,
                                               const ImportData& import_data,
                                               const InferenceTable& table);

// Converts a `ColonRef` with a generic type subject into a `ColonRef` with
// the resolved type as a subject. For example, in:
//   fn foo<T: type>() -> u32 { T::SOME_CONSTANT }
//   const C = foo<MyStruct>();
// If we pass in `T::SOME_CONSTANT`, with the `ParametricContext` for the `foo`
// invocation shown, the result will be `MyStruct::SOME_CONSTANT`.
absl::StatusOr<std::optional<ColonRef*>> ConvertGenericColonRefToDirect(
    const InferenceTable& table, const ImportData& import_data,
    std::optional<const ParametricContext*> parametric_context,
    const ColonRef* colon_ref);

// Determines if the given `type_variable` has any annotations in the table
// that were explicitly written in the DSLX source.
bool VariableHasAnyExplicitTypeAnnotations(
    const InferenceTable& table,
    std::optional<const ParametricContext*> parametric_context,
    const NameRef* type_variable);

// Returns the type annotation explicitly written in the DSLX source for the
// type variable associated with the given `node`, if any.
absl::StatusOr<std::optional<const TypeAnnotation*>> GetExplicitTypeAnnotation(
    const InferenceTable& table,
    std::optional<const ParametricContext*> parametric_context,
    const AstNode* node);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_UTILS_H_
