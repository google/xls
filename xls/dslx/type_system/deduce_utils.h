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

#ifndef XLS_DSLX_TYPE_SYSTEM_DEDUCE_UTILS_H_
#define XLS_DSLX_TYPE_SYSTEM_DEDUCE_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// If the width is known for "type", checks that "number" fits in that type.
absl::Status TryEnsureFitsInType(const Number& number,
                                 const BitsLikeProperties& bits_like,
                                 const Type& type);

// Shorthand for the above where we know a prori the type is a `BitsType` (note
// that there are types that are bits-like that are not exactly `BitsType`, see
// `IsBitsLike`).
absl::Status TryEnsureFitsInBitsType(const Number& number,
                                     const BitsType& type);

// Record that the current function being checked has a side effect and will
// require an implicit token when converted to IR.
void UseImplicitToken(DeduceCtx* ctx);

// Returns whether "e" is a NameRef referring to the given "name_def".
bool IsNameRefTo(const Expr* e, const NameDef* name_def);

// Checks that "number" can legitmately conform to type "type".
absl::Status ValidateNumber(const Number& number, const Type& type);

// Returns the basis of the given ColonRef.
//
// In valid cases this will generally be:
// * a module
// * an enum definition
// * a builtin type (with a constant item on it, a la `u7::MAX`)
// * a constant defined via `impl` on a `StructDef`.
absl::StatusOr<std::variant<Module*, EnumDef*, BuiltinNameDef*,
                            ArrayTypeAnnotation*, StructDef*, ColonRef*>>
ResolveColonRefSubjectForTypeChecking(ImportData* import_data,
                                      const TypeInfo* type_info,
                                      const ColonRef* colon_ref);

// Implementation of the above that can be called after type checking has been
// performed, in which case we can eliminate some of the (invalid) possibilities
// so they no longer need to be handled.
absl::StatusOr<std::variant<Module*, EnumDef*, BuiltinNameDef*,
                            ArrayTypeAnnotation*, Impl*>>
ResolveColonRefSubjectAfterTypeChecking(ImportData* import_data,
                                        const TypeInfo* type_info,
                                        const ColonRef* colon_ref);

// Finds the Function identified by the given node (either NameRef or ColonRef),
// using the associated ImportData for import Module lookup.
// The target function must have been typechecked prior to this call.
absl::StatusOr<Function*> ResolveFunction(Expr* callee,
                                          const TypeInfo* type_info);

// Finds the Proc identified by the given node (either NameRef or ColonRef),
// using the associated ImportData for import Module lookup.
// The target proc must have been typechecked prior to this call.
absl::StatusOr<Proc*> ResolveProc(Expr* callee, const TypeInfo* type_info);

// Returns an AST node typed T from module "m", resolved via name "name".
//
// Errors are attributed to span "span".
//
// Prefer this function to Module::GetMemberOrError(), as this gives a
// positional type-inference-error as its status result when the requested
// resolution cannot be performed.
template <typename T>
inline absl::StatusOr<T*> GetMemberOrTypeInferenceError(Module* m,
                                                        std::string_view name,
                                                        const Span& span) {
  std::optional<ModuleMember*> member = m->FindMemberWithName(name);
  if (!member.has_value()) {
    return TypeInferenceErrorStatus(
        span, nullptr,
        absl::StrFormat("Name '%s' does not exist in module `%s`", name,
                        m->name()),
        *m->file_table());
  }

  if (!std::holds_alternative<T*>(*member.value())) {
    const FileTable& file_table = *m->owner()->file_table();
    return TypeInferenceErrorStatus(
        span, nullptr,
        absl::StrFormat(
            "Name '%s' in module `%s` refers to a %s but a %s is required",
            name, m->name(), GetModuleMemberTypeName(*member.value()),
            T::GetDebugTypeName()),
        file_table);
  }

  T* result = std::get<T*>(*member.value());
  XLS_RET_CHECK(result != nullptr);
  return result;
}

// Deduces the type for a ParametricBinding (via its type annotation).
absl::StatusOr<std::unique_ptr<Type>> ParametricBindingToType(
    const ParametricBinding& binding, DeduceCtx* ctx);

// Decorates parametric binding AST nodes with their deduced types.
//
// This is used externally in things like parametric instantiation of DSLX
// builtins like the higher order function "map".
absl::StatusOr<std::vector<ParametricWithType>> ParametricBindingsToTyped(
    absl::Span<ParametricBinding* const> bindings, DeduceCtx* ctx);

// Dereferences the "original" struct reference to a struct definition or
// returns an error.
//
// Args:
//  span: The span of the original construct trying to dereference the struct
//    (e.g. a StructInstance).
//  original: The original struct reference value (used in error reporting).
//  current: The current type definition being dereferenced towards a struct
//    definition (note there can be multiple levels of typedefs and such).
//  type_info: The type information that the "current" TypeDefinition resolves
//    against.
absl::StatusOr<StructDef*> DerefToStruct(const Span& span,
                                         std::string_view original_ref_text,
                                         TypeDefinition current,
                                         TypeInfo* type_info);

// Wrapper around the DerefToStruct above (that works on TypeDefinitions) that
// takes a `TypeAnnotation` instead.
absl::StatusOr<StructDef*> DerefToStruct(const Span& span,
                                         std::string_view original_ref_text,
                                         const TypeAnnotation& type_annotation,
                                         TypeInfo* type_info);

// Checks that the number of tuple elements in the name def tree matches the
// number of tuple elements in the type; if a "rest of tuple" leaf is
// present, only one is allowed, and it is not counted in the number of names.
//
// Returns the number of tuple elements (first) and the number of names that
// will be bound in the given NameDefTree (second).
//
// The latter may be less than the former if there is a "rest of tuple" leaf.
absl::StatusOr<std::pair<int64_t, int64_t>> GetTupleSizes(
    const NameDefTree* name_def_tree, const TupleType* tuple_type);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_DEDUCE_UTILS_H_
