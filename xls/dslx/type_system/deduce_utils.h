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

#include <optional>
#include <string_view>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// If the width is known for "type", checks that "number" fits in that type.
absl::Status TryEnsureFitsInType(const Number& number, const BitsType& type);

// Record that the current function being checked has a side effect and will
// require an implicit token when converted to IR.
void UseImplicitToken(DeduceCtx* ctx);

// Returns whether "e" is a NameRef referring to the given "name_def".
bool IsNameRefTo(const Expr* e, const NameDef* name_def);

// Checks that "number" can legitmately conform to type "type".
absl::Status ValidateNumber(const Number& number, const ConcreteType& type);

// Returns the basis of the given ColonRef.
//
// In valid cases this will generally be:
// * a module
// * an enum definition
// * a builtin type (with a constant item on it, a la `u7::MAX`)
//
// Struct definitions cannot currently have constant items on them, so this will
// have to be flagged by the type checker.
absl::StatusOr<std::variant<Module*, EnumDef*, BuiltinNameDef*,
                            ArrayTypeAnnotation*, StructDef*, ColonRef*>>
ResolveColonRefSubjectForTypeChecking(ImportData* import_data,
                                      const TypeInfo* type_info,
                                      const ColonRef* colon_ref);

// Implementation of the above that can be called after type checking has been
// performed, in which case we can eliminate some of the (invalid) possibilities
// so they no longer need to be handled.
absl::StatusOr<
    std::variant<Module*, EnumDef*, BuiltinNameDef*, ArrayTypeAnnotation*>>
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
                        m->name()));
  }

  if (!std::holds_alternative<T*>(*member.value())) {
    return TypeInferenceErrorStatus(
        span, nullptr,
        absl::StrFormat(
            "Name '%s' in module `%s` refers to a %s but a %s is required",
            name, m->name(), GetModuleMemberTypeName(*member.value()),
            T::GetDebugTypeName()));
  }

  T* result = std::get<T*>(*member.value());
  XLS_RET_CHECK(result != nullptr);
  return result;
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_DEDUCE_UTILS_H_
