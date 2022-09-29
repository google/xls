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
#ifndef XLS_DSLX_AST_UTILS_H_
#define XLS_DSLX_AST_UTILS_H_

#include "absl/status/statusor.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Returns true if `callee` refers to a builtin function. If `callee` isn't a
// NameRef, then this always returns false.
bool IsBuiltinFn(Expr* callee);

// Returns the name of `callee` if it's a builtin function and an error
// otherwise.
absl::StatusOr<std::string> GetBuiltinName(Expr* callee);

// Finds the Function identified by the given node (either NameRef or ColonRef),
// using the associated ImportData for import Module lookup.
// The target function must have been typechecked prior to this call.
absl::StatusOr<Function*> ResolveFunction(Expr* callee,
                                          const TypeInfo* type_info);

// Finds the Proc identified by the given node (either NameRef or ColonRef),
// using the associated ImportData for import Module lookup.
// The target proc must have been typechecked prior to this call.
absl::StatusOr<Proc*> ResolveProc(Expr* callee, const TypeInfo* type_info);

// Returns the basis of the given ColonRef; either a Module for a constant
// reference or the EnumDef whose attribute is specified, or a builtin type
// (with a constant on it, a la `u7::MAX`).
absl::StatusOr<
    std::variant<Module*, EnumDef*, BuiltinNameDef*, ArrayTypeAnnotation*>>
ResolveColonRefSubject(ImportData* import_data, const TypeInfo* type_info,
                       const ColonRef* colon_ref);

// Verifies that every node's child thinks that that node is its parent.
absl::Status VerifyParentage(const Module* module);
absl::Status VerifyParentage(const AstNode* root);

// Returns the set consisting of all transitive children of the given node (as
// well as that node itself).
absl::flat_hash_set<const AstNode*> FlattenToSet(const AstNode* node);

// Returns the result of accessing a colon-ref member of a builtin type; e.g.
// `s7::MAX`.
absl::StatusOr<InterpValue> GetBuiltinNameDefColonAttr(
    const BuiltinNameDef* builtin_name_def, std::string_view attr);

absl::StatusOr<InterpValue> GetArrayTypeColonAttr(
    const ArrayTypeAnnotation* type, uint64_t constexpr_dim,
    std::string_view attr);

// TMP helper that gets the Nth type from a parameter pack.
template <int N, typename... Ts>
struct GetNth {
  using type = typename std::tuple_element<N, std::tuple<Ts...>>::type;
};

// Recursive helper for WidenVariant below -- attempts to get the Nth type from
// the narrower parameter pack and place it into a variant for ToTypes.
template <int N, typename... ToTypes, typename... FromTypes>
inline std::variant<ToTypes...> TryWidenVariant(
    const std::variant<FromTypes...>& v) {
  using TryT = typename GetNth<N, FromTypes...>::type;
  if (std::holds_alternative<TryT>(v)) {
    return std::get<TryT>(v);
  }
  if constexpr (N == 0) {
    XLS_LOG(FATAL) << "Could not find variant in FromTypes.";
  } else {
    return TryWidenVariant<N - 1, ToTypes...>(v);
  }
}

// "Widens" a variant from a smaller set of types to a larger set of types; e.g.
//
// `variant<int, double>` can be widened to `variant<int, double, std::string>`
// where `int, double` would be FromTypes and `int, double, std::string` would
// be ToTypes.
template <typename... ToTypes, typename... FromTypes>
inline std::variant<ToTypes...> WidenVariant(
    const std::variant<FromTypes...>& v) {
  return TryWidenVariant<sizeof...(FromTypes) - 1, ToTypes...>(v);
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_AST_UTILS_H_
