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

#ifndef XLS_DSLX_FRONTEND_AST_UTILS_H_
#define XLS_DSLX_FRONTEND_AST_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Returns true if `callee` refers to a builtin function. If `callee` isn't a
// NameRef, then this always returns false.
//
// If "target" is provided, only returns true iff it is a builtin function with
// the given "target" name.
bool IsBuiltinFn(Expr* callee,
                 std::optional<std::string_view> target = std::nullopt);

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

// Returns the indentation level of the given AST node.
//
// That is, the contents of the AST node when formatted (flat) should be
// indented by kSpacesPerIndent * $retval.
//
// This is used for determining indentation level at an arbitrary point in the
// AST for formatting.
int64_t DetermineIndentLevel(const AstNode& n);

// -- Template Metaprogramming helpers for dealing with AST node variants

// TMP helper that gets the Nth type from a parameter pack.
template <int N, typename... Ts>
struct GetNth {
  using type = typename std::tuple_element<N, std::tuple<Ts...>>::type;
};

// Recursive helper for WidenVariant below.
//
// * Attempts to get the Nth type from the narrower parameter pack and place it
//   into a variant for ToTypes.
// * If it doesn't hold that alternative, recurses to try to get the `N-1`th
//   type.
//
// Example invocation:
//
//    std::variant<uint64_t, double> v = uint64_t{42};
//    using ToVariantType = std::variant<uint64_t, double, int64_t>;
//    ToVariantType widened = TryWidenVariant<1, ToVariantType>(v);
template <int N, typename ToVariantType, typename... FromTypes>
inline ToVariantType TryWidenVariant(const std::variant<FromTypes...>& v) {
  using TryT = typename GetNth<N, FromTypes...>::type;
  if (std::holds_alternative<TryT>(v)) {
    return std::get<TryT>(v);
  }
  if constexpr (N == 0) {
    XLS_LOG(FATAL) << "Could not find variant in FromTypes.";
  } else {
    return TryWidenVariant<N - 1, ToVariantType>(v);
  }
}

// Type trait for detecting a `std::variant`.
template <typename T>
struct is_variant : std::false_type {};
template <typename... Args>
struct is_variant<std::variant<Args...>> : std::true_type {};

template <typename... ToTypes, typename... FromTypes,
          typename = std::enable_if_t<sizeof...(ToTypes) != 1, int>>
inline std::variant<ToTypes...> TryWidenVariant(
    const std::variant<FromTypes...>& v) {
  return TryWidenVariant<sizeof...(FromTypes) - 1, std::variant<ToTypes...>>(v);
}

// "Widens" a variant from a smaller set of types to a larger set of types; e.g.
//
// `variant<int, double>` can be widened to `variant<int, double, std::string>`
// where `int, double` would be FromTypes and `int, double, std::string` would
// be ToTypes.
template <typename... ToTypes, typename... FromTypes>
inline std::variant<ToTypes...> WidenVariant(
    const std::variant<FromTypes...>& v) {
  return TryWidenVariant<sizeof...(FromTypes) - 1, std::variant<ToTypes...>>(v);
}

template <typename T, typename... FromTypes,
          typename = std::enable_if_t<is_variant<T>::value>>
inline T WidenVariantTo(const std::variant<FromTypes...>& v) {
  return TryWidenVariant<sizeof...(FromTypes) - 1, T>(v);
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_AST_UTILS_H_
