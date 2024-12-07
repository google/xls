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
#include <type_traits>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"

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

// Resolves the given TypeDefinition to a struct definition node local to the
// module.
//
// If the definition is ultimately not a struct definition (e.g. after
// traversing aliases) or is a colon-ref implying a definition outside of the
// current module, returns a status error.
absl::StatusOr<StructDef*> ResolveLocalStructDef(TypeDefinition td);

// Verifies that every node's child thinks that that node is its parent.
absl::Status VerifyParentage(const Module* module);
absl::Status VerifyParentage(const AstNode* root);

// Returns the set consisting of all transitive children of the given node (as
// well as that node itself).
absl::flat_hash_set<const AstNode*> FlattenToSet(const AstNode* node);

// Returns whether the given attribute is a known colon-ref attribute of a
// builtin bits type.
bool IsBuiltinBitsTypeAttr(std::string_view attr);

// Returns whether node n is a parametric function.
//
// "n" may be null.
bool IsParametricFunction(const AstNode* n);

// Returns whether the parent of "n" is an invocation where "n" is the callee.
//
// "n" should not be null.
bool ParentIsInvocationWithCallee(const NameRef* n);

// Returns the result of accessing a colon-ref member of a builtin type; e.g.
// `s7::MAX`.
absl::StatusOr<InterpValue> GetBuiltinNameDefColonAttr(
    const BuiltinNameDef* builtin_name_def, std::string_view attr);

absl::StatusOr<InterpValue> GetArrayTypeColonAttr(
    const ArrayTypeAnnotation* type, uint64_t constexpr_dim,
    std::string_view attr);

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
    LOG(FATAL) << "Could not find variant in FromTypes.";
  } else {
    return TryWidenVariant<N - 1, ToVariantType>(v);
  }
}

// Type trait for detecting a `std::variant`.
template <typename T>
struct is_variant : std::false_type {};
template <typename... Args>
struct is_variant<std::variant<Args...>> : std::true_type {};

// "Widens" a variant from a smaller set of types to a larger set of types; e.g.
//
// `variant<int, double>` can be widened to `variant<int, double, std::string>`
// where `int, double` would be FromTypes and `int, double, std::string` would
// be ToTypes.
template <typename T, typename... FromTypes,
          typename = std::enable_if_t<is_variant<T>::value>>
inline T WidenVariantTo(const std::variant<FromTypes...>& v) {
  return TryWidenVariant<sizeof...(FromTypes) - 1, T>(v);
}

// Enumeration of the different kinds of DSLX types whose underlying data
// representation is a bitvector.
enum BitVectorKind : uint8_t {
  // A native bit type. Examples: u32, s16, bits[3], uN[42], sN[2], bool.
  kBitType,

  // An alias of a bit type., or an alias of an alias of a bit type, etc.
  // Example: `type Foo = u32;.
  kBitTypeAlias,

  // An enum type.
  kEnumType,

  // An alias of an enum type, or an alias of an alias of a enum type, etc.
  // Example: `type MyEnumAlias = MyEnum;`
  kEnumTypeAlias,
};

// Metadata about a DSLX type whose underlying data representation is a
// bitvector.
struct BitVectorMetadata {
  std::variant<int64_t, Expr*> bit_count;
  bool is_signed;
  BitVectorKind kind;
};

// Returns metadata about the bit-vector type if `type_annotation` refers to a
// type whose underlying representation is a bit-vector. Returns std::nullopt
// otherwise.
//
// Examples include `u32`, `s10`, `uN[42]`, `bits[11]`, enums, etc, and
// aliases of these types.
std::optional<BitVectorMetadata> ExtractBitVectorMetadata(
    const TypeAnnotation* type_annotation);

// Collects all nodes under the given root.
absl::StatusOr<std::vector<AstNode*>> CollectUnder(AstNode* root,
                                                   bool want_types);

absl::StatusOr<std::vector<const NameRef*>> CollectNameRefsUnder(
    const AstNode* root, const NameDef* to);

absl::StatusOr<std::vector<const AstNode*>> CollectUnder(const AstNode* root,
                                                         bool want_types);

// Collects NameDefs referred to by NameRefs under "root".
absl::StatusOr<std::vector<const NameDef*>> CollectReferencedUnder(
    const AstNode* root, bool want_types = false);

// Wrapper around GetUnaryParametricBuiltinNames() that checks whether name_ref
// refers to a builtin name def and whether that builtin name is a parametric
// function.
bool IsBuiltinParametricNameRef(const NameRef* name_ref);

// Returns whether "node" is a "bare" number (without an explicit type
// annotation on it).
const Number* IsBareNumber(const AstNode* node, bool* is_boolean = nullptr);

// Returns whether the given "invocation" is contained within the function
// "caller".
//
// Precondition: invocation should be contained within /some/ function (i.e. not
// at module scope). Note that this could be relaxed but it's the only use case
// we need today and it makes for stronger invariant checking.
//
// Implementation note: this traverses parent links from the invocation node to
// see if we arrive at the caller node as the immediately containing function.
bool ContainedWithinFunction(const Invocation& invocation,
                             const Function& caller);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_AST_UTILS_H_
