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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_UTILS_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

struct StructOrProcRef {
  const StructDefBase* def;
  std::vector<ExprOrType> parametrics;
  std::optional<const StructInstanceBase*> instantiator;
};

// The signedness and bit count extracted from a `TypeAnnotation`. The
// `TypeAnnotation` may use primitive values or exprs; we convey the
// representation as is.
struct SignednessAndBitCountResult {
  std::variant<bool, const Expr*> signedness;
  std::variant<int64_t, const Expr*> bit_count;
};

// An `Expr`-ified rendition of the `StartAndWidth` struct used by `TypeInfo`.
// In situations where the values are constexpr, this struct uses the constant
// value instead of an expr.
struct StartAndWidthExprs {
  std::variant<int64_t, const Expr*> start;
  std::variant<int64_t, const Expr*> width;
};

struct InterpValueWithTypeAnnotation {
  const TypeAnnotation* type_annotation;
  InterpValue value;
};

// Creates an annotation for `uN[bit_count]` or `sN[bit_count]` depending on the
// value of `is_signed`.
TypeAnnotation* CreateUnOrSnAnnotation(Module& module, const Span& span,
                                       bool is_signed, int64_t bit_count);

// Variant that uses an `Expr` for the bit count.
TypeAnnotation* CreateUnOrSnAnnotation(Module& module, const Span& span,
                                       bool is_signed, Expr* bit_count);

// Creates an annotation for an "element" of a `uN[N]` or `sN[N]` type, i.e.
// just the `uN` or the `sN` piece with no dimension.
TypeAnnotation* CreateUnOrSnElementAnnotation(Module& module, const Span& span,
                                              bool is_signed);

// Creates a `bool` type annotation.
TypeAnnotation* CreateBoolAnnotation(Module& module, const Span& span);

// Creates a `u32` type annotation.
//
// TODO(https://github.com/google/xls/issues/450): 2025-01-23 In the future we
// should perhaps be using "usize" instead to characterize the places we need
// this, e.g. a type that we evaluate array dimensions and similar undecorated
// values to.
TypeAnnotation* CreateU32Annotation(Module& module, const Span& span);

// Creates an `s32` type annotation.
TypeAnnotation* CreateS32Annotation(Module& module, const Span& span);

// Creates a `u8` type annotation.
TypeAnnotation* CreateU8Annotation(Module& module, const Span& span);

// Creates a type annotation based on the name def of a built-in type like
// `u32`.
TypeAnnotation* CreateBuiltinTypeAnnotation(Module& module,
                                            BuiltinNameDef* name_def,
                                            const Span& span);

// Creates an annotation referring to the given struct definition with the given
// parametric arguments.
TypeAnnotation* CreateStructAnnotation(
    Module& module, StructDef* def, std::vector<ExprOrType> parametrics,
    std::optional<const StructInstanceBase*> instantiator);

// Variant that converts a `StructOrProcRef` into an annotation.
TypeAnnotation* CreateStructAnnotation(Module& module,
                                       const StructOrProcRef& ref);

// Returns the element channel type of the given channel array type.
ChannelTypeAnnotation* GetChannelArrayElementType(
    Module& module, const ChannelTypeAnnotation* channel_array_type);

// Returns the signedness and bit count from the given type annotation, if it is
// a bits-like annotation; otherwise, returns an error. The kind of error is
// either:
// - Not found, if the annotation is not bits-like.
// - Invalid argument, if the annotation is erroneously missing dimensions and
//   `ignore_missing_dimensions` is false.
//
// The `ignore_missing_dimensions` flag is useful for a caller that does its own
// traversal or zipping of the internals of type annotations like `uN[32]`, and
// wants to be able to extract what information it can from just the `uN` piece.
absl::StatusOr<SignednessAndBitCountResult> GetSignednessAndBitCount(
    const TypeAnnotation* annotation, bool ignore_missing_dimensions = false);

// Variant that returns an error suitable to present to the user on any failure.
// The `default_error_factory` is used to convert any "not found" error (i.e.
// "annotation is not bits-like") into a context-relevant error for the user,
// If the annotation is erroneously missing dimensions, then a user-facing error
// is internally provided.
absl::StatusOr<SignednessAndBitCountResult>
GetSignednessAndBitCountWithUserFacingError(
    const TypeAnnotation* annotation, const FileTable& file_table,
    absl::AnyInvocable<absl::Status()> default_error_factory);

// Creates a type annotation for a literal `number`, which is sized to fit the
// value. If the `number` is negative, then the annotation will be signed and
// have room for a sign bit; otherwise, it will not.
absl::StatusOr<TypeAnnotation*> CreateAnnotationSizedToFit(
    Module& module, const Number& number);

// Creates a type annotation for the unit tuple, `()`.
TypeAnnotation* CreateUnitTupleAnnotation(Module& module, const Span& span);

// Creates a type annotation for the given `function`.
FunctionTypeAnnotation* CreateFunctionTypeAnnotation(Module& module,
                                                     const Function& function);

// Returns the explicit or implied return type of `fn`.
const TypeAnnotation* GetReturnType(Module& module, const Function& fn);

// Acts like `dynamic_cast<const ArrayTypeAnnotation*>(annotation)` but only
// succeeds if the annotation is for a true array, as opposed to a bits-like
// type expressed as an array (e.g. `uN` or `xN`).
const ArrayTypeAnnotation* CastToNonBitsArrayTypeAnnotation(
    const TypeAnnotation* annotation);

// Verifies that all `bindings` either have a value in `actual_parametrics` or
// a default expression. Note that this is not a requirement in all situations
// where parametrics can be explicitly passed. In situations where they can be
// inferred, this is only a requirement after inference.
absl::Status VerifyAllParametricsSatisfied(
    const std::vector<ParametricBinding*>& bindings,
    const std::vector<ExprOrType>& actual_parametrics,
    std::string_view binding_owner_identifier, const Span& error_span,
    const FileTable& file_table);

// Creates an `Expr` representing `element_count<annotation>()`.
Expr* CreateElementCountInvocation(Module& module, TypeAnnotation* annotation);

// Creates an `Expr` representing `element_count<lhs>() +
// element_count<rhs>()`.
Expr* CreateElementCountSum(Module& module, TypeAnnotation* lhs,
                            TypeAnnotation* rhs);

// Creates an `Expr` representing `element_count<lhs>() +
// element_count<rhs>()`.
Expr* CreateElementCountOffset(Module& module, TypeAnnotation* lhs,
                               int64_t offset);

// Creates a literal representing 0 without a type annotation.
Number* CreateUntypedZero(Module& module, const Span& span);

// Creates an Expr calculating the number of elements in a Range. If the number
// is negative (i.e. `range->end() < range.start()`) the result is undefined,
// which can be caught by const evaluation of the range.
Expr* CreateRangeElementCount(Module& module, const Range* range);

// Returns the type annotation and and value of the builtin member named
// `member_name` of the given `object_type` (e.g. `uN[32]::MAX`).
absl::StatusOr<InterpValueWithTypeAnnotation> GetBuiltinMember(
    Module& module, bool is_signed, uint32_t bit_count,
    std::string_view member_name, const Span& span,
    std::string_view object_type_for_error, const FileTable& file_table);

// Returns true if the type annotation is a builtin Token
bool IsToken(const TypeAnnotation* annotation);

// If the annotation is a Token BuiltinTypeAnnotation, returns it. Otherwise,
// returns nullptr.
const BuiltinTypeAnnotation* CastToTokenType(const TypeAnnotation* annotation);

// Creates a new FunctionTypeAnnotation from the given one, with the last
// and parameter of the original signature repeated `count` times (which
// can be 0, meaning, remove the last parameter).
const FunctionTypeAnnotation* ExpandVarargs(
    Module& module, const FunctionTypeAnnotation* signature, int count);

// Returns whether `annotation` is a part of a bits-like type annotation that is
// unusable without being combined into a larger annotation. For example,
// returns true for `uN` but false for `uN[4]` or `uN[N]`.
bool IsBitsLikeFragment(const TypeAnnotation* annotation);

// Returns a description of the entity that owns the given binding, suitable for
// user-facing error messages.
std::string GetParametricBindingOwnerDescription(
    const ParametricBinding* binding);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_UTILS_H_
