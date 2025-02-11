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
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

struct StructOrProcRef {
  const StructDefBase* def;
  std::vector<ExprOrType> parametrics;
  std::optional<const StructInstance*> instantiator;
};

// The signedness and bit count extracted from a `TypeAnnotation`. The
// `TypeAnnotation` may use primitive values or exprs; we convey the
// representation as is.
struct SignednessAndBitCountResult {
  std::variant<bool, const Expr*> signedness;
  std::variant<int64_t, const Expr*> bit_count;
};

// Creates an annotation for `uN[bit_count]` or `sN[bit_count]` depending on the
// value of `is_signed`.
TypeAnnotation* CreateUnOrSnAnnotation(Module& module, const Span& span,
                                       bool is_signed, int64_t bit_count);

// Creates a `bool` type annotation.
TypeAnnotation* CreateBoolAnnotation(Module& module, const Span& span);

// Creates a `u32` type annotation.
//
// TODO(https://github.com/google/xls/issues/450): 2025-01-23 In the future we
// should perhaps be using "usize" instead to characterize the places we need
// this, e.g. a type that we evaluate array dimensions and similar undecorated
// values to.
TypeAnnotation* CreateU32Annotation(Module& module, const Span& span);

// Creates an annotation referring to the given struct definition with the given
// parametric arguments.
TypeAnnotation* CreateStructAnnotation(
    Module& module, StructDef* def, std::vector<ExprOrType> parametrics,
    std::optional<const StructInstance*> instantiator);

// Variant that converts a `StructOrProcRef` into an annotation.
TypeAnnotation* CreateStructAnnotation(Module& module,
                                       const StructOrProcRef& ref);

// Returns the signedness and bit count from the given type annotation, if it is
// a bits-like annotation; otherwise, returns an error.
absl::StatusOr<SignednessAndBitCountResult> GetSignednessAndBitCount(
    const TypeAnnotation* annotation);

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

// Resolves the definition and parametrics for the struct or proc type referred
// to by `annotation`.
std::optional<StructOrProcRef> GetStructOrProcRef(
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

// Creates a CloneReplacer that replaces any `NameRef` to a `NameDef` in `map`
// with the corresponding `ExprOrType`. This is used for replacement of
// parametric variables with values.
CloneReplacer NameRefMapper(
    const absl::flat_hash_map<const NameDef*, ExprOrType>& map);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_UTILS_H_
