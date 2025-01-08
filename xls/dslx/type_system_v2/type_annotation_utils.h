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
#include <variant>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

using StructOrProcDef = std::variant<const StructDef*, const ProcDef*>;

// Creates an annotation for `uN[bit_count]` or `sN[bit_count]` depending on the
// value of `is_signed`.
TypeAnnotation* CreateUnOrSnAnnotation(Module& module, const Span& span,
                                       bool is_signed, int64_t bit_count);

// Creates a `bool` type annotation.
TypeAnnotation* CreateBoolAnnotation(Module& module, const Span& span);

// Creates a `s64` type annotation.
TypeAnnotation* CreateS64Annotation(Module& module, const Span& span);

// The signedness and bit count extracted from a `TypeAnnotation`. The
// `TypeAnnotation` may use primitive values or exprs; we convey the
// representation as is.
struct SignednessAndBitCountResult {
  std::variant<bool, const Expr*> signedness;
  std::variant<int64_t, const Expr*> bit_count;
};

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

// Acts like `dynamic_cast<const ArrayTypeAnnotation*>(annotation)` but only
// succeeds if the annotation is for a true array, as opposed to a bits-like
// type expressed as an array (e.g. `uN` or `xN`).
const ArrayTypeAnnotation* CastToNonBitsArrayTypeAnnotation(
    const TypeAnnotation* annotation);

// Gets the `StructDef` or `ProcDef` for the type referenced by `annotation`, if
// it is indeed a struct or impl-style proc type.
std::optional<StructOrProcDef> GetStructOrProcDef(
    const TypeAnnotation* annotation);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_UTILS_H_
