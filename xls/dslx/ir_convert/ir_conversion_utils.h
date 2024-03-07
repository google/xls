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
#ifndef XLS_DSLX_IR_CONVERT_IR_CONVERSION_UTILS_H_
#define XLS_DSLX_IR_CONVERT_IR_CONVERSION_UTILS_H_

// Utilities involved in DSLX-to-IR type conversion that aren't strictly part of
// tree traversal (which is the main concern of ir_converter.h/cc.

#include <cstdint>

#include "absl/status/statusor.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls::dslx {

// Resolves "dim" (from a possible parametric) against the given symbolic
// bindings.
absl::StatusOr<TypeDim> ResolveDim(TypeDim dim, const ParametricEnv& bindings);

// As above, does ResolveDim() but then accesses the dimension value as an
// expected int64_t.
absl::StatusOr<int64_t> ResolveDimToInt(const TypeDim& dim,
                                        const ParametricEnv& bindings);

// Converts a concrete type to its corresponding IR representation.
// TODO(rspringer): Only a fully-concretized type can be converted to IR,
// meaning that ParametricEnv shouldn't be necessary here. Verify that's the
// case, and if so, remove the argument.
absl::StatusOr<xls::Type*> TypeToIr(Package* package, const Type& type,
                                    const ParametricEnv& bindings);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_IR_CONVERSION_UTILS_H_
