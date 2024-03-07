// Copyright 2022 The XLS Authors
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
#ifndef XLS_FUZZER_VALUE_GENERATOR_H_
#define XLS_FUZZER_VALUE_GENERATOR_H_

#include <cstdint>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/ir/bits.h"

namespace xls {

// Generates a Bits object with the given size.
Bits GenerateBits(absl::BitGenRef bit_gen, int64_t bit_count);

// Randomly generates an Expr* holding a value of the given type.
// Note: Currently, AstGenerator only produces single-dimensional arrays with
// [AST] Number-typed or ConstantDef-defined sizes. If that changes, then this
// function will need to be modified.
absl::StatusOr<dslx::Expr*> GenerateDslxConstant(absl::BitGenRef bit_gen,
                                                 dslx::Module* module,
                                                 dslx::TypeAnnotation* type);

// Returns a single value of the given type.
absl::StatusOr<dslx::InterpValue> GenerateInterpValue(
    absl::BitGenRef bit_gen, const dslx::Type& arg_type,
    absl::Span<const dslx::InterpValue> prior);

// Returns randomly generated values of the given types.
absl::StatusOr<std::vector<dslx::InterpValue>> GenerateInterpValues(
    absl::BitGenRef bit_gen, absl::Span<const dslx::Type* const> arg_types);

}  // namespace xls

#endif  // XLS_FUZZER_VALUE_GENERATOR_H_
