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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_SOLVE_FOR_PARAMETRICS_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_SOLVE_FOR_PARAMETRICS_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

// Determines values for the specified `parametrics` by equating the type of an
// actual value with a formal type that depends on the parametrics. For example:
//     fn f<N: u32>(a: uN[N]) { ... }
//     f(u24:12);
//
// Here we can use the "equation" `uN[N] = u24` to solve for N. It may even be
// possible to solve for multiple parametrics given one pair of types, if they
// are all used by a single formal type:
//
//     fn f<S: bool, W: u32, N: u32>(a: xN[S][W][N]) { ... }
//     f([s16:1, s16:2, s16:3]);
//
// The `parametrics` passed in may be a subset of the ones for the function or
// struct they belong to. This function may only be able to resolve some of
// those using the passed in type annotations, in which case the returned map
// will have fewer elements than `parametrics`.
absl::StatusOr<absl::flat_hash_map<const ParametricBinding*, InterpValue>>
SolveForParametrics(const TypeAnnotation* resolvable_type,
                    const TypeAnnotation* parametric_dependent_type,
                    absl::flat_hash_set<const ParametricBinding*> parametrics,
                    absl::AnyInvocable<absl::StatusOr<InterpValue>(const Expr*)>
                        expr_evaluator);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_SOLVE_FOR_PARAMETRICS_H_
