// Copyright 2020 The XLS Authors
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

#include "xls/dslx/type_system/parametric_instantiator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/type_system/parametric_instantiator_internal.h"

namespace xls::dslx {

// Helper ToString()s for debug logging.
static std::string ToTypesString(absl::Span<const InstantiateArg> ts) {
  if (ts.empty()) {
    return "none";
  }
  return absl::StrJoin(ts, ", ", [](std::string* out, const auto& t) {
    absl::StrAppend(out, t.type->ToString());
  });
}
static std::string ToString(
    absl::Span<std::unique_ptr<ConcreteType> const> ts) {
  if (ts.empty()) {
    return "none";
  }
  return absl::StrJoin(ts, ", ", [](std::string* out, const auto& t) {
    absl::StrAppend(out, t->ToString());
  });
}
static std::string ToString(
    const absl::flat_hash_map<std::string, InterpValue>& map) {
  return absl::StrCat(
      "{",
      absl::StrJoin(
          map, ", ",
          [](std::string* out, const std::pair<std::string, InterpValue>& p) {
            out->append(absl::StrCat(p.first, ":", p.second.ToString()));
          }),
      "}");
}

static std::string ToString(
    const absl::Span<const ParametricConstraint>& parametric_constraints) {
  return absl::StrCat(
      "[",
      absl::StrJoin(parametric_constraints, ", ",
                    [](std::string* out, const ParametricConstraint& c) {
                      absl::StrAppend(out, c.ToString());
                    }),
      "]");
}

absl::StatusOr<TypeAndBindings> InstantiateFunction(
    Span span, const FunctionType& function_type,
    absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    absl::Span<const ParametricConstraint> parametric_constraints,
    const absl::flat_hash_map<std::string, InterpValue>& explicit_bindings) {
  XLS_VLOG(5) << "Function instantiation @ " << span
              << " type: " << function_type.ToString();
  XLS_VLOG(5) << " parametric constraints: "
              << ToString(parametric_constraints);
  XLS_VLOG(5) << " arg types:              " << ToTypesString(args);
  XLS_VLOG(5) << " explicit bindings:   " << ToString(explicit_bindings);
  XLS_ASSIGN_OR_RETURN(auto instantiator,
                       internal::FunctionInstantiator::Make(
                           std::move(span), function_type, args, ctx,
                           parametric_constraints, explicit_bindings));
  return instantiator->Instantiate();
}

absl::StatusOr<TypeAndBindings> InstantiateStruct(
    Span span, const StructType& struct_type,
    absl::Span<const InstantiateArg> args,
    absl::Span<std::unique_ptr<ConcreteType> const> member_types,
    DeduceCtx* ctx,
    absl::Span<const ParametricConstraint> parametric_constraints) {
  XLS_VLOG(5) << "Struct instantiation @ " << span
              << " type: " << struct_type.ToString();
  XLS_VLOG(5) << " arg types:           " << ToTypesString(args);
  XLS_VLOG(5) << " member types:        " << ToString(member_types);
  XLS_ASSIGN_OR_RETURN(auto instantiator,
                       internal::StructInstantiator::Make(
                           std::move(span), struct_type, args, member_types,
                           ctx, parametric_constraints));
  return instantiator->Instantiate();
}

}  // namespace xls::dslx
