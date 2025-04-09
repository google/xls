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

#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_instantiator_internal.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"

namespace xls::dslx {
namespace {

// Helper ToString()s for debug logging.
std::string ToTypesString(absl::Span<const InstantiateArg> ts) {
  if (ts.empty()) {
    return "none";
  }
  return absl::StrJoin(ts, ", ", [](std::string* out, const auto& t) {
    absl::StrAppend(out, t.type()->ToString());
  });
}

std::string ToString(absl::Span<std::unique_ptr<Type> const> ts) {
  if (ts.empty()) {
    return "none";
  }
  return absl::StrJoin(ts, ", ", [](std::string* out, const auto& t) {
    absl::StrAppend(out, t->ToString());
  });
}

std::string ToString(const absl::flat_hash_map<std::string, InterpValue>& map) {
  return absl::StrCat(
      "{",
      absl::StrJoin(
          map, ", ",
          [](std::string* out, const std::pair<std::string, InterpValue>& p) {
            out->append(absl::StrCat(p.first, ":", p.second.ToString()));
          }),
      "}");
}

std::string ToString(
    const absl::Span<const ParametricWithType>& typed_parametrics) {
  return absl::StrCat(
      "[",
      absl::StrJoin(typed_parametrics, ", ",
                    [](std::string* out, const ParametricWithType& c) {
                      absl::StrAppend(out, c.ToString());
                    }),
      "]");
}

}  // namespace

absl::StatusOr<TypeAndParametricEnv> InstantiateFunction(
    Span span, Function& callee_fn, const FunctionType& function_type,
    absl::Span<const InstantiateArg> args, DeduceCtx* ctx,
    absl::Span<const ParametricWithType> typed_parametrics,
    const absl::flat_hash_map<std::string, InterpValue>& explicit_bindings,
    absl::Span<const ParametricBinding* ABSL_NONNULL const>
        parametric_bindings) {
  const FileTable& file_table = ctx->file_table();
  VLOG(5) << "Function instantiation @ " << span.ToString(file_table)
          << " type: " << function_type;
  VLOG(5) << " typed-parametrics: " << ToString(typed_parametrics);
  VLOG(5) << " arg types:              " << ToTypesString(args);
  VLOG(5) << " explicit bindings:   " << ToString(explicit_bindings);
  XLS_ASSIGN_OR_RETURN(
      auto instantiator,
      internal::FunctionInstantiator::Make(
          span, callee_fn, function_type, args, ctx, typed_parametrics,
          explicit_bindings, parametric_bindings));
  return instantiator->Instantiate();
}

absl::StatusOr<TypeAndParametricEnv> InstantiateStruct(
    Span span, const StructType& struct_type,
    absl::Span<const InstantiateArg> args,
    absl::Span<std::unique_ptr<Type> const> member_types, DeduceCtx* ctx,
    absl::Span<const ParametricWithType> typed_parametrics,
    absl::Span<const ParametricBinding* ABSL_NONNULL const>
        parametric_bindings) {
  const FileTable& file_table = ctx->file_table();
  VLOG(5) << "Struct instantiation @ " << span.ToString(file_table)
          << " type: " << struct_type;
  VLOG(5) << " arg types:           " << ToTypesString(args);
  VLOG(5) << " member types:        " << ToString(member_types);
  VLOG(5) << " typed-parametrics: " << ToString(typed_parametrics);
  XLS_ASSIGN_OR_RETURN(auto instantiator,
                       internal::StructInstantiator::Make(
                           span, struct_type, args, member_types, ctx,
                           typed_parametrics, parametric_bindings));
  return instantiator->Instantiate();
}

}  // namespace xls::dslx
