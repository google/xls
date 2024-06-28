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

#include "xls/dslx/type_system/deduce_struct_instance.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_instantiator.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {

// Forward declaration from sister implementation file as we are recursively
// bound to the central deduce-and-resolve routine in our node deduction
// routines.
//
// TODO(cdleary): 2024-01-16 We can break this circular resolution with a
// virtual function on DeduceCtx when we get things refactored nicely.
extern absl::StatusOr<std::unique_ptr<Type>> DeduceAndResolve(
    const AstNode* node, DeduceCtx* ctx);

namespace {

struct ValidatedStructMembers {
  // Names seen in the struct instance; e.g. for a SplatStructInstance can be a
  // subset of the struct member names.
  //
  // Note: we use a btree set so we can do set differencing via c_set_difference
  // (which works on ordered sets).
  absl::btree_set<std::string> seen_names;

  std::vector<InstantiateArg> args;
  std::vector<std::unique_ptr<Type>> member_types;
};

// Validates a struct instantiation is a subset of 'members' with no dups.
//
// Args:
//  members: Sequence of members used in instantiation. Note this may be a
//    subset; e.g. in the case of splat instantiation.
//  struct_type: The deduced type for the struct (instantiation).
//  struct_text: Display name to use for the struct in case of an error.
//  ctx: Wrapper containing node to type mapping context.
//
// Returns:
//  A tuple containing:
//  * The set of struct member names that were instantiated
//  * The Types of the provided arguments
//  * The Types of the corresponding struct member definition.
absl::StatusOr<ValidatedStructMembers> ValidateStructMembersSubset(
    absl::Span<const std::pair<std::string, Expr*>> members,
    const StructType& struct_type, std::string_view struct_text,
    DeduceCtx* ctx) {
  ValidatedStructMembers result;
  for (auto& [name, expr] : members) {
    if (!result.seen_names.insert(name).second) {
      return TypeInferenceErrorStatus(
          expr->span(), nullptr,
          absl::StrFormat(
              "Duplicate value seen for '%s' in this '%s' struct instance.",
              name, struct_text));
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> expr_type,
                         DeduceAndResolve(expr, ctx));
    XLS_RET_CHECK(!expr_type->IsMeta())
        << "name: " << name << " expr: " << expr->ToString()
        << " type: " << *expr_type;

    result.args.push_back(InstantiateArg{std::move(expr_type), expr->span()});
    std::optional<const Type*> maybe_type =
        struct_type.GetMemberTypeByName(name);

    if (maybe_type.has_value()) {
      XLS_RET_CHECK(!maybe_type.value()->IsMeta())
          << *maybe_type.value();
      result.member_types.push_back(maybe_type.value()->CloneToUnique());
    } else {
      return TypeInferenceErrorStatus(
          expr->span(), nullptr,
          absl::StrFormat("Struct '%s' has no member '%s', but it was provided "
                          "by this instance.",
                          struct_text, name));
    }
  }

  return result;
}

}  // namespace

absl::StatusOr<std::unique_ptr<Type>> DeduceStructInstance(
    const StructInstance* node, DeduceCtx* ctx) {
  VLOG(5) << "Deducing type for struct instance: " << node->ToString();

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                       ctx->Deduce(ToAstNode(node->struct_ref())));
  XLS_ASSIGN_OR_RETURN(type, UnwrapMetaType(std::move(type), node->span(),
                                            "struct instance type"));

  auto* struct_type = dynamic_cast<const StructType*>(type.get());
  if (struct_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->span(), struct_type,
        "Expected a struct definition to instantiate");
  }

  // Note what names we expect to be present.
  XLS_ASSIGN_OR_RETURN(std::vector<std::string> names,
                       struct_type->GetMemberNames());
  absl::btree_set<std::string> expected_names(names.begin(), names.end());

  XLS_ASSIGN_OR_RETURN(
      ValidatedStructMembers validated,
      ValidateStructMembersSubset(node->GetUnorderedMembers(), *struct_type,
                                  node->struct_ref()->ToString(), ctx));
  if (validated.seen_names != expected_names) {
    absl::btree_set<std::string> missing_set;
    absl::c_set_difference(expected_names, validated.seen_names,
                           std::inserter(missing_set, missing_set.begin()));
    std::vector<std::string> missing(missing_set.begin(), missing_set.end());
    std::sort(missing.begin(), missing.end());
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat(
            "Struct instance is missing member(s): %s",
            absl::StrJoin(missing, ", ",
                          [](std::string* out, const std::string& piece) {
                            absl::StrAppendFormat(out, "'%s'", piece);
                          })));
  }

  const TypeAnnotation* struct_ref = node->struct_ref();
  XLS_RET_CHECK(struct_ref != nullptr);
  XLS_ASSIGN_OR_RETURN(StructDef * struct_def,
                       DerefToStruct(node->span(), struct_ref->ToString(),
                                     *struct_ref, ctx->type_info()));

  XLS_ASSIGN_OR_RETURN(
      std::vector<ParametricWithType> typed_parametrics,
      ParametricBindingsToTyped(struct_def->parametric_bindings(), ctx));
  XLS_ASSIGN_OR_RETURN(
      TypeAndParametricEnv tab,
      InstantiateStruct(node->span(), *struct_type, validated.args,
                        validated.member_types, ctx, typed_parametrics,
                        struct_def->parametric_bindings()));

  return std::move(tab.type);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceSplatStructInstance(
    const SplatStructInstance* node, DeduceCtx* ctx) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> struct_type_ct,
                       ctx->Deduce(ToAstNode(node->struct_ref())));
  XLS_ASSIGN_OR_RETURN(struct_type_ct,
                       UnwrapMetaType(std::move(struct_type_ct), node->span(),
                                      "splatted struct instance type"));

  // The type of the splatted value; e.g. in `MyStruct{..s}` the type of `s`.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> splatted_type_ct,
                       ctx->Deduce(node->splatted()));

  // The splatted type should be (nominally) equivalent to the struct type,
  // because that's where we're filling in the default values from (those values
  // that were not directly provided by the user).
  auto* struct_type = dynamic_cast<StructType*>(struct_type_ct.get());
  if (struct_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->struct_ref()->span(), struct_type_ct.get(),
        absl::StrFormat("Type given to struct instantiation was not a struct"));
  }

  auto* splatted_type = dynamic_cast<StructType*>(splatted_type_ct.get());
  if (splatted_type == nullptr) {
    return TypeInferenceErrorStatus(
        node->splatted()->span(), splatted_type_ct.get(),
        absl::StrFormat(
            "Type given to 'splatted' struct instantiation was not a struct"));
  }

  if (&struct_type->nominal_type() != &splatted_type->nominal_type()) {
    return ctx->TypeMismatchError(
        node->span(), nullptr, *struct_type, nullptr, *splatted_type,
        absl::StrFormat("Attempting to fill values in '%s' instantiation from "
                        "a value of type '%s'",
                        struct_type->nominal_type().identifier(),
                        splatted_type->nominal_type().identifier()));
  }

  XLS_ASSIGN_OR_RETURN(
      ValidatedStructMembers validated,
      ValidateStructMembersSubset(node->members(), *struct_type,
                                  node->struct_ref()->ToString(), ctx));

  XLS_ASSIGN_OR_RETURN(std::vector<std::string> all_names,
                       struct_type->GetMemberNames());
  VLOG(5) << "SplatStructInstance @ " << node->span() << " seen names: ["
          << absl::StrJoin(validated.seen_names, ", ") << "] "
          << " all names: [" << absl::StrJoin(all_names, ", ") << "]";

  if (validated.seen_names.size() == all_names.size()) {
    ctx->warnings()->Add(
        node->splatted()->span(), WarningKind::kUselessStructSplat,
        absl::StrFormat("'Splatted' struct instance has all members of struct "
                        "defined, consider removing the `..%s`",
                        node->splatted()->ToString()));
  }

  for (const std::string& name : all_names) {
    // If we didn't see the name, it comes from the "splatted" argument.
    if (!validated.seen_names.contains(name)) {
      const Type& splatted_member_type =
          *splatted_type->GetMemberTypeByName(name).value();
      const Type& struct_member_type =
          *struct_type->GetMemberTypeByName(name).value();

      validated.args.push_back(InstantiateArg{
          splatted_member_type.CloneToUnique(), node->splatted()->span()});
      validated.member_types.push_back(struct_member_type.CloneToUnique());
    }
  }

  // At this point, we should have the same number of args compared to the
  // number of members defined in the struct.
  XLS_RET_CHECK_EQ(validated.args.size(), validated.member_types.size());

  const TypeAnnotation* struct_ref = node->struct_ref();
  XLS_RET_CHECK(struct_ref != nullptr);
  XLS_ASSIGN_OR_RETURN(StructDef * struct_def,
                       DerefToStruct(node->span(), struct_ref->ToString(),
                                     *struct_ref, ctx->type_info()));

  XLS_ASSIGN_OR_RETURN(
      std::vector<ParametricWithType> typed_parametrics,
      ParametricBindingsToTyped(struct_def->parametric_bindings(), ctx));
  XLS_ASSIGN_OR_RETURN(
      TypeAndParametricEnv tab,
      InstantiateStruct(node->span(), *struct_type, validated.args,
                        validated.member_types, ctx, typed_parametrics,
                        struct_def->parametric_bindings()));

  return std::move(tab.type);
}

}  // namespace xls::dslx
