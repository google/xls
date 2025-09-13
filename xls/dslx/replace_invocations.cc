// Copyright 2025 The XLS Authors
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

#include "xls/dslx/replace_invocations.h"

#include <string>
#include <utility>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_module.h"
#include "xls/dslx/warning_collector.h"
#include "xls/ir/bits.h"

namespace xls::dslx {

namespace {

template <typename... T>
struct Overloaded : T... {
  using T::operator()...;
};
template <typename... T>
Overloaded(T...) -> Overloaded<T...>;

bool MatchesCalleeEnv(const InvocationData& data,
                      const std::optional<ParametricEnv>& want_env) {
  if (!want_env.has_value()) {
    return true;
  }
  if (data.env_to_callee_data().empty()) {
    return want_env->empty();
  }
  for (const auto& kv : data.env_to_callee_data()) {
    const InvocationCalleeData& callee_data = kv.second;
    if (callee_data.callee_bindings == *want_env) {
      return true;
    }
  }
  return false;
}

const InvocationRewriteRule* FindMatchingRule(
    const InvocationData& data, absl::Span<const InvocationRewriteRule> rules) {
  for (const InvocationRewriteRule& r : rules) {
    if (data.callee() == r.from_callee &&
        MatchesCalleeEnv(data, r.match_callee_env)) {
      return &r;
    }
  }
  return nullptr;
}

absl::flat_hash_map<const NameDef*, NameDef*> BuildNameDefMap(
    const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new) {
  absl::flat_hash_map<const NameDef*, NameDef*> name_def_map;
  for (const auto& kv : old_to_new) {
    if (auto* old_nd = dynamic_cast<const NameDef*>(kv.first)) {
      name_def_map.emplace(old_nd, down_cast<NameDef*>(kv.second));
    }
  }
  return name_def_map;
}

absl::StatusOr<Expr*> CloneExprIntoModule(
    Expr* e, Module* target_module,
    const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new,
    const absl::flat_hash_map<const NameDef*, NameDef*>& name_def_map) {
  auto it = old_to_new.find(e);
  if (it != old_to_new.end()) {
    return down_cast<Expr*>(it->second);
  }
  XLS_ASSIGN_OR_RETURN(
      auto pairs, CloneAstAndGetAllPairs(
                      /*root=*/e,
                      /*target_module=*/std::optional<Module*>{target_module},
                      /*replacer=*/NameRefReplacer(&name_def_map)));
  auto it_cloned = pairs.find(e);
  XLS_RET_CHECK(it_cloned != pairs.end());
  return down_cast<Expr*>(it_cloned->second);
}

absl::StatusOr<ColonRef::Subject> MakeColonRefSubjectFromTypeRef(
    TypeRef* type_ref, const Span& inv_span, Module* target_module,
    const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new) {
  using ReturnT = absl::StatusOr<ColonRef::Subject>;
  return absl::visit(Overloaded{
                         [&](ColonRef* old_cref) -> ReturnT {
                           auto it = old_to_new.find(old_cref);
                           XLS_RET_CHECK(it != old_to_new.end());
                           auto* new_cref = down_cast<ColonRef*>(it->second);
                           return ColonRef::Subject(new_cref);
                         },
                         [&](EnumDef* old_enum) -> ReturnT {
                           const NameDef* old_nd = old_enum->name_def();
                           auto it = old_to_new.find(old_nd);
                           XLS_RET_CHECK(it != old_to_new.end());
                           auto* new_nd = down_cast<NameDef*>(it->second);
                           NameRef* nr = target_module->Make<NameRef>(
                               inv_span, new_nd->identifier(), new_nd,
                               /*in_parens=*/false);
                           return ColonRef::Subject(nr);
                         },
                         [&](TypeAlias* ta) -> ReturnT {
                           auto it = old_to_new.find(&ta->name_def());
                           XLS_RET_CHECK(it != old_to_new.end());
                           auto* new_nd = down_cast<NameDef*>(it->second);
                           NameRef* nr = target_module->Make<NameRef>(
                               inv_span, new_nd->identifier(), new_nd,
                               /*in_parens=*/false);
                           return ColonRef::Subject(nr);
                         },
                         [&](UseTreeEntry* ute) -> ReturnT {
                           std::optional<NameDef*> leaf = ute->GetLeafNameDef();
                           XLS_RET_CHECK(leaf.has_value());
                           auto it = old_to_new.find(*leaf);
                           XLS_RET_CHECK(it != old_to_new.end());
                           auto* new_nd = down_cast<NameDef*>(it->second);
                           NameRef* nr = target_module->Make<NameRef>(
                               inv_span, new_nd->identifier(), new_nd,
                               /*in_parens=*/false);
                           return ColonRef::Subject(nr);
                         },
                         [&](auto*) -> ReturnT {
                           return absl::InvalidArgumentError(
                               "Unsupported enum type reference form");
                         },
                     },
                     type_ref->type_definition());
}

absl::StatusOr<TypeInfo::TypeSource> DealiasTypeDefinition(
    TypeRef* tr, TypeInfo& type_info) {
  XLS_ASSIGN_OR_RETURN(TypeInfo::TypeSource ts,
                       type_info.ResolveTypeDefinition(tr->type_definition()));
  while (std::holds_alternative<TypeAlias*>(ts.definition)) {
    TypeAlias* ta = std::get<TypeAlias*>(ts.definition);
    auto* trta2 = dynamic_cast<TypeRefTypeAnnotation*>(&ta->type_annotation());
    if (trta2 == nullptr) {
      return absl::InvalidArgumentError(
          "Unsupported type alias in explicit replacement (non-TypeRef type)");
    }
    TypeRef* tr2 = trta2->type_ref();
    XLS_ASSIGN_OR_RETURN(
        ts, type_info.ResolveTypeDefinition(tr2->type_definition()));
  }
  return ts;
}

absl::StatusOr<std::string> ResolveEnumMemberName(const EnumDef* enum_def,
                                                  const InterpValue& iv,
                                                  TypeInfo& type_info) {
  XLS_ASSIGN_OR_RETURN(Bits want_bits, iv.GetBits());
  for (const EnumMember& em : enum_def->values()) {
    TypeInfo* ti_used = &type_info;
    if (em.value->owner() != type_info.module()) {
      std::optional<TypeInfo*> imported =
          type_info.GetImportedTypeInfo(em.value->owner());
      if (imported.has_value() && *imported != nullptr) {
        ti_used = *imported;
      }
    }
    std::optional<InterpValue> mv = ti_used->GetConstExprOption(em.value);
    if (!mv.has_value()) {
      continue;
    }
    absl::StatusOr<Bits> mb = mv->GetBits();
    if (mb.ok() && *mb == want_bits) {
      return std::string(em.name_def->identifier());
    }
  }
  return absl::InvalidArgumentError(
      "No matching enum member for provided value");
}

// Validates that the provided explicit env keys are a subset of the callee's
// parametric bindings, and that all required (non-defaulted) bindings are
// supplied. Returns InvalidArgumentError with messages matching the inline
// checks previously performed in BuildExplicitParametricsFromEnv.
static absl::Status ValidateExplicitEnvAgainstCallee(
    const absl::flat_hash_map<std::string, InterpValue>& env_map,
    const Function* to_callee) {
  absl::btree_set<std::string> callee_keys;
  for (const ParametricBinding* pb : to_callee->parametric_bindings()) {
    callee_keys.insert(pb->identifier());
  }

  std::vector<std::string> unknown_keys;
  unknown_keys.reserve(env_map.size());
  for (const auto& kv : env_map) {
    if (!callee_keys.contains(kv.first)) {
      unknown_keys.push_back(kv.first);
    }
  }
  if (!unknown_keys.empty()) {
    std::string listed;
    for (size_t i = 0; i < unknown_keys.size(); ++i) {
      absl::StrAppend(&listed, (i == 0 ? "" : ", "), "`", unknown_keys[i], "`");
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown binding(s) ", listed, " for replacement callee `",
                     to_callee->name_def()->identifier(), "`"));
  }

  for (const ParametricBinding* pb : to_callee->parametric_bindings()) {
    if (!pb->expr() && !env_map.contains(pb->identifier())) {
      return absl::InvalidArgumentError(
          absl::StrCat("Missing required binding `", pb->identifier(),
                       "` for replacement callee"));
    }
  }
  return absl::OkStatus();
}

// Builds an expression representing the enum member corresponding to the
// provided InterpValue for the given enum TypeRef.
absl::StatusOr<Expr*> BuildEnumParametricExpr(
    TypeRef* tr, const EnumDef* enum_def, const InterpValue& iv,
    TypeInfo& type_info, const Span& inv_span, Module* target_module,
    const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new) {
  XLS_ASSIGN_OR_RETURN(std::string member_name,
                       ResolveEnumMemberName(enum_def, iv, type_info));
  XLS_ASSIGN_OR_RETURN(
      ColonRef::Subject subject,
      MakeColonRefSubjectFromTypeRef(tr, inv_span, target_module, old_to_new));
  ColonRef* cref = target_module->Make<ColonRef>(inv_span, subject, member_name,
                                                 /*in_parens=*/false);
  return static_cast<Expr*>(cref);
}

// Currently only enums are supported; other kinds will return an error.
absl::StatusOr<Expr*> BuildParametricExprForTypeRef(
    TypeRef* tr, const InterpValue& iv, TypeInfo& type_info,
    const Span& inv_span, Module* target_module,
    const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new) {
  XLS_ASSIGN_OR_RETURN(TypeInfo::TypeSource ts_final,
                       DealiasTypeDefinition(tr, type_info));
  if (std::holds_alternative<EnumDef*>(ts_final.definition)) {
    auto* enum_def = std::get<EnumDef*>(ts_final.definition);
    return BuildEnumParametricExpr(tr, enum_def, iv, type_info, inv_span,
                                   target_module, old_to_new);
  }
  return absl::InvalidArgumentError(
      "Unsupported parametric TypeRef in explicit replacement (only enums "
      "supported at this time)");
}

absl::StatusOr<std::vector<ExprOrType>> BuildExplicitParametricsFromEnv(
    const ParametricEnv& env, const Function* to_callee, TypeInfo& type_info,
    const Span& inv_span, Module* target_module,
    const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new) {
  std::vector<ExprOrType> result;
  absl::flat_hash_map<std::string, InterpValue> env_map = env.ToMap();
  XLS_RETURN_IF_ERROR(ValidateExplicitEnvAgainstCallee(env_map, to_callee));
  for (const ParametricBinding* pb : to_callee->parametric_bindings()) {
    auto it = env_map.find(pb->identifier());
    if (it == env_map.end()) {
      continue;
    }
    const InterpValue& iv = it->second;
    TypeAnnotation* ann = pb->type_annotation();
    if (ann == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Parametric binding `", pb->identifier(),
          "` lacks a type annotation; explicit replacement not supported"));
    }
    if (auto* bta = dynamic_cast<BuiltinTypeAnnotation*>(ann)) {
      if (!iv.IsBits()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Parametric `", pb->identifier(),
            "` expected bits value for builtin type ", bta->ToString()));
      }
      std::string digits = iv.ToString(/*humanize=*/true);
      BuiltinNameDef* bnd =
          target_module->GetOrCreateBuiltinNameDef(bta->builtin_type());
      BuiltinTypeAnnotation* typed = target_module->Make<BuiltinTypeAnnotation>(
          inv_span, bta->builtin_type(), bnd);
      Number* num = target_module->Make<Number>(inv_span, digits,
                                                NumberKind::kOther, typed,
                                                /*in_parens=*/false);
      result.push_back(static_cast<Expr*>(num));
      continue;
    }
    if (auto* trta = dynamic_cast<TypeRefTypeAnnotation*>(ann)) {
      TypeRef* tr = trta->type_ref();
      XLS_ASSIGN_OR_RETURN(Expr * enum_expr, BuildParametricExprForTypeRef(
                                                 tr, iv, type_info, inv_span,
                                                 target_module, old_to_new));
      result.push_back(enum_expr);
      continue;
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported parametric type annotation for `",
                     pb->identifier(), "` in explicit replacement"));
  }
  return result;
}

absl::StatusOr<std::vector<ExprOrType>> RetainExplicitParametrics(
    const Invocation& inv, Module* target_module,
    const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new,
    const absl::flat_hash_map<const NameDef*, NameDef*>& name_def_map) {
  std::vector<ExprOrType> new_parametrics;
  for (const ExprOrType& eot : inv.explicit_parametrics()) {
    if (std::holds_alternative<Expr*>(eot)) {
      XLS_ASSIGN_OR_RETURN(
          Expr * e, CloneExprIntoModule(std::get<Expr*>(eot), target_module,
                                        old_to_new, name_def_map));
      new_parametrics.push_back(e);
    } else {
      TypeAnnotation* ta = std::get<TypeAnnotation*>(eot);
      auto it = old_to_new.find(ta);
      if (it != old_to_new.end()) {
        new_parametrics.push_back(down_cast<TypeAnnotation*>(it->second));
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          auto pairs,
          CloneAstAndGetAllPairs(
              /*root=*/ta,
              /*target_module=*/std::optional<Module*>{target_module},
              /*replacer=*/NameRefReplacer(&name_def_map)));
      auto it_cloned = pairs.find(ta);
      XLS_RET_CHECK(it_cloned != pairs.end());
      new_parametrics.push_back(down_cast<TypeAnnotation*>(it_cloned->second));
    }
  }
  return new_parametrics;
}

absl::StatusOr<TypecheckedModule> TypecheckAndInstallCloned(
    std::unique_ptr<Module> cloned, const TypecheckedModule& tm,
    ImportData& import_data, std::string_view install_subject) {
  WarningCollector warnings(import_data.enabled_warnings());
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ModuleInfo> module_info,
      TypecheckModule(std::move(cloned), tm.module->fs_path().value(),
                      &import_data, &warnings));
  XLS_ASSIGN_OR_RETURN(ImportTokens subject,
                       ImportTokens::FromString(install_subject));
  XLS_ASSIGN_OR_RETURN(ModuleInfo * stored,
                       import_data.Put(subject, std::move(module_info)));
  return TypecheckedModule{
      .module = &stored->module(),
      .type_info = stored->type_info(),
      .warnings = std::move(warnings),
  };
}

}  // namespace

absl::StatusOr<TypecheckedModule> ReplaceInvocationsInModule(
    const TypecheckedModule& tm, absl::Span<const Function* const> callers,
    absl::Span<const InvocationRewriteRule> rules, ImportData& import_data,
    std::string_view install_subject) {
  const Module& module = *tm.module;
  TypeInfo& type_info = *tm.type_info;
  XLS_RET_CHECK(!callers.empty());
  XLS_RET_CHECK(!rules.empty());
  for (const Function* f : callers) {
    XLS_RET_CHECK_NE(f, nullptr);
    XLS_RET_CHECK_EQ(f->owner(), &module);
  }
  for (const InvocationRewriteRule& r : rules) {
    XLS_RET_CHECK_NE(r.from_callee, nullptr);
    XLS_RET_CHECK_NE(r.to_callee, nullptr);
    XLS_RET_CHECK_EQ(r.from_callee->owner(), &module);
    XLS_RET_CHECK_EQ(r.to_callee->owner(), &module);
    if (r.match_callee_env.has_value()) {
      const ParametricEnv& me = *r.match_callee_env;
      const auto& pbs = r.from_callee->parametric_bindings();
      absl::btree_set<std::string> want_keys = me.GetKeySet();
      absl::btree_set<std::string> callee_keys;
      for (const ParametricBinding* pb : pbs) {
        callee_keys.insert(pb->identifier());
      }
      if (want_keys != callee_keys) {
        return absl::InvalidArgumentError(
            "match_callee_env keys do not match callee parametric names");
      }
    }
  }

  CloneReplacer replacer =
      [&type_info, callers, rules](
          const AstNode* node, Module* target_module,
          const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new)
      -> absl::StatusOr<std::optional<AstNode*>> {
    const Invocation* inv = dynamic_cast<const Invocation*>(node);
    if (inv == nullptr) {
      return std::nullopt;
    }
    auto is_within_any_caller = [&]() -> bool {
      for (const Function* c : callers) {
        if (ContainedWithinFunction(*inv, *c)) {
          return true;
        }
      }
      return false;
    }();
    if (!is_within_any_caller) {
      return std::nullopt;
    }

    std::optional<const InvocationData*> data_opt =
        type_info.GetRootInvocationData(inv);
    if (!data_opt.has_value()) {
      return std::nullopt;
    }
    const InvocationData* data = *data_opt;

    const InvocationRewriteRule* matched_rule = FindMatchingRule(*data, rules);
    if (matched_rule == nullptr) {
      return std::nullopt;
    }

    const NameDef* old_target = matched_rule->to_callee->name_def();
    auto it_nd = old_to_new.find(old_target);
    XLS_RET_CHECK(it_nd != old_to_new.end());
    auto* new_target = down_cast<NameDef*>(it_nd->second);
    NameRef* new_callee = target_module->Make<NameRef>(
        inv->callee()->span(), new_target->identifier(), new_target,
        inv->callee()->in_parens());

    // Pre-built the name def map to avoid rebuilding it for each invocation.
    absl::flat_hash_map<const NameDef*, NameDef*> name_def_map =
        BuildNameDefMap(old_to_new);

    auto clone_expr_into = [&](Expr* e) -> absl::StatusOr<Expr*> {
      return CloneExprIntoModule(e, target_module, old_to_new, name_def_map);
    };

    std::vector<Expr*> new_args;
    new_args.reserve(inv->args().size());
    for (Expr* arg : inv->args()) {
      XLS_ASSIGN_OR_RETURN(Expr * cloned, clone_expr_into(arg));
      new_args.push_back(cloned);
    }

    std::vector<ExprOrType> new_parametrics;

    if (matched_rule->to_callee_env.has_value()) {
      if (!matched_rule->to_callee_env->empty()) {
        XLS_ASSIGN_OR_RETURN(
            new_parametrics,
            BuildExplicitParametricsFromEnv(
                *matched_rule->to_callee_env, matched_rule->to_callee,
                type_info, inv->span(), target_module, old_to_new));
      }
    } else {
      XLS_ASSIGN_OR_RETURN(new_parametrics,
                           RetainExplicitParametrics(*inv, target_module,
                                                     old_to_new, name_def_map));
    }

    std::optional<const Invocation*> new_origin = std::nullopt;
    Invocation* replacement = target_module->Make<Invocation>(
        inv->span(), new_callee, std::move(new_args),
        std::move(new_parametrics), inv->in_parens(), new_origin);
    return replacement;
  };

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> cloned,
                       CloneModule(module, std::move(replacer)));
  XLS_RET_CHECK_OK(VerifyClone(&module, cloned.get(), *module.file_table()));

  return TypecheckAndInstallCloned(std::move(cloned), tm, import_data,
                                   install_subject);
}

absl::StatusOr<TypecheckedModule> ReplaceInvocationsInModule(
    const TypecheckedModule& tm, const Function* caller,
    const InvocationRewriteRule& rule, ImportData& import_data,
    std::string_view install_subject) {
  const Function* callers_arr[] = {caller};
  const InvocationRewriteRule rules_arr[] = {rule};
  return ReplaceInvocationsInModule(tm, absl::MakeSpan(callers_arr),
                                    absl::MakeSpan(rules_arr), import_data,
                                    install_subject);
}

}  // namespace xls::dslx
