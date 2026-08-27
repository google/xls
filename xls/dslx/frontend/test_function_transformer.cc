// Copyright 2026 The XLS Authors
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

#include "xls/dslx/frontend/test_function_transformer.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xls/common/attribute_data.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/type_to_type_annotation.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/name_uniquer.h"

namespace xls::dslx {

namespace {

// Determines if a given test function calls spawn.
class SpawnFinder : public AstNodeRecursiveVisitor {
 public:
  // Legacy spawn statement, not trait-based spawn.
  absl::Status HandleSpawn(const Spawn* node) override {
    has_spawn_ = true;
    return absl::OkStatus();
  }

  absl::Status HandleInvocation(const Invocation* node) override {
    if (has_spawn_) {
      return absl::OkStatus();
    }
    // Trait-based spawn
    if (const Attr* attr = dynamic_cast<const Attr*>(node->callee());
        attr != nullptr) {
      has_spawn_ |= attr->attr() == "spawn";
    }
    // It's possible it's a multi-level invocation (e.g., new().spawn()), so we
    // need to recurse.
    return DefaultHandler(node);
  }

  absl::Status HandleTestFunction(const TestFunction* node) override {
    has_spawn_ = false;
    return DefaultHandler(node);
  }

  bool has_spawn() const { return has_spawn_; }

 private:
  bool has_spawn_ = false;
};

constexpr std::string_view kTerminatorChannelName = "__test__terminator";

// Create the 'next' function on the impl node with the non-constexpr
// statements, adding an additional terminator 'send' statement.
absl::StatusOr<Function*> CreateNextFunction(
    const Function& fn, Module& new_module, TypeAnnotation* proc_def_type,
    const std::vector<Statement*>& non_const_statements,
    const absl::flat_hash_set<const NameDef*>& defs_to_promote,
    const absl::flat_hash_map<const NameDef*, std::string>&
        original_to_unique_name,
    const TypeInfo& type_info) {
  NameDef* self_name = new_module.Make<NameDef>(fn.span(), "self",
                                                /*definer=*/nullptr);
  TypeAnnotation* self_param_type = new_module.Make<SelfTypeAnnotation>(
      fn.span(), /*explicit_type=*/false, proc_def_type);
  Param* self_param = new_module.Make<Param>(self_name, self_param_type);
  self_name->set_definer(self_param);

  // Clone next statements with custom replacer and read() wrapping
  auto next_block = new_module.Make<StatementBlock>(
      fn.span(), non_const_statements, /*trailing_semi=*/true);

  // Replacer for cloning next statements. It replaces references to promoted
  // variables (`let foo`) with accesses to the proc fields (e.g. `self.foo`).
  // Non-channel variables are wrapped in `read()` invocations (e.g.
  // `read(self.field)`) to read the state. Channels are accessed directly.
  auto next_replacer =
      [&](const AstNode* node, Module* target_module,
          const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new)
      -> absl::StatusOr<std::optional<AstNode*>> {
    if (const auto* name_ref = dynamic_cast<const NameRef*>(node)) {
      if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
        const NameDef* original_def =
            std::get<const NameDef*>(name_ref->name_def());
        if (defs_to_promote.contains(original_def)) {
          std::string member_name = original_to_unique_name.at(original_def);
          NameRef* self_ref =
              target_module->Make<NameRef>(name_ref->span(), "self", self_name);
          Attr* member_access = target_module->Make<Attr>(
              name_ref->span(), self_ref, member_name);

          XLS_ASSIGN_OR_RETURN(Type * type,
                               type_info.GetItemOrError(original_def));
          if (type->IsChannel()) {
            return member_access;
          }

          BuiltinNameDef* read_builtin =
              target_module->GetOrCreateBuiltinNameDef("read");
          NameRef* read_ref = target_module->Make<NameRef>(
              name_ref->span(), "read", read_builtin);
          Invocation* read_invocation = target_module->Make<Invocation>(
              name_ref->span(), read_ref, std::vector<Expr*>{member_access});
          return read_invocation;
        }
      }
    }
    return std::nullopt;
  };

  CloneReplacer chained_replacer =
      ChainCloneReplacers(next_replacer, PreserveTypeDefinitionsReplacer);

  XLS_ASSIGN_OR_RETURN(auto next_old_to_new,
                       CloneAstAndGetAllPairs(next_block, &new_module,
                                              std::move(chained_replacer)));
  auto* cloned_next_block =
      absl::down_cast<StatementBlock*>(next_old_to_new.at(next_block));
  std::vector<Statement*> cloned_next_statements(
      cloned_next_block->statements().begin(),
      cloned_next_block->statements().end());

  // Create the send(token(), self.__test__terminator, true) statement:
  // Create `token()` invocation
  BuiltinNameDef* token_builtin = new_module.GetOrCreateBuiltinNameDef("token");
  NameRef* token_ref = new_module.Make<NameRef>(
      fn.span(), token_builtin->identifier(), token_builtin);
  Invocation* token_invocation =
      new_module.Make<Invocation>(fn.span(), token_ref, std::vector<Expr*>{});

  // Create `self.__test__terminator` access
  NameRef* self_ref = new_module.Make<NameRef>(fn.span(), "self", self_name);
  Attr* terminator_channel_access = new_module.Make<Attr>(
      fn.span(), self_ref, std::string(kTerminatorChannelName));

  TypeAnnotation* bool_type = new_module.Make<BuiltinTypeAnnotation>(
      fn.span(), BuiltinType::kBool,
      new_module.GetOrCreateBuiltinNameDef(BuiltinType::kBool));
  Number* true_literal =
      new_module.Make<Number>(fn.span(), "true", NumberKind::kBool, bool_type);

  // Create `send(token(), self.__test__terminator, true)`
  BuiltinNameDef* send_builtin = new_module.GetOrCreateBuiltinNameDef("send");
  NameRef* send_ref = new_module.Make<NameRef>(
      fn.span(), send_builtin->identifier(), send_builtin);
  Invocation* send_invocation = new_module.Make<Invocation>(
      fn.span(), send_ref,
      std::vector<Expr*>{token_invocation, terminator_channel_access,
                         true_literal});

  // Wrap the send in a Statement and add to the cloned statements list.
  Statement* send_stmt = new_module.Make<Statement>(send_invocation);
  std::vector<Statement*> next_statements = cloned_next_statements;
  next_statements.push_back(send_stmt);

  // fn next(self) {...}
  NameDef* next_fn_name = new_module.Make<NameDef>(fn.span(), "next",
                                                   /*definer=*/nullptr);
  Function* next_fn = new_module.Make<Function>(
      fn.span(), next_fn_name, std::vector<ParametricBinding*>{},
      std::vector<Param*>{self_param},
      /*return_type=*/nullptr,
      new_module.Make<StatementBlock>(fn.span(), next_statements,
                                      /*trailing_semi=*/true),
      FunctionTag::kNormal,
      /*is_public=*/false,
      /*is_stub=*/false);
  next_fn_name->set_definer(next_fn);
  return next_fn;
}

struct ProcDefAndTypeAnnotation {
  ProcDef* proc_def;
  TypeAnnotation* terminator_channel_type;
};

absl::StatusOr<ProcDefAndTypeAnnotation> CreateProcDef(const Function& fn,
                                                       Module& new_module) {
  ProcDef* proc_def = new_module.Make<ProcDef>(
      fn.span(),
      new_module.Make<NameDef>(
          fn.span(),
          absl::StrCat("__test__proc__", fn.name_def()->identifier()),
          /*type_annotation=*/nullptr),
      std::vector<ParametricBinding*>{},
      /*members=*/std::vector<StructMemberNode*>{},
      /*is_public=*/false);

  // Add the #[test] attribute
  proc_def->AddAttribute(
      new_module.Make<Attribute>(fn.span(), /*arg_span=*/std::nullopt,
                                 AttributeData(AttributeKind::kTest,
                                               /*args=*/{})));

  // Create the terminator channel on the ProcDef.
  NameDef* terminator_channel_name =
      new_module.Make<NameDef>(fn.span(), std::string(kTerminatorChannelName),
                               /*definer=*/nullptr);
  TypeAnnotation* bool_type = new_module.Make<BuiltinTypeAnnotation>(
      fn.span(), BuiltinType::kBool,
      new_module.GetOrCreateBuiltinNameDef(BuiltinType::kBool));
  TypeAnnotation* terminator_channel_type =
      new_module.Make<ChannelTypeAnnotation>(fn.span(), ChannelDirection::kOut,
                                             bool_type, /*dims=*/std::nullopt);
  StructMemberNode* terminator_channel = new_module.Make<StructMemberNode>(
      fn.span(), terminator_channel_name,
      /*colon_span=*/fn.span(), terminator_channel_type);
  proc_def->AddMember(terminator_channel);
  terminator_channel_name->set_definer(terminator_channel);

  XLS_RETURN_IF_ERROR(
      new_module.AddTop(proc_def, /*make_collision_error=*/nullptr));

  return ProcDefAndTypeAnnotation{proc_def, terminator_channel_type};
}

// Returns the set of NameDefs that should be promoted to fields on the
// synthesized proc.
absl::StatusOr<absl::flat_hash_set<const NameDef*>> find_promoted(
    const std::vector<Statement*> const_statements,
    const std::vector<Statement*> non_const_statements) {
  absl::flat_hash_set<const NameDef*> name_defs;
  for (const auto& stmt : const_statements) {
    XLS_ASSIGN_OR_RETURN(absl::flat_hash_set<const NameDef*> stmt_defs,
                         CollectNameDefsUnder(stmt));
    name_defs.insert(stmt_defs.begin(), stmt_defs.end());
  }
  absl::flat_hash_set<const NameRef*> name_refs;
  for (const auto& stmt : non_const_statements) {
    XLS_ASSIGN_OR_RETURN(std::vector<const NameRef*> stmt_refs,
                         CollectNameRefsUnder(stmt));
    name_refs.insert(stmt_refs.begin(), stmt_refs.end());
  }
  absl::flat_hash_set<const NameDef*> promoted;
  for (const auto& name_ref : name_refs) {
    if (!std::holds_alternative<const NameDef*>(name_ref->name_def())) {
      continue;
    }
    const NameDef* name_def = std::get<const NameDef*>(name_ref->name_def());
    if (name_defs.find(name_def) != name_defs.end()) {
      promoted.insert(name_def);
    }
  }
  return promoted;
}

}  // namespace

bool TestFunctionTransformer::IsConstExpr(const Statement* stmt) {
  // Need to unwrap the statement to get the underlying expression to test if
  // it's a known const expr.
  return std::visit(
      [this](const auto& node) { return type_info_.IsKnownConstExpr(node); },
      stmt->wrapped());
}

// Promote variables from the 'new' function that are subsequently used by the
// 'next' function into fields on the synthesized proc. Modify their creation
// (in 'new') and use (in 'new' and 'next') to reflect that they are fields
// and not local variables.
absl::StatusOr<TestFunctionTransformer::PromotionResult>
TestFunctionTransformer::PromoteVariablesToFields(
    const absl::flat_hash_set<const NameDef*>& defs_to_promote,
    Module& new_module, ProcDef& proc_def) {
  // Sort the promoted variables by their appearance in the original function,
  // so that they are added in the same order as fields of the impl.
  std::vector<const NameDef*> sorted_defs_to_promote(defs_to_promote.begin(),
                                                     defs_to_promote.end());
  std::sort(sorted_defs_to_promote.begin(), sorted_defs_to_promote.end(),
            [](const NameDef* a, const NameDef* b) {
              return a->span().start() < b->span().start();
            });

  absl::flat_hash_map<const NameDef*, std::string> original_to_unique_name;
  NameUniquer name_uniquer("_");
  // Seed it with the terminator channel name so that no promoted variable
  // collides with it.
  XLS_RETURN_IF_ERROR(name_uniquer.ReserveIdentifier(kTerminatorChannelName));

  // If the same identifier is used for multiple promoted variables (e.g. in
  // different scopes), uniquify their names to prevent collisions when they
  // become proc fields.
  for (const NameDef* def : sorted_defs_to_promote) {
    std::string base_name = def->identifier();
    std::string unique_name = name_uniquer.GetSanitizedUniqueName(base_name);
    original_to_unique_name[def] = unique_name;

    NameDef* member_name_def =
        new_module.Make<NameDef>(def->span(), unique_name, nullptr);

    XLS_ASSIGN_OR_RETURN(Type * type, type_info_.GetItemOrError(def));
    XLS_ASSIGN_OR_RETURN(TypeAnnotation * type_annot,
                         CreateTypeAnnotation(new_module, *type, def->span(),
                                              &source_module_, &type_info_));

    StructMemberNode* member_node = new_module.Make<StructMemberNode>(
        def->span(), member_name_def, /*colon_span=*/def->span(), type_annot);
    proc_def.AddMember(member_node);
    member_name_def->set_definer(member_node);
  }
  return PromotionResult{original_to_unique_name, sorted_defs_to_promote};
}

absl::StatusOr<Function*> TestFunctionTransformer::CreateNewFunction(
    const Function& fn, Module& new_module, TypeAnnotation* proc_def_type,
    TypeAnnotation* terminator_channel_type, const PromotionResult& promotion,
    const ClonedSetup& setup) {
  NameDef* new_fn_name = new_module.Make<NameDef>(fn.span(), "new",
                                                  /*definer=*/nullptr);

  // Add the terminator channel parameter.
  NameDef* terminator_channel_param_name =
      new_module.Make<NameDef>(fn.span(), std::string(kTerminatorChannelName),
                               /*definer=*/nullptr);
  Param* terminator_channel_param = new_module.Make<Param>(
      terminator_channel_param_name, terminator_channel_type);
  terminator_channel_param_name->set_definer(terminator_channel_param);

  TypeAnnotation* self_return_type = new_module.Make<SelfTypeAnnotation>(
      fn.span(), /*explicit_type=*/true, proc_def_type);

  // Create the struct instance, populated with the terminal channel and
  // promoted members.
  std::vector<std::pair<std::string, Expr*>> members;
  NameRef* terminator_channel_ref =
      new_module.Make<NameRef>(fn.span(), std::string(kTerminatorChannelName),
                               terminator_channel_param_name);
  members.push_back(
      {std::string(kTerminatorChannelName), terminator_channel_ref});

  for (const NameDef* def : promotion.sorted_defs_to_promote) {
    std::string unique_name = promotion.original_to_unique_name.at(def);
    NameDef* cloned_local_def = setup.original_to_cloned_local_def.at(def);
    NameRef* local_ref =
        new_module.Make<NameRef>(def->span(), unique_name, cloned_local_def);
    members.push_back({unique_name, local_ref});
  }

  StructInstance* struct_instance =
      new_module.Make<StructInstance>(fn.span(), proc_def_type,
                                      /*members=*/members);

  // Return the struct instance as the last statement.
  std::vector<Statement*> new_statements = setup.cloned_statements;
  new_statements.push_back(new_module.Make<Statement>(struct_instance));

  // fn new(terminator channel) -> Self {...}
  Function* new_fn = new_module.Make<Function>(
      fn.span(), new_fn_name, std::vector<ParametricBinding*>{},
      std::vector<Param*>{terminator_channel_param}, self_return_type,
      new_module.Make<StatementBlock>(fn.span(), new_statements,
                                      /*trailing_semi=*/false),
      FunctionTag::kNormal,
      /*is_public=*/false,
      /*is_stub=*/false);
  new_fn_name->set_definer(new_fn);

  return new_fn;
}

absl::StatusOr<TestFunctionTransformer::ClonedSetup>
TestFunctionTransformer::CloneSetupBlock(
    const Function& fn, const std::vector<Statement*>& const_statements,
    const PromotionResult& promotion_result, Module& new_module) {
  auto const_block = new_module.Make<StatementBlock>(
      fn.span(), const_statements, /*trailing_semi=*/false);

  absl::flat_hash_set<const NameDef*> defs_to_promote(
      promotion_result.sorted_defs_to_promote.begin(),
      promotion_result.sorted_defs_to_promote.end());

  // This cloner replacer replaces promoted variables' NameDefs with new
  // NameDefs using their unique names, and updates NameRefs pointing to them to
  // use the new unique names and link to the cloned NameDefs (resolved via
  // `old_to_new`).
  auto new_replacer =
      [&](const AstNode* node, Module* target_module,
          const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new)
      -> absl::StatusOr<std::optional<AstNode*>> {
    if (const auto* name_def = dynamic_cast<const NameDef*>(node)) {
      if (defs_to_promote.contains(name_def)) {
        std::string unique_name =
            promotion_result.original_to_unique_name.at(name_def);
        NameDef* new_name_def = target_module->Make<NameDef>(
            name_def->span(), unique_name, nullptr);
        return new_name_def;
      }
    }
    if (const auto* name_ref = dynamic_cast<const NameRef*>(node)) {
      if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
        const NameDef* original_def =
            std::get<const NameDef*>(name_ref->name_def());
        if (defs_to_promote.contains(original_def)) {
          std::string unique_name =
              promotion_result.original_to_unique_name.at(original_def);
          AnyNameDef new_name_def = original_def;
          auto it = old_to_new.find(original_def);
          const bool found = it != old_to_new.end();
          XLS_RET_CHECK(found);
          new_name_def = absl::down_cast<NameDef*>(it->second);
          return target_module->Make<NameRef>(name_ref->span(), unique_name,
                                              new_name_def,
                                              name_ref->in_parens());
        }
      }
    }
    return std::nullopt;
  };

  CloneReplacer chained_replacer =
      ChainCloneReplacers(new_replacer, PreserveTypeDefinitionsReplacer);

  XLS_ASSIGN_OR_RETURN(auto const_old_to_new,
                       CloneAstAndGetAllPairs(const_block, &new_module,
                                              std::move(chained_replacer)));
  auto* cloned_const_block =
      absl::down_cast<StatementBlock*>(const_old_to_new.at(const_block));
  std::vector<Statement*> cloned_const_statements(
      cloned_const_block->statements().begin(),
      cloned_const_block->statements().end());

  absl::flat_hash_map<const NameDef*, NameDef*> original_to_cloned_local_def;
  for (const NameDef* def : promotion_result.sorted_defs_to_promote) {
    original_to_cloned_local_def[def] =
        absl::down_cast<NameDef*>(const_old_to_new.at(def));
  }

  return ClonedSetup{cloned_const_statements, original_to_cloned_local_def};
}

absl::StatusOr<Impl*> TestFunctionTransformer::TransformToTestProc(
    const Function& fn, Module& new_module) {
  // Find all const and non-const statements in 'fn' (which is in the old
  // module); "new" will mostly contain the const statements, and "next" the
  // non-const.
  std::vector<Statement*> const_statements;
  std::vector<Statement*> non_const_statements;
  for (auto stmt : fn.body()->statements()) {
    if (IsConstExpr(stmt)) {
      const_statements.push_back(stmt);
    } else {
      non_const_statements.push_back(stmt);
    }
  }

  // Any name ref that is in the name def needs to be promoted
  XLS_ASSIGN_OR_RETURN(absl::flat_hash_set<const NameDef*> defs_to_promote,
                       find_promoted(const_statements, non_const_statements));

  XLS_ASSIGN_OR_RETURN(ProcDefAndTypeAnnotation proc_def_and_type,
                       CreateProcDef(fn, new_module));
  ProcDef* proc_def = proc_def_and_type.proc_def;

  XLS_ASSIGN_OR_RETURN(
      PromotionResult promotion_result,
      PromoteVariablesToFields(defs_to_promote, new_module, *proc_def));

  TypeRef* type_ref = new_module.Make<TypeRef>(fn.span(), proc_def);
  TypeAnnotation* proc_def_type = new_module.Make<TypeRefTypeAnnotation>(
      fn.span(), type_ref, std::vector<ExprOrType>{});

  XLS_ASSIGN_OR_RETURN(
      ClonedSetup setup,
      CloneSetupBlock(fn, const_statements, promotion_result, new_module));

  XLS_ASSIGN_OR_RETURN(
      Function * new_fn,
      CreateNewFunction(fn, new_module, proc_def_type,
                        proc_def_and_type.terminator_channel_type,
                        promotion_result, setup));

  XLS_ASSIGN_OR_RETURN(
      Function * next_fn,
      CreateNextFunction(fn, new_module, proc_def_type, non_const_statements,
                         defs_to_promote,
                         promotion_result.original_to_unique_name, type_info_));

  Impl* impl = new_module.Make<Impl>(
      fn.span(), proc_def_type,
      /*members=*/std::vector<ImplMember>{new_fn, next_fn},
      /*is_public=*/false);
  proc_def->set_impl(impl);
  new_fn->set_impl(impl);

  return impl;
}

absl::StatusOr<std::unique_ptr<Module>>
TestFunctionTransformer::TransformTestFunctions() {
  // Cannot reserve space in the vector because we don't know how many functions
  // will be transformed.
  std::vector<const AstNode*> test_functions_with_spawn;

  // Find all test functions with spawns; these will be replaced.
  for (const auto& member : source_module_.top()) {
    if (std::holds_alternative<TestFunction*>(member)) {
      TestFunction* func = std::get<TestFunction*>(member);
      SpawnFinder spawn_finder;
      XLS_RETURN_IF_ERROR(func->Accept(&spawn_finder));
      if (spawn_finder.has_spawn()) {
        test_functions_with_spawn.push_back(func);
      }
    }
  }

  if (test_functions_with_spawn.empty()) {
    return nullptr;
  }

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Module> cloned_module,
      CloneModuleRemovingMembers(source_module_, test_functions_with_spawn));
  // For each test function with a spawn from the original module, transform
  // it into a TestProc and add to the *cloned* module.
  for (auto* node : test_functions_with_spawn) {
    const TestFunction* func = dynamic_cast<const TestFunction*>(node);
    XLS_ASSIGN_OR_RETURN(Impl * impl,
                         TransformToTestProc(func->fn(), *cloned_module));
    // TODO(davidplass): Add collision error handler using the node's location
    XLS_RETURN_IF_ERROR(
        cloned_module->AddTop(impl, /*make_collision_error=*/nullptr));
  }
  return cloned_module;
}
}  // namespace xls::dslx
