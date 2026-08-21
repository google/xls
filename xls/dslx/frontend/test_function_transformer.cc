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

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xls/common/attribute_data.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/module.h"

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
    std::vector<Statement*>& non_const_statements) {
  NameDef* self_name = new_module.Make<NameDef>(fn.span(), "self",
                                                /*definer=*/nullptr);
  TypeAnnotation* self_param_type = new_module.Make<SelfTypeAnnotation>(
      fn.span(), /*explicit_type=*/false, proc_def_type);
  Param* self_param = new_module.Make<Param>(self_name, self_param_type);
  self_name->set_definer(self_param);

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

  // Wrap the send in a Statement and add to the non-constexpr statements list.
  Statement* send_stmt = new_module.Make<Statement>(send_invocation);
  std::vector<Statement*> next_statements = non_const_statements;
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

absl::StatusOr<Function*> CreateNewFunction(
    const Function& fn, Module& new_module, TypeAnnotation* proc_def_type,
    TypeAnnotation* terminator_channel_type,
    const std::vector<Statement*>& const_statements) {
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

  // Create the struct instance, populated with (for now) the terminal channel.
  std::vector<std::pair<std::string, Expr*>> members;
  NameRef* terminator_channel_ref =
      new_module.Make<NameRef>(fn.span(), std::string(kTerminatorChannelName),
                               terminator_channel_param_name);
  members.push_back(
      {std::string(kTerminatorChannelName), terminator_channel_ref});
  StructInstance* struct_instance =
      new_module.Make<StructInstance>(fn.span(), proc_def_type,
                                      /*members=*/members);

  // Return the struct instance as the last statement.
  std::vector<Statement*> new_statements = const_statements;
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

}  // namespace

bool TestFunctionTransformer::IsConstExpr(const Statement* stmt) {
  // Need to unwrap the statement to get the underlying expression to test if
  // it's a known const expr.
  return std::visit(
      [this](const auto& node) { return type_info_.IsKnownConstExpr(node); },
      stmt->wrapped());
}

absl::StatusOr<Impl*> TestFunctionTransformer::TransformToTestProc(
    const Function& fn, Module& new_module) {
  // Find all const and non-const statements in 'fn' (which is in the old
  // module); "new" will mostly contain the const statements, and "next" the
  // non-const.
  std::vector<Statement*> const_statements;
  std::vector<Statement*> non_const_statements;
  // Clone all the statements into the new module; otherwise the newly
  // constructed methods (in the new module) will reference statements in the
  // old module. (Note, these statements were not cloned into the new module
  // yet, because we intentionally skipped the test function).
  XLS_ASSIGN_OR_RETURN(auto old_to_new,
                       CloneAstAndGetAllPairs(fn.body(), &new_module));
  auto* cloned_body =
      absl::down_cast<StatementBlock*>(old_to_new.at(fn.body()));
  for (int i = 0; i < cloned_body->statements().size(); ++i) {
    Statement* stmt = fn.body()->statements().at(i);
    Statement* cloned_stmt = cloned_body->statements().at(i);
    // We have to test the original statements for const since the cloned
    // statements haven't been typechecked yet.
    if (IsConstExpr(stmt)) {
      const_statements.push_back(cloned_stmt);
    } else {
      non_const_statements.push_back(cloned_stmt);
    }
  }

  XLS_ASSIGN_OR_RETURN(ProcDefAndTypeAnnotation proc_def_and_type,
                       CreateProcDef(fn, new_module));
  ProcDef* proc_def = proc_def_and_type.proc_def;

  // TODO(davidplass): Promote variables from the 'new' function that are
  // subsequently used by the 'next' function into fields on the synthesized
  // proc. Modify their creation (in 'new') and use (in 'new' and 'next') to
  // reflect that they are fields and not local variables.

  TypeRef* type_ref = new_module.Make<TypeRef>(fn.span(), proc_def);
  TypeAnnotation* proc_def_type = new_module.Make<TypeRefTypeAnnotation>(
      fn.span(), type_ref, std::vector<ExprOrType>{});

  XLS_ASSIGN_OR_RETURN(
      Function * new_fn,
      CreateNewFunction(fn, new_module, proc_def_type,
                        proc_def_and_type.terminator_channel_type,
                        const_statements));

  XLS_ASSIGN_OR_RETURN(
      Function * next_fn,
      CreateNextFunction(fn, new_module, proc_def_type, non_const_statements));

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
  for (const auto& name : source_module_.GetTestNames()) {
    XLS_ASSIGN_OR_RETURN(TestFunction * func, source_module_.GetTest(name));
    SpawnFinder spawn_finder;
    XLS_RETURN_IF_ERROR(func->Accept(&spawn_finder));
    if (spawn_finder.has_spawn()) {
      test_functions_with_spawn.push_back(func);
    }
  }

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Module> cloned_module,
      CloneModuleRemovingMembers(source_module_, test_functions_with_spawn));
  if (test_functions_with_spawn.empty()) {
    // TODO(davidplass): Consider not cloning if there are no test functions.
    return cloned_module;
  }

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
