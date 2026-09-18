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

#include "xls/dslx/frontend/semantics_analysis.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/attribute_data.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/builtin_stubs_utils.h"
#include "xls/dslx/frontend/fuzz_domain_rewriter.h"
#include "xls/dslx/frontend/lambda_rewriter.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {

namespace {
// Warns if the next-to-last statement in a block has a trailing semi and the
// last statement is a nil tuple expression, as this is redundant; i.e.
//
//    {
//      foo;
//      ()  <-- useless, semi on previous statement implies it
//    }
void DetectUselessTrailingTuplePattern(const StatementBlock* block,
                                       WarningCollector& warning_collector) {
  // TODO(https://github.com/google/xls/issues/1124) 2023-08-31 Proc config
  // parsing functions synthesize a tuple at the end, and we don't want to flag
  // that since the user didn't even create it.
  if (block->parent()->kind() == AstNodeKind::kFunction &&
      dynamic_cast<const Function*>(block->parent())->tag() ==
          FunctionTag::kProcConfig) {
    return;
  }

  // Need at least a statement (i.e. with semicolon after it) and an
  // expression-statement at the end to match this pattern.
  if (block->statements().size() < 2) {
    return;
  }

  // Make sure we ignore this if we're only following an implicit prologue (as
  // is used to convert implicit-token-parameter semantics for now).
  // TODO(https://github.com/google/xls/issues/1401): Remove once we no longer
  // support implicit token parameter semantics.
  const Statement* next_to_last_stmt =
      block->statements()[block->statements().size() - 2];
  if (next_to_last_stmt->GetSpan().has_value() &&
      next_to_last_stmt->GetSpan()->limit() <=
          block->span().start().BumpCol()) {
    return;
  }

  // Trailing statement has to be an expression-statement.
  const Statement* last_stmt = block->statements().back();
  if (!std::holds_alternative<Expr*>(last_stmt->wrapped())) {
    return;
  }

  // It has to be a tuple.
  const auto* last_expr = std::get<Expr*>(last_stmt->wrapped());
  auto* trailing_tuple = dynamic_cast<const XlsTuple*>(last_expr);
  if (trailing_tuple == nullptr) {
    return;
  }

  // Tuple has to be nil.
  if (!trailing_tuple->empty()) {
    return;
  }

  warning_collector.Add(
      trailing_tuple->span(), WarningKind::kTrailingTupleAfterSemi,
      absl::StrFormat("Block has a trailing nil (empty) tuple after a "
                      "semicolon -- this is implied, please remove it"));
}

bool IsBlockJustUnitTuple(const StatementBlock& block) {
  if (block.empty()) {
    return true;
  }
  if (block.size() != 1) {
    return false;
  }
  const Statement* statement = block.statements().front();
  const Statement::Wrapped& wrapped = statement->wrapped();
  if (!std::holds_alternative<Expr*>(wrapped)) {
    return false;
  }
  const auto* expr = std::get<Expr*>(wrapped);
  auto* tuple = dynamic_cast<const XlsTuple*>(expr);
  if (tuple == nullptr) {
    return false;
  }
  return tuple->empty();
}

bool IsBlockWithOneFailStmt(const StatementBlock& block) {
  if (block.size() != 1) {
    return false;
  }
  const Statement* statement = block.statements().front();
  const Statement::Wrapped& wrapped = statement->wrapped();
  if (!std::holds_alternative<Expr*>(wrapped)) {
    return false;
  }
  const auto* expr = std::get<Expr*>(wrapped);
  auto* invocation = dynamic_cast<const Invocation*>(expr);
  if (invocation == nullptr) {
    return false;
  }
  Expr* callee = invocation->callee();
  auto* name_ref = dynamic_cast<const NameRef*>(callee);
  if (name_ref == nullptr) {
    return false;
  }
  AnyNameDef any_name_def = name_ref->name_def();
  if (!std::holds_alternative<BuiltinNameDef*>(any_name_def)) {
    return false;
  }
  auto* bnd = std::get<BuiltinNameDef*>(any_name_def);
  return bnd->identifier() == "fail!";
}

void WarnOnConditionalContainingJustFailStatement(
    const Conditional& node, WarningCollector& warning_collector) {
  const StatementBlock* consequent = node.consequent();
  std::variant<StatementBlock*, Conditional*> alternate_ast_node =
      node.alternate();
  if (!std::holds_alternative<StatementBlock*>(alternate_ast_node)) {
    return;
  }
  const StatementBlock* alternate =
      std::get<StatementBlock*>(alternate_ast_node);

  if (IsBlockWithOneFailStmt(*consequent) && IsBlockJustUnitTuple(*alternate)) {
    std::string message = absl::StrFormat(
        "`if test { fail!(...) } else { () }` pattern should be replaced with "
        "`assert!(test, ...)`");
    warning_collector.Add(node.span(), WarningKind::kShouldUseAssert,
                          std::move(message));
  }
}

// Sees if the function is named with a `_test` suffix but not marked with a
// test annotation -- this is likely to be a user mistake, so we give a warning.
void WarnIfConfusinglyNamedLikeTest(const Function& f,
                                    WarningCollector& warning_collector) {
  if (!absl::EndsWith(f.identifier(), "_test")) {
    return;
  }
  AstNode* parent = f.parent();
  if (parent == nullptr || parent->kind() != AstNodeKind::kTestFunction) {
    warning_collector.Add(
        f.span(), WarningKind::kMisleadingFunctionName,
        absl::StrFormat("Function `%s` ends with `_test` but is "
                        "not marked as a unit test via #[test]",
                        f.identifier()));
  }
}

// Warn folks if it's not following
// https://doc.rust-lang.org/1.0.0/style/style/naming/README.html
void WarnOnInappropriateMemberName(std::string_view member_name,
                                   const Span& span, const Module& module,
                                   WarningCollector& warning_collector) {
  if (!IsAcceptablySnakeCase(member_name) &&
      !module.attributes().contains(
          ModuleAttribute::kAllowNonstandardMemberNaming)) {
    warning_collector.Add(
        span, WarningKind::kMemberNaming,
        absl::StrFormat("Standard style is snake_case for struct member names; "
                        "got: `%s`",
                        member_name));
  }
}

// Checks whether an expression may have side-effects. It may have
// false-positives, for example an invocation may actually have no side-effects,
// but because we do not recursively look into the callee, we conservatively
// mark it as potentially side-effect causing. Conversely, if an expression is
// determined to have no side-effects, it is definitely useless and thus should
// be flagged.
class SideEffectExpressionFinder : public AstNodeVisitorWithDefault {
 public:
  SideEffectExpressionFinder() : has_side_effect_(false) {}

  absl::Status HandleInvocation(const Invocation* node) override {
    has_side_effect_ = true;
    return absl::OkStatus();
  }

  absl::Status HandleSpawn(const Spawn* node) override {
    has_side_effect_ = true;
    return absl::OkStatus();
  }

  absl::Status HandleFormatMacro(const FormatMacro* node) override {
    has_side_effect_ = true;
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (const AstNode* child : node->GetChildren(/*want_types=*/false)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
      // We only need to find one side-effect.
      if (has_side_effect_) {
        return absl::OkStatus();
      }
    }
    return absl::OkStatus();
  }

  bool HasSideEffect() const { return has_side_effect_; }

 private:
  bool has_side_effect_;
};

// An impl-style proc automatically derives the `Spawn` attribute whether the
// user says this or not. This visitor adds it to the `derive` list for each
// proc where not present, creating the whole `derive` attribute if necessary to
// achieve this.
class AddSpawnTraitToProcDefs : public AstNodeRecursiveVisitor {
 public:
  AddSpawnTraitToProcDefs() : AstNodeRecursiveVisitor(/*want_types=*/true) {}

  absl::Status HandleProcDef(const ProcDef* node) override {
    std::optional<Attribute*> existing_attribute =
        GetAttribute(node, AttributeKind::kDerive);
    if (existing_attribute.has_value()) {
      for (const AttributeData::Argument& argument :
           (*existing_attribute)->args()) {
        if (std::holds_alternative<std::string>(argument) &&
            std::get<std::string>(argument) == kSpawnTraitName) {
          return absl::OkStatus();
        }
      }

      (*existing_attribute)->AddArgument(std::string(kSpawnTraitName));
      return absl::OkStatus();
    }

    Attribute* new_attribute = node->owner()->Make<Attribute>(
        Span::None(), Span::None(),
        AttributeData(AttributeKind::kDerive,
                      std::vector<AttributeData::Argument>{
                          std::string(kSpawnTraitName)}));
    const_cast<ProcDef*>(node)->AddAttribute(new_attribute);
    return absl::OkStatus();
  }

 private:
  static constexpr std::string_view kSpawnTraitName = "Spawn";
};

class PreTypecheckPass : public AstNodeRecursiveVisitor {
 public:
  PreTypecheckPass(WarningCollector& warning_collector,
                   const FileTable& file_table)
      : AstNodeRecursiveVisitor(/*want_types=*/true),
        warning_collector_(warning_collector),
        file_table_(file_table) {}

  absl::Status HandleStatementBlock(const StatementBlock* node) override {
    for (size_t i = 0; i < node->statements().size(); ++i) {
      const Statement* s = node->statements()[i];
      // We only want to check the last statement for "useless
      // expression-statement" property if it is not yielding a value from a
      // block; e.g.
      //
      //    {
      //      my_invocation!();
      //      u32:42  // <- ok, no trailing semi
      //    }
      //
      // vs
      //
      //    {
      //      my_invocation!();
      //      u32:42;  // <- useless, trailing semi means block yields nil
      //    }
      bool should_check_useless_expression =
          std::holds_alternative<Expr*>(s->wrapped()) &&
          (i != node->statements().size() - 1 || node->trailing_semi());

      if (should_check_useless_expression) {
        SideEffectExpressionFinder visitor;
        XLS_RETURN_IF_ERROR(s->Accept(&visitor));
        if (!visitor.HasSideEffect()) {
          warning_collector_.Add(
              *(s->GetSpan()), WarningKind::kUselessExpressionStatement,
              absl::StrFormat("Expression statement `%s` appears "
                              "useless (i.e. has no side-effects)",
                              s->ToString()));
        }
      }
    }
    DetectUselessTrailingTuplePattern(node, warning_collector_);
    return DefaultHandler(node);
  }

  absl::Status HandleConditional(const Conditional* node) override {
    WarnOnConditionalContainingJustFailStatement(*node, warning_collector_);
    return DefaultHandler(node);
  }

  absl::Status HandleFuzzTestFunction(const FuzzTestFunction* node) override {
    const Function& f = node->fn();
    if (f.IsParametric()) {
      return TypeInferenceErrorStatus(
          f.GetSpan().value(), nullptr,
          absl::StrFormat("Cannot fuzz test parametric function `%s`",
                          f.identifier()),
          file_table_);
    }
    if (f.params().empty()) {
      return TypeInferenceErrorStatus(
          f.GetSpan().value(), nullptr,
          absl::StrFormat("Can only fuzz test functions with at least 1 "
                          "parameter; function `%s` has 0",
                          node->identifier()),
          file_table_);
    }
    if (!node->domains().has_value()) {
      return DefaultHandler(node);
    }

    const XlsTuple* domains = *node->domains();
    int64_t domain_count = domain_count = domains->members().size();
    if (domain_count != f.params().size()) {
      return TypeInferenceErrorStatus(
          node->GetSpan().value(), nullptr,
          absl::StrFormat("fuzz_test attribute has %d domain argument%s, but "
                          "function `%s` has %d parameter%s",
                          domain_count, domain_count == 1 ? "" : "s",
                          f.identifier(), f.params().size(),
                          f.params().size() == 1 ? "" : "s"),
          file_table_);
    }
    return DefaultHandler(node);
  }

  absl::Status HandleFunction(const Function* node) override {
    WarnIfConfusinglyNamedLikeTest(*node, warning_collector_);

    // If this function is an instance method on a parametric struct, ensure
    // that it doesn't duplicate any parametric bindings from the struct.
    if (node->IsMethodOnParametricStruct() && node->IsParametric()) {
      const auto* struct_ref = absl::down_cast<const TypeRefTypeAnnotation*>(
          (*node->impl())->struct_ref());
      StructDef* struct_def =
          std::get<StructDef*>(struct_ref->type_ref()->type_definition());

      ParametricBindings bindings(node->parametric_bindings());
      for (ParametricBinding* parametric_binding :
           struct_def->parametric_bindings()) {
        if (node->parametric_keys().contains(
                parametric_binding->identifier())) {
          return xls::dslx::ParseErrorStatus(
              bindings.at(parametric_binding->identifier())->span(),
              absl::StrFormat("Parametric binding `%s` shadows binding from "
                              "struct definition",
                              parametric_binding->identifier()),
              file_table_);
        }
      }
    }
    return DefaultHandler(node);
  }

  absl::Status HandleStructDef(const StructDef* node) override {
    for (const auto* member : node->members()) {
      WarnOnInappropriateMemberName(member->name(), member->name_def()->span(),
                                    *node->owner(), warning_collector_);
    }
    return DefaultHandler(node);
  }

  absl::Status HandleProc(const Proc* node) override {
    in_legacy_proc_ = true;
    XLS_RETURN_IF_ERROR(DefaultHandler(node));
    in_legacy_proc_ = false;
    return absl::OkStatus();
  }

  absl::Status HandleSelfTypeAnnotation(
      const SelfTypeAnnotation* node) override {
    if (in_legacy_proc_) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          "Use of `Self` inside legacy procs is not supported.", file_table_);
    }
    return absl::OkStatus();
  }

  absl::Status HandleProcDef(const ProcDef* node) override {
    for (const auto* member : node->members()) {
      WarnOnInappropriateMemberName(member->name(), member->name_def()->span(),
                                    *node->owner(), warning_collector_);
    }
    return DefaultHandler(node);
  }

  absl::Status HandleXlsTuple(const XlsTuple* node) override {
    // If the tuple is generated by the parser as part of a FuzzTestFunction,
    // skip the warning.
    if (node->parent() != nullptr &&
        node->parent()->kind() == AstNodeKind::kFuzzTestFunction) {
      return DefaultHandler(node);
    }

    // Give a warning if the tuple is on a single line, is more than one
    // element, but has a trailing comma.
    //
    // Note: warning diagnostics and type checking are currently fused together,
    // but this is a pure post-parsing warning -- currently type checking the
    // pass that has a warning collector available.
    if (node->span().start().lineno() == node->span().limit().lineno() &&
        node->members().size() > 1 && node->has_trailing_comma()) {
      std::string message = absl::StrFormat(
          "Tuple expression (with >1 element) is on a single "
          "line, but has a trailing comma.");
      warning_collector_.Add(node->span(),
                             WarningKind::kSingleLineTupleTrailingComma,
                             std::move(message));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleConstantDef(const ConstantDef* node) override {
    WarnOnInappropriateConstantName(node->identifier(), node->span(),
                                    *node->owner(), &warning_collector_);
    return DefaultHandler(node);
  }

  absl::Status HandleLet(const Let* node) override {
    if (IsWildcardLeaf(node->pattern())) {
      warning_collector_.Add(GetPatternSpan(node->pattern()),
                             WarningKind::kUselessLetBinding,
                             "`let _ = expr;` statement can be simplified to "
                             "`expr;` -- there is no "
                             "need for a `let` binding here");
    }

    if (node->is_const()) {
      NameDef* name_def = GetPatternNameDefs(node->pattern())[0];
      WarnOnInappropriateConstantName(name_def->identifier(), node->span(),
                                      *node->owner(), &warning_collector_);
    }

    return DefaultHandler(node);
  }

 private:
  WarningCollector& warning_collector_;

  const FileTable& file_table_;
  bool in_legacy_proc_ = false;
};

class CollectUseDef : public AstNodeRecursiveVisitor {
 public:
  CollectUseDef() : AstNodeRecursiveVisitor(/*want_types=*/true) {}

  absl::Status HandleNameDef(const NameDef* node) override {
    // Users can silence unused warnings by prefixing an identifier with an
    // underscore to make it more well documented; e.g.
    //  let (one, _two, three) = ...;  // _two can go unused
    if (!absl::StartsWith(node->identifier(), "_")) {
      defs_.insert(node);
    }
    return absl::OkStatus();
  }

  absl::Status HandleNameRef(const NameRef* node) override {
    AddUse(node->name_def());
    return absl::OkStatus();
  }

  absl::Status HandleTypeRef(const TypeRef* node) override {
    AddUse(TypeDefinitionGetNameDef(node->type_definition()));
    return absl::OkStatus();
  }

  absl::Status HandleTypeAlias(const TypeAlias* node) override {
    // Do not mark type alias as unused.
    return node->type_annotation().Accept(this);
  }

  const absl::flat_hash_set<const NameDef*>& Defs() const { return defs_; }
  const absl::flat_hash_set<const NameDef*>& Uses() const { return uses_; }

 private:
  void AddUse(const AnyNameDef& any_name_def) {
    if (const NameDef* const* name_def =
            std::get_if<const NameDef*>(&any_name_def)) {
      if (!uses_.insert(*name_def).second) {
        return;
      }

      // If any name in a tuple binding is used, consider all names in that
      // binding used. Users typically want to keep meaningful names for each
      // component even when only some components are referenced.
      const AstNode* node = *name_def;
      while (node->parent() &&
             node->parent()->kind() == AstNodeKind::kTuplePattern) {
        node = absl::down_cast<const TuplePattern*>(node->parent());
      }
      if (node != *name_def) {
        ConstPatternTree pattern = absl::down_cast<const TuplePattern*>(node);
        for (const NameDef* pattern_name_def : GetPatternNameDefs(pattern)) {
          uses_.insert(pattern_name_def);
        }
      }
    }
  }

  absl::flat_hash_set<const NameDef*> defs_;
  absl::flat_hash_set<const NameDef*> uses_;
};

// Replaces the type annotation for proc state members with
// BuiltinProcState<TheOriginalType>. In legacy procs, this affects the next()
// param nodes. In impl-style procs, it affects the declared state members of
// the proc.
class ProcStateVisitor : public AstNodeRecursiveVisitor {
 public:
  // Creates a visitor using `import_data` and the given `StructDef` for the
  // builtin `BuiltinProcState` struct.
  ProcStateVisitor(ImportData& import_data, StructDef* state_struct_def,
                   const TypecheckModuleFn& typecheck_imported_module)
      : import_data_(import_data),
        state_struct_def_(state_struct_def),
        typecheck_imported_module_(typecheck_imported_module) {}

  absl::Status HandleFunction(const Function* node) override {
    // Legacy proc has multiple pointers to functions within the AST. This could
    // cause the HandleParam visitor to run multiple times on the same function.
    if (!processed_fns_.insert(node).second) {
      return absl::OkStatus();
    }
    return DefaultHandler(node);
  }

  absl::Status HandleParam(const Param* node) override {
    if (node->parent() != nullptr &&
        node->parent()->kind() == AstNodeKind::kFunction) {
      const Function* fn = absl::down_cast<const Function*>(node->parent());
      if (fn->tag() == FunctionTag::kProcNext) {
        const_cast<Param*>(node)->set_type_annotation(
            CreateStateTypeAnnotation(node->owner(), node->type_annotation(),
                                      node->type_annotation()->span()));
      }
    }
    return absl::OkStatus();
  }

  absl::Status HandleProcDef(const ProcDef* node) override {
    for (StructMemberNode* member : node->members()) {
      TypeAnnotation* type = member->type();

      // Don't do the T -> State<T> conversion for channels, channel arrays, or
      // sub-procs.
      if (IsChannelOrChannelArrayAnnotation(type)) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          std::optional<const StructDefBase*> def,
          GetStructOrProcDef(type, import_data_, typecheck_imported_module_));
      if (def.has_value() && (*def)->kind() == AstNodeKind::kProcDef) {
        continue;
      }

      if (auto* tr_type = dynamic_cast<const TypeRefTypeAnnotation*>(type)) {
        if (ToAstNode(tr_type->type_ref()->type_definition()) ==
            state_struct_def_) {
          // Already wrapped
          continue;
        }
      }

      member->set_type(
          CreateStateTypeAnnotation(node->owner(), type, type->span()));
    }
    return absl::OkStatus();
  }

 private:
  TypeAnnotation* CreateStateTypeAnnotation(Module* module,
                                            TypeAnnotation* underlying_type,
                                            const Span& span) {
    TypeRef* state_typeref = module->Make<TypeRef>(span, state_struct_def_);
    std::vector<ExprOrType> parametrics = {underlying_type};
    return module->Make<TypeRefTypeAnnotation>(span, state_typeref,
                                               parametrics);
  }

  ImportData& import_data_;
  StructDef* const state_struct_def_;
  const TypecheckModuleFn& typecheck_imported_module_;
  absl::flat_hash_set<const Function*> processed_fns_;
};

// Generates a trivial `fn next(self) {}` in a ProcDef that does not have one.
// Such a proc may just be stitching subprocs together and not doing any runtime
// work itself. The generated `next` function has an attribute of kind
// `kTrivialNext` for downstream recognition. For example, IR conversion
// requires the top proc to have a real next function.
class ProcDefTrivialNextGenerator : public AstNodeRecursiveVisitor {
 public:
  ProcDefTrivialNextGenerator()
      : AstNodeRecursiveVisitor(/*want_types=*/true) {}

  absl::Status HandleProcDef(const ProcDef* node) override {
    if (!node->impl().has_value()) {
      return absl::OkStatus();
    }
    if (GetProcNextFunction(node).has_value()) {
      return absl::OkStatus();
    }

    Module* module = node->owner();
    XLS_RET_CHECK(node->impl().has_value());
    Impl* impl = *node->impl();

    NameDef* fn_name_def =
        module->Make<NameDef>(node->span(), "next", /*definer=*/nullptr);
    SelfTypeAnnotation* self_type = module->Make<SelfTypeAnnotation>(
        node->span(), /*explicit_type=*/false, impl->struct_ref());
    NameDef* self_name_def =
        module->Make<NameDef>(node->span(), "self", /*definer=*/nullptr);
    Param* param = module->Make<Param>(self_name_def, self_type);
    self_name_def->set_definer(param);
    std::vector<Param*> params = {param};
    StatementBlock* body = module->Make<StatementBlock>(
        node->span(), std::vector<Statement*>{}, /*trailing_semi=*/true,
        /*has_braces=*/true);
    Function* next_fn = module->Make<Function>(
        node->span(), fn_name_def, std::vector<ParametricBinding*>{}, params,
        /*return_type=*/nullptr, body, FunctionTag::kNormal,
        /*is_public=*/false, /*is_stub=*/false);
    next_fn->set_impl(impl);
    fn_name_def->set_definer(next_fn);

    Attribute* trivial_next_attr = module->Make<Attribute>(
        Span::None(), Span::None(),
        AttributeData(AttributeKind::kTrivialNext,
                      std::vector<AttributeData::Argument>{}));
    next_fn->AddAttribute(trivial_next_attr);
    next_fn->SetParentNonLexical(impl);
    next_fn->set_compiler_derived(true);
    impl->AddMember(next_fn);
    return absl::OkStatus();
  }
};

absl::StatusOr<AstNode*> CloneNodeWithSubstitutions(
    const AstNode* root, Module& target_module,
    const absl::flat_hash_map<const NameDef*, ExprOrType>& binding_map) {
  CloneReplacer replacer =
      [&](const AstNode* node, Module* m,
          const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new)
      -> absl::StatusOr<std::optional<AstNode*>> {
    if (node->kind() == AstNodeKind::kTypeAnnotation) {
      const auto* ta = absl::down_cast<const TypeAnnotation*>(node);
      if (ta->IsAnnotation<TypeVariableTypeAnnotation>()) {
        const auto* tvta = ta->AsAnnotation<TypeVariableTypeAnnotation>();
        if (std::holds_alternative<const NameDef*>(
                tvta->type_variable()->name_def())) {
          const NameDef* nd =
              std::get<const NameDef*>(tvta->type_variable()->name_def());
          auto it = binding_map.find(nd);
          if (it != binding_map.end() &&
              std::holds_alternative<TypeAnnotation*>(it->second)) {
            return CloneNodeWithSubstitutions(
                std::get<TypeAnnotation*>(it->second), *m, binding_map);
          }
        }
      }
    } else if (node->kind() == AstNodeKind::kNameRef) {
      const auto* nr = absl::down_cast<const NameRef*>(node);
      if (std::holds_alternative<const NameDef*>(nr->name_def())) {
        const NameDef* nd = std::get<const NameDef*>(nr->name_def());
        auto it = binding_map.find(nd);
        if (it != binding_map.end() &&
            std::holds_alternative<Expr*>(it->second)) {
          return CloneNodeWithSubstitutions(std::get<Expr*>(it->second), *m,
                                            binding_map);
        }
      }
    }
    return PreserveTypeDefinitionsReplacer(node, m, old_to_new);
  };
  XLS_ASSIGN_OR_RETURN(auto pairs, CloneAstAndGetAllPairs(root, &target_module,
                                                          std::move(replacer)));
  return pairs.at(root);
}

absl::Status RewriteFunctionAliasToWrapper(
    Module& module, ImportData& import_data, AliasDef* alias,
    Function* target_fn, const AliasDef::Target& target,
    const std::vector<ExprOrType>& explicit_parametrics) {
  if (explicit_parametrics.size() > target_fn->parametric_bindings().size()) {
    return ArgCountMismatchErrorStatus(
        alias->span(),
        absl::StrFormat(
            "Too many parametric values supplied; limit: %d given: %d",
            target_fn->parametric_bindings().size(),
            explicit_parametrics.size()),
        import_data.file_table());
  }
  absl::flat_hash_map<const NameDef*, ExprOrType> binding_map;
  for (size_t i = 0; i < target_fn->parametric_bindings().size(); ++i) {
    const ParametricBinding* binding = target_fn->parametric_bindings()[i];
    if (i < explicit_parametrics.size()) {
      binding_map[binding->name_def()] = explicit_parametrics[i];
    } else if (binding->default_expr_or_type().has_value()) {
      binding_map[binding->name_def()] = *binding->default_expr_or_type();
    }
  }

  std::vector<Param*> wrapper_params;
  std::vector<Expr*> call_args;
  wrapper_params.reserve(target_fn->params().size());
  call_args.reserve(target_fn->params().size());
  for (const Param* old_param : target_fn->params()) {
    XLS_ASSIGN_OR_RETURN(
        AstNode * cloned_type_node,
        CloneNodeWithSubstitutions(old_param->type_annotation(), module,
                                   binding_map));
    auto* new_param_type = absl::down_cast<TypeAnnotation*>(cloned_type_node);
    NameDef* new_param_name_def = module.Make<NameDef>(
        Span(new_param_type->span().start(), new_param_type->span().start()),
        old_param->identifier(),
        /*definer=*/nullptr);
    Param* new_param = module.Make<Param>(new_param_name_def, new_param_type);
    new_param_name_def->set_definer(new_param);
    wrapper_params.push_back(new_param);
    call_args.push_back(module.Make<NameRef>(
        alias->span(), new_param->identifier(), new_param_name_def));
  }

  TypeAnnotation* wrapper_return_type = nullptr;
  if (target_fn->return_type() != nullptr) {
    XLS_ASSIGN_OR_RETURN(AstNode * cloned_ret_node,
                         CloneNodeWithSubstitutions(target_fn->return_type(),
                                                    module, binding_map));
    wrapper_return_type = absl::down_cast<TypeAnnotation*>(cloned_ret_node);
  }

  XLS_ASSIGN_OR_RETURN(
      AstNode * cloned_callee_node,
      CloneNodeWithSubstitutions(ToAstNode(target), module, {}));
  Expr* callee_expr = absl::down_cast<Expr*>(cloned_callee_node);

  std::vector<ExprOrType> invocation_parametrics;
  invocation_parametrics.reserve(explicit_parametrics.size());
  for (ExprOrType p : explicit_parametrics) {
    XLS_ASSIGN_OR_RETURN(AstNode * cloned_p,
                         CloneNodeWithSubstitutions(ToAstNode(p), module, {}));
    invocation_parametrics.push_back(ToExprOrType(cloned_p));
  }

  Invocation* call = module.Make<Invocation>(alias->span(), callee_expr,
                                             call_args, invocation_parametrics);
  Statement* stmt = module.Make<Statement>(call);
  StatementBlock* body =
      module.Make<StatementBlock>(alias->span(), std::vector<Statement*>{stmt},
                                  /*trailing_semi=*/false);

  NameDef* fn_name_def = alias->name_def();
  Function* wrapper_fn = module.Make<Function>(
      alias->span(), fn_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>{}, wrapper_params,
      wrapper_return_type, body, FunctionTag::kNormal, alias->is_public(),
      /*is_stub=*/false);
  fn_name_def->set_definer(wrapper_fn);
  return module.ReplaceTopMember(alias, wrapper_fn);
}

// User-written `proc` aliases to impl-based procs (`ProcDef`) are rejected in
// favor of standard `type` aliases (`type Alias = ImplProc<...>;`), which act
// as transparent type aliases for `.spawn()` and subproc member composition.
//
// However, compiler-generated synthetic top-level entry aliases
// (`alias->is_synthetic()`, created when `--top` targets a parametric `ProcDef`
// or a `TypeAlias` to a `ProcDef`) still require synthesizing a non-parametric
// wrapper `ProcDef`. This is because a `TypeAlias` does not invoke
// `ImplProc<...>::new(...)`, leaving `TypeInfo::GetCanonicalProcInitializers`
// empty unless a synthetic wrapper proc calls
// `ImplProc<...>::new(...).spawn()`.
absl::Status RewriteProcDefAliasToWrapper(
    Module& module, ImportData& import_data, AliasDef* alias,
    ProcDef* target_proc_def, const AliasDef::Target& target,
    const std::vector<ExprOrType>& explicit_parametrics) {
  if (!alias->is_synthetic()) {
    return TypeInferenceErrorStatus(
        alias->span(), nullptr,
        absl::StrFormat(
            "impl-based procs must be aliased using `type` (e.g. `type %s "
            "= ...;`) rather than `proc`.",
            alias->identifier()),
        import_data.file_table());
  }
  if (explicit_parametrics.size() >
      target_proc_def->parametric_bindings().size()) {
    return ArgCountMismatchErrorStatus(
        alias->span(),
        absl::StrFormat(
            "Too many parametric values supplied; limit: %d given: %d",
            target_proc_def->parametric_bindings().size(),
            explicit_parametrics.size()),
        import_data.file_table());
  }
  if (!target_proc_def->impl().has_value()) {
    return absl::OkStatus();
  }
  Function* target_ctor = nullptr;
  for (ImplMember member : (*target_proc_def->impl())->members()) {
    if (std::holds_alternative<Function*>(member)) {
      Function* fn = std::get<Function*>(member);
      if (fn->identifier() == "new") {
        target_ctor = fn;
        break;
      }
    }
  }
  if (target_ctor == nullptr) {
    return absl::OkStatus();
  }

  absl::flat_hash_map<const NameDef*, ExprOrType> binding_map;
  for (size_t i = 0; i < target_proc_def->parametric_bindings().size(); ++i) {
    const ParametricBinding* binding =
        target_proc_def->parametric_bindings()[i];
    if (i < explicit_parametrics.size()) {
      binding_map[binding->name_def()] = explicit_parametrics[i];
    } else if (binding->default_expr_or_type().has_value()) {
      binding_map[binding->name_def()] = *binding->default_expr_or_type();
    }
  }

  std::vector<Param*> ctor_params;
  std::vector<Expr*> ctor_call_args;
  ctor_params.reserve(target_ctor->params().size());
  ctor_call_args.reserve(target_ctor->params().size());
  for (const Param* old_param : target_ctor->params()) {
    XLS_ASSIGN_OR_RETURN(
        AstNode * cloned_type_node,
        CloneNodeWithSubstitutions(old_param->type_annotation(), module,
                                   binding_map));
    auto* new_param_type = absl::down_cast<TypeAnnotation*>(cloned_type_node);
    NameDef* new_param_name_def = module.Make<NameDef>(
        Span(new_param_type->span().start(), new_param_type->span().start()),
        old_param->identifier(),
        /*definer=*/nullptr);
    Param* new_param = module.Make<Param>(new_param_name_def, new_param_type);
    new_param_name_def->set_definer(new_param);
    ctor_params.push_back(new_param);
    ctor_call_args.push_back(module.Make<NameRef>(
        alias->span(), new_param->identifier(), new_param_name_def));
  }

  NameDef* proc_name_def = alias->name_def();
  ProcDef* wrapper_proc = module.Make<ProcDef>(
      alias->span(), proc_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>{},
      /*members=*/std::vector<StructMemberNode*>{}, alias->is_public());
  wrapper_proc->set_alias_target(target_proc_def);
  proc_name_def->set_definer(wrapper_proc);

  TypeRef* wrapper_type_ref = module.Make<TypeRef>(alias->span(), wrapper_proc);
  TypeRefTypeAnnotation* wrapper_type_ann = module.Make<TypeRefTypeAnnotation>(
      alias->span(), wrapper_type_ref, std::vector<ExprOrType>{});
  Impl* wrapper_impl =
      module.Make<Impl>(alias->span(), wrapper_type_ann,
                        std::vector<ImplMember>{}, alias->is_public());
  wrapper_proc->set_impl(wrapper_impl);

  TypeRef* target_type_ref = nullptr;
  if (target_proc_def->owner() == &module) {
    target_type_ref = module.Make<TypeRef>(alias->span(), target_proc_def);
  } else {
    XLS_ASSIGN_OR_RETURN(
        AstNode * cloned_target,
        CloneNodeWithSubstitutions(ToAstNode(target), module, {}));
    target_type_ref = module.Make<TypeRef>(
        alias->span(), absl::down_cast<ColonRef*>(cloned_target));
  }
  std::vector<ExprOrType> invocation_parametrics;
  invocation_parametrics.reserve(explicit_parametrics.size());
  for (ExprOrType p : explicit_parametrics) {
    XLS_ASSIGN_OR_RETURN(AstNode * cloned_p,
                         CloneNodeWithSubstitutions(ToAstNode(p), module, {}));
    invocation_parametrics.push_back(ToExprOrType(cloned_p));
  }
  TypeRefTypeAnnotation* target_type_ann = module.Make<TypeRefTypeAnnotation>(
      alias->span(), target_type_ref, invocation_parametrics);
  ColonRef* target_ctor_ref = module.Make<ColonRef>(
      alias->span(), target_type_ann, target_ctor->identifier());
  Invocation* target_ctor_call =
      module.Make<Invocation>(alias->span(), target_ctor_ref, ctor_call_args,
                              std::vector<ExprOrType>{});
  Attr* spawn_attr =
      module.Make<Attr>(alias->span(), target_ctor_call, "spawn");
  Invocation* spawn_call =
      module.Make<Invocation>(alias->span(), spawn_attr, std::vector<Expr*>{},
                              std::vector<ExprOrType>{});
  Statement* spawn_stmt = module.Make<Statement>(spawn_call);

  StructInstance* return_instance =
      module.Make<StructInstance>(alias->span(), wrapper_type_ann,
                                  std::vector<std::pair<std::string, Expr*>>{});
  Statement* return_stmt = module.Make<Statement>(return_instance);
  StatementBlock* ctor_body = module.Make<StatementBlock>(
      alias->span(), std::vector<Statement*>{spawn_stmt, return_stmt},
      /*trailing_semi=*/false);

  SelfTypeAnnotation* self_type_ann = module.Make<SelfTypeAnnotation>(
      alias->span(), /*explicit_type=*/false, wrapper_type_ann);
  NameDef* ctor_name_def =
      module.Make<NameDef>(alias->span(), "new", /*definer=*/nullptr);
  Function* wrapper_ctor = module.Make<Function>(
      alias->span(), ctor_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>{}, ctor_params,
      self_type_ann, ctor_body, FunctionTag::kNormal,
      /*is_public=*/true, /*is_stub=*/false);
  ctor_name_def->set_definer(wrapper_ctor);
  wrapper_ctor->set_impl(wrapper_impl);
  wrapper_impl->AddMember(wrapper_ctor);

  NameDef* self_name_def =
      module.Make<NameDef>(alias->span(), "self", /*definer=*/nullptr);
  SelfTypeAnnotation* next_self_type_ann = module.Make<SelfTypeAnnotation>(
      alias->span(), /*explicit_type=*/false, wrapper_type_ann);
  Param* self_param = module.Make<Param>(self_name_def, next_self_type_ann);
  self_name_def->set_definer(self_param);
  StatementBlock* next_body = module.Make<StatementBlock>(
      alias->span(), std::vector<Statement*>{}, /*trailing_semi=*/true);
  NameDef* next_name_def =
      module.Make<NameDef>(alias->span(), "next", /*definer=*/nullptr);
  Function* wrapper_next = module.Make<Function>(
      alias->span(), next_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>{},
      std::vector<Param*>{self_param}, /*return_type=*/nullptr, next_body,
      FunctionTag::kNormal, /*is_public=*/true, /*is_stub=*/false);
  next_name_def->set_definer(wrapper_next);
  wrapper_next->set_impl(wrapper_impl);
  wrapper_impl->AddMember(wrapper_next);
  wrapper_impl->SetParentage();
  wrapper_proc->SetParentage();

  XLS_RETURN_IF_ERROR(module.ReplaceTopMember(alias, wrapper_proc));
  return module.InsertTopAfter(wrapper_proc, wrapper_impl);
}

absl::Status RewriteLegacyProcAliasToWrapper(
    Module& module, ImportData& import_data, AliasDef* alias, Proc* target_proc,
    const AliasDef::Target& target,
    const std::vector<ExprOrType>& explicit_parametrics) {
  if (explicit_parametrics.size() > target_proc->parametric_bindings().size()) {
    return ArgCountMismatchErrorStatus(
        alias->span(),
        absl::StrFormat(
            "Too many parametric values supplied; limit: %d given: %d",
            target_proc->parametric_bindings().size(),
            explicit_parametrics.size()),
        import_data.file_table());
  }

  absl::flat_hash_map<const NameDef*, ExprOrType> binding_map;
  for (size_t i = 0; i < target_proc->parametric_bindings().size(); ++i) {
    const ParametricBinding* binding = target_proc->parametric_bindings()[i];
    if (i < explicit_parametrics.size()) {
      binding_map[binding->name_def()] = explicit_parametrics[i];
    } else if (binding->default_expr_or_type().has_value()) {
      binding_map[binding->name_def()] = *binding->default_expr_or_type();
    }
  }

  std::vector<Param*> config_params;
  std::vector<Expr*> config_call_args;
  config_params.reserve(target_proc->config().params().size());
  config_call_args.reserve(target_proc->config().params().size());
  for (const Param* old_param : target_proc->config().params()) {
    XLS_ASSIGN_OR_RETURN(
        AstNode * cloned_type_node,
        CloneNodeWithSubstitutions(old_param->type_annotation(), module,
                                   binding_map));
    auto* new_param_type = absl::down_cast<TypeAnnotation*>(cloned_type_node);
    NameDef* new_param_name_def = module.Make<NameDef>(
        Span(new_param_type->span().start(), new_param_type->span().start()),
        old_param->identifier(), /*definer=*/nullptr);
    Param* new_param = module.Make<Param>(new_param_name_def, new_param_type);
    new_param_name_def->set_definer(new_param);
    config_params.push_back(new_param);
    config_call_args.push_back(module.Make<NameRef>(
        alias->span(), new_param->identifier(), new_param_name_def));
  }

  auto clone_parametrics = [&]() -> absl::StatusOr<std::vector<ExprOrType>> {
    std::vector<ExprOrType> cloned;
    cloned.reserve(explicit_parametrics.size());
    for (ExprOrType p : explicit_parametrics) {
      XLS_ASSIGN_OR_RETURN(
          AstNode * cp, CloneNodeWithSubstitutions(ToAstNode(p), module, {}));
      cloned.push_back(ToExprOrType(cp));
    }
    return cloned;
  };

  Expr* spawnee = nullptr;
  Expr* config_ref = nullptr;
  Expr* next_ref = nullptr;
  Expr* init_ref = nullptr;
  if (target_proc->owner() == &module) {
    spawnee = module.Make<NameRef>(alias->span(), target_proc->identifier(),
                                   target_proc->name_def());
    config_ref =
        module.Make<NameRef>(alias->span(), target_proc->config().identifier(),
                             target_proc->config().name_def());
    next_ref =
        module.Make<NameRef>(alias->span(), target_proc->next().identifier(),
                             target_proc->next().name_def());
    init_ref =
        module.Make<NameRef>(alias->span(), target_proc->init().identifier(),
                             target_proc->init().name_def());
  } else {
    XLS_ASSIGN_OR_RETURN(
        AstNode * cloned_spawnee,
        CloneNodeWithSubstitutions(ToAstNode(target), module, {}));
    auto* cr = absl::down_cast<ColonRef*>(cloned_spawnee);
    spawnee = cr;
    auto to_subject = [](AstNode* n) -> ColonRef::Subject {
      if (auto* nr = dynamic_cast<NameRef*>(n)) {
        return nr;
      }
      return absl::down_cast<ColonRef*>(n);
    };
    XLS_ASSIGN_OR_RETURN(
        AstNode * subj1,
        CloneNodeWithSubstitutions(ToAstNode(cr->subject()), module, {}));
    config_ref = module.Make<ColonRef>(alias->span(), to_subject(subj1),
                                       absl::StrCat(cr->attr(), ".config"));
    XLS_ASSIGN_OR_RETURN(
        AstNode * subj2,
        CloneNodeWithSubstitutions(ToAstNode(cr->subject()), module, {}));
    next_ref = module.Make<ColonRef>(alias->span(), to_subject(subj2),
                                     absl::StrCat(cr->attr(), ".next"));
    XLS_ASSIGN_OR_RETURN(
        AstNode * subj3,
        CloneNodeWithSubstitutions(ToAstNode(cr->subject()), module, {}));
    init_ref = module.Make<ColonRef>(alias->span(), to_subject(subj3),
                                     absl::StrCat(cr->attr(), ".init"));
  }

  XLS_ASSIGN_OR_RETURN(std::vector<ExprOrType> init_parametrics,
                       clone_parametrics());
  Invocation* init_invoc =
      module.Make<Invocation>(alias->span(), init_ref, std::vector<Expr*>{},
                              std::move(init_parametrics));
  XLS_ASSIGN_OR_RETURN(std::vector<ExprOrType> config_parametrics,
                       clone_parametrics());
  Invocation* config_invoc =
      module.Make<Invocation>(alias->span(), config_ref, config_call_args,
                              std::move(config_parametrics));
  XLS_ASSIGN_OR_RETURN(std::vector<ExprOrType> next_parametrics,
                       clone_parametrics());
  Invocation* next_invoc = module.Make<Invocation>(
      alias->span(), next_ref, std::vector<Expr*>{init_invoc},
      std::move(next_parametrics));
  XLS_ASSIGN_OR_RETURN(std::vector<ExprOrType> spawn_parametrics,
                       clone_parametrics());
  Spawn* spawn_node =
      module.Make<Spawn>(alias->span(), spawnee, config_invoc, next_invoc,
                         std::move(spawn_parametrics));
  Statement* spawn_stmt = module.Make<Statement>(spawn_node);
  XlsTuple* empty_tuple_config = module.Make<XlsTuple>(
      alias->span(), std::vector<Expr*>{}, /*has_trailing_comma=*/false);
  Statement* ret_tuple_stmt = module.Make<Statement>(empty_tuple_config);
  StatementBlock* config_body = module.Make<StatementBlock>(
      alias->span(), std::vector<Statement*>{spawn_stmt, ret_tuple_stmt},
      /*trailing_semi=*/false);
  TupleTypeAnnotation* unit_type_ann_config = module.Make<TupleTypeAnnotation>(
      alias->span(), std::vector<TypeAnnotation*>{});
  NameDef* config_name_def = module.Make<NameDef>(
      alias->span(), absl::StrCat(alias->identifier(), ".config"),
      /*definer=*/nullptr);
  Function* wrapper_config = module.Make<Function>(
      alias->span(), config_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>{}, config_params,
      unit_type_ann_config, config_body, FunctionTag::kProcConfig,
      alias->is_public(), /*is_stub=*/false);
  config_name_def->set_definer(wrapper_config);

  XlsTuple* empty_tuple_init = module.Make<XlsTuple>(
      alias->span(), std::vector<Expr*>{}, /*has_trailing_comma=*/false);
  Statement* init_stmt = module.Make<Statement>(empty_tuple_init);
  StatementBlock* init_body = module.Make<StatementBlock>(
      alias->span(), std::vector<Statement*>{init_stmt},
      /*trailing_semi=*/false);
  TupleTypeAnnotation* unit_type_ann_init = module.Make<TupleTypeAnnotation>(
      alias->span(), std::vector<TypeAnnotation*>{});
  NameDef* init_name_def = module.Make<NameDef>(
      alias->span(), absl::StrCat(alias->identifier(), ".init"),
      /*definer=*/nullptr);
  Function* wrapper_init = module.Make<Function>(
      alias->span(), init_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>{},
      std::vector<Param*>{}, unit_type_ann_init, init_body,
      FunctionTag::kProcInit, alias->is_public(), /*is_stub=*/false);
  init_name_def->set_definer(wrapper_init);

  TupleTypeAnnotation* state_type_ann = module.Make<TupleTypeAnnotation>(
      alias->span(), std::vector<TypeAnnotation*>{});
  NameDef* state_name_def =
      module.Make<NameDef>(alias->span(), "state", /*definer=*/nullptr);
  Param* state_param = module.Make<Param>(state_name_def, state_type_ann);
  state_name_def->set_definer(state_param);
  XlsTuple* empty_tuple_next = module.Make<XlsTuple>(
      alias->span(), std::vector<Expr*>{}, /*has_trailing_comma=*/false);
  Statement* next_stmt = module.Make<Statement>(empty_tuple_next);
  StatementBlock* next_body = module.Make<StatementBlock>(
      alias->span(), std::vector<Statement*>{next_stmt},
      /*trailing_semi=*/false);
  TupleTypeAnnotation* unit_type_ann_next = module.Make<TupleTypeAnnotation>(
      alias->span(), std::vector<TypeAnnotation*>{});
  NameDef* next_name_def = module.Make<NameDef>(
      alias->span(), absl::StrCat(alias->identifier(), ".next"),
      /*definer=*/nullptr);
  Function* wrapper_next = module.Make<Function>(
      alias->span(), next_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>{},
      std::vector<Param*>{state_param}, unit_type_ann_next, next_body,
      FunctionTag::kProcNext, alias->is_public(), /*is_stub=*/false);
  next_name_def->set_definer(wrapper_next);

  ProcLikeBody proc_body = {
      .stmts = {wrapper_init, wrapper_config, wrapper_next},
      .config = wrapper_config,
      .next = wrapper_next,
      .init = wrapper_init,
      .members = {},
  };
  NameDef* proc_name_def = alias->name_def();
  Proc* wrapper_proc = module.Make<Proc>(
      alias->span(), alias->span(), proc_name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>{}, proc_body,
      alias->is_public());
  wrapper_proc->set_alias_target(target_proc);
  proc_name_def->set_definer(wrapper_proc);
  wrapper_config->set_proc(wrapper_proc);
  wrapper_next->set_proc(wrapper_proc);
  wrapper_init->set_proc(wrapper_proc);
  wrapper_proc->SetParentage();
  XLS_RETURN_IF_ERROR(module.ReplaceTopMember(alias, wrapper_proc));
  XLS_RETURN_IF_ERROR(module.InsertTopAfter(wrapper_proc, wrapper_init));
  XLS_RETURN_IF_ERROR(module.InsertTopAfter(wrapper_init, wrapper_config));
  return module.InsertTopAfter(wrapper_config, wrapper_next);
}

absl::Status RewriteAliasesToWrappers(
    Module& module, ImportData& import_data,
    const TypecheckModuleFn& typecheck_imported_module) {
  std::vector<AliasDef*> aliases;
  for (ModuleMember& top : module.top()) {
    if (std::holds_alternative<AliasDef*>(top)) {
      aliases.push_back(std::get<AliasDef*>(top));
    }
  }

  for (AliasDef* alias : aliases) {
    AliasDef::Target target = alias->target();
    std::vector<ExprOrType> explicit_parametrics = alias->parametrics();

    std::optional<ModuleMember> resolved_member;
    if (std::holds_alternative<NameRef*>(target)) {
      NameRef* nr = std::get<NameRef*>(target);
      std::optional<ModuleMember*> member =
          module.FindMemberWithName(nr->identifier());
      if (member.has_value()) {
        resolved_member = **member;
      }
    } else {
      ColonRef* cr = std::get<ColonRef*>(target);
      XLS_ASSIGN_OR_RETURN(
          std::optional<ModuleInfo*> imported_info,
          GetImportedModuleInfo(cr, import_data, typecheck_imported_module));
      if (imported_info.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            resolved_member,
            GetPublicModuleMember((*imported_info)->module(), cr,
                                  import_data.file_table()));
      }
    }

    if (!resolved_member.has_value()) {
      return TypeInferenceErrorStatus(
          alias->span(), nullptr,
          alias->is_function_alias()
              ? "Function alias must have a function as a target."
              : "Proc alias must have a proc as a target.",
          import_data.file_table());
    }

    // Unwrap TypeAlias chains (e.g., `pub type Counter16 =
    // Counter<u32:16>;`) if referenced as an alias target.
    while (resolved_member.has_value() &&
           std::holds_alternative<TypeAlias*>(*resolved_member)) {
      TypeAlias* ta = std::get<TypeAlias*>(*resolved_member);
      resolved_member = std::nullopt;
      if (ta->type_annotation().IsAnnotation<TypeRefTypeAnnotation>()) {
        const auto* trta =
            ta->type_annotation().AsAnnotation<TypeRefTypeAnnotation>();
        if (explicit_parametrics.empty()) {
          explicit_parametrics = trta->parametrics();
        }
        TypeDefinition td = trta->type_ref()->type_definition();
        if (std::holds_alternative<ProcDef*>(td)) {
          resolved_member = std::get<ProcDef*>(td);
        } else if (std::holds_alternative<TypeAlias*>(td)) {
          resolved_member = std::get<TypeAlias*>(td);
        } else if (std::holds_alternative<ColonRef*>(td)) {
          ColonRef* cr = std::get<ColonRef*>(td);
          XLS_ASSIGN_OR_RETURN(std::optional<ModuleInfo*> imported_info,
                               GetImportedModuleInfo(
                                   cr, import_data, typecheck_imported_module));
          if (imported_info.has_value()) {
            XLS_ASSIGN_OR_RETURN(
                resolved_member,
                GetPublicModuleMember((*imported_info)->module(), cr,
                                      import_data.file_table()));
          }
        }
      }
    }

    if (!resolved_member.has_value()) {
      return TypeInferenceErrorStatus(
          alias->span(), nullptr,
          alias->is_function_alias()
              ? "Function alias must have a function as a target."
              : "Proc alias must have a proc as a target.",
          import_data.file_table());
    }

    if (std::holds_alternative<Function*>(*resolved_member) &&
        (alias->is_function_alias() || alias->is_synthetic())) {
      XLS_RETURN_IF_ERROR(RewriteFunctionAliasToWrapper(
          module, import_data, alias, std::get<Function*>(*resolved_member),
          target, explicit_parametrics));
    } else if (std::holds_alternative<ProcDef*>(*resolved_member) &&
               (!alias->is_function_alias() || alias->is_synthetic())) {
      XLS_RETURN_IF_ERROR(RewriteProcDefAliasToWrapper(
          module, import_data, alias, std::get<ProcDef*>(*resolved_member),
          target, explicit_parametrics));
    } else if (std::holds_alternative<Proc*>(*resolved_member) &&
               (!alias->is_function_alias() || alias->is_synthetic())) {
      XLS_RETURN_IF_ERROR(RewriteLegacyProcAliasToWrapper(
          module, import_data, alias, std::get<Proc*>(*resolved_member), target,
          explicit_parametrics));
    } else {
      return TypeInferenceErrorStatus(
          alias->span(), nullptr,
          alias->is_function_alias()
              ? "Function alias must have a function as a target."
              : "Proc alias must have a proc as a target.",
          import_data.file_table());
    }
  }
  return absl::OkStatus();
}

}  // namespace

SemanticsAnalysis::SemanticsAnalysis(bool suppress_warnings)
    : suppress_warnings_(suppress_warnings) {}

absl::Status SemanticsAnalysis::RunPreTypeCheckPass(
    Module& module, WarningCollector& warning_collector,
    ImportData& import_data,
    const TypecheckModuleFn& typecheck_imported_module) {
  XLS_RETURN_IF_ERROR(
      RewriteAliasesToWrappers(module, import_data, typecheck_imported_module));
  ProcDefTrivialNextGenerator next_generator;
  XLS_RETURN_IF_ERROR(module.Accept(&next_generator));

  if (module.attributes().contains(ModuleAttribute::kExplicitStateAccess)) {
    XLS_ASSIGN_OR_RETURN(Module * builtins,
                         import_data.GetBuiltinStubsModule());
    XLS_ASSIGN_OR_RETURN(
        StructDef * state_struct_def,
        builtins->GetMemberOrError<StructDef>(kBuiltinProcStateStructName));
    ProcStateVisitor state_visitor(import_data, state_struct_def,
                                   typecheck_imported_module);
    XLS_RETURN_IF_ERROR(module.Accept(&state_visitor));
  }
  XLS_RETURN_IF_ERROR(RewriteLambdas(module, import_data));
  XLS_RETURN_IF_ERROR(RewriteDomainStructs(module, import_data));

  AddSpawnTraitToProcDefs add_spawn_trait;
  XLS_RETURN_IF_ERROR(module.Accept(&add_spawn_trait));

  if (suppress_warnings_) {
    return absl::OkStatus();
  }
  PreTypecheckPass pass(warning_collector, import_data.file_table());

  for (const ModuleMember& top : module.top()) {
    if (const Function* const* func = std::get_if<Function*>(&top)) {
      CollectUseDef visitor;
      XLS_RETURN_IF_ERROR((*func)->body()->Accept(&visitor));

      maybe_unreferenced_defs.emplace_back(
          std::make_pair(*func, std::vector<const NameDef*>()));
      std::vector<const NameDef*>& defs_in_func =
          maybe_unreferenced_defs.back().second;

      for (const NameDef* def : visitor.Defs()) {
        if (!visitor.Uses().contains(def)) {
          defs_in_func.emplace_back(def);
          def_to_type_.try_emplace(def, nullptr);
        }
      }
    }
  }

  return module.Accept(&pass);
}

// If a possibly unused def is concretized to a non-token type at any possible
// context, it is truly unused.
void SemanticsAnalysis::SetNameDefType(const NameDef* def, const Type* type) {
  auto found = def_to_type_.find(def);
  if (found != def_to_type_.end()) {
    if ((found->second && found->second->IsToken()) || !found->second) {
      found->second = type->CloneToUnique();
    }
  }
}

absl::Status SemanticsAnalysis::RunPostTypeCheckPass(
    WarningCollector& warning_collector) {
  if (suppress_warnings_) {
    return absl::OkStatus();
  }
  // Report unused defs.
  for (auto& [f, unused_defs] : maybe_unreferenced_defs) {
    // Sort them for reporting stability.
    std::sort(
        unused_defs.begin(), unused_defs.end(),
        [](const NameDef* a, const NameDef* b) {
          return a->span() < b->span() ||
                 (a->span() == b->span() && a->identifier() < b->identifier());
        });
    for (const NameDef* def : unused_defs) {
      std::unique_ptr<Type>& type = def_to_type_.at(def);
      // Tokens are implicitly joined at the end of a proc `next()`, so we
      // don't warn on these.
      if (type && !type->IsToken()) {
        warning_collector.Add(
            def->span(), WarningKind::kUnusedDefinition,
            absl::StrFormat(
                "Definition of `%s` (type `%s`) is not used in function `%s`",
                def->identifier(), type->ToString(), f->identifier()));
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace xls::dslx
