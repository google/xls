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

#include "xls/dslx/fmt/type_annotation_simplifier.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/builtins_metadata.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

// A stable ID for a node across clones of the module. We need this because
// TypeAnnotationSimplifier starts with a clone of the target module, which has
// format-disabled parts replaced with `VerbatimNode`, but the `TypeInfo`, which
// we consult, has the original module's nodes.
struct NodeKey {
  AstNodeKind kind;
  Span span;

  bool operator==(const NodeKey& other) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const NodeKey& key) {
    return H::combine(std::move(h), key.kind, key.span.fileno(),
                      key.span.start().lineno(), key.span.start().colno(),
                      key.span.limit().lineno(), key.span.limit().colno());
  }
};

class CollectNodesByNodeKey : public AstNodeRecursiveVisitor {
 public:
  CollectNodesByNodeKey(absl::flat_hash_map<NodeKey, const AstNode*>* map)
      : AstNodeRecursiveVisitor(/*want_types=*/true), map_(map) {}

  absl::Status DefaultHandler(const AstNode* node) override {
    if (std::optional<Span> span = node->GetSpan(); span.has_value()) {
      map_->try_emplace(NodeKey{node->kind(), *span}, node);
    }
    return AstNodeRecursiveVisitor::DefaultHandler(node);
  }

 private:
  absl::flat_hash_map<NodeKey, const AstNode*>* const map_;
};

// Returns true if `expr` starts with (has as its leftmost leaf) a Number
// literal with an explicit type annotation.
bool StartsWithTypedNumber(const Expr* expr) {
  if (expr == nullptr) {
    return false;
  }
  if (const auto* num = dynamic_cast<const Number*>(expr)) {
    return num->type_annotation() != nullptr &&
           num->number_kind() != NumberKind::kBool;
  }
  if (const auto* binop = dynamic_cast<const Binop*>(expr)) {
    return StartsWithTypedNumber(binop->lhs());
  }
  return false;
}

std::optional<const Expr*> GetBlockValueExpr(const StatementBlock* block) {
  if (block == nullptr || block->trailing_semi() ||
      block->statements().empty()) {
    return std::nullopt;
  }
  const Statement* last = block->statements().back();
  if (std::holds_alternative<Expr*>(last->wrapped())) {
    return std::get<Expr*>(last->wrapped());
  }
  return std::nullopt;
}

std::optional<const Expr*> GetAlternateValueExpr(const Conditional& cond) {
  if (std::holds_alternative<StatementBlock*>(cond.alternate())) {
    return GetBlockValueExpr(std::get<StatementBlock*>(cond.alternate()));
  }
  return std::get<Conditional*>(cond.alternate());
}

bool HasNonLiteralType(const Expr* expr);

bool HasNonLiteralType(std::optional<const Expr*> expr) {
  return expr.has_value() && HasNonLiteralType(*expr);
}

// Returns true if `expr` has a type that is determined without relying on
// standalone integer or array literal type annotations (or is a shift
// expression starting with a typed integer literal, which preserves its type).
bool HasNonLiteralType(const Expr* expr) {
  if (expr == nullptr) {
    return false;
  }
  if (const auto* num = dynamic_cast<const Number*>(expr)) {
    return num->number_kind() == NumberKind::kBool;
  }
  if (dynamic_cast<const Array*>(expr) != nullptr) {
    return false;
  }
  if (const auto* block = dynamic_cast<const StatementBlock*>(expr)) {
    return HasNonLiteralType(GetBlockValueExpr(block));
  }
  if (const auto* cond = dynamic_cast<const Conditional*>(expr)) {
    return HasNonLiteralType(GetBlockValueExpr(cond->consequent())) ||
           HasNonLiteralType(GetAlternateValueExpr(*cond));
  }
  if (const auto* match = dynamic_cast<const Match*>(expr)) {
    for (const MatchArm* arm : match->arms()) {
      if (HasNonLiteralType(arm->expr())) {
        return true;
      }
    }
    return false;
  }
  if (const auto* unop = dynamic_cast<const Unop*>(expr)) {
    return HasNonLiteralType(unop->operand());
  }
  if (const auto* binop = dynamic_cast<const Binop*>(expr)) {
    if (binop->binop_kind() == BinopKind::kShl ||
        binop->binop_kind() == BinopKind::kShr) {
      return StartsWithTypedNumber(binop->lhs()) ||
             HasNonLiteralType(binop->lhs());
    }
    return HasNonLiteralType(binop->lhs()) || HasNonLiteralType(binop->rhs());
  }
  return true;
}

std::optional<const StructDefBase*> ResolveStructDef(
    const TypeInfo& ti, const TypeAnnotation* type,
    bool* all_parametrics_specified);

std::optional<const StructDefBase*> ResolveTypeDefToStruct(
    const TypeInfo& ti, const TypeDefinition& td,
    size_t explicit_parametrics_count, bool* all_parametrics_specified) {
  if (std::holds_alternative<ColonRef*>(td) &&
      !std::get<ColonRef*>(td)->ResolveImportSubject().has_value()) {
    *all_parametrics_specified = false;
    return std::nullopt;
  }
  absl::StatusOr<TypeInfo::TypeSource> source =
      const_cast<TypeInfo&>(ti).ResolveTypeDefinition(td);
  if (!source.ok()) {
    *all_parametrics_specified = false;
    return std::nullopt;
  }
  return std::visit(
      Visitor{
          [&](TypeAlias* alias) -> std::optional<const StructDefBase*> {
            return ResolveStructDef(*source->type_info,
                                    &alias->type_annotation(),
                                    all_parametrics_specified);
          },
          [&](StructDef* struct_def) -> std::optional<const StructDefBase*> {
            *all_parametrics_specified =
                !struct_def->is_domain_struct() &&
                explicit_parametrics_count ==
                    struct_def->parametric_bindings().size();
            return struct_def;
          },
          [&](ProcDef* proc_def) -> std::optional<const StructDefBase*> {
            *all_parametrics_specified = explicit_parametrics_count ==
                                         proc_def->parametric_bindings().size();
            return proc_def;
          },
          [&](auto*) -> std::optional<const StructDefBase*> {
            *all_parametrics_specified = true;
            return std::nullopt;
          },
      },
      source->definition);
}

std::optional<const StructDefBase*> ResolveStructDef(
    const TypeInfo& ti, const TypeAnnotation* type,
    bool* all_parametrics_specified) {
  if (type == nullptr) {
    *all_parametrics_specified = true;
    return std::nullopt;
  }
  if (const auto* self_type = dynamic_cast<const SelfTypeAnnotation*>(type)) {
    std::optional<const StructDefBase*> def = ResolveStructDef(
        ti, self_type->struct_ref(), all_parametrics_specified);
    *all_parametrics_specified = true;
    return def;
  }
  if (dynamic_cast<const TypeVariableTypeAnnotation*>(type) != nullptr) {
    *all_parametrics_specified = true;
    return std::nullopt;
  }
  const auto* tr = dynamic_cast<const TypeRefTypeAnnotation*>(type);
  if (tr == nullptr || tr->type_ref() == nullptr) {
    *all_parametrics_specified = true;
    return std::nullopt;
  }
  return ResolveTypeDefToStruct(ti, tr->type_ref()->type_definition(),
                                tr->parametrics().size(),
                                all_parametrics_specified);
}

bool AreAllStructParametricsSpecified(const TypeInfo& ti,
                                      const TypeAnnotation* type) {
  bool all_specified = true;
  ResolveStructDef(ti, type, &all_specified);
  return all_specified;
}

std::optional<const StructDefBase*> ResolveColonRefSubjectToStruct(
    const TypeInfo& ti, const ColonRef::Subject& subject,
    bool* all_parametrics_specified) {
  return std::visit(
      Visitor{
          [&](TypeRefTypeAnnotation* tr)
              -> std::optional<const StructDefBase*> {
            return ResolveStructDef(ti, tr, all_parametrics_specified);
          },
          [&](SelfTypeAnnotation* self_type)
              -> std::optional<const StructDefBase*> {
            return ResolveStructDef(ti, self_type, all_parametrics_specified);
          },
          [&](TypeVariableTypeAnnotation*)
              -> std::optional<const StructDefBase*> {
            *all_parametrics_specified = true;
            return std::nullopt;
          },
          [&](NameRef* subj_name) -> std::optional<const StructDefBase*> {
            if (!std::holds_alternative<const NameDef*>(
                    subj_name->name_def())) {
              *all_parametrics_specified = false;
              return std::nullopt;
            }
            const NameDef* name_def =
                std::get<const NameDef*>(subj_name->name_def());
            if (auto* alias = dynamic_cast<TypeAlias*>(name_def->definer())) {
              return ResolveTypeDefToStruct(ti, alias,
                                            /*explicit_parametrics_count=*/0,
                                            all_parametrics_specified);
            }
            if (auto* sd = dynamic_cast<StructDef*>(name_def->definer())) {
              return ResolveTypeDefToStruct(ti, sd,
                                            /*explicit_parametrics_count=*/0,
                                            all_parametrics_specified);
            }
            if (auto* pd = dynamic_cast<ProcDef*>(name_def->definer())) {
              return ResolveTypeDefToStruct(ti, pd,
                                            /*explicit_parametrics_count=*/0,
                                            all_parametrics_specified);
            }
            if (dynamic_cast<GenericTypeAnnotation*>(name_def->definer()) !=
                nullptr) {
              *all_parametrics_specified = true;
              return std::nullopt;
            }
            if (auto* ute = dynamic_cast<UseTreeEntry*>(name_def->parent())) {
              return ResolveTypeDefToStruct(ti, ute,
                                            /*explicit_parametrics_count=*/0,
                                            all_parametrics_specified);
            }
            *all_parametrics_specified = false;
            return std::nullopt;
          },
          [&](ColonRef* subj_colon_ref) -> std::optional<const StructDefBase*> {
            return ResolveTypeDefToStruct(ti, subj_colon_ref,
                                          /*explicit_parametrics_count=*/0,
                                          all_parametrics_specified);
          },
      },
      subject);
}

std::optional<const Function*> ResolveImportedFunction(
    const TypeInfo& ti, const ImportSubject& subject, std::string_view name) {
  std::optional<const ImportedInfo*> imported = ti.GetImported(subject);
  if (!imported.has_value()) {
    return std::nullopt;
  }
  return (*imported)->module->GetFunction(name);
}

std::optional<const Function*> ResolveStaticCalleeFunction(
    const TypeInfo& ti, const Expr* callee,
    bool* struct_parametrics_specified) {
  *struct_parametrics_specified = true;
  if (const auto* name_ref = dynamic_cast<const NameRef*>(callee)) {
    if (!std::holds_alternative<const NameDef*>(name_ref->name_def())) {
      return std::nullopt;
    }
    const NameDef* name_def = std::get<const NameDef*>(name_ref->name_def());
    if (const auto* fn = dynamic_cast<const Function*>(name_def->definer())) {
      return fn;
    }
    if (auto* ute = dynamic_cast<UseTreeEntry*>(name_def->parent())) {
      return ResolveImportedFunction(ti, ute, name_def->identifier());
    }
    return std::nullopt;
  }

  const auto* colon_ref = dynamic_cast<const ColonRef*>(callee);
  if (colon_ref == nullptr) {
    return std::nullopt;
  }
  if (std::optional<ImportSubject> import_subject =
          colon_ref->ResolveImportSubject();
      import_subject.has_value()) {
    return ResolveImportedFunction(ti, *import_subject, colon_ref->attr());
  }
  std::optional<const StructDefBase*> struct_def =
      ResolveColonRefSubjectToStruct(ti, colon_ref->subject(),
                                     struct_parametrics_specified);
  if (!struct_def.has_value()) {
    return std::nullopt;
  }
  return (*struct_def)->GetImplFunction(colon_ref->attr());
}

bool AreAllFunctionParametricsSpecified(const TypeInfo& ti,
                                        const Invocation& n) {
  if (IsParametricBuiltinWithInferrableType(n.callee())) {
    return true;
  }
  std::vector<InvocationCalleeData> callee_data =
      ti.GetUniqueInvocationCalleeData(&n);
  if (callee_data.empty()) {
    // Fallback for invocations inside uninstantiated parametric functions,
    // which are not populated in `TypeInfo::GetUniqueInvocationCalleeData`.
    bool struct_parametrics_specified = true;
    std::optional<const Function*> target_fn = ResolveStaticCalleeFunction(
        ti, n.callee(), &struct_parametrics_specified);
    return target_fn.has_value() && struct_parametrics_specified &&
           n.explicit_parametrics().size() ==
               (*target_fn)->parametric_bindings().size();
  }

  for (const InvocationCalleeData& data : callee_data) {
    if (data.callee == nullptr ||
        n.explicit_parametrics().size() !=
            data.callee->parametric_bindings().size()) {
      return false;
    }
    const auto* colon_ref = dynamic_cast<const ColonRef*>(n.callee());
    if (data.callee->IsFunctionOnParametricStruct() && colon_ref != nullptr) {
      bool struct_parametrics_specified = true;
      ResolveColonRefSubjectToStruct(ti, colon_ref->subject(),
                                     &struct_parametrics_specified);
      if (!struct_parametrics_specified) {
        return false;
      }
    }
  }
  return true;
}

class TypeAnnotationSimplifier : public Formatter {
 public:
  TypeAnnotationSimplifier(Comments& comments, DocArena& arena,
                           const TypeInfo& type_info)
      : Formatter(comments, arena), type_info_(type_info) {}

  ~TypeAnnotationSimplifier() override = default;

  absl::StatusOr<DocRef> FormatModule(const Module& n) override {
    // `AutoFmt` clones the module before formatting. Index the original
    // typechecked module's AST nodes by `(AstNodeKind, Span)` so handlers can
    // map cloned nodes back to original nodes when querying `TypeInfo`.
    original_nodes_.clear();
    if (std::optional<const Module*> orig_module = FindOriginalModule(n);
        orig_module.has_value() && *orig_module != &n) {
      CollectNodesByNodeKey collector(&original_nodes_);
      XLS_RETURN_IF_ERROR((*orig_module)->Accept(&collector));
    }
    return Formatter::FormatModule(n);
  }

  DocRef FormatExpr(const Expr& n, bool suppress_parens) override {
    // Apply any per-expression context override (`type_inferred` and/or
    // `forced_type_annotation`) registered by an enclosing AST node handler.
    if (auto it = expr_overrides_.find(&n); it != expr_overrides_.end()) {
      ScopedContext scoped(*this, it->second.type_inferred,
                           it->second.forced_type_annotation);
      return Formatter::FormatExpr(n, suppress_parens);
    }
    return Formatter::FormatExpr(n, suppress_parens);
  }

  DocRef FormatNumber(const Number& n) override {
    // Drop the number's type annotation if the surrounding context already
    // determines its type (`type_inferred_` is true) or if it is a boolean
    // literal. Otherwise, keep its explicit type annotation (or attach
    // `forced_type_annotation_` if pushed down from an enclosing array).
    DocRef num_text;
    if (n.number_kind() == NumberKind::kCharacter) {
      std::string guts;
      if (n.text() == "\"") {
        guts = "\"";
      } else {
        guts = Escape(n.text());
      }
      num_text = arena().MakeText(absl::StrFormat("'%s'", guts));
    } else {
      num_text = arena().MakeText(n.text());
    }

    const TypeAnnotation* type = n.type_annotation() != nullptr
                                     ? n.type_annotation()
                                     : forced_type_annotation_;
    if (type != nullptr && !type_inferred_ &&
        n.number_kind() != NumberKind::kBool) {
      return ConcatNGroup(
          arena(), {FormatTypeAnnotation(*type), arena().colon(), num_text});
    }
    return num_text;
  }

  DocRef FormatMakeArrayLeader(const Array& n) override {
    // Emit or omit the leading array type annotation (`Type:[`) based on the
    // decision recorded in `array_leader_types_` by `ScopedArrayConfig`.
    if (auto it = array_leader_types_.find(&n);
        it != array_leader_types_.end()) {
      if (it->second == nullptr) {
        return arena().obracket();
      }
      return ConcatN(arena(), {FormatTypeAnnotation(*it->second),
                               arena().colon(), arena().obracket()});
    }
    return Formatter::FormatMakeArrayLeader(n);
  }

  DocRef FormatBlockedExprLeader(const Expr& e) override {
    if (e.kind() == AstNodeKind::kArray) {
      return FormatMakeArrayLeader(static_cast<const Array&>(e));
    }
    return Formatter::FormatBlockedExprLeader(e);
  }

  bool IsBlockedExprNoLeader(const Expr& e) override {
    if (e.kind() == AstNodeKind::kArray) {
      const auto& arr = static_cast<const Array&>(e);
      if (auto it = array_leader_types_.find(&arr);
          it != array_leader_types_.end()) {
        return it->second == nullptr;
      }
    }
    return Formatter::IsBlockedExprNoLeader(e);
  }

  bool IsBlockedExprWithLeader(const Expr& e) override {
    if (e.kind() == AstNodeKind::kArray) {
      const auto& arr = static_cast<const Array&>(e);
      if (auto it = array_leader_types_.find(&arr);
          it != array_leader_types_.end()) {
        return it->second != nullptr;
      }
    }
    return Formatter::IsBlockedExprWithLeader(e);
  }

  DocRef FormatArray(const Array& n) override {
    // Delegate array and element type-annotation simplification to
    // `ScopedArrayConfig`.
    ScopedArrayConfig config(*this, n);
    return Formatter::FormatArray(n);
  }

  DocRef FormatArrayWithoutComments(const Array& n) override {
    // Apply the same array simplification rules as `FormatArray`.
    ScopedArrayConfig config(*this, n);
    return Formatter::FormatArrayWithoutComments(n);
  }

  DocRef FormatConstantDef(const ConstantDef& n) override {
    // Drop type annotations in the constant's initializer if and only if the
    // `const` declaration has an explicit type annotation.
    ScopedContext scoped(*this,
                         /*type_inferred=*/n.type_annotation() != nullptr);
    return Formatter::FormatConstantDef(n);
  }

  DocRef FormatLet(const Let& n, bool trailing_semi) override {
    // Drop type annotations in the `let` RHS expression if and only if the
    // `let` binding has an explicit type annotation.
    ScopedContext scoped(*this,
                         /*type_inferred=*/n.type_annotation() != nullptr);
    return Formatter::FormatLet(n, trailing_semi);
  }

  DocRef FormatEnumMember(const EnumMember& n) override {
    // Always drop type annotations on enum member values, as their type is
    // determined by the enclosing enum's underlying type.
    ScopedContext scoped(*this, /*type_inferred=*/true);
    return Formatter::FormatEnumMember(n);
  }

  DocRef FormatStructInstance(const StructInstance& n) override {
    // Drop type annotations on struct field initializers if and only if all
    // parametric bindings of the struct are explicitly specified at the
    // instantiation site (otherwise field types may be needed to infer omitted
    // struct parametrics).
    const StructInstance* orig = OriginalNode(&n);
    ScopedContext scoped(*this,
                         /*type_inferred=*/AreAllStructParametricsSpecified(
                             type_info_, orig->struct_ref()));
    return Formatter::FormatStructInstance(n);
  }

  DocRef FormatSplatStructInstance(const SplatStructInstance& n) override {
    // Same as `FormatStructInstance`: drop type annotations on field
    // initializers only when all struct parametrics are explicitly specified.
    const SplatStructInstance* orig = OriginalNode(&n);
    ScopedContext scoped(*this,
                         /*type_inferred=*/AreAllStructParametricsSpecified(
                             type_info_, orig->struct_ref()));
    return Formatter::FormatSplatStructInstance(n);
  }

  DocRef FormatXlsTuple(const XlsTuple& n) override {
    // Never propagate `type_inferred_` into tuple elements; retain literal
    // type annotations inside tuples even when the tuple's type is inferred.
    ScopedContext scoped(*this, /*type_inferred=*/false);
    return Formatter::FormatXlsTuple(n);
  }

  DocRef FormatTypeAnnotation(const TypeAnnotation& n) override {
    // Always drop type annotations on expressions inside type annotations
    // (such as array dimension expressions like `u32[4]`).
    ScopedContext scoped(*this, /*type_inferred=*/true);
    return Formatter::FormatTypeAnnotation(n);
  }

  DocRef FormatParametricBinding(const ParametricBinding& n) override {
    // Always drop type annotations on a parametric binding's default
    // expression, as its type is declared on the binding itself.
    ScopedContext scoped(*this, /*type_inferred=*/true);
    return Formatter::FormatParametricBinding(n);
  }

  std::optional<DocRef> FormatExplicitParametrics(
      absl::Span<const ExprOrType> parametrics) override {
    // Always drop type annotations on explicit parametric arguments (e.g.,
    // `foo<32>()`), as the callee's parametric declarations fix their types.
    ScopedContext scoped(*this, /*type_inferred=*/true);
    return Formatter::FormatExplicitParametrics(parametrics);
  }

  DocRef FormatBinop(const Binop& n) override {
    ScopedExprOverrides overrides(*this);
    const Expr* lhs = n.lhs();
    const Expr* rhs = n.rhs();
    switch (n.binop_kind()) {
      case BinopKind::kConcat:
        // Never drop type annotations on either operand of `++`, since operand
        // bit-widths independently determine the result width.
        overrides.Set(lhs, {.type_inferred = false});
        overrides.Set(rhs, {.type_inferred = false});
        break;
      case BinopKind::kShl:
      case BinopKind::kShr:
        // Always drop the type annotation on the RHS shift amount. Keep the LHS
        // type annotation if the LHS starts with a typed number literal;
        // otherwise inherit `type_inferred_` for the LHS.
        overrides.Set(
            lhs, {.type_inferred =
                      StartsWithTypedNumber(lhs) ? false : type_inferred_});
        overrides.Set(rhs, {.type_inferred = true});
        break;
      case BinopKind::kLogicalAnd:
      case BinopKind::kLogicalOr:
        // Always drop type annotations on both operands of `&&` and `||` (both
        // are `bool`).
        overrides.Set(lhs, {.type_inferred = true});
        overrides.Set(rhs, {.type_inferred = true});
        break;
      case BinopKind::kEq:
      case BinopKind::kNe:
      case BinopKind::kLt:
      case BinopKind::kLe:
      case BinopKind::kGt:
      case BinopKind::kGe: {
        // Always drop the RHS type annotation; drop the LHS type annotation
        // only if at least one operand has a non-literal type.
        bool has_non_literal = HasNonLiteralType(lhs) || HasNonLiteralType(rhs);
        overrides.Set(lhs, {.type_inferred = has_non_literal});
        overrides.Set(rhs, {.type_inferred = true});
        break;
      }
      case BinopKind::kAdd:
      case BinopKind::kSub:
      case BinopKind::kMul:
      case BinopKind::kDiv:
      case BinopKind::kMod:
      case BinopKind::kAnd:
      case BinopKind::kOr:
      case BinopKind::kXor: {
        // Always drop the RHS type annotation; drop the LHS type annotation if
        // the enclosing context infers the result type (`type_inferred_`) or if
        // at least one operand has a non-literal type.
        bool lhs_inferred =
            type_inferred_ || HasNonLiteralType(lhs) || HasNonLiteralType(rhs);
        overrides.Set(lhs, {.type_inferred = lhs_inferred});
        overrides.Set(rhs, {.type_inferred = true});
        break;
      }
    }
    return Formatter::FormatBinop(n);
  }

  DocRef FormatInvocation(const Invocation& n) override {
    // Drop type annotations on call arguments if and only if all parametric
    // bindings of the callee (and of the target struct, for static method
    // calls on parametric structs) are explicitly specified at the call site.
    ScopedExprOverrides overrides(*this);
    const Invocation* orig = OriginalNode(&n);
    bool args_inferred = AreAllFunctionParametricsSpecified(type_info_, *orig);
    for (const Expr* arg : n.args()) {
      overrides.Set(arg, {.type_inferred = args_inferred});
    }
    return Formatter::FormatInvocation(n);
  }

  DocRef FormatFunction(const Function& n, bool is_test) override {
    // Drop type annotations on the function body's return value expression if
    // and only if the function declares an explicit return type.
    ScopedContext scoped(*this, /*type_inferred=*/n.return_type() != nullptr);
    return Formatter::FormatFunction(n, is_test);
  }

  DocRef FormatProc(const Proc& n, bool is_test) override {
    // Drop type annotations on the return value expressions of a legacy proc's
    // `config`, `init`, and `next` blocks, as their return types are fixed by
    // the proc's member declarations and `next` state parameter type.
    ScopedContext scoped(*this, /*type_inferred=*/true);
    return Formatter::FormatProc(n, is_test);
  }

  DocRef FormatChannelDecl(const ChannelDecl& n) override {
    // Always drop type annotations on a channel declaration's FIFO depth and
    // array dimensions (e.g., `chan<u32, 4>[2]("ch")`), which are always `u32`.
    ScopedContext scoped(*this, /*type_inferred=*/true);
    return Formatter::FormatChannelDecl(n);
  }

  DocRef FormatStatement(const Statement& n, bool trailing_semi) override {
    // Semicolon-terminated expression statements do not return a value to the
    // enclosing block, so reset `type_inferred_` to false for them.
    if (std::holds_alternative<Expr*>(n.wrapped()) && trailing_semi) {
      ScopedContext scoped(*this, /*type_inferred=*/false);
      return Formatter::FormatStatement(n, trailing_semi);
    }
    return Formatter::FormatStatement(n, trailing_semi);
  }

  DocRef FormatConditional(const Conditional& n) override {
    // Always drop type annotations on the `if` condition (`bool`). For
    // non-`const` conditionals, always drop type annotations on the `else`
    // branch, and drop them on the `then` branch if the conditional's result
    // type is inferred from context or if either branch has a non-literal type.
    ScopedExprOverrides overrides(*this);
    overrides.Set(n.test(), {.type_inferred = true});
    if (!n.IsConst()) {
      std::optional<const Expr*> cons_expr = GetBlockValueExpr(n.consequent());
      std::optional<const Expr*> alt_expr = GetAlternateValueExpr(n);
      bool any_non_literal =
          HasNonLiteralType(cons_expr) || HasNonLiteralType(alt_expr);
      overrides.Set(cons_expr,
                    {.type_inferred = type_inferred_ || any_non_literal});
      overrides.Set(alt_expr, {.type_inferred = true});
    }
    return Formatter::FormatConditional(n);
  }

  DocRef FormatMatch(const Match& n) override {
    // For match arm expressions: drop type annotations on all arms if the
    // `match` result type is inferred from context or if any arm has a
    // non-literal type; otherwise (for non-`const` match expressions), keep
    // the type annotation on the first arm and drop it on subsequent arms.
    // For match patterns: drop type annotations on pattern literals if and
    // only if the matched value expression has a non-literal type.
    ScopedExprOverrides overrides(*this);
    bool any_arm_non_literal = false;
    for (const MatchArm* arm : n.arms()) {
      if (HasNonLiteralType(arm->expr())) {
        any_arm_non_literal = true;
        break;
      }
    }
    for (size_t i = 0; i < n.arms().size(); ++i) {
      bool arm_inferred =
          type_inferred_ || any_arm_non_literal || (!n.IsConst() && i > 0);
      overrides.Set(n.arms()[i]->expr(), {.type_inferred = arm_inferred});
    }
    bool pattern_inferred = HasNonLiteralType(n.matched());
    ScopedMatchPatternContext pattern_ctx(*this, pattern_inferred);
    return Formatter::FormatMatch(n);
  }

  DocRef FormatPatternTree(const PatternTree& n) override {
    // Apply the match pattern inference state computed by the enclosing
    // `FormatMatch` when formatting literals inside match arm patterns.
    if (match_pattern_inferred_.has_value()) {
      ScopedContext scoped(*this, *match_pattern_inferred_);
      return Formatter::FormatPatternTree(n);
    }
    return Formatter::FormatPatternTree(n);
  }

  DocRef FormatIndex(const Index& n) override {
    // Never drop type annotations on the indexed LHS expression, because the
    // result type of `lhs[i]` does not determine the full type of `lhs`.
    ScopedExprOverrides overrides(*this);
    overrides.Set(n.lhs(), {.type_inferred = false});
    return Formatter::FormatIndex(n);
  }

  DocRef FormatIndexRhs(const IndexRhs& n) override {
    // Always drop type annotations on index and slice bound expressions on the
    // RHS of an indexing operation (`a[i]`, `a[lo:hi]`, `a[start+:width]`).
    ScopedContext scoped(*this, /*type_inferred=*/true);
    return Formatter::FormatIndexRhs(n);
  }

  DocRef FormatRange(const Range& n) override {
    // Always drop the type annotation on the range end expression; drop it on
    // the range start expression if the range type is inferred from context or
    // if either bound has a non-literal type.
    ScopedExprOverrides overrides(*this);
    bool start_inferred = type_inferred_ || HasNonLiteralType(n.start()) ||
                          HasNonLiteralType(n.end());
    overrides.Set(n.start(), {.type_inferred = start_inferred});
    overrides.Set(n.end(), {.type_inferred = true});
    return Formatter::FormatRange(n);
  }

  DocRef FormatFor(const For& n) override {
    // Drop type annotations on `iterable` if the loop has an explicit
    // `(index, accum)` type annotation, on `init` if either the loop's result
    // type is inferred from context or the loop has an explicit type
    // annotation, and always on the loop body's accumulator expression.
    ScopedExprOverrides overrides(*this);
    bool has_type_annot = n.type_annotation() != nullptr;
    overrides.Set(n.iterable(), {.type_inferred = has_type_annot});
    overrides.Set(n.init(),
                  {.type_inferred = type_inferred_ || has_type_annot});
    ScopedContext body_scoped(*this, /*type_inferred=*/true);
    return Formatter::FormatFor(n);
  }

  DocRef FormatConstFor(const ConstFor& n) override {
    // Same as `FormatFor`: drop type annotations on `iterable` when the loop
    // has an explicit type annotation, on `init` when the loop's type is
    // inferred or explicitly annotated, and always on the loop body's result.
    ScopedExprOverrides overrides(*this);
    bool has_type_annot = n.type_annotation() != nullptr;
    overrides.Set(n.iterable(), {.type_inferred = has_type_annot});
    overrides.Set(n.init(),
                  {.type_inferred = type_inferred_ || has_type_annot});
    ScopedContext body_scoped(*this, /*type_inferred=*/true);
    return Formatter::FormatConstFor(n);
  }

  DocRef FormatCast(const Cast& n) override {
    // Never drop type annotations on the cast operand (`expr as Type`), as the
    // source type cannot be inferred from the target cast type.
    ScopedExprOverrides overrides(*this);
    overrides.Set(n.expr(), {.type_inferred = false});
    return Formatter::FormatCast(n);
  }

  DocRef FormatTupleIndex(const TupleIndex& n) override {
    // Never drop type annotations on the tuple LHS of `tuple.i`, as knowing one
    // element's type does not determine the types of the remaining elements.
    ScopedExprOverrides overrides(*this);
    overrides.Set(n.lhs(), {.type_inferred = false});
    return Formatter::FormatTupleIndex(n);
  }

  DocRef FormatConstAssert(const ConstAssert& n) override {
    // Always drop type annotations on the `const_assert!` condition, which is
    // always `bool`.
    ScopedContext scoped(*this, /*type_inferred=*/true);
    return Formatter::FormatConstAssert(n);
  }

 private:
  // Finds the module in `type_info_` which is the version of `n` before any
  // cloning after type inference.
  std::optional<const Module*> FindOriginalModule(const Module& n) const {
    for (const auto& [import_subject, _] : type_info_.GetRootImports()) {
      const AstNode* node = ToAstNode(import_subject);
      if (node != nullptr && node->owner() != nullptr &&
          node->owner()->name() == n.name() &&
          node->owner()->fs_path() == n.fs_path()) {
        return node->owner();
      }
    }
    for (const auto& [node, _] : type_info_.dict()) {
      if (node != nullptr && node->owner() != nullptr &&
          node->owner()->name() == n.name() &&
          node->owner()->fs_path() == n.fs_path()) {
        return node->owner();
      }
    }
    return std::nullopt;
  }

  // Finds the version of `node` from `original_nodes_`, i.e. its version from
  // the time when type inference finished. If `original_nodes_` is empty, then
  // `node` is assumed to be "original".
  template <typename T>
  const T* OriginalNode(const T* node) const {
    if (node == nullptr || original_nodes_.empty()) {
      return node;
    }
    std::optional<Span> span = node->GetSpan();
    if (!span.has_value()) {
      return node;
    }
    auto it = original_nodes_.find(NodeKey{node->kind(), *span});
    if (it != original_nodes_.end()) {
      if (const auto* orig = dynamic_cast<const T*>(it->second)) {
        return orig;
      }
    }
    return node;
  }

  // Per-expression simplification state used to override `type_inferred_` and
  // `forced_type_annotation_` when formatting a specific child `Expr`.
  struct ExprContext {
    bool type_inferred = false;
    const TypeAnnotation* forced_type_annotation = nullptr;
  };

  // RAII guard that temporarily sets `type_inferred_` and
  // `forced_type_annotation_` on `TypeAnnotationSimplifier` for the duration of
  // a subtree's formatting and restores the previous values upon destruction.
  class ScopedContext {
   public:
    ScopedContext(TypeAnnotationSimplifier& s, bool type_inferred,
                  const TypeAnnotation* forced_type = nullptr)
        : s_(s),
          old_type_inferred_(s.type_inferred_),
          old_forced_type_(s.forced_type_annotation_) {
      s_.type_inferred_ = type_inferred;
      s_.forced_type_annotation_ = forced_type;
    }
    ~ScopedContext() {
      s_.type_inferred_ = old_type_inferred_;
      s_.forced_type_annotation_ = old_forced_type_;
    }

   private:
    TypeAnnotationSimplifier& s_;
    bool old_type_inferred_;
    const TypeAnnotation* old_forced_type_;
  };

  // RAII guard that registers temporary `ExprContext` overrides in
  // `expr_overrides_` for specific child expressions (applied when `FormatExpr`
  // visits them) and restores their prior entries upon destruction.
  class ScopedExprOverrides {
   public:
    explicit ScopedExprOverrides(TypeAnnotationSimplifier& s) : s_(s) {}
    void Set(std::optional<const Expr*> expr, ExprContext ctx) {
      if (!expr.has_value() || *expr == nullptr) {
        return;
      }
      auto it = s_.expr_overrides_.find(*expr);
      if (it != s_.expr_overrides_.end()) {
        previous_.emplace_back(*expr, it->second);
        it->second = ctx;
      } else {
        previous_.emplace_back(*expr, std::nullopt);
        s_.expr_overrides_.emplace(*expr, ctx);
      }
    }

    ~ScopedExprOverrides() {
      for (auto it = previous_.rbegin(); it != previous_.rend(); ++it) {
        if (it->second.has_value()) {
          s_.expr_overrides_[it->first] = *it->second;
        } else {
          s_.expr_overrides_.erase(it->first);
        }
      }
    }

   private:
    TypeAnnotationSimplifier& s_;
    std::vector<std::pair<const Expr*, std::optional<ExprContext>>> previous_;
  };

  // RAII guard that temporarily sets `match_pattern_inferred_` while formatting
  // a `Match` expression so `FormatPatternTree` knows whether pattern literals
  // can drop their type annotations, restoring the previous state on exit.
  class ScopedMatchPatternContext {
   public:
    ScopedMatchPatternContext(TypeAnnotationSimplifier& s, bool inferred)
        : s_(s), old_(s.match_pattern_inferred_) {
      s_.match_pattern_inferred_ = inferred;
    }

    ~ScopedMatchPatternContext() { s_.match_pattern_inferred_ = old_; }

   private:
    TypeAnnotationSimplifier& s_;
    std::optional<bool> old_;
  };

  // RAII guard that computes and records whether an `Array` literal should
  // emit a leading array type annotation (`array_leader_types_`) and sets
  // per-element `ExprContext` overrides (including pushing the element type
  // down to the first element when dropping the outer array annotation).
  // Guards against re-entry when `FormatArray` delegates to
  // `FormatArrayWithoutComments`.
  class ScopedArrayConfig {
   public:
    ScopedArrayConfig(TypeAnnotationSimplifier& s, const Array& n)
        : s_(s), array_(&n), overrides_(s) {
      if (!s_.active_arrays_.insert(&n).second) {
        active_ = false;
        return;
      }
      active_ = true;
      const TypeAnnotation* effective_type = n.type_annotation() != nullptr
                                                 ? n.type_annotation()
                                                 : s_.forced_type_annotation_;
      const auto* array_type =
          dynamic_cast<const ArrayTypeAnnotation*>(effective_type);
      const TypeAnnotation* elem_type =
          array_type != nullptr ? array_type->element_type() : nullptr;

      bool any_non_literal = false;
      for (const Expr* member : n.members()) {
        if (HasNonLiteralType(member)) {
          any_non_literal = true;
          break;
        }
      }

      if (s_.type_inferred_ && !n.has_ellipsis()) {
        // The surrounding context already determines the array's full type and
        // length (and there is no `...`), so drop both the outer array type
        // annotation and all element type annotations.
        s_.array_leader_types_[&n] = nullptr;
        for (const Expr* member : n.members()) {
          overrides_.Set(member, {.type_inferred = true});
        }
      } else if (!n.has_ellipsis() && !n.members().empty() &&
                 (effective_type == nullptr || array_type != nullptr)) {
        // The array's length is fixed by its explicit member list (no `...`),
        // and its element type can be inferred from the members themselves:
        // drop the outer array type annotation, and drop element type
        // annotations on all members except element 0 when all members are
        // literals (pushing `elem_type` onto element 0 if it is not already
        // annotated).
        s_.array_leader_types_[&n] = nullptr;
        for (size_t i = 0; i < n.members().size(); ++i) {
          if (!any_non_literal && i == 0) {
            overrides_.Set(
                n.members()[i],
                {.type_inferred = false, .forced_type_annotation = elem_type});
          } else {
            overrides_.Set(n.members()[i], {.type_inferred = true});
          }
        }
      } else {
        // Either the array has an ellipsis (`...`), is empty, or its outer type
        // annotation is a type alias (e.g., `MyArray:[1, 2]`): keep the outer
        // array type annotation and drop all element type annotations.
        s_.array_leader_types_[&n] = effective_type;
        for (const Expr* member : n.members()) {
          overrides_.Set(member, {.type_inferred = true});
        }
      }
    }

    ~ScopedArrayConfig() {
      if (active_) {
        s_.active_arrays_.erase(array_);
      }
    }

   private:
    TypeAnnotationSimplifier& s_;
    const Array* array_;
    bool active_ = false;
    ScopedExprOverrides overrides_;
  };

  const TypeInfo& type_info_;
  absl::flat_hash_map<NodeKey, const AstNode*> original_nodes_;
  bool type_inferred_ = false;
  const TypeAnnotation* forced_type_annotation_ = nullptr;
  std::optional<bool> match_pattern_inferred_;
  absl::flat_hash_map<const Expr*, ExprContext> expr_overrides_;
  absl::flat_hash_set<const Array*> active_arrays_;
  absl::flat_hash_map<const Array*, const TypeAnnotation*> array_leader_types_;
};

}  // namespace

std::unique_ptr<Formatter> CreateTypeAnnotationSimplifier(
    Comments& comments, DocArena& arena, const TypeInfo& type_info) {
  return std::make_unique<TypeAnnotationSimplifier>(comments, arena, type_info);
}

}  // namespace xls::dslx
