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

#include "xls/dslx/type_system_v2/typecheck_module_v2.h"

#include <iterator>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_to_type_info.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

// Determines what function is being invoked by a `callee` expression that is
// not invoking an `impl` instance method.
absl::StatusOr<const Function*> ResolveFreeFunction(
    const Expr* callee, const FileTable& file_table) {
  if (const auto& name_ref = dynamic_cast<const NameRef*>(callee);
      name_ref != nullptr &&
      std::holds_alternative<const NameDef*>(name_ref->name_def())) {
    const NameDef* def = std::get<const NameDef*>(name_ref->name_def());
    const auto* fn = dynamic_cast<Function*>(def->definer());
    if (fn == nullptr) {
      return TypeInferenceErrorStatus(
          callee->span(), nullptr,
          absl::Substitute("Invocation callee `$0` is not a function",
                           callee->ToString()),
          file_table);
    }
    return fn;
  }
  return absl::UnimplementedError(
      "Type inference version 2 is a work in progress and only supports "
      "invoking free functions in the same module so far.");
}

// A visitor that walks an AST and populates an `InferenceTable` with the
// encountered info.
class PopulateInferenceTableVisitor : public AstNodeVisitorWithDefault {
 public:
  PopulateInferenceTableVisitor(Module& module, InferenceTable& table,
                                const FileTable& file_table)
      : module_(module), table_(table), file_table_(file_table) {}

  absl::Status HandleConstantDef(const ConstantDef* node) override {
    VLOG(5) << "HandleConstantDef: " << node->ToString();
    XLS_ASSIGN_OR_RETURN(const NameRef* variable,
                         DefineTypeVariableForVariableOrConstant(node));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->value(), variable));
    return DefaultHandler(node);
  }

  absl::Status HandleParam(const Param* node) override {
    VLOG(5) << "HandleParam: " << node->ToString();
    XLS_RETURN_IF_ERROR(DefineTypeVariableForVariableOrConstant(node).status());
    return DefaultHandler(node);
  }

  absl::Status HandleNameRef(const NameRef* node) override {
    return PropagateDefToRef(node);
  }

  absl::Status HandleNumber(const Number* node) override {
    VLOG(5) << "HandleNumber: " << node->ToString();
    TypeAnnotation* annotation = node->type_annotation();
    if (annotation == nullptr) {
      XLS_ASSIGN_OR_RETURN(annotation,
                           CreateAnnotationSizedToFit(module_, *node));
      // Treat `true` and `false` like they have intrinsic bool annotations.
      // Otherwise, consider an annotation we add to be an auto-annotation that
      // is "negotiable".
      if (node->number_kind() != NumberKind::kBool) {
        auto_literal_annotations_.insert(annotation);
      }
    }
    return table_.SetTypeAnnotation(node, annotation);
  }

  absl::Status HandleBinop(const Binop* node) override {
    VLOG(5) << "HandleBinop: " << node->ToString();

    // Any `Binop` should be a descendant of some context-setting node and
    // should have a type that was set when its parent was visited.
    const NameRef* type_variable = *table_.GetTypeVariable(node);
    if (GetBinopSameTypeKinds().contains(node->binop_kind()) ||
        GetBinopLogicalKinds().contains(node->binop_kind())) {
      // In the example `const C = a + b;`, the `ConstantDef` establishes a type
      // variable that is just propagated down to `a` and `b` here, meaning that
      // `a`, `b`, and the result must ultimately be the same type.
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->lhs(), type_variable));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->rhs(), type_variable));
    } else if (GetBinopComparisonKinds().contains(node->binop_kind())) {
      // In a comparison example, like `const C = a > b;`, the `>` establishes a
      // new type variable for `a` and `b` (meaning the two of them must be the
      // same type), and attaches a bool annotation to the overall expression,
      // which will then be assumed by the type variable for the `ConstantDef`.
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node, CreateBoolAnnotation(module_, node->span())));
      XLS_ASSIGN_OR_RETURN(
          const NameRef* operand_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Binop*>(node),
              GenerateInternalTypeVariableName(node)));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(node->lhs(), operand_variable));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(node->rhs(), operand_variable));
    } else {
      return absl::UnimplementedError(
          absl::StrCat("Type inference version 2 is a work in progress and "
                       "does not yet support the expression: ",
                       node->ToString()));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleUnop(const Unop* node) override {
    VLOG(5) << "HandleUnop: " << node->ToString();

    // Any `Unop` should be a descendant of some context-setting node and
    // should have a type that was set when its parent was visited.
    const NameRef* type_variable = *table_.GetTypeVariable(node);
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->operand(), type_variable));

    return DefaultHandler(node);
  }

  absl::Status HandleCast(const Cast* node) override {
    VLOG(5) << "HandleCast: " << node->ToString();

    // Create a new type variable for the casted expression.
    XLS_ASSIGN_OR_RETURN(const NameRef* casted_variable,
                         table_.DefineInternalVariable(
                             InferenceVariableKind::kType, node->expr(),
                             GenerateInternalTypeVariableName(node->expr())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->expr(), casted_variable));

    // The cast node has the target type annotation (assuming it is valid, which
    // will be checked at conversion time).
    const TypeAnnotation* target_type = node->type_annotation();
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node, target_type));
    return DefaultHandler(node);
  }

  absl::Status HandleConditional(const Conditional* node) override {
    VLOG(5) << "HandleConditional: " << node->ToString();
    // In the example `const D = if (a) {b} else {c};`, the `ConstantDef`
    // establishes a type variable that is just propagated down to `b` and
    // `c` here, meaning that `b`, `c`, and the result must ultimately be
    // the same type as 'D'. The test 'a' must be a bool, so we annotate it as
    // such.
    const NameRef* type_variable = *table_.GetTypeVariable(node);
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->consequent(), type_variable));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(ToAstNode(node->alternate()), type_variable));

    // Mark the test as bool.
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node->test(), CreateBoolAnnotation(module_, node->test()->span())));
    XLS_ASSIGN_OR_RETURN(
        const NameRef* test_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->test()),
            GenerateInternalTypeVariableName(node->test())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->test(), test_variable));

    return DefaultHandler(node);
  }

  absl::Status HandleMatch(const Match* node) override {
    VLOG(5) << "HandleMatch: " << node->ToString();

    // Any `match` should be a descendant of some context-setting node and
    // should have a type that was set when its parent was visited. Each
    // arm of the `match` must match the type of the `match` itself.
    const NameRef* arm_type = *table_.GetTypeVariable(node);
    for (const MatchArm* arm : node->arms()) {
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(arm->expr(), arm_type));
      // TODO: also set type variables on each arm's patterns from
      // node->matched()'s type, but that depends on destructured let, which
      // isn't implemented yet.
    }
    return DefaultHandler(node);
  }

  absl::Status HandleXlsTuple(const XlsTuple* node) override {
    VLOG(5) << "HandleXlsTuple: " << node->ToString();

    // When we come in here with an example like:
    //   const FOO: (u32, (s8, u32)) = (4, (-2, 5));
    //
    // the table will look like this before descent into this function:
    //   Node               Annotation          Variable
    //   -----------------------------------------------
    //   FOO                (u32, (s8, u32))    T0
    //   (4, (-2, 5))                           T0
    //
    // and this function will make it look like this:
    //   Node               Annotation          Variable
    //   -----------------------------------------------
    //   FOO                (u32, (s8, u32))    T0
    //   (4, (-2, 5))       (var:M0, var:M1)    T0
    //   4                  u32                 M0
    //   (-2, 5)            (s8, u32)           M1
    //
    // Recursive descent will ultimately put auto annotations for the literals
    // in the table. Upon conversion of the table to type info, unification of
    // the LHS annotation with the variable-based RHS annotation will be
    // attempted.

    XLS_ASSIGN_OR_RETURN(
        std::optional<const TypeAnnotation*> tuple_annotation,
        GetDeclarationTypeAnnotation<TupleTypeAnnotation>(node));

    // Create the M0, M1, ... variables and apply them to the members.
    std::vector<TypeAnnotation*> member_types;
    member_types.reserve(node->members().size());
    for (int i = 0; i < node->members().size(); ++i) {
      Expr* member = node->members()[i];
      XLS_ASSIGN_OR_RETURN(const NameRef* member_variable,
                           table_.DefineInternalVariable(
                               InferenceVariableKind::kType, member,
                               GenerateInternalTypeVariableName(member)));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(member, member_variable));
      if (tuple_annotation.has_value()) {
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
            member,
            module_.Make<ElementTypeAnnotation>(
                *tuple_annotation,
                module_.Make<Number>((*tuple_annotation)->span(),
                                     absl::StrCat(i), NumberKind::kOther,
                                     /*type_annotation=*/nullptr))));
      }
      member_types.push_back(
          module_.Make<TypeVariableTypeAnnotation>(member_variable));
    }
    // Annotate the whole tuple expression as (var:M0, var:M1, ...).
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, module_.Make<TupleTypeAnnotation>(node->span(), member_types)));
    return DefaultHandler(node);
  }

  absl::Status HandleTypeRefTypeAnnotation(
      const TypeRefTypeAnnotation* node) override {
    VLOG(5) << "HandleTypeRefTypeAnnotation: " << node->ToString();

    std::optional<StructOrProcRef> struct_or_proc_ref =
        GetStructOrProcRef(node);
    if (!struct_or_proc_ref.has_value() ||
        struct_or_proc_ref->parametrics.empty()) {
      return DefaultHandler(node);
    }
    const StructDefBase* struct_def =
        dynamic_cast<const StructDefBase*>(ToAstNode(struct_or_proc_ref->def));
    if (struct_or_proc_ref->parametrics.size() >
        struct_def->parametric_bindings().size()) {
      return ArgCountMismatchErrorStatus(
          node->span(),
          absl::Substitute(
              "Too many parametric values supplied; limit: $0 given: $1",
              struct_def->parametric_bindings().size(),
              struct_or_proc_ref->parametrics.size()),
          file_table_);
    }

    // If any parametrics are explicitly specified, then they must all be
    // explicit or defaulted. We technically could infer the rest, as with
    // functions, but historically we choose not to. We must also constrain the
    // actual parametric values to the binding type.
    for (int i = 0; i < struct_def->parametric_bindings().size(); i++) {
      const ParametricBinding* binding = struct_def->parametric_bindings()[i];
      if (i < struct_or_proc_ref->parametrics.size()) {
        const Expr* actual_expr =
            i < struct_or_proc_ref->parametrics.size()
                ? std::get<Expr*>(struct_or_proc_ref->parametrics[i])
                : binding->expr();
        XLS_ASSIGN_OR_RETURN(
            const NameRef* actual_expr_variable,
            table_.DefineInternalVariable(
                InferenceVariableKind::kType, const_cast<Expr*>(actual_expr),
                GenerateInternalTypeVariableName(actual_expr)));
        XLS_RETURN_IF_ERROR(
            table_.SetTypeVariable(actual_expr, actual_expr_variable));
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(actual_expr, binding->type_annotation()));
      } else if (binding->expr() == nullptr) {
        return ArgCountMismatchErrorStatus(
            node->span(),
            absl::Substitute("No parametric value provided for `$0` in `$1`",
                             binding->identifier(), struct_def->identifier()),
            file_table_);
      }
    }
    return DefaultHandler(node);
  }

  absl::Status HandleStructInstance(const StructInstance* node) override {
    // As far as we're concerned here, type-checking a struct instance is like
    // type-checking a function invocation (see `HandleFreeFunctionInvocation`),
    // but with named arguments instead of parallel ordering. The naming of
    // arguments creates additional pitfalls, like erroneously naming two
    // different arguments the same thing.
    VLOG(5) << "HandleStructInstance: " << node->ToString();

    XLS_RETURN_IF_ERROR(node->struct_ref()->Accept(this));
    std::optional<StructOrProcRef> struct_or_proc_ref =
        GetStructOrProcRef(node->struct_ref());
    if (!struct_or_proc_ref.has_value()) {
      return TypeInferenceErrorStatusForAnnotation(
          node->span(), node->struct_ref(),
          absl::Substitute(
              "Attempted to instantiate non-struct type `$0` as a struct.",
              node->struct_ref()->ToString()),
          file_table_);
    }
    if (!std::holds_alternative<const StructDef*>(struct_or_proc_ref->def)) {
      return TypeInferenceErrorStatusForAnnotation(
          node->span(), node->struct_ref(),
          "Impl-style procs are a work in progress and cannot yet be "
          "instantiated.",
          file_table_);
    }

    const NameRef* type_variable = *table_.GetTypeVariable(node);
    const TypeAnnotation* struct_variable_type =
        module_.Make<TypeVariableTypeAnnotation>(type_variable);
    const StructDef* struct_def =
        std::get<const StructDef*>(struct_or_proc_ref->def);
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node,
        CreateStructAnnotation(module_, const_cast<StructDef*>(struct_def),
                               struct_or_proc_ref->parametrics, node)));
    XLS_RETURN_IF_ERROR(ValidateStructInstanceMemberNames(*node, *struct_def));

    std::vector<std::pair<std::string, Expr*>> members =
        node->GetOrderedMembers(struct_def);
    for (int i = 0; i < members.size(); i++) {
      const auto& [name, actual_member] = members[i];
      const StructMemberNode* formal_member = struct_def->members()[i];
      XLS_ASSIGN_OR_RETURN(
          const NameRef* member_type_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(actual_member),
              GenerateInternalTypeVariableName(formal_member, actual_member)));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(actual_member, member_type_variable));
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          actual_member, module_.Make<MemberTypeAnnotation>(
                             struct_variable_type, struct_def, formal_member)));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleArray(const Array* node) override {
    VLOG(5) << "HandleArray: " << node->ToString();

    // When we come in here with an example like:
    //   const FOO = [u32:4, u32:5];
    //
    // the table will look like this before descent into this function:
    //   Node               Annotation          Variable
    //   -----------------------------------------------
    //   FOO                                    T0
    //   [u32:4, u32:5]                         T0
    //
    // and this function will make it look like this:
    //   Node               Annotation          Variable
    //   -----------------------------------------------
    //   FOO                                    T0
    //   [u32:4, u32:5]     var:T1[2]           T0
    //   u32:4                                  T1
    //   u32:5                                  T1
    //
    // Recursive descent will ultimately put annotations on the elements in the
    // table. Upon conversion of the table to type info, unification of any LHS
    // annotation with the variable-based RHS annotation will be attempted, and
    // this unification will fail if the array is inadequately annotated (e.g.
    // no explicit annotation on a zero-size or elliptical array).

    XLS_ASSIGN_OR_RETURN(
        std::optional<const TypeAnnotation*> array_annotation,
        GetDeclarationTypeAnnotation<ArrayTypeAnnotation>(node));

    // An empty array can't end with an ellipsis, even if unification is
    // possible.
    if (node->has_ellipsis() && node->members().empty()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          "Array cannot have an ellipsis (`...`) without an element to repeat.",
          file_table_);
    }

    // If the array has no members, we can't infer an RHS element type and
    // therefore can't annotate the RHS. The success of the later unification
    // will depend on whether the LHS is annotated.
    if (node->members().empty()) {
      return absl::OkStatus();
    }

    // Create a variable for the element type, and assign it to all the
    // elements.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* element_type_variable,
        table_.DefineInternalVariable(InferenceVariableKind::kType,
                                      const_cast<Array*>(node),
                                      GenerateInternalTypeVariableName(node)));
    for (Expr* member : node->members()) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(member, element_type_variable));
      if (array_annotation.has_value()) {
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
            member, module_.Make<ElementTypeAnnotation>(*array_annotation)));
      }
    }
    Expr* element_count = module_.Make<Number>(
        node->span(), absl::StrCat(node->members().size()), NumberKind::kOther,
        /*type_annotation=*/nullptr);
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node,
        module_.Make<ArrayTypeAnnotation>(
            node->span(),
            module_.Make<TypeVariableTypeAnnotation>(element_type_variable),
            element_count,
            /*dim_is_min=*/node->has_ellipsis())));
    return DefaultHandler(node);
  }

  absl::Status HandleIndex(const Index* node) override {
    if (std::holds_alternative<Slice*>(node->rhs()) ||
        std::holds_alternative<WidthSlice*>(node->rhs())) {
      return absl::UnimplementedError(
          "Type inference version 2 is a work in progress and does not yet "
          "support slices.");
    }

    // A node like `array[i]` is basically a binary operator with independent
    // contexts on the LHS and RHS. The RHS is constrained to u32, while the LHS
    // must be some kind of array. The "some kind of array" part is not
    // capturable in the table, but readily verifiable at the end of type
    // inference, so we defer that.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* lhs_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->lhs()),
            GenerateInternalTypeVariableName(node->lhs())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->lhs(), lhs_variable));
    Expr* index = std::get<Expr*>(node->rhs());
    XLS_ASSIGN_OR_RETURN(
        const NameRef* rhs_variable,
        table_.DefineInternalVariable(InferenceVariableKind::kType,
                                      const_cast<Expr*>(index),
                                      GenerateInternalTypeVariableName(index)));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(index, rhs_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        index, CreateU32Annotation(module_, index->span())));

    // The type of the entire expr is then ElementType(lhs_variable).
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, module_.Make<ElementTypeAnnotation>(
                  module_.Make<TypeVariableTypeAnnotation>(lhs_variable))));
    return DefaultHandler(node);
  }

  absl::Status HandleFunction(const Function* node) override {
    VLOG(5) << "HandleFunction: " << node->ToString()
            << ", parametric: " << node->IsParametric();
    if (node->IsParametric()) {
      // We only process parametric functions when we see invocations of them in
      // real functions. `HandleInvocation` ends up calling
      // `HandleFunctionInternal` for the callee function in that case.
      return absl::OkStatus();
    }
    current_function_stack_.push(node);
    XLS_RETURN_IF_ERROR(HandleFunctionInternal(node));
    current_function_stack_.pop();
    return absl::OkStatus();
  }

  absl::Status HandleParametricBinding(const ParametricBinding* node) override {
    VLOG(5) << "HandleParametricBinding: " << node->ToString();
    XLS_RETURN_IF_ERROR(table_.DefineParametricVariable(*node).status());
    if (node->expr() != nullptr) {
      // To handle the default expression correctly, we need to impose a type
      // variable pretending that there is something like a `let` or `const`
      // LHS, and the expression type will later have to be unified to that.
      XLS_ASSIGN_OR_RETURN(const NameRef* type_of_parametric,
                           table_.DefineInternalVariable(
                               InferenceVariableKind::kType, node->expr(),
                               GenerateInternalTypeVariableName(node->expr())));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(node->expr(), type_of_parametric));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node->expr(), node->type_annotation()));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleStatementBlock(const StatementBlock* node) override {
    // A statement block may have a type variable imposed at a higher level of
    // the tree. For example, in
    //     `const X = { statement0; ...; statementN }`
    // or
    //     `fn foo() -> u32 { statement0; ...; statementN }`
    //
    // we will have imposed a type variable on the statement block upon hitting
    // the `ConstantDef` or `Function`. In such cases, we need to propagate the
    // statement block's type variable to `statementN`, if it is an `Expr`, in
    // order for unification to ensure that it's producing the expected type.
    std::optional<const NameRef*> variable = table_.GetTypeVariable(node);
    if (!node->trailing_semi() && !node->statements().empty() &&
        variable.has_value()) {
      const Statement* last_statement =
          node->statements()[node->statements().size() - 1];
      if (std::holds_alternative<Expr*>(last_statement->wrapped())) {
        XLS_RETURN_IF_ERROR(table_.SetTypeVariable(
            std::get<Expr*>(last_statement->wrapped()), *variable));
      }
    }
    return DefaultHandler(node);
  }

  absl::Status HandleInvocation(const Invocation* node) override {
    VLOG(5) << "HandleInvocation: " << node->ToString();
    XLS_ASSIGN_OR_RETURN(const Function* fn,
                         ResolveFreeFunction(node->callee(), file_table_));
    std::optional<const ParametricInvocation*> current_parametric_invocation =
        GetCurrentParametricInvocation();
    // If we are already in a parametric function when we hit an invocation,
    // remember this node as one of the caller's outbound invocations, because
    // such invocations need reprocessing on any subsequent analyses of the
    // caller function when reached from other contexts.
    if (current_parametric_invocation.has_value()) {
      invocations_per_parametric_function_[&(*current_parametric_invocation)
                                                ->callee()]
          .push_back(node);
    }
    return fn->IsParametric()
               ? HandleParametricInvocation(node, *fn)
               : HandleFreeFunctionInvocation(
                     node, *fn, /*parametric_invocation=*/std::nullopt);
  }

  absl::Status HandleZeroMacro(const ZeroMacro* node) override {
    VLOG(5) << "HandleZeroMacro: " << node->ToString();

    ExprOrType type = node->type();
    if (std::holds_alternative<TypeAnnotation*>(type)) {
      // If the "type" is a type annotation, that's the type.
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node, std::get<TypeAnnotation*>(type)));
      return DefaultHandler(node);
    }
    // If it's an "expr", that's an error (just like in V1)
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::Substitute("Expected a type in zero! macro type; saw `$0`.",
                         std::get<Expr*>(type)->ToString()),
        file_table_);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (AstNode* child : node->GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

  const absl::flat_hash_set<const TypeAnnotation*>& auto_literal_annotations()
      const {
    return auto_literal_annotations_;
  }

  const absl::flat_hash_map<const TypeAnnotation*, const ParametricInvocation*>&
  invocation_scoped_type_annotations() const {
    return invocation_scoped_type_annotations_;
  }

 private:
  // Helper that creates an internal type variable for a `ConstantDef`, `Param`,
  // or similar type of node that contains a `NameDef` and optional
  // `TypeAnnotation`.
  template <typename T>
  absl::StatusOr<const NameRef*> DefineTypeVariableForVariableOrConstant(
      const T* node) {
    XLS_ASSIGN_OR_RETURN(const NameRef* variable,
                         table_.DefineInternalVariable(
                             InferenceVariableKind::kType, const_cast<T*>(node),
                             GenerateInternalTypeVariableName(node)));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node, variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->name_def(), variable));
    if (node->type_annotation() != nullptr) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node->name_def(), node->type_annotation()));
    }
    return variable;
  }

  // Helper that handles invocation nodes calling free functions, i.e. functions
  // that do not require callee object type info to be looked up. If a
  // `parametric_invocation` is specified, it is for the invocation actually
  // done by `node`.
  absl::Status HandleFreeFunctionInvocation(
      const Invocation* node, const Function& fn,
      std::optional<const ParametricInvocation*> parametric_invocation) {
    VLOG(5) << "HandleFreeFunctionInvocation: " << node->ToString()
            << ", fn: " << fn.identifier();

    // When we come in here with an example like:
    //   let x: u32 = foo(a, b);
    //
    // the table will look like this before descent into this function:
    //   Node               Annotation             Variable
    //   --------------------------------------------------
    //   x                  u32                    T0
    //   foo(a, b)                                 T0
    //
    // and this function will make it look like this:
    //   Node               Annotation             Variable
    //   --------------------------------------------------
    //   x                  u32                    T0
    //   foo(a, b)          formal ret type        T0
    //   a                  formal arg0 type       T1
    //   b                  formal arg1 type       T2
    //
    // Recursive descent will attach the types of the actual arg exprs to `T1`
    // and `T2`. Unification at the end will ensure that the LHS (`x` in this
    // example) agrees with the formal return type, and that each actual
    // argument type matches the formal argument type.

    // The argument count must be correct in order to do anything else.
    if (node->args().size() != fn.params().size()) {
      return ArgCountMismatchErrorStatus(
          node->span(),
          absl::Substitute("Expected $0 argument(s) but got $1.",
                           fn.params().size(), node->args().size()),
          file_table_);
    }

    XLS_ASSIGN_OR_RETURN(
        const TypeAnnotation* return_type,
        ScopeToParametricInvocation(parametric_invocation, GetReturnType(fn)));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node, return_type));
    for (int i = 0; i < node->args().size(); i++) {
      const Expr* actual_param = node->args()[i];
      const Param* formal_param = fn.params()[i];
      XLS_ASSIGN_OR_RETURN(
          const NameRef* arg_type_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(actual_param),
              GenerateInternalTypeVariableName(formal_param, actual_param)));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(actual_param, arg_type_variable));
      XLS_ASSIGN_OR_RETURN(
          const TypeAnnotation* formal_param_type,
          ScopeToParametricInvocation(parametric_invocation,
                                      formal_param->type_annotation()));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(actual_param, formal_param_type));
      XLS_RETURN_IF_ERROR(actual_param->Accept(this));
    }
    return absl::OkStatus();
  }

  // Performs the common handling for any type of function. The caller must push
  // the function onto `current_function_stack_` beforehand and pop it
  // sometime afterwards; this is left up to the caller because it may want the
  // function on the stack for longer than the `HandleFunctionInternal` call.
  absl::Status HandleFunctionInternal(const Function* node) {
    VLOG(5) << "HandleFunctionInternal: " << node->ToString();

    CHECK(!current_function_stack_.empty());
    CHECK(current_function_stack_.top() == node);

    // Create a variable for the function return type, and use it to unify the
    // formal and actual return type. Eventually, we should create the notion of
    // "function type" annotations, and use them here instead of annotating the
    // function with just the return type. However, until we have constructs
    // like lambdas, that would be overkill.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* type_variable,
        table_.DefineInternalVariable(InferenceVariableKind::kType,
                                      const_cast<Function*>(node),
                                      GenerateInternalTypeVariableName(node)));
    const TypeAnnotation* return_type = GetReturnType(*node);
    XLS_RETURN_IF_ERROR(return_type->Accept(this));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node, type_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->body(), type_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node, return_type));

    for (const Param* param : node->params()) {
      XLS_RETURN_IF_ERROR(param->Accept(this));
    }
    XLS_RETURN_IF_ERROR(node->body()->Accept(this));
    return absl::OkStatus();
  }

  // Re-processes the invocation nodes in the given parametric function, in the
  // context of the current parametric invocation on the stack.
  absl::Status ReprocessParametricInvocations(const Function* node) {
    CHECK(!parametric_invocation_stack_.empty());
    CHECK(&parametric_invocation_stack_.top()->callee() == node);

    const auto it = invocations_per_parametric_function_.find(node);
    CHECK(it != invocations_per_parametric_function_.end());
    for (const Invocation* invocation : it->second) {
      XLS_RETURN_IF_ERROR(HandleParametricInvocation(invocation, *node));
    }
    return absl::OkStatus();
  }

  // Top-level helper to handle an invocation of a parametric function.
  absl::Status HandleParametricInvocation(const Invocation* node,
                                          const Function& fn) {
    VLOG(5) << "HandleParametricInvocation: " << node->ToString()
            << ", fn: " << fn.identifier();
    CHECK(fn.IsParametric());
    const std::vector<ParametricBinding*>& bindings = fn.parametric_bindings();
    const std::vector<ExprOrType>& explicit_parametrics =
        node->explicit_parametrics();
    const std::optional<const Function*> caller = GetCurrentFunction();
    current_function_stack_.push(&fn);
    const bool function_processed_before =
        !invocations_per_parametric_function_
             .emplace(&fn, std::vector<const Invocation*>{})
             .second;
    if (!function_processed_before) {
      // The bindings need to be defined in the table up front, because the rest
      // of the header may depend on them, and we can't even create a
      // `ParametricInvocation` without them being registered.
      for (const ParametricBinding* binding : bindings) {
        XLS_RETURN_IF_ERROR(binding->Accept(this));
      }
    }

    if (explicit_parametrics.size() > bindings.size()) {
      return ArgCountMismatchErrorStatus(
          node->span(),
          absl::Substitute(
              "Too many parametric values supplied; limit: $0 given: $1",
              bindings.size(), explicit_parametrics.size()),
          file_table_);
    }

    // Type-check the subtrees for any explicit parametric values. Note that the
    // addition of the invocation above will have verified that a valid number
    // of explicit parametrics was passed in.
    for (int i = 0; i < explicit_parametrics.size(); i++) {
      ExprOrType explicit_parametric = explicit_parametrics[i];
      const ParametricBinding* formal_parametric = bindings[i];
      if (std::holds_alternative<Expr*>(explicit_parametric)) {
        const Expr* parametric_value_expr =
            std::get<Expr*>(explicit_parametric);
        XLS_ASSIGN_OR_RETURN(
            const NameRef* type_variable,
            table_.DefineInternalVariable(
                InferenceVariableKind::kType,
                const_cast<Expr*>(parametric_value_expr),
                GenerateInternalTypeVariableName(parametric_value_expr)));
        XLS_RETURN_IF_ERROR(
            table_.SetTypeVariable(parametric_value_expr, type_variable));
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
            parametric_value_expr, formal_parametric->type_annotation()));
        XLS_RETURN_IF_ERROR(parametric_value_expr->Accept(this));
      }
    }

    // Register the parametric invocation in the table, regardless of whether
    // we have seen the function before.
    XLS_ASSIGN_OR_RETURN(
        const ParametricInvocation* parametric_invocation,
        table_.AddParametricInvocation(*node, fn, caller,
                                       GetCurrentParametricInvocation()));

    // We don't need to process the entire function multiple times, if it's
    // used in multiple contexts. Only the invocation nodes in it need to be
    // dealt with multiple times.
    parametric_invocation_stack_.push(parametric_invocation);
    if (function_processed_before) {
      VLOG(5) << "Reprocessing outbound invocations in this context from: "
              << fn.identifier();
      XLS_RETURN_IF_ERROR(ReprocessParametricInvocations(&fn));
    } else {
      VLOG(5) << "Doing first-time processing for parametric function: "
              << fn.identifier();
      XLS_RETURN_IF_ERROR(HandleFunctionInternal(&fn));
    }
    current_function_stack_.pop();
    parametric_invocation_stack_.pop();

    // Then, we handle the actual invocation of it, as we would a real function.
    XLS_RETURN_IF_ERROR(
        HandleFreeFunctionInvocation(node, fn, parametric_invocation));
    return absl::OkStatus();
  }

  // Returns the function currently being handled (or the containing function,
  // if we are handling a descendant node). The result should only be `nullopt`
  // for global nodes.
  std::optional<const Function*> GetCurrentFunction() {
    return current_function_stack_.empty()
               ? std::nullopt
               : std::make_optional(current_function_stack_.top());
  }

  // Returns the parametric invocation currently being handled.
  std::optional<const ParametricInvocation*> GetCurrentParametricInvocation() {
    return parametric_invocation_stack_.empty()
               ? std::nullopt
               : std::make_optional(parametric_invocation_stack_.top());
  }

  // Generates a name for an internal inference variable that will be used as
  // the type for the given node. The name is only relevant for traceability.
  template <typename T>
  std::string GenerateInternalTypeVariableName(const T* node) {
    return absl::Substitute("internal_type_$0_at_$1", node->identifier(),
                            node->span().ToString(file_table_));
  }
  // Specialization for `Expr` nodes, which do not have an identifier.
  template <>
  std::string GenerateInternalTypeVariableName(const Expr* node) {
    return absl::StrCat("internal_type_expr_at_",
                        node->span().ToString(file_table_));
  }
  // Specialization for `Array` nodes.
  template <>
  std::string GenerateInternalTypeVariableName(const Array* node) {
    return absl::StrCat("internal_type_array_element_at_",
                        node->span().ToString(file_table_));
  }
  // Variant for an actual function argument.
  std::string GenerateInternalTypeVariableName(const Param* formal_param,
                                               const Expr* actual_arg) {
    return absl::StrCat("internal_type_actual_arg_", formal_param->identifier(),
                        "_at_", actual_arg->span().ToString(file_table_));
  }
  // Variant for an actual struct member expr.
  std::string GenerateInternalTypeVariableName(
      const StructMemberNode* formal_member, const Expr* actual_member) {
    return absl::StrCat("internal_type_actual_member_", formal_member->name(),
                        "_at_", actual_member->span().ToString(file_table_));
  }
  // Variant for operands of a binary operator.
  std::string GenerateInternalTypeVariableName(const Binop* binop) {
    return absl::StrCat("internal_type_operand_",
                        BinopKindToString(binop->binop_kind()), "_at_",
                        binop->span().ToString(file_table_));
  }

  // Propagates the type from the def for `ref`, to `ref` itself in the
  // inference table. This may result in a `TypeAnnotation` being added to the
  // table, but never a variable. If the type of the def is governed by a
  // variable, then `ref` will get a `TypeVariableTypeAnnotation`. This allows
  // the caller to assign a variable to `ref` which unifies it with its
  // context, while also carrying the type information over from its def.
  template <typename T>
  absl::Status PropagateDefToRef(const T* ref) {
    const AstNode* def;
    if constexpr (is_variant<decltype(ref->name_def())>::value) {
      def = ToAstNode(ref->name_def());
    } else {
      def = ref->name_def();
    }
    std::optional<const NameRef*> variable = table_.GetTypeVariable(def);
    if (variable.has_value()) {
      return table_.SetTypeAnnotation(
          ref, module_.Make<TypeVariableTypeAnnotation>(*variable));
    }
    std::optional<const TypeAnnotation*> annotation =
        table_.GetTypeAnnotation(def);
    if (annotation.has_value()) {
      return table_.SetTypeAnnotation(ref, *annotation);
    }
    return absl::OkStatus();
  }

  // Returns the explicit or implied return type of `fn`.
  const TypeAnnotation* GetReturnType(const Function& fn) {
    return fn.return_type() != nullptr
               ? fn.return_type()
               : CreateUnitTupleAnnotation(module_, fn.span());
  }

  // Clones the given `annotation` and marks it as scoped to the given
  // `parametric_invocation` in `invocation_scoped_type_annotations_`. The
  // reason for doing this is because in a situation like:
  //
  //    fn foo<N: u32>(a: uN[N]) -> uN[N] { ... }
  //    fn bar() { foo<32>(5); }
  //
  // The invocation node `foo<32>(5)` has the implicit type annotation `uN[N]`
  // where `N` must be evaluated within the context of the invocation done by
  // that node, even though the invocation node itself resides in the global
  // `TypeInfo`. The subtree in the position where we have the `5` in this
  // example also has this property.
  absl::StatusOr<const TypeAnnotation*> ScopeToParametricInvocation(
      std::optional<const ParametricInvocation*> parametric_invocation,
      const TypeAnnotation* annotation) {
    if (!parametric_invocation.has_value()) {
      return annotation;
    }
    VLOG(5) << "Scoping annotation " << annotation->ToString();
    XLS_ASSIGN_OR_RETURN(
        AstNode * cloned,
        CloneAst(annotation, &PreserveTypeDefinitionsReplacer));
    annotation = dynamic_cast<const TypeAnnotation*>(cloned);
    CHECK(annotation != nullptr);
    for (const AstNode* next : FlattenToSet(annotation)) {
      if (const auto* next_annotation =
              dynamic_cast<const TypeAnnotation*>(next)) {
        invocation_scoped_type_annotations_.emplace(next_annotation,
                                                    *parametric_invocation);
      }
    }
    return annotation;
  }

  // Ensures that a `StructInstance` nodes provides exprs for all the names in a
  // struct definition, with no extraneous or duplicate names.
  absl::Status ValidateStructInstanceMemberNames(const StructInstance& instance,
                                                 const StructDefBase& def) {
    std::vector<std::string> formal_name_vector = def.GetMemberNames();
    absl::btree_set<std::string> formal_names(formal_name_vector.begin(),
                                              formal_name_vector.end());
    absl::btree_set<std::string> actual_names;
    for (const auto& [name, expr] : instance.GetUnorderedMembers()) {
      if (!formal_names.contains(name)) {
        return TypeInferenceErrorStatus(
            expr->span(), nullptr,
            absl::Substitute("Struct `$0` has no member `$1`, but it was "
                             "provided by this instance.",
                             def.identifier(), name),
            file_table_);
      }
      if (!actual_names.insert(name).second) {
        return TypeInferenceErrorStatus(
            expr->span(), nullptr,
            absl::Substitute(
                "Duplicate value seen for `$0` in this `$1` struct instance.",
                name, def.identifier()),
            file_table_);
      }
    }
    if (actual_names.size() != formal_names.size()) {
      absl::btree_set<std::string> missing_set;
      absl::c_set_difference(formal_names, actual_names,
                             std::inserter(missing_set, missing_set.begin()));
      std::vector<std::string> missing(missing_set.begin(), missing_set.end());
      return TypeInferenceErrorStatus(
          instance.span(), nullptr,
          absl::Substitute(
              "Instance of struct `$0` is missing member(s): $1",
              def.identifier(),
              absl::StrJoin(missing, ", ",
                            [](std::string* out, const std::string& piece) {
                              absl::StrAppendFormat(out, "`%s`", piece);
                            })),
          file_table_);
    }
    return absl::OkStatus();
  }

  // Gets the explicit type annotation (expected to be of type `T` if it is
  // direct) for a node by querying the type variable that it shares with a
  // declaration, if any. This must be done before imposing any synthetic type
  // annotation on the value.
  template <typename T>
  absl::StatusOr<std::optional<const TypeAnnotation*>>
  GetDeclarationTypeAnnotation(const AstNode* node) {
    std::optional<const NameRef*> type_variable = table_.GetTypeVariable(node);
    if (!type_variable.has_value()) {
      return std::nullopt;
    }
    XLS_ASSIGN_OR_RETURN(
        std::vector<const TypeAnnotation*> annotations,
        table_.GetTypeAnnotationsForTypeVariable(*type_variable));
    if (annotations.empty()) {
      return std::nullopt;
    }
    // If > 1, the caller is ignoring the "before imposing an annotation on the
    // RHS" precondition.
    CHECK_EQ(annotations.size(), 1);

    // Constraining the annotation type here improves error messages in
    // situations where there is a type mismatch for an entire array/tuple.
    // We allow indirect member/element annotations through at this point,
    // because we can't yet prove whether they amount to something expected.
    if (dynamic_cast<const T*>(annotations[0]) ||
        dynamic_cast<const MemberTypeAnnotation*>(annotations[0]) ||
        dynamic_cast<const ElementTypeAnnotation*>(annotations[0])) {
      return annotations[0];
    }
    return std::nullopt;
  }

  Module& module_;
  InferenceTable& table_;
  const FileTable& file_table_;
  absl::flat_hash_set<const TypeAnnotation*> auto_literal_annotations_;
  std::stack<const Function*> current_function_stack_;
  std::stack<const ParametricInvocation*> parametric_invocation_stack_;
  absl::flat_hash_map<const Function*, std::vector<const Invocation*>>
      invocations_per_parametric_function_;
  absl::flat_hash_map<const TypeAnnotation*, const ParametricInvocation*>
      invocation_scoped_type_annotations_;
};

}  // namespace

absl::StatusOr<TypeInfo*> TypecheckModuleV2(Module* module,
                                            ImportData* import_data,
                                            WarningCollector* warnings) {
  std::unique_ptr<InferenceTable> table = InferenceTable::Create(*module);
  PopulateInferenceTableVisitor visitor(*module, *table,
                                        import_data->file_table());
  XLS_RETURN_IF_ERROR(module->Accept(&visitor));
  return InferenceTableToTypeInfo(*table, *module, *import_data, *warnings,
                                  import_data->file_table(),
                                  visitor.auto_literal_annotations(),
                                  visitor.invocation_scoped_type_annotations());
}

}  // namespace xls::dslx
