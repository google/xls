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

#include "xls/dslx/type_system_v2/populate_table_visitor.h"

#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/ir/format_strings.h"

namespace xls::dslx {
namespace {

// A visitor that walks an AST and populates an `InferenceTable` with the
// encountered info.
class PopulateInferenceTableVisitor : public PopulateTableVisitor,
                                      AstNodeVisitorWithDefault {
 public:
  PopulateInferenceTableVisitor(Module& module, InferenceTable& table,
                                ImportData& import_data,
                                TypecheckModuleFn typecheck_imported_module)
      : module_(module),
        table_(table),
        file_table_(import_data.file_table()),
        import_data_(import_data),
        typecheck_imported_module_(std::move(typecheck_imported_module)) {}

  absl::Status PopulateFromModule(const Module* module) override {
    return module->Accept(this);
  }

  absl::Status PopulateFromInvocation(const Invocation* invocation) override {
    return invocation->Accept(this);
  }

  absl::Status PopulateFromUnrolledLoopBody(
      const StatementBlock* root) override {
    XLS_RET_CHECK(!handle_proc_functions_);
    std::optional<const Function*> containing_function =
        GetContainingFunction(root);
    handle_proc_functions_ =
        containing_function.has_value() && (*containing_function)->IsInProc();

    absl::Status result = root->Accept(this);
    handle_proc_functions_ = false;
    return result;
  }

  absl::Status HandleImport(const Import* node) override {
    VLOG(5) << "HandleImport: " << node->ToString();
    ImportTokens import_subject = ImportTokens(node->subject());
    if (import_data_.Contains(import_subject)) {
      return DefaultHandler(node);
    }
    XLS_RETURN_IF_ERROR(DoImport(typecheck_imported_module_, import_subject,
                                 &import_data_, node->span(),
                                 import_data_.vfs())
                            .status());
    return DefaultHandler(node);
  }

  absl::Status HandleUse(const Use* node) override {
    VLOG(5) << "HandleUse: " << node->ToString();
    for (UseSubject& subject : const_cast<Use*>(node)->LinearizeToSubjects()) {
      XLS_ASSIGN_OR_RETURN(
          UseImportResult result,
          DoImportViaUse(typecheck_imported_module_, subject, &import_data_,
                         subject.name_def().span(), import_data_.file_table(),
                         import_data_.vfs()));
      XLS_RET_CHECK(result.imported_member != nullptr);
      for (NameDef* name_def :
           ModuleMemberGetNameDefs(*result.imported_member)) {
        std::optional<const NameRef*> type_var =
            table_.GetTypeVariable(name_def);
        std::optional<NameDef*> subject_name_def =
            subject.use_tree_entry().GetLeafNameDef();
        if (type_var.has_value() && subject_name_def.has_value()) {
          XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
              ToAstNode(*subject_name_def),
              module_.Make<TypeVariableTypeAnnotation>(*type_var)));
        }
      }
    }
    return DefaultHandler(node);
  }

  absl::Status HandleConstantDef(const ConstantDef* node) override {
    VLOG(5) << "HandleConstantDef: " << node->ToString();
    XLS_ASSIGN_OR_RETURN(const NameRef* variable,
                         DefineTypeVariableForVariableOrConstant(node));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->value(), variable));
    return DefaultHandler(node);
  }

  absl::Status HandleChannelDecl(const ChannelDecl* node) override {
    VLOG(5) << "HandleChannelDecl: " << node->ToString()
            << " with type: " << node->type()->ToString();
    if (node->fifo_depth().has_value()) {
      Expr* fifo_depth = *node->fifo_depth();
      XLS_ASSIGN_OR_RETURN(const NameRef* fifo_depth_var,
                           table_.DefineInternalVariable(
                               InferenceVariableKind::kType, fifo_depth,
                               GenerateInternalTypeVariableName(fifo_depth)));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(fifo_depth, fifo_depth_var));
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          fifo_depth, CreateU32Annotation(module_, fifo_depth->span())));
    }
    if (node->dims().has_value()) {
      for (Expr* dim : *node->dims()) {
        XLS_ASSIGN_OR_RETURN(const NameRef* dim_variable,
                             table_.DefineInternalVariable(
                                 InferenceVariableKind::kType, dim,
                                 GenerateInternalTypeVariableName(dim)));
        XLS_RETURN_IF_ERROR(table_.SetTypeVariable(dim, dim_variable));
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
            dim, CreateU32Annotation(module_, dim->span())));
      }
    }
    CHECK_NE(node->type(), nullptr);
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, module_.Make<TupleTypeAnnotation>(
                  node->span(), std::vector<TypeAnnotation*>{
                                    module_.Make<ChannelTypeAnnotation>(
                                        node->span(), ChannelDirection::kOut,
                                        node->type(), node->dims()),
                                    module_.Make<ChannelTypeAnnotation>(
                                        node->span(), ChannelDirection::kIn,
                                        node->type(), node->dims())})));

    return DefaultHandler(node);
  }

  absl::Status HandleParam(const Param* node) override {
    VLOG(5) << "HandleParam: " << node->ToString();
    XLS_RETURN_IF_ERROR(DefineTypeVariableForVariableOrConstant(node).status());
    return DefaultHandler(node);
  }

  absl::Status HandleProc(const Proc* node) override {
    VLOG(5) << "HandleProc: " << node->ToString();
    handle_proc_functions_ = true;
    XLS_RETURN_IF_ERROR(DefaultHandler(node));
    handle_proc_functions_ = false;
    return absl::OkStatus();
  }

  absl::Status HandleProcMember(const ProcMember* node) override {
    VLOG(5) << "HandleProcMember: " << node->ToString();
    XLS_RETURN_IF_ERROR(DefineTypeVariableForVariableOrConstant(node).status());
    return DefaultHandler(node);
  }

  absl::Status HandleNameRef(const NameRef* node) override {
    VLOG(5) << "HandleNameRef: " << node->ToString();
    return PropagateDefToRef(node);
  }

  absl::Status HandleColonRef(const ColonRef* node) override {
    // Generally this handler sets both a type annotation and a colon ref target
    // for `node` (the target is a dedicated piece of data specific to
    // `ColonRef` nodes). If the `ColonRef` refers to a type, then these two
    // things are the same. If the `ColonRef` is to a constant or function, then
    // the target is e.g. a `ConstantDef` or `Function` and the type annotation
    // is its type.

    VLOG(5) << "HandleColonRef: " << node->ToString() << " of subject kind: "
            << AstNodeKindToString(ToAstNode(node->subject())->kind());

    // All single `ColonRef` cases are inside here.
    if (std::holds_alternative<NameRef*>(node->subject())) {
      // `SomeEnum::SOME_CONSTANT` case.
      std::variant<const NameDef*, BuiltinNameDef*> any_name_def =
          std::get<NameRef*>(node->subject())->name_def();
      if (const NameDef** name_def = std::get_if<const NameDef*>(&any_name_def);
          name_def && (*name_def)->definer()->kind() == AstNodeKind::kEnumDef) {
        const auto* enum_def =
            down_cast<const EnumDef*>((*name_def)->definer());
        const TypeAnnotation* type_ref_annotation =
            module_.Make<TypeRefTypeAnnotation>(
                (*name_def)->span(),
                module_.Make<TypeRef>(
                    enum_def->span(),
                    TypeDefinition(const_cast<EnumDef*>(enum_def))),
                std::vector<ExprOrType>(), std::nullopt);
        table_.SetColonRefTarget(node, *name_def);
        return table_.SetTypeAnnotation(node, type_ref_annotation);
      }

      // `imported_module::SomeStruct` case.
      XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_ref,
                           GetStructOrProcRef(node, import_data_));
      if (struct_ref.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            TypeDefinition type_def,
            ToTypeDefinition(const_cast<StructDefBase*>(struct_ref->def)));
        const TypeAnnotation* type_ref_annotation =
            module_.Make<TypeRefTypeAnnotation>(
                node->span(), module_.Make<TypeRef>(node->span(), type_def),
                struct_ref->parametrics, std::nullopt);
        table_.SetColonRefTarget(node, type_ref_annotation);
        return table_.SetTypeAnnotation(node, type_ref_annotation);
      }

      // `SomeStruct::CONSTANT` or `SomeStruct::function` case.
      XLS_ASSIGN_OR_RETURN(struct_ref,
                           GetStructOrProcRefForSubject(node, import_data_));
      if (struct_ref.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            std::optional<const AstNode*> def,
            HandleStructAttributeReferenceInternal(
                node, *struct_ref->def, struct_ref->parametrics, node->attr()));
        if (def.has_value()) {
          return PropagateDefToRef(*def, node);
        }
      }

      // Built-in member of a built-in type case, like `u32::ZERO`.
      AstNode* def = ToAstNode(std::get<NameRef*>(node->subject())->name_def());
      if (auto* builtin_name_def = dynamic_cast<BuiltinNameDef*>(def)) {
        return table_.SetTypeAnnotation(
            node, module_.Make<MemberTypeAnnotation>(
                      CreateBuiltinTypeAnnotation(module_, builtin_name_def,
                                                  node->span()),
                      node->attr()));
      }

      // Built-in member of a built-in type being accessed via a type alias.
      const AstNode* definer = dynamic_cast<const NameDef*>(def)->definer();
      if (const auto* alias = dynamic_cast<const TypeAlias*>(definer)) {
        return table_.SetTypeAnnotation(
            node, module_.Make<MemberTypeAnnotation>(&alias->type_annotation(),
                                                     node->attr()));
      }
    }

    // A double colon ref should resolve to a struct definition with an
    // associated impl or to an imported enum.
    if (std::holds_alternative<ColonRef*>(node->subject())) {
      auto* sub_col_ref = std::get<ColonRef*>(node->subject());
      XLS_ASSIGN_OR_RETURN(std::optional<ModuleInfo*> import_module,
                           GetImportedModuleInfo(sub_col_ref, import_data_));
      XLS_RET_CHECK(import_module.has_value());
      XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_ref,
                           GetStructOrProcRef(sub_col_ref, import_data_));
      if (struct_ref.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            std::optional<const AstNode*> ref,
            HandleStructAttributeReferenceInternal(
                node, *struct_ref->def, struct_ref->parametrics, node->attr()));
        XLS_RET_CHECK(ref.has_value());

        return SetCrossModuleTypeAnnotation(node, *ref);
      }
      XLS_ASSIGN_OR_RETURN(ModuleMember member,
                           GetPublicModuleMember((*import_module)->module(),
                                                 sub_col_ref, file_table_));
      return SetCrossModuleTypeAnnotation(node, ToAstNode(member));
    }

    // `S<parametrics>::CONSTANT` or `S<parametrics>::static_fn`. We can't fully
    // resolve these things on the spot, so we do some basic validation and then
    // produce a `MemberTypeAnnotation` for deferred resolution.
    if (std::holds_alternative<TypeRefTypeAnnotation*>(node->subject())) {
      const auto* annotation =
          std::get<TypeRefTypeAnnotation*>(node->subject());
      XLS_RETURN_IF_ERROR(annotation->Accept(this));
      XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                           GetStructOrProcRef(annotation, import_data_));
      if (struct_or_proc_ref.has_value()) {
        XLS_RETURN_IF_ERROR(HandleStructAttributeReferenceInternal(
                                node, *struct_or_proc_ref->def,
                                struct_or_proc_ref->parametrics, node->attr())
                                .status());
      }
      return table_.SetTypeAnnotation(
          node, module_.Make<MemberTypeAnnotation>(annotation, node->attr()));
    }

    // Any imported_module::entity case not covered above.
    XLS_ASSIGN_OR_RETURN(std::optional<ModuleInfo*> import_module,
                         GetImportedModuleInfo(node, import_data_));
    if (import_module.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          ModuleMember member,
          GetPublicModuleMember((*import_module)->module(), node, file_table_));
      absl::StatusOr<TypeDefinition> type_def =
          ToTypeDefinition(ToAstNode(member));
      if (type_def.ok()) {
        const TypeAnnotation* type_ref_annotation =
            module_.Make<TypeRefTypeAnnotation>(
                node->span(), module_.Make<TypeRef>(node->span(), *type_def),
                std::vector<ExprOrType>(), std::nullopt);
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(node, type_ref_annotation));
      } else {
        XLS_RETURN_IF_ERROR(
            SetCrossModuleTypeAnnotation(node, ToAstNode(member)));
      }
      table_.SetColonRefTarget(node, ToAstNode(member));
      return absl::OkStatus();
    }

    return TypeInferenceErrorStatus(
        node->span(), /*type=*/nullptr,
        absl::Substitute("Invalid colon reference: `$0`", node->ToString()),
        file_table_);
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
        table_.SetAnnotationFlag(annotation, TypeInferenceFlag::kMinSize);
        if (node->HasPrefix()) {
          table_.SetAnnotationFlag(annotation, TypeInferenceFlag::kHasPrefix);
        }
      }
    } else {
      XLS_RETURN_IF_ERROR(annotation->Accept(this));
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
    } else if (GetBinopShifts().contains(node->binop_kind())) {
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->lhs(), type_variable));
      XLS_ASSIGN_OR_RETURN(const NameRef* rhs_variable,
                           table_.DefineInternalVariable(
                               InferenceVariableKind::kType, node->rhs(),
                               GenerateInternalTypeVariableName(node->rhs())));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->rhs(), rhs_variable));
    } else if (node->binop_kind() == BinopKind::kConcat) {
      // The type of a concat is
      //   ArrayType(ElementType(lhs),
      //             element_count<lhs_var>() + element_count<rhs_var>())
      //
      // which is bits-like if the element type amounts to a built-in bits type;
      // otherwise, it's a real array.
      //
      // There is a nontrivial set of rules for what input types are actually
      // allowed, and the application of those rules is deferred until
      // `ValidateConcreteType` at the end.
      XLS_ASSIGN_OR_RETURN(
          const NameRef* lhs_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(node->lhs()),
              GenerateInternalTypeVariableName(node->lhs())));
      XLS_ASSIGN_OR_RETURN(
          const NameRef* rhs_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(node->rhs()),
              GenerateInternalTypeVariableName(node->rhs())));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->lhs(), lhs_variable));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->rhs(), rhs_variable));
      auto* lhs_tvta = module_.Make<TypeVariableTypeAnnotation>(lhs_variable);
      auto* rhs_tvta = module_.Make<TypeVariableTypeAnnotation>(rhs_variable);

      // Create the synthetic AST node for the expression, and process it.
      Expr* sum = CreateElementCountSum(module_, lhs_tvta, rhs_tvta);
      XLS_ASSIGN_OR_RETURN(
          const NameRef* sum_variable,
          table_.DefineInternalVariable(InferenceVariableKind::kType, sum,
                                        GenerateInternalTypeVariableName(sum)));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(sum, sum_variable));
      XLS_RETURN_IF_ERROR(sum->Accept(this));

      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node, module_.Make<ArrayTypeAnnotation>(
                    node->span(),
                    module_.Make<ElementTypeAnnotation>(
                        lhs_tvta, /*tuple_index=*/std::nullopt,
                        /*allow_bit_vector_destructuring=*/true),
                    sum)));
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

    XLS_ASSIGN_OR_RETURN(
        const NameRef* matched_var_type,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, node->matched(),
            GenerateInternalTypeVariableName(node->matched())));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->matched(), matched_var_type));

    if (node->arms().empty()) {
      return TypeInferenceErrorStatus(node->span(), /*type=*/nullptr,
                                      "`match` expression has no arms.",
                                      file_table_);
    }

    absl::flat_hash_set<std::string> seen_patterns;
    for (MatchArm* arm : node->arms()) {
      // Identify syntactically identical match arms.
      std::string patterns_string = PatternsToString(arm);
      if (auto [it, inserted] = seen_patterns.insert(patterns_string);
          !inserted) {
        return TypeInferenceErrorStatus(
            arm->GetPatternSpan(), nullptr,
            absl::StrFormat("Exact-duplicate pattern match detected `%s`; only "
                            "the first could possibly match",
                            patterns_string),
            file_table_);
      }

      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(arm->expr(), arm_type));
      for (const NameDefTree* pattern : arm->patterns()) {
        XLS_RETURN_IF_ERROR(table_.SetTypeVariable(pattern, matched_var_type));
        if (pattern->is_leaf()) {
          XLS_RETURN_IF_ERROR(table_.SetTypeVariable(ToAstNode(pattern->leaf()),
                                                     matched_var_type));
        }
      }
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
      std::optional<TypeAnnotation*> element_annotation;
      if (tuple_annotation.has_value()) {
        element_annotation = module_.Make<ElementTypeAnnotation>(
            *tuple_annotation,
            module_.Make<Number>((*tuple_annotation)->span(), absl::StrCat(i),
                                 NumberKind::kOther,
                                 /*type_annotation=*/nullptr));
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(member, *element_annotation));
      }
      XLS_ASSIGN_OR_RETURN(
          const NameRef* member_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, member,
              GenerateInternalTypeVariableName(member), element_annotation));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(member, member_variable));
      member_types.push_back(
          module_.Make<TypeVariableTypeAnnotation>(member_variable));
    }
    // Annotate the whole tuple expression as (var:M0, var:M1, ...).
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, module_.Make<TupleTypeAnnotation>(node->span(), member_types)));
    return DefaultHandler(node);
  }

  absl::Status HandleRestOfTuple(const RestOfTuple* node) override {
    VLOG(5) << "HandleRestOfTuple: " << node->ToString();
    TypeAnnotation* any_type =
        module_.Make<AnyTypeAnnotation>(/*multiple=*/true);
    return table_.SetTypeAnnotation(node, any_type);
  }

  // Recursively handles a `NameDefTree`, propagating the unified type of the
  // tree downward to children (i.e. the children's types are element types of
  // the tree's type variable). This function sets the root-level type to be a
  // tuple of `Any` types, the rationale being that a `NameDefTree` itself does
  // not provide any top-level type information but the allowable size of the
  // tuple. A `NameDefTree` must be used in a context where this tuple of `Any`
  // will get unified against a source of actual type information.
  absl::Status HandleNameDefTree(const NameDefTree* node) override {
    VLOG(5) << "HandleNameDefTree: " << node->ToString();

    if (node->is_leaf()) {
      return DefaultHandler(node);
    }

    std::vector<TypeAnnotation*> member_types;
    const NameRef* variable = *table_.GetTypeVariable(node);

    for (int i = 0; i < node->nodes().size(); i++) {
      const NameDefTree* child = node->nodes()[i];
      member_types.push_back(
          module_.Make<AnyTypeAnnotation>(child->IsRestOfTupleLeaf()));
    }
    TupleTypeAnnotation* tuple_annotation =
        module_.Make<TupleTypeAnnotation>(node->span(), member_types);
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node, tuple_annotation));
    return HandleNameDefTreeChildren(
        node, module_.Make<TypeVariableTypeAnnotation>(variable));
  }

  // Handles all the children of `subtree`, with `type` being the type of
  // `subtree` itself (a TVTA for unification of the whole tuple, or
  // user-specified type).
  absl::Status HandleNameDefTreeChildren(const NameDefTree* subtree,
                                         const TypeAnnotation* type) {
    std::optional<int> rest_of_tuple_index;
    for (int i = 0; i < subtree->nodes().size(); i++) {
      if (subtree->nodes()[i]->IsRestOfTupleLeaf()) {
        rest_of_tuple_index = i;
        break;
      }
      XLS_RETURN_IF_ERROR(HandleNameDefTreeChild(subtree, type, i));
    }

    if (rest_of_tuple_index.has_value()) {
      for (int i = subtree->nodes().size() - 1; i > *rest_of_tuple_index; i--) {
        if (subtree->nodes()[i]->IsRestOfTupleLeaf()) {
          return TypeInferenceErrorStatus(
              subtree->nodes()[i]->span(), /*type=*/nullptr,
              "`..` can only be used once per tuple pattern.", file_table_);
        }
        XLS_RETURN_IF_ERROR(HandleNameDefTreeChild(
            subtree, type, i, /*use_right_based_index_in_type=*/true));
      }
    }
    return absl::OkStatus();
  }

  // Annotates child `i` of `tree` to be `ElementType(tree_type, i)`, meaning it
  // derives its type from the unified type of its parent. The element type
  // annotation will refer to it using a right-based index if
  // `use_right_based_index_in_type` is true, which implies there is a
  // rest-of-tuple node somewhere before child `i`. This function also sets a
  // type variable on the child so that its parent-derived type will be unified
  // with any other information that traversing it picks up (e.g. if it is a
  // literal).
  absl::Status HandleNameDefTreeChild(
      const NameDefTree* tree, const TypeAnnotation* tree_type, int i,
      bool use_right_based_index_in_type = false) {
    const NameDefTree* child = tree->nodes()[i];
    const AstNode* actual_child =
        child->is_leaf() ? ToAstNode(child->leaf()) : child;
    const TypeAnnotation* element_type = nullptr;
    if (use_right_based_index_in_type) {
      // Index from the right uses element count - i.
      Expr* offset = CreateElementCountOffset(
          module_, const_cast<TypeAnnotation*>(tree_type),
          tree->nodes().size() - i);
      XLS_ASSIGN_OR_RETURN(
          const NameRef* offset_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(offset),
              GenerateInternalTypeVariableName(offset)));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(offset, offset_variable));
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          offset, CreateU32Annotation(module_, child->span())));
      XLS_RETURN_IF_ERROR(offset->Accept(this));
      element_type = module_.Make<ElementTypeAnnotation>(tree_type, offset);
    } else {
      // Index from the left just uses a literal.
      XLS_ASSIGN_OR_RETURN(
          Number * index,
          MakeTypeCheckedNumber(module_, table_, child->span(), i,
                                CreateU32Annotation(module_, child->span())));
      element_type = module_.Make<ElementTypeAnnotation>(tree_type, index);
    }

    XLS_ASSIGN_OR_RETURN(
        const NameRef* child_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<AstNode*>(actual_child),
            absl::Substitute("internal_type_ndt_at_$0_in_$1",
                             child->span().ToString(file_table_),
                             module_.name())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(actual_child, child_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(actual_child, element_type));
    if (child->is_leaf()) {
      XLS_RETURN_IF_ERROR(ToAstNode(child->leaf())->Accept(this));
    } else {
      XLS_RETURN_IF_ERROR(HandleNameDefTreeChildren(child, element_type));
    }
    return absl::OkStatus();
  }

  absl::Status HandleForLoopBase(const ForLoopBase* node) {
    // If a type annotation is explicitly specified, it overrides the default
    // type annotation for other components in the for loop.
    TypeAnnotation* iterator_accumulator_type_annotation =
        node->type_annotation();
    TypeAnnotation* iterator_type_annotation = nullptr;
    TypeAnnotation* accumulator_type_annotation = nullptr;
    ArrayTypeAnnotation* iterable_type_annotation = nullptr;
    if (iterator_accumulator_type_annotation) {
      XLS_RETURN_IF_ERROR(iterator_accumulator_type_annotation->Accept(this));

      TupleTypeAnnotation* tuple_type_annotation =
          dynamic_cast<TupleTypeAnnotation*>(
              iterator_accumulator_type_annotation);
      if (!tuple_type_annotation || tuple_type_annotation->size() != 2) {
        return TypeInferenceErrorStatusForAnnotation(
            iterator_accumulator_type_annotation->span(),
            iterator_accumulator_type_annotation,
            " For-loop annotated type should be a tuple containing a type for "
            "the iterable and a type for the accumulator.",
            file_table_);
      }
      iterator_type_annotation = tuple_type_annotation->members()[0];
      accumulator_type_annotation = tuple_type_annotation->members()[1];
      iterable_type_annotation = module_.Make<ArrayTypeAnnotation>(
          iterator_type_annotation->span(), iterator_type_annotation,
          module_.Make<Number>(
              iterator_type_annotation->span(), "0", NumberKind::kOther,
              CreateU32Annotation(module_, iterator_type_annotation->span())),
          /*dim_is_min=*/true);
    }

    // Handle iterable.
    XLS_ASSIGN_OR_RETURN(const NameRef* iterable_type_variable,
                         table_.DefineInternalVariable(
                             InferenceVariableKind::kType, node->iterable(),
                             GenerateInternalTypeVariableName(node->iterable()),
                             iterable_type_annotation
                                 ? std::make_optional(iterable_type_annotation)
                                 : std::nullopt));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->iterable(), iterable_type_variable));
    if (iterable_type_annotation) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node->iterable(), iterable_type_annotation));
    }

    // Handle namedef of iterator and accumulator.
    if (!node->names()->IsIrrefutable() || node->names()->nodes().size() != 2) {
      return TypeInferenceErrorStatus(
          node->names()->span(),
          /*type=*/nullptr,
          absl::Substitute("For-loop iterator and accumulator name tuple must "
                           "contain 2 top-level elements; got: `$0`",
                           node->names()->ToString()),
          file_table_);
    }
    NameDefTree* iterator_ndt = node->names()->nodes()[0];
    AstNode* iterator = iterator_ndt->is_leaf()
                            ? ToAstNode(iterator_ndt->leaf())
                            : iterator_ndt;
    XLS_ASSIGN_OR_RETURN(
        const NameRef* iterator_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, iterator,
            absl::Substitute("internal_type_iterator_at_$0_in_$1",
                             node->span().ToString(file_table_),
                             module_.name()),
            iterator_accumulator_type_annotation
                ? std::make_optional(iterator_accumulator_type_annotation)
                : std::nullopt));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(iterator, iterator_type_variable));
    // The type of iterator and accumulator should be covariant with iterable's
    // element type and For node type respectively.
    const NameRef* for_node_type_variable = *table_.GetTypeVariable(node);
    NameDefTree* accumulator_ndt = node->names()->nodes()[1];
    AstNode* accumulator = accumulator_ndt->is_leaf()
                               ? ToAstNode(accumulator_ndt->leaf())
                               : accumulator_ndt;

    if (node->body()->trailing_semi() && !accumulator_ndt->IsWildcardLeaf() &&
        !accumulator_ndt->IsRestOfTupleLeaf() && accumulator_ndt->is_leaf()) {
      return TypeInferenceErrorStatus(
          accumulator_ndt->span(), /*type=*/nullptr,
          "Loop has an accumulator but the body does not produce a value. The "
          "semicolon at the end of the last body statement may be unintended.",
          file_table_);
    }

    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(accumulator, for_node_type_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        iterator,
        module_.Make<ElementTypeAnnotation>(
            module_.Make<TypeVariableTypeAnnotation>(iterable_type_variable))));

    // Handle explicit type annotation.
    if (iterator_accumulator_type_annotation) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(iterator, iterator_type_annotation));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(accumulator, accumulator_type_annotation));
    }

    // Both init expr and body statement block have the same type as For itself.
    if (accumulator_type_annotation) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node, accumulator_type_annotation));
    }
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->init(), for_node_type_variable));

    XLS_ASSIGN_OR_RETURN(
        const NameRef* body_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, node->body(),
            absl::Substitute("internal_type_loop_body_at_$0_in_$1",
                             node->span().ToString(file_table_),
                             module_.name())));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->body(), body_type_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node->body(),
        module_.Make<TypeVariableTypeAnnotation>(for_node_type_variable)));

    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node->names(),
        module_.Make<TupleTypeAnnotation>(
            node->names()->span(), std::vector<TypeAnnotation*>{
                                       module_.Make<TypeVariableTypeAnnotation>(
                                           iterator_type_variable),
                                       module_.Make<TypeVariableTypeAnnotation>(
                                           for_node_type_variable)})));

    XLS_RETURN_IF_ERROR(node->iterable()->Accept(this));
    XLS_RETURN_IF_ERROR(node->init()->Accept(this));
    XLS_RETURN_IF_ERROR(iterator->Accept(this));
    XLS_RETURN_IF_ERROR(accumulator->Accept(this));
    XLS_RETURN_IF_ERROR(node->body()->Accept(this));
    return absl::OkStatus();
  }

  absl::Status HandleFor(const For* node) override {
    VLOG(5) << "HandleFor: " << node->ToString();
    return HandleForLoopBase(node);
  }

  absl::Status HandleUnrollFor(const UnrollFor* node) override {
    VLOG(5) << "HandleUnrollFor: " << node->ToString();
    return HandleForLoopBase(node);
  }

  absl::Status HandleArrayTypeAnnotation(
      const ArrayTypeAnnotation* node) override {
    VLOG(5) << "HandleArrayTypeAnnotation: " << node->ToString();
    XLS_ASSIGN_OR_RETURN(
        const NameRef* dim_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->dim()),
            GenerateInternalTypeVariableName(node->dim())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->dim(), dim_variable));
    if (node->element_type()->IsAnnotation<BuiltinTypeAnnotation>() &&
        node->element_type()
                ->AsAnnotation<BuiltinTypeAnnotation>()
                ->builtin_type() == BuiltinType::kXN) {
      // For an `xN[S][N]`-style annotation, there is one ArrayTypeAnnotation
      // wrapping another, and so we get into this function twice. The "outer"
      // one has the dimension `N` and an ArrayTypeAnnotation for the element
      // type, and does not come into this if-statement. The "inner" one has the
      // dimension `S` and the `BuiltinType` `kXN` for the element type.
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node->dim(), CreateBoolAnnotation(module_, node->dim()->span())));
    } else {
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node->dim(), CreateU32Annotation(module_, node->dim()->span())));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleChannelTypeAnnotation(
      const ChannelTypeAnnotation* node) override {
    if (node->dims()) {
      for (Expr* dim : *node->dims()) {
        XLS_ASSIGN_OR_RETURN(const NameRef* dim_variable,
                             table_.DefineInternalVariable(
                                 InferenceVariableKind::kType, dim,
                                 GenerateInternalTypeVariableName(dim)));
        XLS_RETURN_IF_ERROR(table_.SetTypeVariable(dim, dim_variable));
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
            dim, CreateU32Annotation(module_, dim->span())));
      }
    }
    return DefaultHandler(node);
  }

  absl::Status HandleTypeRefTypeAnnotation(
      const TypeRefTypeAnnotation* node) override {
    VLOG(5) << "HandleTypeRefTypeAnnotation: " << node->ToString();

    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                         GetStructOrProcRef(node, import_data_));
    if (!struct_or_proc_ref.has_value() ||
        struct_or_proc_ref->parametrics.empty()) {
      return DefaultHandler(node);
    }
    const StructDefBase* struct_def = struct_or_proc_ref->def;
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

  absl::Status HandleTypeRef(const TypeRef* node) override {
    // `TypeRef::GetChildren` does not yield the type definition it contains. We
    // want that to be processed here in case it's a `ColonRef`.
    return ToAstNode(node->type_definition())->Accept(this);
  }

  absl::Status HandleSelfTypeAnnotation(
      const SelfTypeAnnotation* node) override {
    VLOG(5) << "HandleSelfTypeAnnotation: " << node->ToString();
    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                         GetStructOrProcRef(node->struct_ref(), import_data_));
    // There are two paths for handling of `Self`.
    // - Within a parametric struct, it gets left alone here, and when the
    //   conversion step scrubs struct parametrics via
    //   GetParametricFreeStructMemberType, we finally turn it into
    //   `TheStruct<ActualParametricValues>`.
    // - Within a non-parametric struct, we just equate it to `TheStruct` now,
    //   because the conversion logic will not send it down the parametric
    //   scrubbing path.
    if (!struct_or_proc_ref->def->IsParametric()) {
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node, node->struct_ref()));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleStructInstance(const StructInstance* node) override {
    VLOG(5) << "HandleStructInstance: " << node->ToString();
    return HandleStructInstanceInternal(node, /*source=*/std::nullopt);
  }

  absl::Status HandleSplatStructInstance(
      const SplatStructInstance* node) override {
    VLOG(5) << "HandleSplatStructInstance: " << node->ToString();
    XLS_RETURN_IF_ERROR(HandleStructInstanceInternal(node, node->splatted()));
    return node->splatted()->Accept(this);
  }

  absl::Status HandleAttr(const Attr* node) override {
    // Establish a context for the unification of the struct type.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* struct_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->lhs()),
            GenerateInternalTypeVariableName(node->lhs())));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->lhs(), struct_type_variable));

    // The type of the node itself is basically
    // decltype(struct_type_variable.member).
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node,
        module_.Make<MemberTypeAnnotation>(
            module_.Make<TypeVariableTypeAnnotation>(struct_type_variable),
            node->attr())));
    return DefaultHandler(node);
  }

  absl::Status HandleString(const String* node) override {
    VLOG(5) << "HandleString: " << node->ToString();
    // Strings are always constants, and we always know their size, so we can
    // just set their annotation to u8[N] (with a constant N)
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, module_.Make<ArrayTypeAnnotation>(
                  node->span(), CreateU8Annotation(module_, node->span()),
                  module_.Make<Number>(
                      node->span(), absl::StrCat(node->text().size()),
                      NumberKind::kOther,
                      CreateU32Annotation(module_, node->span())))));
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

    if (node->type_annotation() != nullptr) {
      array_annotation = node->type_annotation();
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node, node->type_annotation()));

      // If it's a zero-length array literal with a type annotation directly
      // attached, we can at least presume it's meant to be a zero-length array
      // of the element type in the annotation. Otherwise, we know nothing about
      // it, and the early return below will just let it be unified with any LHS
      // annotation later.
      if (node->members().empty()) {
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
            node,
            module_.Make<ArrayTypeAnnotation>(
                node->span(),
                module_.Make<ElementTypeAnnotation>(node->type_annotation()),
                CreateUntypedZero(module_, node->span()))));
      }
    }

    if (node->members().empty()) {
      return absl::OkStatus();
    }

    // Create a variable for the element type, and assign it to all the
    // elements.
    std::optional<TypeAnnotation*> element_annotation;
    if (array_annotation.has_value()) {
      element_annotation =
          module_.Make<ElementTypeAnnotation>(*array_annotation);
    }
    XLS_ASSIGN_OR_RETURN(
        const NameRef* element_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Array*>(node),
            GenerateInternalTypeVariableName(node), element_annotation));
    for (Expr* member : node->members()) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(member, element_type_variable));
      if (element_annotation.has_value()) {
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(member, *element_annotation));
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

  absl::Status HandleRange(const Range* node) override {
    VLOG(5) << "HandleRange: " << node->ToString();

    // In a match pattern, a range means "match against anything in this range"
    // as opposed to "fabricate an array with everything in this range," so the
    // type of the range expr is just the type of the range endpoints, and
    // default handling of the children will get us that.
    if (node->has_pattern_semantics()) {
      return DefaultHandler(node);
    }

    // The type of a range expression is inferenced as if it is an array of
    // enumerated values. For example
    //   const FOO = s32:1..s32:3;
    //
    // the table will look like this before descent into this function:
    //   Node               Annotation          Variable
    //   -----------------------------------------------
    //   FOO                                    T0
    //   s32:1..s32:3                           T0
    //
    // and this function will make it look like this:
    //   Node              Annotation                           Variable
    //   ---------------------------------------------------------------
    //   FOO                                                    T0
    //   s32:1..s32:3      var:T1[s32:3 as s32 - s32:1 as s32]  T0
    //   s32:1                                                  T1
    //   s32:3                                                  T1
    //   s32:1 as s32      s32                                  T2
    //   s32:3 as s32      s32                                  T2
    //   subtract expr                                          T2
    // Array size is determined by concretization as it can be a constexpr.

    Expr* element_count = CreateRangeElementCount(module_, node);

    // Create and assign a type variable for generated element count expr and
    // its sub expressions.
    XLS_ASSIGN_OR_RETURN(const NameRef* element_count_type_variable,
                         table_.DefineInternalVariable(
                             InferenceVariableKind::kType, element_count,
                             GenerateInternalTypeVariableName(element_count)));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(element_count, element_count_type_variable));
    XLS_RETURN_IF_ERROR(element_count->Accept(this));

    // Create a variable for the element type, and assign it to start and end.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* endpoint_type_variable,
        table_.DefineInternalVariable(InferenceVariableKind::kType,
                                      const_cast<Range*>(node),
                                      GenerateInternalTypeVariableName(node)));
    XLS_ASSIGN_OR_RETURN(
        std::optional<const TypeAnnotation*> range_annotation,
        GetDeclarationTypeAnnotation<ArrayTypeAnnotation>(node));
    for (Expr* endpoint : {node->start(), node->end()}) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(endpoint, endpoint_type_variable));
      // If the declaration annotation is a `ParamTypeAnnotation`, don't apply
      // it here since it indirectly references this type variable and will
      // create a loop.
      if (range_annotation.has_value() &&
          !(*range_annotation)->IsAnnotation<ParamTypeAnnotation>()) {
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
            endpoint, module_.Make<ElementTypeAnnotation>(*range_annotation)));
      }
    }

    ArrayTypeAnnotation* type_annotation = module_.Make<ArrayTypeAnnotation>(
        node->span(),
        module_.Make<TypeVariableTypeAnnotation>(endpoint_type_variable),
        element_count, /*dim_is_min=*/false);
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node, type_annotation));
    return DefaultHandler(node);
  }

  absl::Status HandleIndex(const Index* node) override {
    // Whether it's a normal index op or a slice, the LHS, which is the original
    // array, always has its own unification context.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* lhs_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->lhs()),
            GenerateInternalTypeVariableName(node->lhs()) + "_index_lhs"));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->lhs(), lhs_variable));
    auto* lhs_tvta = module_.Make<TypeVariableTypeAnnotation>(lhs_variable);

    XLS_RETURN_IF_ERROR(absl::visit(
        Visitor{
            [&](Slice* slice) -> absl::Status {
              return table_.SetTypeAnnotation(
                  node, module_.Make<SliceTypeAnnotation>(node->span(),
                                                          lhs_tvta, slice));
            },
            [&](WidthSlice* width_slice) -> absl::Status {
              table_.SetAnnotationFlag(lhs_tvta,
                                       TypeInferenceFlag::kBitsLikeType);
              XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
                  node, module_.Make<SliceTypeAnnotation>(
                            node->span(), lhs_tvta, width_slice)));
              return HandleWidthSliceInternal(lhs_tvta, width_slice);
            },
            [&](Expr* expr) -> absl::Status {
              // A node like `array[i]` is basically a binary operator with
              // independent contexts on the LHS and RHS. The RHS is constrained
              // to u32, while the LHS must be some kind of array. The "some
              // kind of array" part is not capturable in the table, but readily
              // verifiable at the end of type inference, so we defer that.
              XLS_ASSIGN_OR_RETURN(
                  const NameRef* rhs_variable,
                  table_.DefineInternalVariable(
                      InferenceVariableKind::kType, const_cast<Expr*>(expr),
                      GenerateInternalTypeVariableName(expr) + "_index"));
              XLS_RETURN_IF_ERROR(table_.SetTypeVariable(expr, rhs_variable));
              const TypeAnnotation* u32 =
                  CreateU32Annotation(module_, expr->span());
              XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(expr, u32));
              table_.SetAnnotationFlag(u32, TypeInferenceFlag::kStandardType);

              // The type of the entire expr is then
              // ElementType(lhs_variable).
              XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
                  node, module_.Make<ElementTypeAnnotation>(lhs_tvta)));
              return absl::OkStatus();
            }},
        node->rhs()));

    return DefaultHandler(node);
  }

  absl::Status HandleWidthSlice(const WidthSlice* node) override {
    // We handle this out-of-band via `HandleWidthSliceInternal` while looking
    // at the `Index` node that contains the slice, because the default type of
    // the slice index depends on the type of the container being sliced.
    return absl::OkStatus();
  }

  absl::Status HandleWidthSliceInternal(TypeVariableTypeAnnotation* lhs_tvta,
                                        const WidthSlice* node) {
    // A width slice uses an unsigned start index.
    Expr* start = node->start();
    XLS_ASSIGN_OR_RETURN(
        const NameRef* start_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(start),
            GenerateInternalTypeVariableName(start) + "_width_slice_start"));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(start, start_variable));

    Expr* lhs_element_count = CreateElementCountInvocation(module_, lhs_tvta);
    XLS_ASSIGN_OR_RETURN(
        const NameRef* element_count_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, lhs_element_count,
            GenerateInternalTypeVariableName(lhs_element_count)));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(lhs_element_count, element_count_variable));
    XLS_RETURN_IF_ERROR(lhs_element_count->Accept(this));

    const TypeAnnotation* rhs_standard_type = CreateUnOrSnAnnotation(
        module_, lhs_tvta->span(), /*is_signed=*/false, lhs_element_count);
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(start, rhs_standard_type));
    TypeInferenceFlag flag = TypeInferenceFlag::kStandardType;
    flag.SetFlag(TypeInferenceFlag::kSliceContainerSize);
    table_.SetAnnotationFlag(rhs_standard_type, flag);
    return DefaultHandler(node);
  }

  absl::Status HandleSlice(const Slice* node) override {
    // A general slice uses a signed start and/or limit.

    const NameRef* bound_variable = nullptr;
    if (node->start() != nullptr || node->limit() != nullptr) {
      XLS_ASSIGN_OR_RETURN(
          bound_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Slice*>(node),
              GenerateInternalTypeVariableName(node)));
    }

    if (node->start() != nullptr) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(node->start(), bound_variable));
      const TypeAnnotation* s32 =
          CreateS32Annotation(module_, node->start()->span());
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node->start(), s32));
      table_.SetAnnotationFlag(s32, TypeInferenceFlag::kStandardType);
    }

    if (node->limit() != nullptr) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(node->limit(), bound_variable));
      const TypeAnnotation* s32 =
          CreateS32Annotation(module_, node->limit()->span());
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node->limit(), s32));
      table_.SetAnnotationFlag(s32, TypeInferenceFlag::kStandardType);
    }

    return DefaultHandler(node);
  }

  absl::Status HandleTupleIndex(const TupleIndex* node) override {
    VLOG(5) << "HandleTupleIndex: " << node->ToString();

    // Establish a context for the unification of the tuple type.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* tuple_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->lhs()),
            GenerateInternalTypeVariableName(node->lhs())));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->lhs(), tuple_type_variable));

    // The index itself must be u32.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* index_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Number*>(node->index()),
            GenerateInternalTypeVariableName<Expr>(node->index())));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->index(), index_type_variable));
    const TypeAnnotation* u32 =
        CreateU32Annotation(module_, node->index()->span());
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node->index(), u32));
    table_.SetAnnotationFlag(u32, TypeInferenceFlag::kStandardType);

    // The type of the entire expr is then ElementType(tuple_type_variable,
    // index).
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, module_.Make<ElementTypeAnnotation>(
                  module_.Make<TypeVariableTypeAnnotation>(tuple_type_variable),
                  node->index())));
    return DefaultHandler(node);
  }

  absl::Status HandleFunction(const Function* node) override {
    // Proc functions are reachable via both the `Module` and the `Proc`, as an
    // oddity of how procs are set up in the AST. We only want to handle them in
    // the context of the `Proc`, because at that point we will have processed
    // the members they may be using.
    if (node->IsInProc() && !handle_proc_functions_) {
      return absl::OkStatus();
    }

    VLOG(5) << "HandleFunction: " << node->ToString()
            << ", parametric: " << node->IsParametric();
    for (const ParametricBinding* binding : node->parametric_bindings()) {
      XLS_RETURN_IF_ERROR(binding->Accept(this));
    }

    const TypeAnnotation* return_type = GetReturnType(module_, *node);
    XLS_RETURN_IF_ERROR(return_type->Accept(this));
    for (const Param* param : node->params()) {
      XLS_RETURN_IF_ERROR(param->Accept(this));
    }

    const FunctionTypeAnnotation* function_type_annotation =
        CreateFunctionTypeAnnotation(module_, *node);

    // Create a variable for the function return type, and use it to unify the
    // formal return type and what is returned by the actual body.
    XLS_ASSIGN_OR_RETURN(
        const NameRef* return_type_variable,
        table_.DefineInternalVariable(InferenceVariableKind::kType,
                                      const_cast<Function*>(node),
                                      GenerateInternalTypeVariableName(node),
                                      function_type_annotation->return_type()));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->body(), return_type_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node->body(), function_type_annotation->return_type()));

    // Only apply a type annotation to the function itself if it's
    // non-parametric. This is to avoid leaking types like `uN[N]` into type
    // variables that are outside the function.
    if (!node->IsParametric()) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node, function_type_annotation));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node->name_def(), function_type_annotation));
    }

    // Descend into the function body.
    XLS_RETURN_IF_ERROR(node->body()->Accept(this));
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

  absl::Status HandleStatement(const Statement* node) override {
    VLOG(5) << "HandleStatement: " << node->ToString();
    // If it's just an expr, assign it a type variable.
    if (std::holds_alternative<Expr*>(node->wrapped())) {
      Expr* expr = std::get<Expr*>(node->wrapped());
      std::optional<const NameRef*> type_variable =
          table_.GetTypeVariable(expr);
      if (!type_variable.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            const NameRef* type_variable,
            table_.DefineInternalVariable(
                InferenceVariableKind::kType, const_cast<Statement*>(node),
                GenerateInternalTypeVariableName(expr)));
        XLS_RETURN_IF_ERROR(table_.SetTypeVariable(expr, type_variable));
      }
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
    if (node->trailing_semi()) {
      // A statement block implicitly produces a unit tuple if the last
      // statement ends with a semicolon.
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node, CreateUnitTupleAnnotation(module_, node->span())));
    } else if (!node->statements().empty() && variable.has_value()) {
      const Statement* last_statement =
          node->statements()[node->statements().size() - 1];
      if (std::holds_alternative<Expr*>(last_statement->wrapped())) {
        XLS_RETURN_IF_ERROR(table_.SetTypeVariable(
            std::get<Expr*>(last_statement->wrapped()), *variable));
      }
    }
    return DefaultHandler(node);
  }

  absl::Status HandleSpawn(const Spawn* node) override {
    VLOG(5) << "HandleSpawn: " << node->ToString();
    XLS_ASSIGN_OR_RETURN(
        const NameRef* config_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType,
            const_cast<Invocation*>(node->config()),
            absl::StrCat(GenerateInternalTypeVariableName(
                             static_cast<const Expr*>(node->config())),
                         "_config")));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->config(), config_type_variable));
    XLS_ASSIGN_OR_RETURN(
        const NameRef* next_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Invocation*>(node->next()),
            absl::StrCat(GenerateInternalTypeVariableName(
                             static_cast<const Expr*>(node->next())),
                         "_next")));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->next(), next_type_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, CreateUnitTupleAnnotation(module_, node->span())));
    return DefaultHandler(node);
  }

  // Returns true if the given callee can only be called within a proc.
  bool ProcOnlyFunction(const Expr* callee) {
    static const absl::flat_hash_set<std::string> kShouldBeInProc = {
        "join",
        "recv",
        "recv_if",
        "send",
        "send_if",
        "recv_non_blocking",
        "recv_if_non_blocking",
    };
    std::optional<std::string_view> builtin_name =
        GetBuiltinFnName(const_cast<Expr*>(callee));
    if (!builtin_name.has_value()) {
      return false;
    }

    return kShouldBeInProc.contains(*builtin_name);
  }

  absl::Status HandleInvocation(const Invocation* node) override {
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
    //   foo(a, b)          <unspecified>          T0
    //   a                  ParamType(T3, 0)       T1
    //   b                  ParamType(T3, 1)       T2
    //   foo                (T1, T2) -> T0         T3
    //
    // The core task here is to produce a `FunctionTypeAnnotation` for the
    // actual arguments/return type: the `(T1, T2) -> T0` annotation in the
    // example. At the time of conversion to type info, the target function will
    // be resolved, and a `FunctionTypeAnnotation` for the resolved target
    // `Function` object will be determined. Conversion will apply formal types
    // to the argument nodes and the invocation node just in time for
    // conversion.

    VLOG(5) << "HandleInvocation: " << node->ToString();

    // If we're outside a proc, we can't call proc-only builtins.
    if (!handle_proc_functions_ && ProcOnlyFunction(node->callee())) {
      return TypeInferenceErrorStatus(
          *node->GetSpan(), nullptr,
          absl::Substitute("Cannot call `$0` outside a `proc`",
                           node->callee()->ToString()),
          file_table_);
    }

    for (ExprOrType parametric : node->explicit_parametrics()) {
      if (std::holds_alternative<Expr*>(parametric)) {
        const Expr* parametric_value_expr = std::get<Expr*>(parametric);
        XLS_ASSIGN_OR_RETURN(
            const NameRef* type_variable,
            table_.DefineInternalVariable(
                InferenceVariableKind::kType,
                const_cast<Expr*>(parametric_value_expr),
                GenerateInternalTypeVariableName(parametric_value_expr)));
        XLS_RETURN_IF_ERROR(
            table_.SetTypeVariable(parametric_value_expr, type_variable));
        XLS_RETURN_IF_ERROR(parametric_value_expr->Accept(this));
      } else {
        XLS_RETURN_IF_ERROR(
            std::get<TypeAnnotation*>(parametric)->Accept(this));
      }
    }

    XLS_ASSIGN_OR_RETURN(
        const NameRef* function_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->callee()),
            absl::StrCat(GenerateInternalTypeVariableName(node->callee()),
                         "_callee")));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->callee(), function_type_variable));

    std::vector<const TypeAnnotation*> arg_types;
    arg_types.reserve(node->args().size());
    int self_arg_offset = 0;
    if (node->callee()->kind() == AstNodeKind::kAttr) {
      // An invocation like foo.bar(args), which is targeting an instance
      // function of a struct, needs the actual object type added to the
      // signature in place of the formal `Self`.
      const Attr* attr = down_cast<const Attr*>(node->callee());
      XLS_ASSIGN_OR_RETURN(
          const NameRef* obj_type_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(attr->lhs()),
              absl::StrCat(GenerateInternalTypeVariableName(attr->lhs()),
                           "_target_obj")));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(attr->lhs(), obj_type_variable));
      XLS_RETURN_IF_ERROR(attr->lhs()->Accept(this));
      arg_types.push_back(
          module_.Make<TypeVariableTypeAnnotation>(obj_type_variable));
      self_arg_offset = 1;
    }
    for (int i = 0; i < node->args().size(); i++) {
      const Expr* arg = node->args()[i];
      VLOG(5) << "HandleInvocation arg [" << i << "]= " << arg->ToString();
      // In a case like `foo.fn(arg0, arg1)`, `foo` is the implicit first actual
      // argument, hence `arg0` and `arg1` are actually at index 1 and 2 among
      // the params in the `FunctionTypeAnnotation`.
      const int arg_index_including_implicit_self = i + self_arg_offset;
      TypeAnnotation* arg_annotation = module_.Make<ParamTypeAnnotation>(
          module_.Make<TypeVariableTypeAnnotation>(function_type_variable),
          arg_index_including_implicit_self);
      XLS_ASSIGN_OR_RETURN(
          const NameRef* arg_type_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(arg),
              absl::Substitute("$0_actual_arg_$1",
                               GenerateInternalTypeVariableName(arg), i),
              arg_annotation));
      arg_types.push_back(
          module_.Make<TypeVariableTypeAnnotation>(arg_type_variable));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(arg, arg_type_variable));

      XLS_RETURN_IF_ERROR(arg->Accept(this));
    }

    const NameRef* return_type_variable = *table_.GetTypeVariable(node);
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node->callee(), module_.Make<FunctionTypeAnnotation>(
                            arg_types, module_.Make<TypeVariableTypeAnnotation>(
                                           return_type_variable))));
    return node->callee()->Accept(this);
  }

  absl::Status HandleEnumDef(const EnumDef* node) override {
    // When we come in here with an example like:
    //   enum MyEnum : u8 {
    //     A = 1;
    //     B = 2;
    //   }
    //
    // and this function will make it look like this:
    //   Node               Annotation                    Variable
    //   ---------------------------------------------------------
    //   MyEnum             TypeRefTypeAnnotation(MyEnum) T0
    //   1                  u8                            T1
    //   2                  u8                            T1
    //
    // The EnumDef itself is annotated with a type ref to itself, while its
    // member values are annotated with EnumDef's annotation and any annotations
    // they have.
    VLOG(5) << "HandleEnumDef: " << node->ToString();

    XLS_ASSIGN_OR_RETURN(
        const NameRef* enum_type_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<EnumDef*>(node),
            GenerateInternalTypeVariableName(node),
            node->type_annotation()
                ? std::make_optional(node->type_annotation())
                : std::nullopt));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node, enum_type_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, module_.Make<TypeRefTypeAnnotation>(
                  node->span(),
                  module_.Make<TypeRef>(
                      node->span(), TypeDefinition(const_cast<EnumDef*>(node))),
                  std::vector<ExprOrType>(), std::nullopt)));

    if (node->values().empty()) {
      if (!node->type_annotation()) {
        return TypeInferenceErrorStatus(
            *node->GetSpan(), nullptr,
            absl::Substitute("Enum `$0` has no type annotation and no value.",
                             node->ToString()),
            file_table_);
      }
    } else {
      // Enum values share a separate type variable for the underlying numeric
      // type.
      XLS_ASSIGN_OR_RETURN(
          const NameRef* value_type_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType,
              const_cast<Expr*>(node->values()[0].value),
              GenerateInternalTypeVariableName(node->values()[0].value)));
      absl::flat_hash_set<std::string> names;
      for (const EnumMember& value : node->values()) {
        // Check for duplicated value names.
        if (!names.emplace(value.name_def->identifier()).second) {
          return RedefinedNameErrorStatus(value.name_def->span(), node,
                                          value.name_def->identifier(),
                                          file_table_);
        }
        XLS_RETURN_IF_ERROR(
            table_.SetTypeVariable(value.value, value_type_variable));
      }
      if (node->type_annotation()) {
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node->values()[0].value,
                                                     node->type_annotation()));
      }
    }

    return DefaultHandler(node);
  }

  absl::Status HandleFormatMacro(const FormatMacro* node) override {
    // The verbosity, if specified, has its own unification context.
    if (node->verbosity().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          const NameRef* verbosity_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType,
              const_cast<Expr*>(*node->verbosity()),
              GenerateInternalTypeVariableName(*node->verbosity())));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(*node->verbosity(), verbosity_variable));
    }

    // The number of actual args is determined by the format string.
    const int64_t arg_count = OperandsExpectedByFormat(node->format());
    if (arg_count != node->args().size()) {
      return ArgCountMismatchErrorStatus(
          node->span(),
          absl::StrFormat("%s macro expects %d argument(s) from format but has "
                          "%d argument(s)",
                          node->macro(), arg_count, node->args().size()),
          file_table_);
    }

    // Each arg has an independent unification context.
    for (const Expr* arg : node->args()) {
      XLS_ASSIGN_OR_RETURN(
          const NameRef* arg_variable,
          table_.DefineInternalVariable(InferenceVariableKind::kType,
                                        const_cast<Expr*>(arg),
                                        GenerateInternalTypeVariableName(arg)));
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(arg, arg_variable));
    }

    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, CreateTokenTypeAnnotation(module_, node->span())));
    return DefaultHandler(node);
  }

  absl::Status HandleZeroOrOneMacro(const AstNode* node, ExprOrType type) {
    if (std::holds_alternative<Expr*>(type)) {
      Expr* expr = std::get<Expr*>(type);
      if (expr->kind() == AstNodeKind::kColonRef) {
        XLS_ASSIGN_OR_RETURN(
            const NameRef* expr_variable,
            table_.DefineInternalVariable(
                InferenceVariableKind::kType, const_cast<Expr*>(expr),
                absl::StrCat(GenerateInternalTypeVariableName(expr),
                             "_colon_ref_type_param")));
        XLS_RETURN_IF_ERROR(table_.SetTypeVariable(expr, expr_variable));
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
            node, module_.Make<TypeVariableTypeAnnotation>(expr_variable)));
        XLS_RETURN_IF_ERROR(expr->Accept(this));

        // The target must be a type or it's an invalid parametric to the macro.
        if (IsColonRefWithTypeTarget(table_, expr)) {
          return absl::OkStatus();
        }
      }

      // An expr that isn't a type is not allowed.
      return TypeInferenceErrorStatus(
          *node->GetSpan(), nullptr,
          absl::Substitute("Expected a type argument in `$0`; saw `$1`.",
                           node->ToString(), std::get<Expr*>(type)->ToString()),
          file_table_);
    }

    // If the "type" is not an expr, then it is the type annotation.
    XLS_RETURN_IF_ERROR(
        table_.SetTypeAnnotation(node, std::get<TypeAnnotation*>(type)));
    return DefaultHandler(node);
  }

  absl::Status HandleZeroMacro(const ZeroMacro* node) override {
    VLOG(5) << "HandleZeroMacro: " << node->ToString();
    return HandleZeroOrOneMacro(node, node->type());
  }

  absl::Status HandleAllOnesMacro(const AllOnesMacro* node) override {
    VLOG(5) << "HandleAllOnesMacro: " << node->ToString();
    return HandleZeroOrOneMacro(node, node->type());
  }

  absl::Status HandleConstAssert(const ConstAssert* node) override {
    VLOG(5) << "HandleConstAssert: " << node->ToString();
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node->arg(), CreateBoolAnnotation(module_, node->span())));
    XLS_ASSIGN_OR_RETURN(
        const NameRef* operand_variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Expr*>(node->arg()),
            GenerateInternalTypeVariableName(node->arg())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->arg(), operand_variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, CreateUnitTupleAnnotation(module_, node->span())));
    return DefaultHandler(node);
  }

  absl::Status HandleLet(const Let* node) override {
    VLOG(5) << "HandleLet: " << node->ToString();
    XLS_ASSIGN_OR_RETURN(
        const NameRef* variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<Let*>(node),
            GenerateInternalTypeVariableName(node),
            node->type_annotation() == nullptr
                ? std::nullopt
                : std::make_optional(node->type_annotation())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->rhs(), variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node, variable));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeVariable(node->name_def_tree(), variable));
    if (node->type_annotation() != nullptr) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node, node->type_annotation()));
    }
    if (node->name_def_tree()->is_leaf()) {
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(
          ToAstNode(node->name_def_tree()->leaf()), variable));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleTypeAlias(const TypeAlias* node) override {
    VLOG(5) << "HandleTypeAlias: " << node->ToString();
    XLS_RETURN_IF_ERROR(
        table_.SetTypeAnnotation(node, &node->type_annotation()));
    XLS_RETURN_IF_ERROR(
        table_.SetTypeAnnotation(&node->name_def(), &node->type_annotation()));
    return DefaultHandler(node);
  }

  absl::Status HandleQuickCheck(const QuickCheck* node) override {
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node, CreateBoolAnnotation(module_, node->span())));
    XLS_ASSIGN_OR_RETURN(
        const NameRef* quickcheck_variable,
        table_.DefineInternalVariable(InferenceVariableKind::kType,
                                      const_cast<QuickCheck*>(node),
                                      GenerateInternalTypeVariableName(node)));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node, quickcheck_variable));

    return DefaultHandler(node);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (AstNode* child : node->GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

 private:
  // Determines the target of the given `ColonRef` that is already known to be
  // referencing a member with the name `attribute` of the given `struct_def`.
  // Associates the target node with the `ColonRef` in the `InferenceTable` for
  // later reference, and returns it.
  absl::StatusOr<std::optional<const AstNode*>>
  HandleStructAttributeReferenceInternal(
      const ColonRef* node, const StructDefBase& struct_def,
      const std::vector<ExprOrType>& parametrics, std::string_view attribute) {
    if (!struct_def.impl().has_value()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::Substitute("Struct '$0' has no impl defining '$1'",
                           struct_def.identifier(), attribute),
          file_table_);
    }
    std::optional<ImplMember> member =
        (*struct_def.impl())->GetMember(attribute);
    if (!member.has_value()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::Substitute(
              "Name '$0' is not defined by the impl for struct '$1'.",
              attribute, struct_def.identifier()),
          file_table_);
    }
    if (struct_def.IsParametric()) {
      // The type-checking of a `TypeRefTypeAnnotation` containing any
      // parametrics will prove that there aren't too many parametrics given.
      // However, for general validation, a type reference does not need all
      // bindings satisfied. In a case like `S { a, b }`, we can infer some or
      // all `S` binding values from `a` and `b` at conversion time. However, in
      // `S::SOME_CONSTANT` or `S::static_fn(a)`, we will not infer the `S`
      // bindings; only the bindings for `static_fn` itself, if it has any.
      // Hence all the `S` bindings must be satisfied.
      XLS_RETURN_IF_ERROR(VerifyAllParametricsSatisfied(
          struct_def.parametric_bindings(), parametrics,
          struct_def.identifier(), node->span(), file_table_));
    }
    table_.SetColonRefTarget(node, ToAstNode(*member));
    if (std::holds_alternative<ConstantDef*>(*member) ||
        std::holds_alternative<Function*>(*member)) {
      return ToAstNode(*member);
    }
    return std::nullopt;
  }

  // Helper that creates an internal type variable for a `ConstantDef`, `Param`,
  // or similar type of node that contains a `NameDef` and optional
  // `TypeAnnotation`.
  template <typename T>
  absl::StatusOr<const NameRef*> DefineTypeVariableForVariableOrConstant(
      const T* node) {
    XLS_ASSIGN_OR_RETURN(
        const NameRef* variable,
        table_.DefineInternalVariable(
            InferenceVariableKind::kType, const_cast<T*>(node),
            GenerateInternalTypeVariableName(node),
            node->type_annotation() == nullptr
                ? std::nullopt
                : std::make_optional(node->type_annotation())));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node, variable));
    XLS_RETURN_IF_ERROR(table_.SetTypeVariable(node->name_def(), variable));
    if (node->type_annotation() != nullptr) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(node->name_def(), node->type_annotation()));
    }
    return variable;
  }

  // Generates a name for an internal inference variable that will be used as
  // the type for the given node. The name is only relevant for traceability.
  template <typename T>
  std::string GenerateInternalTypeVariableName(const T* node) {
    return absl::Substitute("internal_type_$0_at_$1_in_$2", node->identifier(),
                            node->span().ToString(file_table_), module_.name());
  }
  // Specialization for `Expr` nodes, which do not have an identifier.
  template <>
  std::string GenerateInternalTypeVariableName(const Expr* node) {
    return absl::Substitute("internal_type_expr_at_$0_in_$1",
                            node->span().ToString(file_table_), module_.name());
  }
  // Specialization for `Quickcheck` nodes, which do not have an identifier.
  template <>
  std::string GenerateInternalTypeVariableName(const QuickCheck* node) {
    return absl::Substitute("internal_type_quickcheck_at_$0_in_$1",
                            node->span().ToString(file_table_), module_.name());
  }

  // Specialization for `Let` nodes, which do not have an identifier.
  template <>
  std::string GenerateInternalTypeVariableName(const Let* node) {
    return absl::Substitute("internal_type_let_at_$0_in_$1",
                            node->span().ToString(file_table_), module_.name());
  }
  // Specialization for `Array` nodes.
  template <>
  std::string GenerateInternalTypeVariableName(const Array* node) {
    return absl::Substitute("internal_type_array_element_at_$0_in_$1",
                            node->span().ToString(file_table_), module_.name());
  }
  // Specialization for `Range` nodes.
  template <>
  std::string GenerateInternalTypeVariableName(const Range* node) {
    return absl::StrCat("internal_type_range_element_at_",
                        node->span().ToString(file_table_));
  }
  // Specialization for `Slice` nodes.
  template <>
  std::string GenerateInternalTypeVariableName(const Slice* node) {
    return absl::StrCat("internal_type_slice_bound_at_",
                        node->GetSpan()->ToString(file_table_));
  }
  // Variant for an actual struct member expr.
  std::string GenerateInternalTypeVariableName(
      const StructMemberNode* formal_member, const Expr* actual_member) {
    return absl::Substitute(
        "internal_type_actual_member_$0_at_$1_in_$2", formal_member->name(),
        actual_member->span().ToString(file_table_), module_.name());
  }
  // Variant for operands of a binary operator.
  std::string GenerateInternalTypeVariableName(const Binop* binop) {
    return absl::Substitute("internal_type_operand_$0_at_$1_in_$2",
                            BinopKindToString(binop->binop_kind()),
                            binop->span().ToString(file_table_),
                            module_.name());
  }
  // Variant for `NameDefTree`.
  std::string GenerateInternalTypeVariableName(const NameDefTree* node) {
    return absl::Substitute("internal_type_ndt_at_$0_in_$1",
                            node->span().ToString(file_table_), module_.name());
  }

  absl::Status SetCrossModuleTypeAnnotation(const AstNode* node,
                                            const AstNode* external_reference) {
    std::optional<const NameRef*> type_var =
        table_.GetTypeVariable(external_reference);
    // In parametric cases, we may not have a type variable for the member.
    if (type_var.has_value()) {
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          node, module_.Make<TypeVariableTypeAnnotation>(*type_var)));
    } else {
      std::optional<const TypeAnnotation*> annotation =
          table_.GetTypeAnnotation(external_reference);
      if (annotation.has_value()) {
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(node, *annotation));
      }
    }
    return absl::OkStatus();
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
    return PropagateDefToRef(def, ref);
  }

  absl::Status PropagateDefToRef(const AstNode* def, const AstNode* ref) {
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

  // Ensures that a `StructInstance` nodes provides exprs for all the names in a
  // struct definition, with no extraneous or duplicate names.
  absl::Status ValidateStructInstanceMemberNames(
      const StructInstanceBase& instance, const StructDefBase& def) {
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
    if (instance.requires_all_members() &&
        actual_names.size() != formal_names.size()) {
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
    XLS_ASSIGN_OR_RETURN(std::optional<const TypeAnnotation*> annotation,
                         table_.GetDeclarationTypeAnnotation(*type_variable));
    // Constraining the annotation type here improves error messages in
    // situations where there is a type mismatch for an entire array/tuple.
    // We allow indirect member/element annotations through at this point,
    // because we can't yet prove whether they amount to something expected.
    if (annotation.has_value()) {
      if ((*annotation)->IsAnnotation<T>() ||
          (*annotation)->IsAnnotation<MemberTypeAnnotation>() ||
          (*annotation)->IsAnnotation<ElementTypeAnnotation>() ||
          (*annotation)->IsAnnotation<ParamTypeAnnotation>()) {
        return annotation;
      }
      VLOG(5) << "Declaration type is unsupported kind: "
              << (*annotation)->ToString() << " for " << node->ToString();
    }
    return std::nullopt;
  }

  // Helper function that does all common logic for either a `StructInstance` or
  // `SplatStructInstance` node.
  absl::Status HandleStructInstanceInternal(const StructInstanceBase* node,
                                            std::optional<Expr*> source) {
    // As far as we're concerned here, type-checking a struct instance is like
    // type-checking a function invocation (see `HandleInvocation`), but with
    // named arguments instead of parallel ordering. The naming of arguments
    // creates additional pitfalls, like erroneously naming two different
    // arguments the same thing.
    XLS_RETURN_IF_ERROR(node->struct_ref()->Accept(this));
    XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                         GetStructOrProcRef(node->struct_ref(), import_data_));
    if (!struct_or_proc_ref.has_value()) {
      return TypeInferenceErrorStatusForAnnotation(
          node->span(), node->struct_ref(),
          absl::Substitute(
              "Attempted to instantiate non-struct type `$0` as a struct.",
              node->struct_ref()->ToString()),
          file_table_);
    }
    if (struct_or_proc_ref->def->kind() == AstNodeKind::kProcDef) {
      return TypeInferenceErrorStatusForAnnotation(
          node->span(), node->struct_ref(),
          "Impl-style procs are a work in progress and cannot yet be "
          "instantiated.",
          file_table_);
    }

    const StructDef* struct_def =
        down_cast<const StructDef*>(struct_or_proc_ref->def);
    const NameRef* type_variable = *table_.GetTypeVariable(node);
    if (source.has_value()) {
      XLS_RETURN_IF_ERROR(table_.SetTypeVariable(*source, type_variable));
    }
    XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
        node,
        CreateStructAnnotation(module_, const_cast<StructDef*>(struct_def),
                               struct_or_proc_ref->parametrics, node)));
    XLS_RETURN_IF_ERROR(ValidateStructInstanceMemberNames(*node, *struct_def));

    absl::flat_hash_map<std::string, const StructMemberNode*> formal_member_map;
    for (const StructMemberNode* formal_member : struct_def->members()) {
      formal_member_map.emplace(formal_member->name(), formal_member);
    }
    const TypeAnnotation* struct_variable_type =
        module_.Make<TypeVariableTypeAnnotation>(type_variable);
    for (const auto& [name, actual_member] : node->members()) {
      const StructMemberNode* formal_member = formal_member_map.at(name);
      TypeAnnotation* member_type = module_.Make<MemberTypeAnnotation>(
          struct_variable_type, formal_member->name());
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(actual_member, member_type));
      XLS_ASSIGN_OR_RETURN(
          const NameRef* member_type_variable,
          table_.DefineInternalVariable(
              InferenceVariableKind::kType, const_cast<Expr*>(actual_member),
              GenerateInternalTypeVariableName(formal_member, actual_member),
              member_type));
      XLS_RETURN_IF_ERROR(
          table_.SetTypeVariable(actual_member, member_type_variable));
      XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
          actual_member, module_.Make<MemberTypeAnnotation>(
                             struct_variable_type, formal_member->name())));
      XLS_RETURN_IF_ERROR(actual_member->Accept(this));
    }
    return absl::OkStatus();
  }

  Module& module_;
  InferenceTable& table_;
  const FileTable& file_table_;
  ImportData& import_data_;
  TypecheckModuleFn typecheck_imported_module_;
  bool handle_proc_functions_ = false;
};

}  // namespace

std::unique_ptr<PopulateTableVisitor> CreatePopulateTableVisitor(
    Module* module, InferenceTable* table, ImportData* import_data,
    TypecheckModuleFn typecheck_imported_module) {
  return std::make_unique<PopulateInferenceTableVisitor>(
      *module, *table, *import_data, std::move(typecheck_imported_module));
}

}  // namespace xls::dslx
