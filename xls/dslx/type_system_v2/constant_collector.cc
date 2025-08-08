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

#include "xls/dslx/type_system_v2/constant_collector.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_zero_value.h"
#include "xls/dslx/type_system_v2/evaluator.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/parametric_struct_instantiator.h"
#include "xls/dslx/type_system_v2/populate_table_visitor.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

// A non-recursive visitor that structures the `ConstantCollector` logic by node
// type.
class Visitor : public AstNodeVisitorWithDefault {
 public:
  Visitor(InferenceTable& table, Module& module, ImportData& import_data,
          WarningCollector& warning_collector, const FileTable& file_table,
          InferenceTableConverter& converter, Evaluator& evaluator,
          ParametricStructInstantiator& parametric_struct_instantiator,
          TypeSystemTracer& tracer,
          std::optional<const ParametricContext*> parametric_context,
          const Type& type, TypeInfo* ti, TypeSystemTrace trace)
      : table_(table),
        module_(module),
        import_data_(import_data),
        warning_collector_(warning_collector),
        file_table_(file_table),
        converter_(converter),
        evaluator_(evaluator),
        parametric_struct_instantiator_(parametric_struct_instantiator),
        tracer_(tracer),
        parametric_context_(parametric_context),
        type_(type),
        ti_(ti),
        trace_(std::move(trace)) {}

  absl::Status HandleConstantDef(const ConstantDef* constant_def) override {
    VLOG(6) << "Checking constant def value: " << constant_def->ToString()
            << " with type: " << type_.ToString();
    absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
        &import_data_, ti_, &warning_collector_,
        table_.GetParametricEnv(parametric_context_), constant_def->value());
    if (value.ok()) {
      VLOG(6) << "Constant def: " << constant_def->ToString()
              << " has value: " << value->ToString();
      trace_.SetResult(*value);
      ti_->NoteConstExpr(constant_def, *value);
      ti_->NoteConstExpr(constant_def->value(), *value);
      ti_->NoteConstExpr(constant_def->name_def(), *value);
    }
    return absl::OkStatus();
  }

  absl::Status HandleZeroMacro(const ZeroMacro* zero_macro) override {
    VLOG(6) << "Checking zero_macro value: " << zero_macro->ToString()
            << " with type: " << type_.ToString();

    XLS_ASSIGN_OR_RETURN(InterpValue value, MakeZeroValue(type_, import_data_,
                                                          zero_macro->span()));
    trace_.SetResult(value);
    ti_->NoteConstExpr(zero_macro, value);
    return absl::OkStatus();
  }

  absl::Status HandleAllOnesMacro(const AllOnesMacro* all_ones_macro) override {
    VLOG(6) << "Checking all_ones_macro value: " << all_ones_macro->ToString()
            << " with type: " << type_.ToString();

    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        MakeAllOnesValue(type_, import_data_, all_ones_macro->span()));
    trace_.SetResult(value);
    ti_->NoteConstExpr(all_ones_macro, value);
    return absl::OkStatus();
  }

  absl::Status HandleNameRef(const NameRef* name_ref) override {
    if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
      const NameDef* name_def = std::get<const NameDef*>(name_ref->name_def());
      if (ti_->IsKnownConstExpr(name_def)) {
        InterpValue value = *ti_->GetConstExprOption(name_def);
        trace_.SetResult(value);
        ti_->NoteConstExpr(name_ref, value);
      }
    }
    return absl::OkStatus();
  }

  absl::Status HandleColonRef(const ColonRef* colon_ref) override {
    // Imported Enums have been handled by their own module. If they can be
    // constexpr evaluated, record value.
    if (IsImport(colon_ref) && type_.IsEnum()) {
      absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
          &import_data_, ti_, &warning_collector_,
          table_.GetParametricEnv(parametric_context_), colon_ref);
      if (value.ok()) {
        trace_.SetResult(*value);
        ti_->NoteConstExpr(colon_ref, *value);
      }
      return absl::OkStatus();
    }

    // Handle enum here, at this point its type is concretized and all its
    // values are evaluated as constexpr.
    if (type_.IsEnum()) {
      const auto& enum_type = type_.AsEnum();
      const EnumDef& enum_def = enum_type.nominal_type();
      absl::StatusOr<Expr*> enum_value = enum_def.GetValue(colon_ref->attr());
      if (!enum_value.ok()) {
        return UndefinedNameErrorStatus(colon_ref->span(), colon_ref,
                                        colon_ref->attr(), file_table_);
      }
      XLS_ASSIGN_OR_RETURN(
          TypeInfo * eval_ti,
          converter_.GetTypeInfo((*enum_value)->owner(), parametric_context_));
      XLS_ASSIGN_OR_RETURN(InterpValue value,
                           eval_ti->GetConstExpr(*enum_value));
      trace_.SetResult(value);
      ti_->NoteConstExpr(colon_ref, value);
      return absl::OkStatus();
    }

    const std::optional<const AstNode*> target =
        table_.GetColonRefTarget(colon_ref);

    if (!target.has_value()) {
      std::optional<BitsLikeProperties> bits_like = GetBitsLike(type_);
      if (bits_like.has_value()) {
        VLOG(6) << "ColonRef is a universal constant referenced with an "
                   "indirect annotation: "
                << colon_ref->ToString();
        XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like->is_signed.GetAsBool());
        XLS_ASSIGN_OR_RETURN(uint32_t bit_count, bits_like->size.GetAsInt64());
        XLS_ASSIGN_OR_RETURN(
            InterpValueWithTypeAnnotation member,
            GetBuiltinMember(module_, is_signed, bit_count, colon_ref->attr(),
                             colon_ref->span(), type_.ToString(), file_table_));
        trace_.SetResult(member.value);
        ti_->NoteConstExpr(colon_ref, std::move(member.value));
      } else {
        VLOG(6) << "ColonRef has no constexpr value: " << colon_ref->ToString();
      }
      return absl::OkStatus();
    }

    // In a case like `S<parametrics>::CONSTANT`, what we do here is
    // constexpr-evaluate `CONSTANT` against the parametric context `TypeInfo`
    // for `S<parametrics>`. This will be `evaluation_ti`. Then we map the
    // result of that to the `ColonRef` node, in the `TypeInfo` where the
    // `ColonRef` resides (which is `ti`). In a non-parametric case like
    // `S::CONSTANT`, there is only one `TypeInfo` involved, so this logic
    // that figures out `evaluation_ti` is a no-op.
    XLS_ASSIGN_OR_RETURN(TypeInfo * evaluation_ti,
                         import_data_.GetRootTypeInfoForNode(*target));
    VLOG(6) << "Checking ColonRef constexpr value for: "
            << colon_ref->ToString()
            << " with target: " << (*target)->ToString();
    if ((*target)->kind() == AstNodeKind::kConstantDef) {
      XLS_ASSIGN_OR_RETURN(
          std::optional<StructOrProcRef> struct_or_proc,
          GetStructOrProcRefForSubject(colon_ref, import_data_));
      if (struct_or_proc.has_value() && struct_or_proc->def->IsParametric()) {
        XLS_ASSIGN_OR_RETURN(
            const ParametricContext* struct_context,
            parametric_struct_instantiator_.GetOrCreateParametricStructContext(
                parametric_context_, *struct_or_proc, colon_ref));
        evaluation_ti = struct_context->type_info();
      }

      // Evaluate the value, and note it if successful.
      absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
          &import_data_, evaluation_ti, &warning_collector_,
          table_.GetParametricEnv(parametric_context_),
          down_cast<const ConstantDef*>(*target)->value());
      if (value.ok()) {
        VLOG(6) << "Noting constexpr for ColonRef: " << colon_ref->ToString()
                << ", value: " << value->ToString();
        trace_.SetResult(*value);
        ti_->NoteConstExpr(colon_ref, *value);
      }
    }
    return absl::OkStatus();
  }

  absl::Status HandleNumber(const Number* number) override {
    XLS_ASSIGN_OR_RETURN(InterpValue value, EvaluateNumber(*number, type_));
    trace_.SetResult(value);
    ti_->NoteConstExpr(number, value);
    return absl::OkStatus();
  }

  absl::Status HandleLet(const Let* let) override {
    absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
        &import_data_, ti_, &warning_collector_,
        table_.GetParametricEnv(parametric_context_), let->rhs());
    if (let->is_const()) {
      if (!value.ok()) {
        return value.status();
      }
      // Reminder: we don't allow name destructuring in constant defs, so this
      // is expected to never fail.
      XLS_RET_CHECK_EQ(let->name_def_tree()->GetNameDefs().size(), 1);
      NameDef* name_def = let->name_def_tree()->GetNameDefs()[0];
      WarnOnInappropriateConstantName(name_def->identifier(), let->span(),
                                      *let->owner(), &warning_collector_);
    } else if (!value.ok()) {
      return absl::OkStatus();
    }
    trace_.SetResult(*value);
    ti_->NoteConstExpr(let, *value);
    ti_->NoteConstExpr(let->rhs(), *value);
    const auto note_members =
        [&](AstNode* name_def, TypeOrAnnotation _,
            std::optional<InterpValue> const_expr) -> absl::Status {
      if (const_expr.has_value()) {
        ti_->NoteConstExpr(name_def, *const_expr);
      }
      return absl::OkStatus();
    };
    XLS_RETURN_IF_ERROR(MatchTupleNodeToType(note_members, let->name_def_tree(),
                                             &type_, file_table_, *value));
    return absl::OkStatus();
  }

  absl::Status HandleIndex(const Index* index) override {
    // A `Slice` actually has its bounds stored in `TypeInfo` out-of-band from
    // the real type info, mirroring the `StartAndWidthExprs` that we store in
    // the `InferenceTable`.
    if (!std::holds_alternative<Slice*>(index->rhs())) {
      return absl::OkStatus();
    }
    std::optional<StartAndWidthExprs> start_and_width_exprs =
        table_.GetSliceStartAndWidthExprs(ToAstNode(index->rhs()));
    XLS_RET_CHECK(start_and_width_exprs.has_value());
    StartAndWidth start_and_width;
    XLS_ASSIGN_OR_RETURN(
        start_and_width.start,
        evaluator_.EvaluateU32OrExpr(parametric_context_,
                                     start_and_width_exprs->start));
    XLS_ASSIGN_OR_RETURN(
        start_and_width.width,
        evaluator_.EvaluateU32OrExpr(parametric_context_,
                                     start_and_width_exprs->width));
    ti_->AddSliceStartAndWidth(std::get<Slice*>(index->rhs()),
                               table_.GetParametricEnv(parametric_context_),
                               start_and_width);
    return absl::OkStatus();
  }

  absl::Status HandleConstAssert(const ConstAssert* node) override {
    absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
        &import_data_, ti_, &warning_collector_,
        table_.GetParametricEnv(parametric_context_), node->arg());
    if (!value.ok()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::Substitute("const_assert! expression is not constexpr: `$0`",
                           node->arg()->ToString()),
          file_table_);
    }
    VLOG(6) << "Evaluated const assert: " << node->arg()->ToString()
            << " to: " << value->ToString();
    if (value->IsFalse()) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          absl::Substitute("const_assert! failure: `$0`",
                           node->arg()->ToString()),
          file_table_);
    }
    return absl::OkStatus();
  }

  // Creates a Let node shadowing an unroll_for! input (iterator or
  // accumulator), which assigns that to the computed `iteration_value` for one
  // iteration of the loop. Adds pairs in `old_to_new_name_defs` mapping all
  // original name defs in `input` to their shadowing ones in the returned `Let`
  // subtree. In a basic case like `unroll_for!((i, acc)...)`, the `input` here
  // is just `i` or `acc`, but each of those is allowed to be a further
  // destructured tuple.
  absl::StatusOr<Let*> ShadowUnrollForInput(
      const NameDefTree* input, Expr* iteration_value,
      absl::flat_hash_map<const NameDef*, NameDef*>& old_to_new_name_defs) {
    absl::flat_hash_map<const AstNode*, AstNode*> pairs;
    XLS_ASSIGN_OR_RETURN(
        pairs, CloneAstAndGetAllPairs(input, /*in_place=*/false,
                                      &PreserveTypeDefinitionsReplacer));
    NameDefTree* iteration_ndt = down_cast<NameDefTree*>(pairs.at(input));
    for (const auto& [old_node, new_node] : pairs) {
      if (old_node->kind() == AstNodeKind::kNameDef) {
        old_to_new_name_defs.emplace(down_cast<const NameDef*>(old_node),
                                     down_cast<NameDef*>(new_node));
      }
    }
    return module_.Make<Let>(input->span(), iteration_ndt, /*type=*/nullptr,
                             iteration_value,
                             /*is_const=*/false);
  }

  absl::Status HandleUnrollFor(const UnrollFor* unroll_for) override {
    // Unroll for is expanded to a sequence of statements as following.
    // ```
    // let X = unroll_for! (i, a) in iterable {
    //   body_statements
    //   last_body_expr
    // } (init);
    // ```
    // becomes
    // ```
    // let a = init;
    // let i = iterable[0];
    // body_statements
    // let a = last_body_expr;
    // let i = iterable[1];
    // body_statements
    // let a = last_body_expr;
    // ... // repeat for each element in iterable
    // let i = iterable[iterable.size() - 1];
    // body_statements
    // let X = last_body_expr;
    // ```
    TypeSystemTrace trace = tracer_.TraceUnroll(unroll_for);
    std::vector<Statement*> unrolled_statements;

    CHECK_EQ(unroll_for->names()->nodes().size(), 2);
    NameDefTree* iterator_name = unroll_for->names()->nodes()[0];
    NameDefTree* accumulator_name = unroll_for->names()->nodes()[1];
    TypeAnnotation* iterator_type = nullptr;
    TypeAnnotation* accumulator_type = nullptr;
    std::optional<const TypeAnnotation*> name_type_annotation =
        table_.GetTypeAnnotation(unroll_for->names());
    if (name_type_annotation.has_value()) {
      // Note that common type checking for ForLoopBase should verify this
      // before it gets here.
      XLS_RET_CHECK(
          (*name_type_annotation)->IsAnnotation<TupleTypeAnnotation>());

      const auto* types =
          (*name_type_annotation)->AsAnnotation<TupleTypeAnnotation>();
      CHECK_EQ(types->size(), 2);
      iterator_type = types->members()[0];
      accumulator_type = types->members()[1];
    }

    // The only thing that needs to be evaluated to a constant here is the size
    // of the iterable, while the elements need not to be constants. However if
    // the entire iterable is a constant array, we can use their evaluated
    // values directly.
    absl::StatusOr<InterpValue> const_iterable =
        ConstexprEvaluator::EvaluateToValue(
            &import_data_, ti_, &warning_collector_,
            table_.GetParametricEnv(parametric_context_),
            unroll_for->iterable());
    const std::vector<InterpValue>* iterable_values = nullptr;
    uint64_t size = 0;
    if (const_iterable.ok()) {
      XLS_ASSIGN_OR_RETURN(iterable_values, const_iterable->GetValues());
      size = iterable_values->size();
    } else {
      absl::StatusOr<uint64_t> size_from_type(TypeMissingErrorStatus(
          *unroll_for->iterable(), nullptr, file_table_));
      std::optional<Type*> iterable_type = ti_->GetItem(unroll_for->iterable());
      if (iterable_type.has_value() && (*iterable_type)->IsArray() &&
          std::holds_alternative<InterpValue>(
              (*iterable_type)->AsArray().size().value())) {
        InterpValue dim =
            std::get<InterpValue>((*iterable_type)->AsArray().size().value());
        size_from_type = dim.GetBitValueUnsigned();
      }
      XLS_ASSIGN_OR_RETURN(size, size_from_type);
    }

    bool has_result_value = !accumulator_name->IsWildcardLeaf() &&
                            !unroll_for->body()->trailing_semi();
    Expr* accumulator_value = unroll_for->init();
    for (uint64_t i = 0; i < size; i++) {
      absl::flat_hash_map<const NameDef*, NameDef*> iteration_name_def_mapping;
      if (has_result_value) {
        XLS_ASSIGN_OR_RETURN(
            Let * accumulator,
            ShadowUnrollForInput(accumulator_name, accumulator_value,
                                 iteration_name_def_mapping));
        XLS_RETURN_IF_ERROR(
            table_.SetTypeAnnotation(accumulator, accumulator_type));
        unrolled_statements.push_back(module_.Make<Statement>(accumulator));
      }
      if (!iterator_name->IsWildcardLeaf()) {
        Let* iterator = nullptr;
        if (iterable_values && (*iterable_values)[i].FitsInUint64()) {
          Number* value =
              module_.Make<Number>(unroll_for->iterable()->span(),
                                   (*iterable_values)[i].ToString(true),
                                   NumberKind::kOther, nullptr);
          XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(value, iterator_type));

          XLS_RETURN_IF_ERROR(converter_.ConvertSubtree(value, std::nullopt,
                                                        parametric_context_));
          XLS_ASSIGN_OR_RETURN(Type * value_type, ti_->GetItemOrError(value));
          ti_->SetItem(value, MetaType(value_type->CloneToUnique()));

          XLS_ASSIGN_OR_RETURN(
              iterator, ShadowUnrollForInput(iterator_name, value,
                                             iteration_name_def_mapping));
        } else {
          Expr* index = module_.Make<Number>(
              unroll_for->iterable()->span(), absl::StrCat(i),
              NumberKind::kOther,
              CreateU32Annotation(module_, unroll_for->span()), false);
          XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(
              index, CreateU32Annotation(module_, unroll_for->span())));
          XLS_RETURN_IF_ERROR(converter_.ConvertSubtree(index, std::nullopt,
                                                        parametric_context_));
          XLS_ASSIGN_OR_RETURN(Type * index_type, ti_->GetItemOrError(index));
          ti_->SetItem(index, MetaType(index_type->CloneToUnique()));
          Expr* element =
              module_.Make<Index>(unroll_for->iterable()->span(),
                                  unroll_for->iterable(), index, false);
          XLS_ASSIGN_OR_RETURN(
              iterator, ShadowUnrollForInput(iterator_name, element,
                                             iteration_name_def_mapping));
        }
        XLS_RETURN_IF_ERROR(table_.SetTypeAnnotation(iterator, iterator_type));
        unrolled_statements.push_back(module_.Make<Statement>(iterator));
      }

      // Clone the body and replace name refs to the iterator/accumulator with
      // refs to the most current shadowing definition. We intentionally do not
      // clone the table data, because it may not be valid for the cloned body.
      // That is because references to the iterator and accumulator within type
      // annotations in the table are not edited with clone replacement.
      // Instead, we will populate the table data for the whole unrolled loop
      // from scratch, further below.
      XLS_ASSIGN_OR_RETURN(
          AstNode * copy_body,
          CloneAst(unroll_for->body(),
                   ChainCloneReplacers(
                       &PreserveTypeDefinitionsReplacer,
                       NameRefReplacer(&iteration_name_def_mapping))));
      StatementBlock* copy_body_statementblock =
          down_cast<StatementBlock*>(copy_body);
      absl::Span<Statement* const> statements =
          copy_body_statementblock->statements();

      if (has_result_value) {
        // If accumulator is not wildcard, we expect the last body statement to
        // be an expr updating the accumulator, which will be handled in the
        // next iteration.
        CHECK_GT(statements.size(), 0);
        unrolled_statements.insert(unrolled_statements.end(),
                                   statements.begin(), statements.end() - 1);
        accumulator_value = std::get<Expr*>(statements.back()->wrapped());
      } else {
        unrolled_statements.insert(unrolled_statements.end(),
                                   statements.begin(), statements.end());
      }
    }
    // Handle the final result of unroll_for! expr.
    if (has_result_value) {
      XLS_RETURN_IF_ERROR(
          table_.SetTypeAnnotation(accumulator_value, accumulator_type));
      Statement* last = module_.Make<Statement>(accumulator_value);
      unrolled_statements.push_back(last);
    }
    StatementBlock* unrolled_statement_block = module_.Make<StatementBlock>(
        unroll_for->span(), unrolled_statements, !has_result_value);

    // This enables us to figure out whether a nested unroll_for! is in a proc.
    unrolled_statement_block->SetParentNonLexical(unroll_for->parent());

    VLOG(6) << "Unrolled loop: " << unrolled_statement_block->ToString();
    auto populate_visitor = CreatePopulateTableVisitor(
        unroll_for->owner(), &table_, &import_data_,
        [](std::unique_ptr<Module>, std::filesystem::path path)
            -> absl::StatusOr<std::unique_ptr<ModuleInfo>> {
          XLS_RET_CHECK_FAIL()
              << "Typecheck for an import should not be triggered while "
                 "populating an expanded unroll_for! body.";
        });

    XLS_RETURN_IF_ERROR(populate_visitor->PopulateFromUnrolledLoopBody(
        unrolled_statement_block));
    XLS_RETURN_IF_ERROR(converter_.ConvertSubtree(
        unrolled_statement_block, std::nullopt, parametric_context_));
    if (has_result_value) {
      absl::StatusOr<InterpValue> const_result =
          ConstexprEvaluator::EvaluateToValue(
              &import_data_, ti_, &warning_collector_,
              table_.GetParametricEnv(parametric_context_),
              std::get<Expr*>(unrolled_statements.back()->wrapped()));
      if (const_result.ok()) {
        trace_.SetResult(*const_result);
        ti_->NoteConstExpr(unroll_for, *const_result);
      }
    }

    ti_->NoteUnrolledLoop(unroll_for,
                          table_.GetParametricEnv(parametric_context_),
                          unrolled_statement_block);
    return absl::OkStatus();
  }

  absl::Status HandleFormatMacro(const FormatMacro* node) override {
    if (!node->verbosity().has_value()) {
      return absl::OkStatus();
    }
    absl::StatusOr<InterpValue> value = ConstexprEvaluator::EvaluateToValue(
        &import_data_, ti_, &warning_collector_, ParametricEnv(),
        *node->verbosity());
    if (!value.ok()) {
      return TypeInferenceErrorStatus(
          (*node->verbosity())->span(), &type_,
          absl::Substitute("$0 verbosity values must be compile-time "
                           "constants; got `$1`.",
                           node->macro(), (*node->verbosity())->ToString()),
          file_table_);
    }
    absl::StatusOr<int64_t> value_as_int64 = value->GetBitValueViaSign();
    if (!value_as_int64.ok() || *value_as_int64 < 0) {
      return TypeInferenceErrorStatus(
          (*node->verbosity())->span(), &type_,
          absl::Substitute(
              "$0 verbosity values must be positive integers; got `$1`.",
              node->macro(), (*node->verbosity())->ToString()),
          file_table_);
    }
    ti_->NoteConstExpr(*node->verbosity(), *value);
    return absl::OkStatus();
  }

  absl::Status HandleFunction(const Function* function) override {
    if (function->tag() == FunctionTag::kProcInit) {
      XLS_RETURN_IF_ERROR(EvaluateAndNoteExpr(function->body()));
    }
    return absl::OkStatus();
  }

  absl::Status HandleInvocation(const Invocation* invocation) override {
    if (!IsBuiltinFn(invocation->callee())) {
      absl::StatusOr<Function*> f = ResolveFunction(invocation->callee(), ti_);
      if (f.ok() && (*f)->tag() == FunctionTag::kProcInit) {
        XLS_RETURN_IF_ERROR(EvaluateAndNoteExpr(invocation));
      }
      return absl::OkStatus();
    }

    NameRef* callee_nameref = down_cast<NameRef*>(invocation->callee());
    std::optional<const Type*> callee_type = ti_->GetItem(invocation->callee());
    if (callee_type.has_value()) {
      const auto& function_type = (*callee_type)->AsFunction();
      XLS_RETURN_IF_ERROR(NoteBuiltinInvocationConstExpr(
          callee_nameref->identifier(), invocation, function_type, ti_,
          &import_data_));
      std::optional<InterpValue> value = ti_->GetConstExprOption(invocation);
      if (value.has_value()) {
        trace_.SetResult(*value);
      }
    }
    return absl::OkStatus();
  }

  absl::Status EvaluateAndNoteExpr(const Expr* expr) {
    absl::StatusOr<InterpValue> val = ConstexprEvaluator::EvaluateToValue(
        &import_data_, ti_, &warning_collector_,
        table_.GetParametricEnv(parametric_context_), expr);
    if (val.ok()) {
      trace_.SetResult(*val);
      ti_->NoteConstExpr(expr, *val);
    }
    return absl::OkStatus();
  }

 private:
  InferenceTable& table_;
  Module& module_;
  ImportData& import_data_;
  WarningCollector& warning_collector_;
  const FileTable& file_table_;
  InferenceTableConverter& converter_;
  Evaluator& evaluator_;
  ParametricStructInstantiator& parametric_struct_instantiator_;
  TypeSystemTracer& tracer_;
  std::optional<const ParametricContext*> parametric_context_;
  const Type& type_;
  TypeInfo* ti_;
  TypeSystemTrace trace_;
};

class ConstantCollectorImpl : public ConstantCollector {
 public:
  ConstantCollectorImpl(
      InferenceTable& table, Module& module, ImportData& import_data,
      WarningCollector& warning_collector, const FileTable& file_table,
      InferenceTableConverter& converter, Evaluator& evaluator,
      ParametricStructInstantiator& parametric_struct_instantiator,
      TypeSystemTracer& tracer)
      : table_(table),
        module_(module),
        import_data_(import_data),
        warning_collector_(warning_collector),
        file_table_(file_table),
        converter_(converter),
        evaluator_(evaluator),
        parametric_struct_instantiator_(parametric_struct_instantiator),
        tracer_(tracer) {}

  absl::Status CollectConstants(
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node, const Type& type, TypeInfo* ti) override {
    TypeSystemTrace trace =
        tracer_.TraceCollectConstants(parametric_context, node);
    Visitor visitor(table_, module_, import_data_, warning_collector_,
                    file_table_, converter_, evaluator_,
                    parametric_struct_instantiator_, tracer_,
                    parametric_context, type, ti, std::move(trace));
    return node->Accept(&visitor);
  }

 private:
  InferenceTable& table_;
  Module& module_;
  ImportData& import_data_;
  WarningCollector& warning_collector_;
  const FileTable& file_table_;
  InferenceTableConverter& converter_;
  Evaluator& evaluator_;
  ParametricStructInstantiator& parametric_struct_instantiator_;
  TypeSystemTracer& tracer_;
};

}  // namespace

std::unique_ptr<ConstantCollector> CreateConstantCollector(
    InferenceTable& table, Module& module, ImportData& import_data,
    WarningCollector& warning_collector, const FileTable& file_table,
    InferenceTableConverter& converter, Evaluator& evaluator,
    ParametricStructInstantiator& parametric_struct_instantiator,
    TypeSystemTracer& tracer) {
  return std::make_unique<ConstantCollectorImpl>(
      table, module, import_data, warning_collector, file_table, converter,
      evaluator, parametric_struct_instantiator, tracer);
}

}  // namespace xls::dslx
