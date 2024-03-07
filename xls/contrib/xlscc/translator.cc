// Copyright 2021 The XLS Authors
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

#include "xls/contrib/xlscc/translator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <regex>  // NOLINT
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "clang/include/clang/AST/APValue.h"
#include "clang/include/clang/AST/ASTContext.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/DeclCXX.h"
#include "clang/include/clang/AST/DeclTemplate.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/ExprCXX.h"
#include "clang/include/clang/AST/Mangle.h"
#include "clang/include/clang/AST/OperationKinds.h"
#include "clang/include/clang/AST/Stmt.h"
#include "clang/include/clang/AST/TemplateBase.h"
#include "clang/include/clang/AST/Type.h"
#include "clang/include/clang/Basic/ABI.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "clang/include/clang/Basic/OperatorKinds.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "clang/include/clang/Basic/SourceManager.h"
#include "clang/include/clang/Basic/TypeTraits.h"
#include "llvm/include/llvm/ADT/APInt.h"
#include "llvm/include/llvm/ADT/FloatingPointMode.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/fileno.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/solvers/z3_utils.h"
#include "re2/re2.h"

using std::list;
using std::ostringstream;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {
template <typename Lambda>
class LambdaGuard {
 public:
  explicit LambdaGuard(const Lambda& lambda) : lambda_(lambda) {}
  ~LambdaGuard() { lambda_(); }

 private:
  Lambda lambda_;
};

// For automatic template deduction
template <class Lambda>
inline LambdaGuard<Lambda> MakeLambdaGuard(const Lambda& free) {
  return LambdaGuard<Lambda>(free);
}

const clang::Expr* GetCastsSubexpr(const clang::Expr* expr) {
  while (clang::isa<clang::CastExpr>(expr)) {
    expr = clang::cast<const clang::CastExpr>(expr)->getSubExpr();
  }
  return expr;
}

}  // namespace

namespace xlscc {

bool Translator::ContainsKeyValuePair(
    const absl::btree_multimap<const IOChannel*, ChannelBundle>& map,
    const std::pair<const IOChannel*, ChannelBundle>& pair) {
  for (absl::btree_multimap<const IOChannel*, ChannelBundle>::const_iterator
           it = map.begin();
       it != map.end(); ++it) {
    if (pair.first == it->first && pair.second == it->second) {
      return true;
    }
  }
  return false;
}

bool Translator::IOChannelInCurrentFunction(IOChannel* to_find,
                                            const xls::SourceInfo& loc) {
  // Assertion for linear scan. If it fires, consider changing data structure
  XLSCC_CHECK_LE(context().sf->io_channels.size(), 128, loc);
  for (IOChannel& channel : context().sf->io_channels) {
    if (&channel == to_find) {
      return true;
    }
  }
  return false;
}

Translator::Translator(bool error_on_init_interval, bool error_on_uninitialized,
                       bool generate_fsms_for_pipelined_loops,
      DebugIrTraceFlags debug_ir_trace_flags,
                       int64_t max_unroll_iters, int64_t warn_unroll_iters,
                       int64_t z3_rlimit, IOOpOrdering op_ordering,
                       std::unique_ptr<CCParser> existing_parser)
    : max_unroll_iters_(max_unroll_iters),
      warn_unroll_iters_(warn_unroll_iters),
      z3_rlimit_(z3_rlimit),
      error_on_init_interval_(error_on_init_interval),
      error_on_uninitialized_(error_on_uninitialized),
      generate_fsms_for_pipelined_loops_(generate_fsms_for_pipelined_loops),
      debug_ir_trace_flags_(debug_ir_trace_flags),
      op_ordering_(op_ordering) {
  context_stack_.push_front(TranslationContext());
  if (existing_parser != nullptr) {
    parser_ = std::move(existing_parser);
  } else {
    parser_ = std::make_unique<CCParser>();
  }
}

Translator::~Translator() = default;

TranslationContext& Translator::PushContext() {
  auto ocond = context().full_condition;
  context_stack_.push_front(context());
  context().full_condition_on_enter_block = ocond;
  context().relative_condition = xls::BValue();
  context().propagate_up = true;
  context().propagate_break_up = true;
  context().propagate_continue_up = true;
  // Declaration propagation does not get propagated down
  context().propagate_declarations = false;
  return context();
}

absl::Status Translator::PopContext(const xls::SourceInfo& loc) {
  const bool propagate_up = context().propagate_up;
  const bool propagate_break_up = context().propagate_break_up;
  const bool propagate_continue_up = context().propagate_continue_up;

  // Copy updated variables
  TranslationContext popped = context();
  context_stack_.pop_front();

  XLSCC_CHECK(!context_stack_.empty(), loc);
  if (!propagate_up) {
    return absl::OkStatus();
  }

  XLS_RETURN_IF_ERROR(
      PropagateVariables(/*from=*/popped, /*to=*/context(), loc));

  context().return_cval = popped.return_cval;
  context().last_return_condition = popped.last_return_condition;
  context().have_returned_condition = popped.have_returned_condition;

  for (const auto& [decl, count] : popped.variables_accessed) {
    if (!context().variables_accessed.contains(decl)) {
      context().variables_accessed[decl] = popped.variables_accessed.at(decl);
      continue;
    }
    context().variables_accessed[decl] =
        std::max(context().variables_accessed.at(decl),
                 popped.variables_accessed.at(decl));
  }

  context().any_side_effects_requested =
      context().any_side_effects_requested || popped.any_side_effects_requested;
  context().any_writes_generated =
      context().any_writes_generated || popped.any_writes_generated;
  context().any_io_ops_requested =
      context().any_io_ops_requested || popped.any_io_ops_requested;

  if (popped.have_returned_condition.valid()) {
    XLS_RETURN_IF_ERROR(and_condition(
        context().fb->Not(popped.have_returned_condition, loc), loc));
  }

  if (propagate_break_up && popped.relative_break_condition.valid()) {
    xls::BValue saved_popped_relative_break_condition =
        popped.relative_break_condition;

    if (popped.relative_break_condition.valid() &&
        context().relative_condition.valid()) {
      context().and_condition_util(context().relative_condition,
                                   saved_popped_relative_break_condition, loc);
    }

    XLS_RETURN_IF_ERROR(
        and_condition(context().fb->Not(popped.relative_break_condition), loc));

    context().or_condition_util(saved_popped_relative_break_condition,
                                context().relative_break_condition, loc);
  }

  if (propagate_continue_up && popped.relative_continue_condition.valid()) {
    xls::BValue saved_popped_relative_continue_condition =
        popped.relative_continue_condition;

    if (popped.relative_continue_condition.valid() &&
        context().relative_condition.valid()) {
      context().and_condition_util(context().relative_condition,
                                   saved_popped_relative_continue_condition,
                                   loc);
    }

    XLS_RETURN_IF_ERROR(and_condition(
        context().fb->Not(popped.relative_continue_condition), loc));

    context().or_condition_util(saved_popped_relative_continue_condition,
                                context().relative_continue_condition, loc);
  }
  return absl::OkStatus();
}

absl::Status Translator::PropagateVariables(const TranslationContext& from,
                                            TranslationContext& to,
                                            const xls::SourceInfo& loc) {
  if (from.sf == nullptr) {
    return absl::OkStatus();
  }
  for (const clang::NamedDecl* name :
       from.sf->DeterministicKeyNames(from.variables)) {
    if (to.variables.contains(name)) {
      if (to.variables.at(name) != from.variables.at(name)) {
        XLS_ASSIGN_OR_RETURN(CValue prepared,
                             PrepareRValueWithSelect(
                                 to.variables.at(name), from.variables.at(name),
                                 from.relative_condition, loc));

        // Don't use Assign(), it uses context()
        to.variables.at(name) = prepared;
      }
    } else if (to.sf->static_values.contains(name) ||
               from.propagate_declarations) {
      to.variables[name] = from.variables.at(name);
    }
  }

  return absl::OkStatus();
}

std::function<std::optional<std::string>(xls::Fileno)>
Translator::LookUpInPackage() {
  return [=](xls::Fileno file_number) {
    AddSourceInfoToPackage(*package_);
    return package_->GetFilename(file_number);
  };
}

void Translator::AddSourceInfoToPackage(xls::Package& package) {
  parser_->AddSourceInfoToPackage(package);
}

TranslationContext& Translator::context() {
  CHECK(!context_stack_.empty());
  return context_stack_.front();
}

absl::Status Translator::and_condition(xls::BValue and_condition,
                                       const xls::SourceInfo& loc) {
  if (!and_condition.valid()) {
    return absl::OkStatus();
  }

  // Whenever the condition changes,
  // selects need to be generated in the enclosing context (if any)
  if (context().propagate_up && (context_stack_.size() > 2)) {
    auto iter = context_stack_.begin();
    TranslationContext& top = *iter;
    XLSCC_CHECK_EQ(&top, &context(), loc);

    ++iter;

    TranslationContext& second_to_top = *iter;
    XLSCC_CHECK_NE(&second_to_top, &context(), loc);

    XLSCC_CHECK_NE(second_to_top.sf, nullptr, loc);
    XLSCC_CHECK_NE(top.sf, nullptr, loc);
    XLSCC_CHECK_NE(&second_to_top, &top, loc);
    XLSCC_CHECK_EQ(top.fb, second_to_top.fb, loc);
    XLSCC_CHECK_EQ(top.sf, second_to_top.sf, loc);

    XLS_RETURN_IF_ERROR(PropagateVariables(/*from=*/top,
                                           /*to=*/second_to_top, loc));
  }

  context().and_condition_util(and_condition, context().relative_condition,
                               loc);
  context().and_condition_util(and_condition, context().full_condition, loc);
  return absl::OkStatus();
}

absl::StatusOr<const clang::NamedDecl*> Translator::GetThisDecl(
    const xls::SourceInfo& loc, bool for_declaration) {
  const clang::NamedDecl* decl = context().override_this_decl_;

  if (decl == nullptr) {
    decl = context().sf->clang_decl;
  }

  XLSCC_CHECK_NE(decl, nullptr, loc);

  if (!for_declaration && !context().variables.contains(decl)) {
    return absl::UnimplementedError(absl::StrFormat(
        "Tried to access 'this' in a context without any enclosing class "
        "(top level methods are not supported unless using block-from-class "
        "mode) at %s",
        LocString(loc)));
  }
  return decl;
}

absl::StatusOr<CValue> Translator::StructUpdate(
    CValue struct_before, CValue rvalue, const clang::NamedDecl* field_name,
    const CStructType& stype, const xls::SourceInfo& loc) {
  const absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<CField>>&
      fields_by_name = stype.fields_by_name();
  auto found_field = fields_by_name.find(field_name);
  if (found_field == fields_by_name.end()) {
    return absl::NotFoundError(
        ErrorMessage(loc, "Assignment to unknown field %s in type %s",
                     field_name->getNameAsString(), string(stype)));
  }
  const CField& cfield = *found_field->second;

  if (*cfield.type() != *rvalue.type()) {
    return absl::InvalidArgumentError(
        ErrorMessage(loc,
                     "Cannot assign rvalue of type %s to struct "
                     "field of type %s",
                     string(*rvalue.type()), string(*cfield.type())));
  }

  // No tuple update, so we need to rebuild the tuple
  std::vector<xls::BValue> bvals;
  absl::flat_hash_map<int64_t, std::shared_ptr<LValue>> compounds;

  if (struct_before.lvalue() != nullptr) {
    compounds = struct_before.lvalue()->get_compounds();
  }

  XLS_ASSIGN_OR_RETURN(int64_t counted_from_type,
                       stype.count_lvalue_compounds(*this));

  XLSCC_CHECK(
      (struct_before.lvalue().get() == nullptr) ||
          (struct_before.lvalue()->get_compounds().empty()) ||
          (struct_before.lvalue()->get_compounds().size() == counted_from_type),
      loc);

  {
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> field_type,
                         ResolveTypeInstance(cfield.type()));
    if (field_type->Is<CStructType>()) {
      XLS_ASSIGN_OR_RETURN(
          int64_t counted_from_field_type,
          field_type->As<CStructType>()->count_lvalue_compounds(*this));
      XLSCC_CHECK((rvalue.lvalue().get() == nullptr) ||
                      (rvalue.lvalue()->get_compounds().empty()) ||
                      (rvalue.lvalue()->get_compounds().size() ==
                       counted_from_field_type),
                  loc);
    }
  }

  for (auto it = stype.fields().begin(); it != stype.fields().end(); it++) {
    std::shared_ptr<CField> fp = *it;
    XLS_ASSIGN_OR_RETURN(bool has_lvals, fp->type()->ContainsLValues(*this));
    xls::BValue bval;
    if (fp->index() != cfield.index()) {
      bval = GetStructFieldXLS(struct_before.rvalue(), fp->index(), stype, loc);
    } else {
      bval = rvalue.rvalue().valid() ? rvalue.rvalue()
                                     : context().fb->Tuple({}, loc);
      if (has_lvals) {
        compounds[fp->index()] = rvalue.lvalue();
      }
    }
    bvals.push_back(bval);
  }

  xls::BValue new_tuple = MakeStructXLS(bvals, stype, loc);

  std::shared_ptr<LValue> lval;

  if (struct_before.lvalue() != nullptr) {
    lval = std::make_shared<LValue>(compounds);
  }

  XLSCC_CHECK((lval.get() == nullptr) || (lval->get_compounds().empty()) ||
                  (lval->get_compounds().size() == counted_from_type),
              loc);

  CValue ret(new_tuple, struct_before.type(), /*disable_type_check=*/false,
             lval);

  return ret;
}

xls::BValue Translator::MakeStructXLS(
    const std::vector<xls::BValue>& bvals_reverse, const CStructType& stype,
    const xls::SourceInfo& loc) {
  std::vector<xls::BValue> bvals = bvals_reverse;
  std::reverse(bvals.begin(), bvals.end());
  XLSCC_CHECK_EQ(bvals.size(), stype.fields().size(), loc);
  xls::BValue ret =
      stype.no_tuple_flag() ? bvals[0] : context().fb->Tuple(bvals, loc);
  return ret;
}

xls::Value Translator::MakeStructXLS(
    const std::vector<xls::Value>& vals_reverse, const CStructType& stype) {
  std::vector<xls::Value> vals = vals_reverse;
  std::reverse(vals.begin(), vals.end());
  CHECK_EQ(vals.size(), stype.fields().size());
  xls::Value ret = stype.no_tuple_flag() ? vals[0] : xls::Value::Tuple(vals);
  return ret;
}

xls::BValue Translator::GetStructFieldXLS(xls::BValue val, int64_t index,
                                          const CStructType& type,
                                          const xls::SourceInfo& loc) {
  XLSCC_CHECK_LT(index, type.fields().size(), loc);
  return type.no_tuple_flag() ? val
                              : context().fb->TupleIndex(
                                    val, type.fields().size() - 1 - index, loc);
}

absl::StatusOr<xls::Value> Translator::GetStructFieldXLS(
    xls::Value val, int index, const CStructType& type) {
  CHECK_LT(index, type.fields().size());
  if (type.no_tuple_flag()) {
    return val;
  }
  XLS_ASSIGN_OR_RETURN(std::vector<xls::Value> values, val.GetElements());
  return values.at(type.fields().size() - 1 - index);
}

absl::StatusOr<xls::Type*> Translator::GetStructXLSType(
    const std::vector<xls::Type*>& members, const CStructType& type,
    const xls::SourceInfo& loc) {
  if (type.no_tuple_flag() && members.size() != 1) {
    return absl::FailedPreconditionError(
        ErrorMessage(loc,
                     "Pragma hls_no_tuple must be used on structs with only "
                     "1 field, but %s has %i\n",
                     string(type), members.size()));
  }
  return type.no_tuple_flag() ? members[0] : package_->GetTupleType(members);
}

xls::BValue Translator::MakeFlexTuple(const std::vector<xls::BValue>& bvals,
                                      const xls::SourceInfo& loc) {
  XLSCC_CHECK(!bvals.empty(), loc);
  return (bvals.size() == 1) ? bvals[0] : context().fb->Tuple(bvals, loc);
}

xls::BValue Translator::GetFlexTupleField(xls::BValue val, int64_t index,
                                          int64_t n_fields,
                                          const xls::SourceInfo& loc,
                                          std::string_view op_name) {
  XLSCC_CHECK_GT(n_fields, 0, loc);
  return (n_fields == 1) ? val
                         : context().fb->TupleIndex(val, index, loc, op_name);
}

xls::Type* Translator::GetFlexTupleType(
    const std::vector<xls::Type*>& members) {
  CHECK(!members.empty());
  return (members.size() == 1) ? members[0] : package_->GetTupleType(members);
}

xls::BValue Translator::MakeFunctionReturn(
    const std::vector<xls::BValue>& bvals, const xls::SourceInfo& loc) {
  return MakeFlexTuple(bvals, loc);
}

xls::BValue Translator::UpdateFlexTupleField(xls::BValue tuple_val,
                                             xls::BValue new_val, int index,
                                             int n_fields,
                                             const xls::SourceInfo& loc) {
  if (n_fields == 1) {
    return new_val;
  }

  std::vector<xls::BValue> new_args;
  new_args.reserve(n_fields);
  for (int i = 0; i < n_fields; ++i) {
    new_args.push_back(
        (i == index) ? new_val : context().fb->TupleIndex(tuple_val, i, loc));
  }
  return context().fb->Tuple(new_args, loc);
}

xls::BValue Translator::GetFunctionReturn(xls::BValue val, int index,
                                          int expected_returns,
                                          const clang::FunctionDecl* /*func*/,
                                          const xls::SourceInfo& loc) {
  return GetFlexTupleField(val, index, expected_returns, loc);
}

std::string Translator::XLSNameMangle(clang::GlobalDecl decl) const {
  std::string res;
  llvm::raw_string_ostream os(res);
  if (!mangler_) {
    mangler_.reset(decl.getDecl()->getASTContext().createMangleContext());
  }
  mangler_->mangleCXXName(decl, os);
  // mangleCXXName can generate "$" which is invalid in function names
  return absl::StrReplaceAll(res, {{"$", "S"}});
}

absl::Status Translator::GenerateThisLValues(
    const clang::RecordDecl* this_struct_decl,
    const std::shared_ptr<CType> thisctype,
    bool member_references_become_channels, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(auto thisctype_resolved, ResolveTypeInstance(thisctype));
  auto struct_type = thisctype_resolved->As<CStructType>();
  XLSCC_CHECK_NE(struct_type, nullptr, loc);

  XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl,
                       GetThisDecl(loc, /*for_declaration=*/false));

  for (const clang::FieldDecl* field_decl : this_struct_decl->fields()) {
    std::shared_ptr<CField> field = struct_type->get_field(field_decl);
    std::shared_ptr<CType> field_type = field->type();
    auto field_decl_loc = GetLoc(*field_decl);

    XLS_ASSIGN_OR_RETURN(bool type_contains_lval,
                         field->type()->ContainsLValues(*this));

    if (!type_contains_lval) {
      continue;
    }

    std::shared_ptr<LValue> lval;

    if (field_decl->getInClassInitializer() != nullptr) {
      PushContextGuard guard(*this, field_decl_loc);
      context().any_side_effects_requested = false;

      XLS_ASSIGN_OR_RETURN(
          CValue field_val,
          GenerateIR_Expr(field_decl->getInClassInitializer(), field_decl_loc));

      if (context().any_side_effects_requested) {
        return absl::UnimplementedError(
            ErrorMessage(field_decl_loc,
                         "Side effects in initializer for top class field %s",
                         field_decl->getQualifiedNameAsString()));
      }

      lval = field_val.lvalue();
    } else {
      if (field->type()->Is<CChannelType>()) {
        XLS_ASSIGN_OR_RETURN(
            lval, CreateChannelParam(
                      field->name(),
                      std::dynamic_pointer_cast<CChannelType>(field->type()),
                      /*declare_variable=*/false, field_decl_loc));

      } else if (field->type()->Is<CReferenceType>() &&
                 member_references_become_channels) {
        return absl::UnimplementedError(ErrorMessage(
            field_decl_loc,
            "Pass through for direct-ins not implemented yet for field %s",
            field->name()->getNameAsString()));
      } else {
        return absl::UnimplementedError(ErrorMessage(
            field_decl_loc, "Don't know how to create LValue for member %s",
            field->name()->getNameAsString()));
      }
    }

    CHECK_NE(lval.get(), nullptr);

    XLS_ASSIGN_OR_RETURN(CValue prev_this_cval, GetIdentifier(this_decl, loc));

    XLSCC_CHECK(prev_this_cval.rvalue().valid(), loc);

    xls::BValue prev_field_rvalue;

    XLS_ASSIGN_OR_RETURN(bool field_must_have_rvalue,
                         TypeMustHaveRValue(*field_type));
    if (field_must_have_rvalue) {
      prev_field_rvalue = GetStructFieldXLS(prev_this_cval.rvalue(),
                                            field->index(), *struct_type, loc);
    }

    CValue new_field_cval(prev_field_rvalue, field_type,
                          /*disable_type_check=*/false, lval);

    {
      UnmaskAndIgnoreSideEffectsGuard unmask_guard(*this);
      XLS_RETURN_IF_ERROR(
          AssignMember(this_decl, field_decl, new_field_cval, loc));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> Translator::TypeMustHaveRValue(const CType& type) {
  if (type.Is<CVoidType>()) {
    return false;
  }
  XLS_ASSIGN_OR_RETURN(bool contains_lvalues, type.ContainsLValues(*this));
  if (!contains_lvalues) {
    return true;
  }
  return type.Is<CInstantiableTypeAlias>() || type.Is<CStructType>();
}

absl::StatusOr<FunctionInProgress> Translator::GenerateIR_Function_Header(
    GeneratedFunction& sf, const clang::FunctionDecl* funcdecl,
    std::string_view name_override, bool force_static,
    bool member_references_become_channels) {
  std::string xls_name;

  if (!name_override.empty()) {
    xls_name = name_override;
  } else {
    clang::GlobalDecl global_decl;
    if (auto c_decl =
            clang::dyn_cast<const clang::CXXConstructorDecl>(funcdecl)) {
      global_decl = clang::GlobalDecl(c_decl, clang::Ctor_Complete);
    } else {
      global_decl = clang::GlobalDecl(funcdecl);
    }
    xls_name = XLSNameMangle(global_decl);
  }

  XLSCC_CHECK(!xls_names_for_functions_generated_.contains(funcdecl),
              GetLoc(*funcdecl));

  xls_names_for_functions_generated_[funcdecl] = xls_name;

  auto builder = std::make_unique<xls::FunctionBuilder>(xls_name, package_);

  PushContextGuard context_guard(*this, GetLoc(*funcdecl));

  std::vector<const clang::NamedDecl*> ref_returns;

  sf.clang_decl = funcdecl;

  // Pragma at class or method level
  XLS_ASSIGN_OR_RETURN(sf.in_synthetic_int, FunctionIsInSyntheticInt(funcdecl));
  XLS_ASSIGN_OR_RETURN(Pragma pragma,
                       FindPragmaForLoc(GetPresumedLoc(*funcdecl)));
  const bool synthetic_int_pragma = pragma.type() == Pragma_SyntheticInt;
  sf.in_synthetic_int = sf.in_synthetic_int || synthetic_int_pragma;

  // Functions need a clean context
  context() = TranslationContext();
  context().propagate_up = false;

  context().fb = absl::implicit_cast<xls::BuilderBase*>(builder.get());
  context().sf = &sf;
  context().ast_context = &funcdecl->getASTContext();

  // Unroll for loops in default function bodies without pragma
  context().for_loops_default_unroll = funcdecl->isDefaulted();
  context().outer_pipelined_loop_init_interval = default_init_interval_;
  XLS_ASSIGN_OR_RETURN(
      context().return_type,
      TranslateTypeFromClang(funcdecl->getReturnType(), GetLoc(*funcdecl)));

  // If add_this_return is true, then a return value is added for the
  //  "this" object, pointed to be the "this" pointer in methods
  bool add_this_return = false;

  xls::SourceInfo body_loc = GetLoc(*funcdecl);

  // "this" input for methods
  if (auto method = clang::dyn_cast<const clang::CXXMethodDecl>(funcdecl)) {
    if (!method->isStatic() && !force_static) {
      // "This" is a PointerType, ignore and treat as reference
      clang::QualType thisQual = method->getThisType();
      XLSCC_CHECK(thisQual->isPointerType(), body_loc);

      add_this_return = !thisQual->getPointeeType().isConstQualified();

      clang::QualType q = thisQual->getPointeeOrArrayElementType()
                              ->getCanonicalTypeUnqualified();

      XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> thisctype,
                           TranslateTypeFromClang(q, body_loc));

      XLSCC_CHECK(!thisctype->Is<CChannelType>(), body_loc);

      XLS_ASSIGN_OR_RETURN(xls::Type * xls_type,
                           TranslateTypeToXLS(thisctype, body_loc));

      xls::BValue this_bval = context().fb->Param("this", xls_type, body_loc);

      XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl,
                           GetThisDecl(body_loc, /*for_declaration=*/true));

      XLS_ASSIGN_OR_RETURN(CValue this_default_val,
                           CreateDefaultCValue(thisctype, body_loc));

      CValue this_val(this_bval, thisctype, /*disable_type_check=*/false,
                      this_default_val.lvalue());

      XLS_RETURN_IF_ERROR(DeclareVariable(this_decl, this_val, body_loc));

      XLS_RETURN_IF_ERROR(GenerateThisLValues(
          /*this_struct_decl=*/q->getAsRecordDecl(), thisctype,
          member_references_become_channels, body_loc));

      XLS_ASSIGN_OR_RETURN(this_val, GetIdentifier(this_decl, body_loc));
      context().sf->this_lvalue = this_val.lvalue();
    }
  }

  absl::flat_hash_set<std::string> used_parameter_names;

  for (const clang::ParmVarDecl* p : funcdecl->parameters()) {
    auto namedecl = absl::implicit_cast<const clang::NamedDecl*>(p);

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));

    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> obj_type,
                         TranslateTypeFromClang(stripped.base, GetLoc(*p)));

    if (obj_type->Is<CPointerType>()) {
      return absl::UnimplementedError(
          ErrorMessage(GetLoc(*p), "Pointer function parameters unsupported"));
    }

    xls::Type* xls_type = nullptr;

    XLS_ASSIGN_OR_RETURN(bool is_channel,
                         TypeIsChannel(p->getType(), GetLoc(*p)));

    if (is_channel) {
      // TODO(seanhaskell): This can be further generalized to allow
      // structs with channels inside
      XLS_ASSIGN_OR_RETURN(
          auto channel_type,
          GetChannelType(p->getType(), p->getASTContext(), GetLoc(*p)));
      XLS_ASSIGN_OR_RETURN(
          std::shared_ptr<LValue> lval,
          CreateChannelParam(p, channel_type,
                             /*declare_variable=*/true, GetLoc(*p)));
      (void)lval;
      continue;
    }

    // Const references don't need a return
    if (stripped.is_ref && (!stripped.base.isConstQualified())) {
      ref_returns.push_back(namedecl);
    }

    if (xls_type == nullptr) {
      XLS_ASSIGN_OR_RETURN(xls_type, TranslateTypeToXLS(obj_type, GetLoc(*p)));
    }

    std::string safe_param_name = namedecl->getNameAsString();
    if (safe_param_name.empty()) {
      safe_param_name = "implicit";
    }

    for (int iter = 0; used_parameter_names.contains(safe_param_name); ++iter) {
      safe_param_name += absl::StrFormat("%i", used_parameter_names.size());

      if (iter > 10) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Couldn't find a safe parameter name at %s",
                            LocString(GetLoc(*p))));
      }
    }
    XLSCC_CHECK(!used_parameter_names.contains(safe_param_name), GetLoc(*p));
    used_parameter_names.insert(safe_param_name);

    xls::BValue pbval =
        context().fb->Param(safe_param_name, xls_type, body_loc);

    // Create CValue without type check
    XLS_RETURN_IF_ERROR(
        DeclareVariable(namedecl, CValue(pbval, obj_type, true), body_loc));
  }

  // Generate constructor initializers
  if (auto constructor =
          clang::dyn_cast<const clang::CXXConstructorDecl>(funcdecl)) {
    XLSCC_CHECK(add_this_return, GetLoc(*constructor));
    XLS_RETURN_IF_ERROR(GenerateIR_Ctor_Initializers(constructor));
  }

  auto context_copy = std::make_unique<TranslationContext>(context());

  FunctionInProgress header = {.add_this_return = add_this_return,
                               .ref_returns = ref_returns,
                               .builder = std::move(builder),
                               .translation_context = std::move(context_copy)};
  return header;
}

absl::Status Translator::GenerateIR_Ctor_Initializers(
    const clang::CXXConstructorDecl* constructor) {
  xls::SourceInfo ctor_loc = GetLoc(*constructor);
  XLSCC_CHECK(constructor != nullptr, ctor_loc);
  XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl,
                       GetThisDecl(GetLoc(*constructor)));

  for (const clang::CXXCtorInitializer* init : constructor->inits()) {
    xls::SourceInfo init_loc = GetLoc(*init->getInit());

    XLS_ASSIGN_OR_RETURN(CValue prev_this_val,
                         GetIdentifier(this_decl, init_loc));
    XLS_ASSIGN_OR_RETURN(auto resolved_type,
                         ResolveTypeInstance(prev_this_val.type()));
    auto struct_type = std::dynamic_pointer_cast<CStructType>(resolved_type);
    XLSCC_CHECK(struct_type, init_loc);

    const auto& fields_by_name = struct_type->fields_by_name();
    absl::flat_hash_map<int64_t, std::shared_ptr<LValue>> compounds;

    // Base class constructors don't have member names
    const clang::NamedDecl* member_name = nullptr;
    if (init->getMember() != nullptr) {
      member_name =
          absl::implicit_cast<const clang::NamedDecl*>(init->getMember());
    } else {
      member_name = absl::implicit_cast<const clang::NamedDecl*>(
          init->getInit()->getType()->getAsRecordDecl());
      XLSCC_CHECK(member_name, init_loc);
    }
    auto found = fields_by_name.find(member_name);
    XLSCC_CHECK(found != fields_by_name.end(), init_loc);
    std::shared_ptr<CField> field = found->second;
    XLSCC_CHECK(field->name() == member_name, init_loc);

    auto ctor_init = clang::dyn_cast<clang::CXXConstructExpr>(init->getInit());

    CValue rvalue;

    if (field->type()->Is<CChannelType>() && (ctor_init != nullptr) &&
        (ctor_init->getNumArgs() == 0)) {
      auto channel_type =
          std::dynamic_pointer_cast<CChannelType>(field->type());
      if (channel_type->GetOpType() != OpType::kSendRecv) {
        return absl::InvalidArgumentError(
            ErrorMessage(init_loc,
                         "Field %s is interpreted as an internal channel "
                         "declaration, but it is not marked as InOut",
                         field->name()->getQualifiedNameAsString()));
      }

      XLS_ASSIGN_OR_RETURN(rvalue, GenerateIR_LocalChannel(
                                       field->name(), channel_type, init_loc));
    } else {
      XLS_ASSIGN_OR_RETURN(rvalue, GenerateIR_Expr(init->getInit(), init_loc));
    }

    XLS_ASSIGN_OR_RETURN(CValue new_this_val,
                         StructUpdate(prev_this_val, rvalue,
                                      /*field_name=*/member_name,
                                      /*type=*/*struct_type, init_loc));

    XLS_RETURN_IF_ERROR(Assign(this_decl, new_this_val, init_loc));
    context().sf->this_lvalue = new_this_val.lvalue();
  }
  return absl::OkStatus();
}

absl::Status Translator::PushLValueSelectConditions(
    std::shared_ptr<LValue> lvalue, std::vector<xls::BValue>& return_bvals,
    const xls::SourceInfo& loc) {
  if (!lvalue->is_select()) {
    return absl::OkStatus();
  }

  std::shared_ptr<LValue> true_lval = lvalue->lvalue_true();
  std::shared_ptr<LValue> false_lval = lvalue->lvalue_false();

  return_bvals.push_back(lvalue->cond());

  if (true_lval->is_select()) {
    XLS_RETURN_IF_ERROR(
        PushLValueSelectConditions(true_lval, return_bvals, loc));
  }
  if (false_lval->is_select()) {
    XLS_RETURN_IF_ERROR(
        PushLValueSelectConditions(false_lval, return_bvals, loc));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<LValue>> Translator::PopLValueSelectConditions(
    std::list<xls::BValue>& unpacked_returns,
    std::shared_ptr<LValue> lvalue_translated, const xls::SourceInfo& loc) {
  if (!lvalue_translated->is_select()) {
    return lvalue_translated;
  }

  XLSCC_CHECK(!unpacked_returns.empty(), loc);
  xls::BValue selector = unpacked_returns.front();
  unpacked_returns.pop_front();

  std::shared_ptr<LValue> true_lval = lvalue_translated->lvalue_true();
  std::shared_ptr<LValue> false_lval = lvalue_translated->lvalue_false();

  if (true_lval->is_select()) {
    XLS_ASSIGN_OR_RETURN(
        true_lval, PopLValueSelectConditions(unpacked_returns, true_lval, loc));
  }
  if (false_lval->is_select()) {
    XLS_ASSIGN_OR_RETURN(false_lval, PopLValueSelectConditions(
                                         unpacked_returns, false_lval, loc));
  }

  return std::make_shared<LValue>(selector, true_lval, false_lval);
}

absl::StatusOr<xls::BValue> Translator::Generate_LValue_Return(
    std::shared_ptr<LValue> lvalue, const xls::SourceInfo& loc) {
  if (lvalue->channel_leaf() != nullptr) {
    return context().fb->Tuple({}, loc);
  }
  if (lvalue->leaf() != nullptr) {
    bool is_this_expr = clang::isa<clang::CXXThisExpr>(lvalue->leaf());
    if (is_this_expr) {
      return context().fb->Tuple({}, loc);
    }

    if (auto decl_ref =
            clang::dyn_cast<const clang::DeclRefExpr>(lvalue->leaf())) {
      const clang::ValueDecl* decl = decl_ref->getDecl();

      if (context().sf->static_values.contains(decl)) {
        // Returned through side-effecting params
        return context().fb->Tuple({}, loc);
      }
    }

    lvalue->leaf()->dump();
    // Fall through to error
  } else if (lvalue->is_select()) {
    std::vector<xls::BValue> tuple_bvals;
    XLS_RETURN_IF_ERROR(PushLValueSelectConditions(lvalue, tuple_bvals, loc));
    return context().fb->Tuple(tuple_bvals, loc);
  }

  return absl::UnimplementedError(
      ErrorMessage(loc, "Don't know how to handle lvalue return %s",
                   lvalue->debug_string()));
}

absl::StatusOr<CValue> Translator::Generate_LValue_Return_Call(
    std::shared_ptr<LValue> lval_untranslated, xls::BValue unpacked_return,
    std::shared_ptr<CType> return_type, xls::BValue* this_inout,
    std::shared_ptr<LValue>* this_lval,
    const absl::flat_hash_map<IOChannel*, IOChannel*>&
        caller_channels_by_callee_channel,
    const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(
      std::shared_ptr<LValue> lval,
      TranslateLValueChannels(lval_untranslated,
                              caller_channels_by_callee_channel, loc));

  if (lval->channel_leaf() != nullptr) {
    return CValue(xls::BValue(), return_type, /*disable_type_check=*/true,
                  lval);
  }

  if (lval->is_select()) {
    xls::BValue selector_value_tuple = unpacked_return;

    XLS_ASSIGN_OR_RETURN(std::list<xls::BValue> selector_values,
                         UnpackTuple(selector_value_tuple, loc));
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<LValue> ret_lval,
                         PopLValueSelectConditions(selector_values, lval, loc));

    return CValue(xls::BValue(), return_type, /*disable_type_check=*/true,
                  ret_lval);
  }

  if (lval->leaf() != nullptr) {
    XLSCC_CHECK(return_type->Is<CReferenceType>(), loc);

    if (IsThisExpr(lval->leaf())) {
      // TODO(seanhaskell): Remove special 'this' handling
      XLSCC_CHECK_NE(this_inout, nullptr, loc);
      XLSCC_CHECK_NE(this_lval, nullptr, loc);

      return CValue(xls::BValue(), return_type, /*disable_type_check=*/true,
                    *this_lval);
    }

    if (auto decl_ref =
            clang::dyn_cast<const clang::DeclRefExpr>(lval->leaf())) {
      const clang::ValueDecl* decl = decl_ref->getDecl();

      if (context().sf->static_values.contains(decl)) {
        return CValue(xls::BValue(), return_type, /*disable_type_check=*/true,
                      lval);
      }
    }

    lval->leaf()->dump();
    // Fall through to error
  }

  return absl::UnimplementedError(ErrorMessage(
      loc, "Don't know how to translate lvalue of type %s from callee context",
      lval->leaf()->getStmtClassName()));
}

absl::StatusOr<std::list<xls::BValue>> Translator::UnpackTuple(
    xls::BValue tuple_val, const xls::SourceInfo& loc) {
  xls::Type* type = tuple_val.GetType();
  XLSCC_CHECK(type->IsTuple(), loc);
  std::list<xls::BValue> ret;
  for (int64_t i = 0; i < type->AsTupleOrDie()->size(); ++i) {
    ret.push_back(context().fb->TupleIndex(tuple_val, i, loc));
  }
  return ret;
}

absl::Status Translator::GenerateIR_Function_Body(
    GeneratedFunction& sf, const clang::FunctionDecl* funcdecl,
    const FunctionInProgress& header) {
  XLS_ASSIGN_OR_RETURN(const clang::Stmt* body, GetFunctionBody(funcdecl));
  xls::SourceInfo body_loc = GetLoc(*funcdecl);

  PushContextGuard context_guard(*this, *header.translation_context, body_loc);

  // Extra context layer to generate selects
  {
    PushContextGuard top_select_guard(*this, GetLoc(*funcdecl));
    context().propagate_up = true;

    if (body != nullptr) {
      XLS_RETURN_IF_ERROR(GenerateIR_Compound(body, funcdecl->getASTContext()));
    }
  }

  vector<xls::BValue> return_bvals;

  // First static returns
  for (const clang::NamedDecl* decl :
       context().sf->GetDeterministicallyOrderedStaticValues()) {
    XLS_ASSIGN_OR_RETURN(CValue value, GetIdentifier(decl, body_loc));
    return_bvals.push_back(value.rvalue());
  }

  // Then this return
  if (header.add_this_return) {
    XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl,
                         GetThisDecl(body_loc));
    XLS_ASSIGN_OR_RETURN(CValue this_val, GetIdentifier(this_decl, body_loc));
    return_bvals.push_back(this_val.rvalue());
  }

  // Then explicit return
  if (!funcdecl->getReturnType()->isVoidType()) {
    if (context().return_cval.lvalue() != nullptr) {
      if (context().return_cval.rvalue().valid()) {
        return absl::UnimplementedError(ErrorMessage(
            body_loc, "Mixed RValue and LValue returns not yet supported"));
      }
      // LValue related returns (select conditions)
      XLS_ASSIGN_OR_RETURN(
          xls::BValue lvalue_return,
          Generate_LValue_Return(context().return_cval.lvalue(), body_loc));
      return_bvals.emplace_back(lvalue_return);
    } else {
      XLSCC_CHECK(context().return_cval.rvalue().valid(), body_loc);
      return_bvals.emplace_back(context().return_cval.rvalue());
    }
  }

  // Then reference parameter returns
  for (const clang::NamedDecl* ret_ident : header.ref_returns) {
    XLS_ASSIGN_OR_RETURN(CValue found, GetIdentifier(ret_ident, body_loc));
    return_bvals.emplace_back(found.rvalue());
  }

  // IO returns
  for (const IOOp& op : sf.io_ops) {
    XLSCC_CHECK(op.ret_value.valid(), body_loc);
    return_bvals.push_back(op.ret_value);
  }

  sf.return_value_count = return_bvals.size();

  if (return_bvals.empty()) {
    return absl::OkStatus();
  }

  sf.return_lvalue = context().return_cval.lvalue();

  xls::BValue return_bval;

  if (return_bvals.size() == 1) {
    // XLS functions return the last value added to the FunctionBuilder
    // So this makes sure the correct value is last.
    return_bval = return_bvals[0];
  } else {
    return_bval = MakeFunctionReturn(return_bvals, body_loc);
  }

  if (!sf.io_ops.empty() && funcdecl->isOverloadedOperator()) {
    return absl::UnimplementedError(
        ErrorMessage(body_loc, "IO ops in operator calls are not supported"));
  }

  XLS_ASSIGN_OR_RETURN(sf.xls_func,
                       header.builder->BuildWithReturnValue(return_bval));

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<CType>> Translator::InterceptBuiltInStruct(
    const clang::RecordDecl* sd) {
  // "__xls_bits" is a special built-in type: CBitsType
  // The number of bits, or width, is specified as a single integer-type
  //  template parameter.
  if (sd->getNameAsString() == "__xls_bits") {
    auto temp_spec =
        clang::dyn_cast<const clang::ClassTemplateSpecializationDecl>(sd);
    // __xls_bits must always have a template parameter
    if (temp_spec == nullptr) {
      return absl::UnimplementedError(absl::StrFormat(
          "__xls_bits should be used in template specialization %s",
          LocString(GetLoc(*sd))));
    }
    const clang::TemplateArgumentList& temp_args = temp_spec->getTemplateArgs();
    if ((temp_args.size() != 1) ||
        (temp_args.get(0).getKind() !=
         clang::TemplateArgument::ArgKind::Integral)) {
      return absl::UnimplementedError(absl::StrFormat(
          "__xls_bits should have on integral template argument (width) %s",
          LocString(GetLoc(*sd))));
    }

    llvm::APSInt width_aps = temp_args.get(0).getAsIntegral();
    return std::shared_ptr<CType>(new CBitsType(width_aps.getExtValue()));
  }
  return nullptr;
}

absl::Status Translator::ScanStruct(const clang::RecordDecl* sd) {
  std::shared_ptr<CInstantiableTypeAlias> signature(new CInstantiableTypeAlias(
      absl::implicit_cast<const clang::NamedDecl*>(sd)));

  std::shared_ptr<CType> new_type;

  // Check for built-in XLS[cc] types
  XLS_ASSIGN_OR_RETURN(new_type, InterceptBuiltInStruct(sd));
  if (new_type != nullptr) {
    inst_types_[signature] = new_type;
    return absl::OkStatus();
  }
  // If no built-in type was found, interpret as a normal C++ struct
  std::vector<std::shared_ptr<CField>> fields;
  const clang::CXXRecordDecl* cxx_record = nullptr;
  cxx_record = clang::dyn_cast<const clang::CXXRecordDecl>(sd);
  if (cxx_record == nullptr) {
    return absl::UnavailableError(ErrorMessage(
        GetLoc(*sd),
        "Definition for CXXRecord '%s' isn't available from Clang. A "
        "possible work-around is to declare an instance of this class.",
        signature->base()->getNameAsString()));
  }

  // Interpret forward declarations as empty structs
  if (cxx_record->hasDefinition()) {
    for (auto base : cxx_record->bases()) {
      const clang::RecordDecl* base_struct = base.getType()->getAsRecordDecl();
      XLS_ASSIGN_OR_RETURN(
          auto field_type,
          TranslateTypeFromClang(
              base_struct->getTypeForDecl()->getCanonicalTypeInternal(),
              GetLoc(*base_struct)));

      fields.push_back(std::make_shared<CField>(
          absl::implicit_cast<const clang::NamedDecl*>(base_struct),
          fields.size(), field_type));
    }

    for (const clang::FieldDecl* it : sd->fields()) {
      clang::QualType field_qtype = it->getType();
      // TODO(seanhaskell): Remove special channel handling
      XLS_ASSIGN_OR_RETURN(
          bool is_channel,
          TypeIsChannel(field_qtype, GetLoc(*it->getCanonicalDecl())));
      if (is_channel) {
        XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                             StripTypeQualifiers(field_qtype));
        field_qtype = stripped.base;
      }

      XLS_ASSIGN_OR_RETURN(
          std::shared_ptr<CType> field_type,
          TranslateTypeFromClang(field_qtype, GetLoc(*it->getCanonicalDecl())));

      // Up cast FieldDecl to NamedDecl because NamedDecl pointers are used to
      //  track identifiers by XLS[cc], no matter the type of object being
      //  identified
      fields.push_back(std::make_shared<CField>(
          absl::implicit_cast<const clang::NamedDecl*>(it), fields.size(),
          field_type));
    }
  } else {
    XLS_LOG(WARNING) << ErrorMessage(
        GetLoc(*cxx_record),
        "Warning: interpreting definition-less struct '%s' as empty",
        signature->base()->getNameAsString());
  }
  XLS_ASSIGN_OR_RETURN(Pragma pragma, FindPragmaForLoc(GetPresumedLoc(*sd)));
  const bool no_tuple_pragma = pragma.type() == Pragma_NoTuples;
  const bool synthetic_int_pragma = pragma.type() == Pragma_SyntheticInt;

  new_type = std::make_shared<CStructType>(
      fields, synthetic_int_pragma || no_tuple_pragma, synthetic_int_pragma);

  inst_types_[signature] = new_type;
  return absl::OkStatus();
}

absl::StatusOr<xls::Op> Translator::XLSOpcodeFromClang(
    clang::BinaryOperatorKind clang_op, const CType& left_type,
    const CType& result_type, const xls::SourceInfo& loc) {
  if (clang_op == clang::BinaryOperatorKind::BO_Comma) {
    return xls::Op::kIdentity;
  }
  if (result_type.Is<CIntType>()) {
    auto result_int_type = result_type.As<CIntType>();
    switch (clang_op) {
      case clang::BinaryOperatorKind::BO_Assign:
        return xls::Op::kIdentity;
      case clang::BinaryOperatorKind::BO_Add:
      case clang::BinaryOperatorKind::BO_AddAssign:
        return xls::Op::kAdd;
      case clang::BinaryOperatorKind::BO_Sub:
      case clang::BinaryOperatorKind::BO_SubAssign:
        return xls::Op::kSub;
      case clang::BinaryOperatorKind::BO_Mul:
      case clang::BinaryOperatorKind::BO_MulAssign:
        return result_int_type->is_signed() ? xls::Op::kSMul : xls::Op::kUMul;
      case clang::BinaryOperatorKind::BO_Div:
      case clang::BinaryOperatorKind::BO_DivAssign:
        return result_int_type->is_signed() ? xls::Op::kSDiv : xls::Op::kUDiv;
      case clang::BinaryOperatorKind::BO_Rem:
      case clang::BinaryOperatorKind::BO_RemAssign:
        return result_int_type->is_signed() ? xls::Op::kSMod : xls::Op::kUMod;
      case clang::BinaryOperatorKind::BO_Shl:
      case clang::BinaryOperatorKind::BO_ShlAssign:
        return xls::Op::kShll;
      case clang::BinaryOperatorKind::BO_Shr:
      case clang::BinaryOperatorKind::BO_ShrAssign:
        return result_int_type->is_signed() ? xls::Op::kShra : xls::Op::kShrl;
      case clang::BinaryOperatorKind::BO_And:
      case clang::BinaryOperatorKind::BO_AndAssign:
        return xls::Op::kAnd;
      case clang::BinaryOperatorKind::BO_Or:
      case clang::BinaryOperatorKind::BO_OrAssign:
        return xls::Op::kOr;
      case clang::BinaryOperatorKind::BO_Xor:
      case clang::BinaryOperatorKind::BO_XorAssign:
        return xls::Op::kXor;
      default:
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unimplemented binary operator %i for result %s",
                         clang_op, std::string(result_type)));
    }
  }
  if (result_type.Is<CEnumType>()) {
    switch (clang_op) {
      case clang::BinaryOperatorKind::BO_Assign:
        return xls::Op::kIdentity;
      default:
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unimplemented binary operator %i for result %s",
                         clang_op, std::string(result_type)));
    }
  }
  if (result_type.Is<CBoolType>()) {
    if (left_type.Is<CIntType>()) {
      auto input_int_type = left_type.As<CIntType>();
      switch (clang_op) {
        case clang::BinaryOperatorKind::BO_GT:
          return input_int_type->is_signed() ? xls::Op::kSGt : xls::Op::kUGt;
        case clang::BinaryOperatorKind::BO_LT:
          return input_int_type->is_signed() ? xls::Op::kSLt : xls::Op::kULt;
        case clang::BinaryOperatorKind::BO_GE:
          return input_int_type->is_signed() ? xls::Op::kSGe : xls::Op::kUGe;
        case clang::BinaryOperatorKind::BO_LE:
          return input_int_type->is_signed() ? xls::Op::kSLe : xls::Op::kULe;
        default:
          break;
      }
    }
    switch (clang_op) {
      case clang::BinaryOperatorKind::BO_Assign:
        return xls::Op::kIdentity;
      case clang::BinaryOperatorKind::BO_EQ:
        return xls::Op::kEq;
      case clang::BinaryOperatorKind::BO_NE:
        return xls::Op::kNe;
      // Clang generates an ImplicitCast to bool for the parameters to
      //  logical expressions (eg && ||), so logical ops (eg & |) are sufficient
      case clang::BinaryOperatorKind::BO_LAnd:
      case clang::BinaryOperatorKind::BO_AndAssign:
        return xls::Op::kAnd;
      case clang::BinaryOperatorKind::BO_LOr:
      case clang::BinaryOperatorKind::BO_OrAssign:
        return xls::Op::kOr;
      default:
        return absl::UnimplementedError(absl::StrFormat(
            "Unimplemented binary operator %i for result %s with input %s at "
            "%s",
            clang_op, std::string(result_type), std::string(left_type),
            LocString(loc)));
    }
  }

  if (result_type.Is<CPointerType>()) {
    if (clang_op == clang::BinaryOperatorKind::BO_Assign) {
      return xls::Op::kIdentity;
    }
    return absl::UnimplementedError(ErrorMessage(
        loc, "Unimplemented binary operator %i for pointer", clang_op));
  }

  return absl::UnimplementedError(
      ErrorMessage(loc, "Binary operators unimplemented for type %s",
                   std::string(result_type)));
}

absl::StatusOr<CValue> Translator::TranslateVarDecl(
    const clang::VarDecl* decl, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                       TranslateTypeFromClang(decl->getType(), loc));
  const clang::Expr* initializer = decl->getAnyInitializer();

  XLS_ASSIGN_OR_RETURN(CValue ret, CreateInitValue(ctype, initializer, loc));

  return ret;
}

absl::StatusOr<std::shared_ptr<LValue>> Translator::CreateReferenceValue(
    const clang::Expr* initializer, const xls::SourceInfo& loc) {
  if (initializer == nullptr) {
    return absl::InvalidArgumentError(
        ErrorMessage(loc, "References must be initialized"));
  }

  // TODO(seanhaskell): Remove special 'this' handling
  if (clang::isa<clang::CXXThisExpr>(initializer)) {
    return std::make_shared<LValue>(initializer);
  }

  if (!initializer->isLValue()) {
    return absl::InvalidArgumentError(
        ErrorMessage(loc, "References must be initialized to an lvalue"));
  }

  {
    XLS_ASSIGN_OR_RETURN(CValue cv, GenerateIR_Expr(initializer, loc));

    if (cv.type()->Is<CReferenceType>() || cv.type()->Is<CChannelType>()) {
      XLSCC_CHECK_NE(cv.lvalue(), nullptr, loc);
      return cv.lvalue();
    }
  }

  // TODO(seanhaskell): Remove special 'this' handling
  if (const clang::CXXThisExpr* this_expr = IsThisExpr(initializer)) {
    return std::make_shared<LValue>(this_expr);
  }

  const clang::Expr* nested_implicit = RemoveParensAndCasts(initializer);

  if (auto cond_expr =
          clang::dyn_cast<clang::ConditionalOperator>(nested_implicit)) {
    XLS_ASSIGN_OR_RETURN(CValue cond_val,
                         GenerateIR_Expr(cond_expr->getCond(), loc));
    XLSCC_CHECK(cond_val.rvalue().valid(), loc);

    XLS_ASSIGN_OR_RETURN(std::shared_ptr<LValue> true_lval,
                         CreateReferenceValue(cond_expr->getTrueExpr(), loc));
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<LValue> false_lval,
                         CreateReferenceValue(cond_expr->getFalseExpr(), loc));
    return std::make_shared<LValue>(cond_val.rvalue(), true_lval, false_lval);
  }

  return std::make_shared<LValue>(nested_implicit);
}

absl::StatusOr<CValue> Translator::CreateInitValue(
    const std::shared_ptr<CType>& ctype, const clang::Expr* initializer,
    const xls::SourceInfo& loc) {
  std::shared_ptr<LValue> lvalue = nullptr;
  xls::BValue init_val;

  bool is_channel = false;

  if (initializer != nullptr) {
    XLS_ASSIGN_OR_RETURN(is_channel,
                         TypeIsChannel(initializer->getType(), loc));
  }

  if (is_channel) {
    XLS_ASSIGN_OR_RETURN(CValue cv, GenerateIR_Expr(initializer, loc));

    if (!cv.type()->Is<CChannelType>()) {
      return absl::InvalidArgumentError(ErrorMessage(
          loc, "Channel declaration initialized to non-channel type: %s",
          cv.type()->debug_string()));
    }
    if (cv.lvalue() == nullptr) {
      return absl::InvalidArgumentError(
          ErrorMessage(loc,
                       "Channel declaration uninitialized (local channel "
                       "declarations should be static)"));
    }
    return cv;
  }

  if (ctype->Is<CReferenceType>()) {
    context().any_io_ops_requested = false;
    MaskIOOtherThanMemoryWritesGuard guard1(*this);
    MaskMemoryWritesGuard guard2(*this);
    std::shared_ptr<LValue> lval;
    XLS_ASSIGN_OR_RETURN(lval, CreateReferenceValue(initializer, loc));
    if (context().any_io_ops_requested) {
      return absl::UnimplementedError(ErrorMessage(
          loc, "References to side effecting operations not supported"));
    }
    return CValue(xls::BValue(), ctype,
                  /*disable_type_check=*/false, lval);
  }

  if (initializer != nullptr) {
    LValueModeGuard lvalue_mode(*this);

    XLS_ASSIGN_OR_RETURN(CValue cv, GenerateIR_Expr(initializer, loc));

    XLS_ASSIGN_OR_RETURN(init_val, GenTypeConvert(cv, ctype, loc));
    lvalue = cv.lvalue();
    if (ctype->Is<CPointerType>() && !lvalue) {
      return absl::UnimplementedError(
          ErrorMessage(loc,
                       "Initializer for pointer has no lvalue (unsupported "
                       "construct such as ternary?)"));
    }

    XLS_ASSIGN_OR_RETURN(bool type_contains_lval,
                         ctype->ContainsLValues(*this));

    if (type_contains_lval) {
      XLSCC_CHECK(lvalue != nullptr, loc);
      XLSCC_CHECK(!lvalue->is_null(), loc);
    }
  } else {
    if (error_on_uninitialized_) {
      return absl::InvalidArgumentError(ErrorMessage(
          loc, "Default initialization with error_on_uninitialized set"));
    }
    XLS_ASSIGN_OR_RETURN(init_val, CreateDefaultValue(ctype, loc));
    if (ctype->Is<CPointerType>()) {
      lvalue = std::make_shared<LValue>();
    }
  }

  return CValue(init_val, ctype, /*disable_type_check=*/false, lvalue);
}

absl::StatusOr<CValue> Translator::TranslateEnumConstantDecl(
    const clang::EnumConstantDecl* decl, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                       TranslateTypeFromClang(decl->getType(), loc));
  auto int_type = std::dynamic_pointer_cast<CIntType>(ctype);
  int64_t ext_val = decl->getInitVal().getExtValue();
  int width = ctype->GetBitWidth();
  auto val = int_type->is_signed() ? xls::Value(xls::SBits(ext_val, width))
                                   : xls::Value(xls::UBits(ext_val, width));
  xls::BValue init_val = context().fb->Literal(val, loc);
  return CValue(init_val, ctype, /*disable_type_check=*/false);
}

absl::StatusOr<CValue> Translator::GetOnReset(const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(const clang::VarDecl* on_reset_var_decl,
                       parser_->GetXlsccOnReset());
  auto on_reset_decl = static_cast<const clang::NamedDecl*>(on_reset_var_decl);

  if (!context().variables.contains(on_reset_decl)) {
    XLSCC_CHECK(!context().sf->static_values.contains(on_reset_decl), loc);
    ConstValue init_val(xls::Value(xls::UBits(1, 1)),
                        std::make_shared<CBoolType>());
    XLS_RETURN_IF_ERROR(DeclareStatic(on_reset_decl, init_val,
                                      /*init_lvalue=*/nullptr, loc,
                                      /*check_unique_ids=*/false));
  }
  context().sf->uses_on_reset = true;

  return context().variables.at(on_reset_decl);
}

absl::StatusOr<bool> Translator::DeclIsOnReset(const clang::NamedDecl* decl) {
  XLS_ASSIGN_OR_RETURN(const clang::VarDecl* on_reset_var_decl,
                       parser_->GetXlsccOnReset());
  return decl == static_cast<const clang::NamedDecl*>(on_reset_var_decl);
}

absl::StatusOr<CValue> Translator::GetIdentifier(const clang::NamedDecl* decl,
                                                 const xls::SourceInfo& loc,
                                                 bool record_access) {
  XLS_ASSIGN_OR_RETURN(bool is_on_reset, DeclIsOnReset(decl));
  if (is_on_reset) {
    return GetOnReset(loc);
  }

  if (record_access &&
      !context().variables_masked_by_assignment.contains(decl)) {
    // Record the access for analysis purposes
    if (!context().variables_accessed.contains(decl)) {
      context().variables_accessed[decl] = 1;
    } else {
      ++context().variables_accessed[decl];
    }
  }

  // CValue on file in the context?
  auto found = context().variables.find(decl);
  if (found != context().variables.end()) {
    return found->second;
  }

  // Is it an enum?
  auto enum_decl = dynamic_cast<const clang::EnumConstantDecl*>(decl);
  // Is this static/global?
  auto var_decl = dynamic_cast<const clang::VarDecl*>(decl);

  if (var_decl != nullptr && var_decl->isStaticDataMember() &&
      (!var_decl->getType().isConstQualified())) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Mutable static data members not implemented %s",
                     decl->getNameAsString()));
  }

  XLSCC_CHECK(var_decl == nullptr || !var_decl->isStaticLocal() ||
                  var_decl->getType().isConstQualified(),
              loc);

  XLSCC_CHECK(!(enum_decl && var_decl), loc);

  if (var_decl == nullptr && enum_decl == nullptr) {
    return absl::NotFoundError(
        ErrorMessage(loc, "Undeclared identifier %s", decl->getNameAsString()));
  }

  // Don't re-build the global value for each reference
  // They need to be built once for each Function[Builder]
  auto found_global = context().sf->global_values.find(decl);
  if (found_global != context().sf->global_values.end()) {
    return found_global->second;
  }

  const xls::SourceInfo global_loc = GetLoc(*decl);

  CValue value;

  if (enum_decl != nullptr) {
    XLS_ASSIGN_OR_RETURN(value,
                         TranslateEnumConstantDecl(enum_decl, global_loc));
  } else {
    XLSCC_CHECK(var_decl->hasGlobalStorage(), global_loc);

    XLSCC_CHECK(context().fb, global_loc);

    if (var_decl->getInit() != nullptr) {
      XLS_ASSIGN_OR_RETURN(value,
                           GenerateIR_Expr(var_decl->getInit(), global_loc));
      if (var_decl->isStaticLocal() || var_decl->isStaticDataMember()) {
        // Statics must have constant initialization
        if (!EvaluateBVal(value.rvalue(), global_loc).ok()) {
          return absl::InvalidArgumentError(
              ErrorMessage(loc, "Statics must have constant initializers"));
        }
      }
    } else {
      if (error_on_uninitialized_) {
        return absl::InvalidArgumentError(ErrorMessage(
            loc, "Variable '%s' uninitialized", decl->getNameAsString()));
      }
      XLS_ASSIGN_OR_RETURN(
          std::shared_ptr<CType> type,
          TranslateTypeFromClang(var_decl->getType(), global_loc));
      XLS_ASSIGN_OR_RETURN(xls::BValue bval,
                           CreateDefaultValue(type, global_loc));
      value = CValue(bval, type);
    }
  }
  context().sf->global_values[decl] = value;
  XLSCC_CHECK(!context().sf->declaration_order_by_name_.contains(decl), loc);
  context().sf->declaration_order_by_name_[decl] =
      ++context().sf->declaration_count;
  return value;
}

absl::StatusOr<CValue> Translator::PrepareRValueWithSelect(
    const CValue& lvalue, const CValue& rvalue,
    const xls::BValue& relative_condition, const xls::SourceInfo& loc) {
  CValue rvalue_to_use = rvalue;
  // Avoid generating unnecessary selects
  absl::StatusOr<xls::Value> const_var_cond = xls::Value(xls::UBits(1, 1));
  if (relative_condition.valid()) {
    const_var_cond = EvaluateBVal(relative_condition, loc, /*do_check=*/false);
  }
  if (const_var_cond.ok() && const_var_cond.value().IsAllOnes()) {
    return rvalue_to_use;
  }

  // Typical rvalue case
  if (!lvalue.type()->Is<CPointerType>()) {
    XLSCC_CHECK(rvalue.rvalue().valid(), loc);
    XLSCC_CHECK(lvalue.rvalue().valid(), loc);
    XLSCC_CHECK_EQ(rvalue.rvalue().GetType()->kind(),
                   lvalue.rvalue().GetType()->kind(), loc);

    XLSCC_CHECK_EQ(rvalue_to_use.lvalue(), rvalue.lvalue(), loc);
    auto cond_sel = context().fb->Select(relative_condition, rvalue.rvalue(),
                                         lvalue.rvalue(), loc);
    rvalue_to_use =
        CValue(cond_sel, rvalue_to_use.type(),
               /*disable_type_check=*/false, rvalue_to_use.lvalue());
  } else {
    // LValue (pointer) case
    XLSCC_CHECK_NE(rvalue_to_use.lvalue(), nullptr, loc);
    XLSCC_CHECK_NE(lvalue.lvalue(), nullptr, loc);

    auto select_lvalue = std::make_shared<LValue>(
        relative_condition, rvalue_to_use.lvalue(), lvalue.lvalue());

    rvalue_to_use = CValue(xls::BValue(), rvalue_to_use.type(),
                           /*disable_type_check=*/false, select_lvalue);
  }
  return rvalue_to_use;
}

absl::Status Translator::Assign(const clang::NamedDecl* lvalue,
                                const CValue& rvalue,
                                const xls::SourceInfo& loc,
                                bool force_no_lvalue_assign) {
  context().any_side_effects_requested = true;
  if (context().mask_side_effects || context().mask_assignments) {
    return absl::OkStatus();
  }

  // Don't allow assignment to globals. This doesn't work because
  //  each function has a different FunctionBuilder.
  if (auto var_decl = dynamic_cast<const clang::VarDecl*>(lvalue);
      var_decl != nullptr) {
    if (var_decl->hasGlobalStorage() && (!var_decl->isStaticLocal()) &&
        (!var_decl->isStaticDataMember())) {
      return absl::UnimplementedError(ErrorMessage(
          loc, "Assignments to global variables not supported for %s",
          lvalue->getNameAsString()));
    }
  }

  // An assignment doesn't count as access
  XLS_ASSIGN_OR_RETURN(CValue found,
                       GetIdentifier(lvalue, loc, /*record_access=*/false));

  context().variables_masked_by_assignment.insert(lvalue);

  if (found.type()->Is<CReferenceType>()) {
    XLSCC_CHECK_NE(found.lvalue(), nullptr, loc);
    return Assign(found.lvalue(), rvalue, loc);
  }

  if (found.type()->Is<CPointerType>()) {
    // If re-assigning the pointer, then treat it as usual
    if (rvalue.type()->Is<CPointerType>()) {
      if (rvalue.lvalue() == nullptr) {
        return absl::UnimplementedError(
            ErrorMessage(loc,
                         "Initializer for pointer has no lvalue "
                         "(unsupported construct such as ternary?)"));
      }
    } else {
      XLSCC_CHECK(rvalue.rvalue().valid(), loc);

      // Otherwise, assign to the lvalue pointed to. This is necessary for
      // array-slicing, for example: void foo(int arr[4])
      // ...
      // foo(&arr[2]); // Modifies indices 2-6
      if (found.lvalue() == nullptr) {
        return absl::UnimplementedError(ErrorMessage(
            loc, "Pointer is uninitialized (no lvalue associated)"));
      }
      return Assign(found.lvalue(), rvalue, loc);
    }
  } else {
    // Not a pointer assignment
    XLSCC_CHECK(rvalue.rvalue().valid(), loc);
  }

  if (*found.type() != *rvalue.type()) {
    lvalue->dump();
    return absl::InvalidArgumentError(ErrorMessage(
        loc, "Cannot assign rvalue of type %s to lvalue of type %s",
        std::string(*rvalue.type()), std::string(*found.type())));
  }

  XLSCC_CHECK(context().variables.contains(lvalue), loc);

  XLS_ASSIGN_OR_RETURN(bool type_contains_lval,
                       rvalue.type()->ContainsLValues(*this));

  // TODO(seanhaskell): Remove special 'this' handling
  if (!type_contains_lval || force_no_lvalue_assign) {
    context().variables.at(lvalue) = CValue(rvalue.rvalue(), rvalue.type());
  } else {
    XLSCC_CHECK(rvalue.lvalue() != nullptr, loc);
    XLSCC_CHECK(!rvalue.lvalue()->is_null(), loc);

    XLS_RETURN_IF_ERROR(ValidateLValue(rvalue.lvalue(), loc));

    context().variables.at(lvalue) = rvalue;
  }

  return absl::OkStatus();
}

int64_t Translator::ArrayBValueWidth(xls::BValue array_bval) {
  xls::Type* type = array_bval.node()->GetType();
  CHECK(type->IsArray());
  return type->AsArrayOrDie()->size();
}

absl::StatusOr<int64_t> Translator::EvaluateBValInt64(
    xls::BValue bval, const xls::SourceInfo& loc, bool do_check) {
  absl::StatusOr<xls::Value> val_result = EvaluateBVal(bval, loc, do_check);
  if (!val_result.ok()) {
    return val_result.status();
  }
  xls::Value val = val_result.value();
  XLSCC_CHECK(val.IsBits(), loc);
  XLS_ASSIGN_OR_RETURN(int64_t const_index, val.bits().ToInt64());
  return const_index;
}

absl::StatusOr<xls::BValue> Translator::UpdateArraySlice(
    xls::BValue array_to_update, xls::BValue start_index,
    xls::BValue slice_to_write, const xls::SourceInfo& loc) {
  const int64_t total_width = ArrayBValueWidth(array_to_update);
  const int64_t slice_width = ArrayBValueWidth(slice_to_write);

  absl::StatusOr<int64_t> const_index_result =
      EvaluateBValInt64(start_index, loc, /*do_check=*/false);

  // Constant index case: use array concatenation
  if (const_index_result.ok()) {
    const int64_t const_index = const_index_result.value();

    if (total_width < (const_index + slice_width)) {
      return absl::OutOfRangeError(
          ErrorMessage(loc, "Array slice out of bounds"));
    }

    int64_t remaining_width = total_width;

    std::vector<xls::BValue> parts;

    if (const_index > 0) {
      parts.push_back(context().fb->ArraySlice(
          array_to_update, context().fb->Literal(xls::SBits(0, 32), loc),
          const_index, loc));
      remaining_width -= const_index;
    }

    parts.push_back(slice_to_write);
    remaining_width -= slice_width;

    if (remaining_width > 0) {
      parts.push_back(context().fb->ArraySlice(
          array_to_update,
          context().fb->Literal(xls::SBits(total_width - remaining_width, 32),
                                loc),
          remaining_width, loc));
    }

    xls::BValue updated_array_const = context().fb->ArrayConcat(parts, loc);
    const int64_t updated_array_width = ArrayBValueWidth(updated_array_const);

    XLSCC_CHECK_EQ(total_width, updated_array_width, loc);

    XLSCC_CHECK(updated_array_const.valid(), loc);
    return updated_array_const;
  }

  xls::BValue updated_array_dynamic = array_to_update;

  // Dynamic index case: update elements one by one
  if (total_width <= slice_width) {
    return absl::OutOfRangeError(
        ErrorMessage(loc, "Array slice sure to be out of bounds"));
  }

  for (uint64_t si = 0; si < slice_width; ++si) {
    xls::BValue slice_idx_bval =
        context().fb->Literal(xls::UBits(si, start_index.BitCountOrDie()), loc);
    XLSCC_CHECK(slice_idx_bval.valid(), loc);
    xls::BValue index_bval =
        context().fb->Add(start_index, slice_idx_bval, loc);
    XLSCC_CHECK(index_bval.valid(), loc);
    xls::BValue slice_bval =
        context().fb->ArrayIndex(slice_to_write, {slice_idx_bval}, loc);
    XLSCC_CHECK(slice_bval.valid(), loc);
    updated_array_dynamic = context().fb->ArrayUpdate(
        updated_array_dynamic, slice_bval, {index_bval}, loc);
  }

  XLSCC_CHECK(updated_array_dynamic.valid(), loc);
  return updated_array_dynamic;
}

absl::Status Translator::Assign(std::shared_ptr<LValue> lvalue,
                                const CValue& rvalue,
                                const xls::SourceInfo& loc) {
  if (!lvalue->is_select()) {
    return Assign(lvalue->leaf(), rvalue, loc);
  }

  // Apply the select condition to assign to true expression
  {
    PushContextGuard condition_guard(*this, loc);
    XLS_RETURN_IF_ERROR(and_condition(lvalue->cond(), loc));
    XLS_RETURN_IF_ERROR(Assign(lvalue->lvalue_true(), rvalue, loc));
  }
  // Apply ! the select condition to assign to false expression
  {
    xls::BValue sel_cond = context().fb->Not(lvalue->cond(), loc);
    PushContextGuard condition_guard(*this, loc);
    XLS_RETURN_IF_ERROR(and_condition(sel_cond, loc));
    XLS_RETURN_IF_ERROR(Assign(lvalue->lvalue_false(), rvalue, loc));
  }

  return absl::OkStatus();
}

absl::Status Translator::AssignMember(const clang::NamedDecl* lvalue,
                                      const clang::NamedDecl* member,
                                      const CValue& rvalue,
                                      const xls::SourceInfo& loc) {
  CValue struct_prev_val;

  XLS_ASSIGN_OR_RETURN(struct_prev_val, GetIdentifier(lvalue, loc));

  if (struct_prev_val.type()->Is<CReferenceType>()) {
    XLSCC_CHECK_NE(struct_prev_val.lvalue().get(), nullptr, loc);
    XLSCC_CHECK(!struct_prev_val.lvalue()->is_null(), loc);
    XLS_ASSIGN_OR_RETURN(struct_prev_val,
                         GenerateIR_Expr(struct_prev_val.lvalue(), loc));
  }

  XLS_ASSIGN_OR_RETURN(auto resolved_type,
                       ResolveTypeInstance(struct_prev_val.type()));

  if (auto sitype = std::dynamic_pointer_cast<CStructType>(resolved_type)) {
    XLS_ASSIGN_OR_RETURN(CValue newval, StructUpdate(struct_prev_val, rvalue,
                                                     member, *sitype, loc));

    return Assign(lvalue, newval, loc);
  }
  return absl::UnimplementedError(
      ErrorMessage(loc,
                   "Unimplemented fielddecl assignment to "
                   "non-struct typed lvalue of type %s",
                   string(*struct_prev_val.type())));
}

absl::Status Translator::AssignMember(const clang::Expr* lvalue,
                                      const clang::NamedDecl* member,
                                      const CValue& rvalue,
                                      const xls::SourceInfo& loc) {
  CValue struct_prev_val;

  XLS_ASSIGN_OR_RETURN(struct_prev_val, GenerateIR_Expr(lvalue, loc));

  if (struct_prev_val.type()->Is<CReferenceType>()) {
    XLSCC_CHECK_NE(struct_prev_val.lvalue().get(), nullptr, loc);
    XLSCC_CHECK(!struct_prev_val.lvalue()->is_null(), loc);
    XLS_ASSIGN_OR_RETURN(struct_prev_val,
                         GenerateIR_Expr(struct_prev_val.lvalue(), loc));
  }

  XLS_ASSIGN_OR_RETURN(auto resolved_type,
                       ResolveTypeInstance(struct_prev_val.type()));

  if (auto sitype = std::dynamic_pointer_cast<CStructType>(resolved_type)) {
    XLS_ASSIGN_OR_RETURN(CValue newval, StructUpdate(struct_prev_val, rvalue,
                                                     member, *sitype, loc));

    return Assign(lvalue, newval, loc);
  }
  return absl::UnimplementedError(
      ErrorMessage(loc,
                   "Unimplemented fielddecl assignment to "
                   "non-struct typed lvalue of type %s",
                   string(*struct_prev_val.type())));
}

absl::Status Translator::Assign(const clang::Expr* lvalue, const CValue& rvalue,
                                const xls::SourceInfo& loc) {
  // Assign to a variable using the identifier it was declared with
  // foo = rvalue
  if (auto cast = clang::dyn_cast<const clang::DeclRefExpr>(lvalue)) {
    const clang::NamedDecl* named = cast->getFoundDecl();
    return Assign(named, rvalue, loc);
  }
  if (auto cast = clang::dyn_cast<const clang::ParenExpr>(lvalue)) {
    // Assignment to a parenthetical expression
    // (...) = rvalue
    return Assign(cast->getSubExpr(), rvalue, loc);
    // cast<type>(...) = rvalue
  }
  if (auto cast = clang::dyn_cast<const clang::CastExpr>(lvalue)) {
    // Don't generate pointer errors for C++ "this" keyword
    IgnorePointersGuard ignore_pointers(*this);
    if (clang::isa<clang::CXXThisExpr>(cast->getSubExpr())) {
      ignore_pointers.enable();
    }

    XLS_ASSIGN_OR_RETURN(CValue sub, GenerateIR_Expr(cast->getSubExpr(), loc));

    auto from_arr_type = std::dynamic_pointer_cast<CArrayType>(sub.type());

    // Inheritance
    XLS_ASSIGN_OR_RETURN(shared_ptr<CType> to_type,
                         TranslateTypeFromClang(cast->getType(), loc));

    XLS_ASSIGN_OR_RETURN(ResolvedInheritance inheritance,
                         ResolveInheritance(sub.type(), to_type));

    CValue adjusted_rvalue = rvalue;

    // Are we casting to a derived class?
    if (inheritance.base_field != nullptr) {
      XLSCC_CHECK(inheritance.resolved_struct != nullptr, loc);
      XLSCC_CHECK((*rvalue.type()) == *inheritance.base_field->type(), loc);
      XLS_ASSIGN_OR_RETURN(
          CValue updated_derived,
          StructUpdate(sub, rvalue, inheritance.base_field_name,
                       *inheritance.resolved_struct, loc));
      adjusted_rvalue = updated_derived;
    }

    return Assign(cast->getSubExpr(), adjusted_rvalue, loc);
  }
  if (clang::isa<clang::MaterializeTemporaryExpr>(lvalue)) {
    // This happens when copy constructors with non-const reference inputs are
    // invoked. class Temporary { Temporary() { } }; class Blah { Blah(Temporary
    // &in) { } }; Blah x(Temporary());
    // Ignore assignment to temporaries,
    // but still generate the sub-expression in case it has side-effects.
    XLS_RETURN_IF_ERROR(GenerateIR_Expr(lvalue, loc).status());
    return absl::OkStatus();
  }
  if (auto* cast = clang::dyn_cast<const clang::ArraySubscriptExpr>(lvalue)) {
    // Assign to an array element
    // (...)[index] = rvalue
    XLS_ASSIGN_OR_RETURN(CValue arr_val, GenerateIR_Expr(cast->getBase(), loc));
    if (!arr_val.type()->Is<CArrayType>()) {
      return absl::UnimplementedError(
          ErrorMessage(loc,
                       "Only array subscript assignments directly to "
                       "arrays supported (not pointers, yet)"));
    }
    auto arr_type = arr_val.type()->As<CArrayType>();
    XLS_ASSIGN_OR_RETURN(CValue idx_val, GenerateIR_Expr(cast->getIdx(), loc));
    if (*rvalue.type() != *arr_type->GetElementType()) {
      return absl::InvalidArgumentError(ErrorMessage(
          loc, "Cannot assign rvalue of type %s to element of array of type %s",
          std::string(*rvalue.type()),
          std::string(*arr_type->GetElementType())));
    }
    XLS_ASSIGN_OR_RETURN(bool element_has_lvalues,
                         arr_type->GetElementType()->ContainsLValues(*this));
    XLSCC_CHECK(!element_has_lvalues, loc);
    XLSCC_CHECK(rvalue.rvalue().valid(), loc);
    auto arr_rvalue =
        CValue(context().fb->ArrayUpdate(arr_val.rvalue(), rvalue.rvalue(),
                                         {idx_val.rvalue()}, loc),
               arr_val.type());
    return Assign(cast->getBase(), arr_rvalue, loc);
  }
  if (auto member_expr = clang::dyn_cast<const clang::MemberExpr>(lvalue)) {
    // Assign to a struct element
    // (...).member = rvalue
    clang::FieldDecl* member =
        clang::dyn_cast<clang::FieldDecl>(member_expr->getMemberDecl());

    if (member == nullptr) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Unimplemented assignment to lvalue member kind %s",
                       member_expr->getMemberDecl()->getDeclKindName()));
    }

    if (member->getType()->isLValueReferenceType()) {
      XLS_ASSIGN_OR_RETURN(CValue member_val,
                           GenerateIR_Expr(member_expr, loc));
      XLSCC_CHECK(member_val.type()->Is<CReferenceType>(), loc);
      XLSCC_CHECK_NE(member_val.lvalue().get(), nullptr, loc);
      XLSCC_CHECK_NE(member_val.lvalue()->leaf(), nullptr, loc);
      return Assign(member_val.lvalue()->leaf(), rvalue, loc);
    }

    return AssignMember(member_expr->getBase(), member, rvalue, loc);
  }
  if (auto uop = clang::dyn_cast<const clang::UnaryOperator>(lvalue)) {
    if (uop->getOpcode() == clang::UnaryOperatorKind::UO_AddrOf) {
      const clang::ArraySubscriptExpr* subscript_sub_expr =
          clang::dyn_cast<const clang::ArraySubscriptExpr>(uop->getSubExpr());
      if (subscript_sub_expr == nullptr) {
        return absl::UnimplementedError(ErrorMessage(
            loc, "Only assignment to array slices supported via pointers"));
      }
      XLS_ASSIGN_OR_RETURN(CValue base_cv,
                           GenerateIR_Expr(subscript_sub_expr->getBase(), loc));
      XLSCC_CHECK(base_cv.type()->Is<CArrayType>(), loc);
      XLS_ASSIGN_OR_RETURN(CValue index_cv,
                           GenerateIR_Expr(subscript_sub_expr->getIdx(), loc));
      XLSCC_CHECK(index_cv.type()->Is<CIntType>(), loc);

      XLS_ASSIGN_OR_RETURN(xls::BValue updated_array,
                           UpdateArraySlice(base_cv.rvalue(), index_cv.rvalue(),
                                            rvalue.rvalue(), loc));

      return Assign(subscript_sub_expr->getBase(),
                    CValue(updated_array, base_cv.type()), loc);
    }
    if (uop->getOpcode() != clang::UnaryOperatorKind::UO_Deref) {
      return absl::UnimplementedError(
          ErrorMessage(loc,
                       "Unimplemented assignment to unary operator lvalue "
                       "with opcode %i",
                       uop->getOpcode()));
    }

    // Deref is the pointer dereferencing operator: *ptr
    // We simply ignore this for "this", so *this just evaluates to the
    //  "this" BValue from the TranslationContext
    if (clang::isa<clang::CXXThisExpr>(uop->getSubExpr())) {
      XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl, GetThisDecl(loc));
      return Assign(this_decl, rvalue, loc);
    }

    return absl::UnimplementedError(absl::StrFormat(
        "Unsupported assignment to dereference of statement of class %i at "
        "%s",
        static_cast<int>(uop->getSubExpr()->getStmtClass()), LocString(loc)));
  }
  if (clang::isa<clang::CXXThisExpr>(lvalue)) {
    XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl, GetThisDecl(loc));
    return Assign(this_decl, rvalue, loc);
  }
  if (auto cond = clang::dyn_cast<const clang::ConditionalOperator>(lvalue)) {
    XLS_ASSIGN_OR_RETURN(
        shared_ptr<CType> result_type,
        TranslateTypeFromClang(cond->getType().getCanonicalType(), loc));

    if (!result_type->Is<CPointerType>() || result_type->Is<CReferenceType>()) {
      return absl::UnimplementedError(
          ErrorMessage(loc,
                       "Ternaries in lvalues only supported for pointers or "
                       "references, type used is %s",
                       std::string(*result_type)));
    }

    XLS_ASSIGN_OR_RETURN(
        CValue lcv,
        Generate_TernaryOp(result_type, cond->getCond(), cond->getTrueExpr(),
                           cond->getFalseExpr(), loc));

    XLSCC_CHECK_NE(lcv.lvalue().get(), nullptr, loc);
    return Assign(lcv.lvalue(), rvalue, loc);
  }
  if (auto call = clang::dyn_cast<const clang::CallExpr>(lvalue)) {
    XLS_ASSIGN_OR_RETURN(
        IOOpReturn ret,
        InterceptIOOp(call, GetLoc(*call), /*assignment_value=*/rvalue));
    // If this call is an IO op, then return the IO value, rather than
    //  generating the call.
    if (!ret.generate_expr) {
      return absl::OkStatus();
    }
  }
  return absl::UnimplementedError(
      ErrorMessage(loc, "Unimplemented assignment to lvalue of type %s",
                   lvalue->getStmtClassName()));
}

absl::Status Translator::DeclareVariable(const clang::NamedDecl* lvalue,
                                         const CValue& rvalue,
                                         const xls::SourceInfo& loc,
                                         bool check_unique_ids) {
  if (context().variables.contains(lvalue)) {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Declaration '%s' duplicated\n", lvalue->getNameAsString()));
  }

  if (check_unique_ids) {
    if (unique_decl_ids_.contains(lvalue)) {
      return absl::InternalError(
          ErrorMessage(loc, "Code assumes NamedDecls are unique, but %s isn't",
                       lvalue->getNameAsString()));
    }
  }
  unique_decl_ids_.insert(lvalue);

  context().sf->declaration_order_by_name_[lvalue] =
      ++context().sf->declaration_count;

  XLS_RETURN_IF_ERROR(ValidateLValue(rvalue.lvalue(), loc));

  context().variables[lvalue] = rvalue;
  return absl::OkStatus();
}

absl::Status Translator::DeclareStatic(
    const clang::NamedDecl* lvalue, const ConstValue& init,
    const std::shared_ptr<LValue>& init_lvalue, const xls::SourceInfo& loc,
    bool check_unique_ids) {
  XLSCC_CHECK(!context().sf->static_values.contains(lvalue) ||
                  context().sf->static_values.at(lvalue) == init,
              loc);

  // Static declarations containing only LValues
  if (init.rvalue().kind() == xls::ValueKind::kInvalid) {
    return DeclareVariable(lvalue,
                           CValue(xls::BValue(), init.type(),
                                  /*disable_type_check=*/false, init_lvalue),
                           loc, check_unique_ids);
  }

  context().sf->declaration_order_by_name_[lvalue] =
      ++context().sf->declaration_count;

  context().sf->static_values[lvalue] = init;

  // Mangle the name since statics with the same name may occur in different
  // contexts
  std::string xls_name = XLSNameMangle(clang::GlobalDecl(lvalue));

  XLS_ASSIGN_OR_RETURN(xls::Type * xls_type,
                       TranslateTypeToXLS(init.type(), loc));

  const xls::BValue bval = context().fb->Param(xls_name, xls_type, loc);

  XLSCC_CHECK(bval.valid(), loc);

  SideEffectingParameter side_effecting_param;
  side_effecting_param.type = SideEffectingParameterType::kStatic;
  side_effecting_param.param_name = bval.node()->As<xls::Param>()->GetName();
  side_effecting_param.static_value = lvalue;
  context().sf->side_effecting_parameters.push_back(side_effecting_param);

  const CValue init_cval(bval, init.type(), /*disable_type_check=*/false,
                         init_lvalue);

  return DeclareVariable(lvalue, init_cval, loc, check_unique_ids);
}

absl::StatusOr<CValue> Translator::Generate_Synthetic_ByOne(
    xls::Op xls_op, bool is_pre, CValue sub_value, const clang::Expr* sub_expr,
    const xls::SourceInfo& loc) {
  auto sub_type = std::dynamic_pointer_cast<CIntType>(sub_value.type());
  const int width = sub_type->width();
  xls::BValue literal_one = context().fb->Literal(
      sub_type->is_signed() ? xls::SBits(1, width) : xls::UBits(1, width));
  xls::BValue result_val =
      context().fb->AddBinOp(xls_op, sub_value.rvalue(), literal_one, loc);
  // No extra bits because this is only for built-ins
  std::shared_ptr<CType> result_type = sub_value.type();

  XLS_RETURN_IF_ERROR(Assign(sub_expr, CValue(result_val, result_type), loc));

  xls::BValue ret = is_pre ? result_val : sub_value.rvalue();
  return CValue(ret, result_type);
}

absl::StatusOr<CValue> Translator::Generate_UnaryOp(
    const clang::UnaryOperator* uop, const xls::SourceInfo& loc) {
  auto clang_op = uop->getOpcode();

  XLS_ASSIGN_OR_RETURN(CValue lhs_cv, GenerateIR_Expr(uop->getSubExpr(), loc));
  XLS_ASSIGN_OR_RETURN(
      shared_ptr<CType> result_type,
      TranslateTypeFromClang(uop->getType().getCanonicalType(), loc));

  if (clang_op == clang::UnaryOperatorKind::UO_AddrOf) {
    auto result_pointer_type =
        std::dynamic_pointer_cast<CPointerType>(result_type);
    XLSCC_CHECK_NE(result_pointer_type.get(), nullptr, loc);

    if (context().lvalue_mode) {
      // Include & in the lvalue expression, so that Assign()
      // can just look for that
      return CValue(xls::BValue(), result_pointer_type,
                    /*disable_type_check=*/false,
                    std::make_shared<LValue>(uop));
    }

    const clang::Expr* sub_expr = uop->getSubExpr();
    auto array_subscript_expr =
        clang::dyn_cast<const clang::ArraySubscriptExpr>(sub_expr);
    if (array_subscript_expr == nullptr) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Address of sub expression of class %i",
                       static_cast<int>(sub_expr->getStmtClass())));
    }

    const clang::Expr* base_expr = array_subscript_expr->getBase();
    XLS_ASSIGN_OR_RETURN(CValue base_cv, GenerateIR_Expr(base_expr, loc));

    if (!base_cv.type()->Is<CArrayType>()) {
      return absl::InvalidArgumentError(
          ErrorMessage(loc,
                       "Address-of (&) operator "
                       "only supported on arrays, for array slicing"));
    }

    XLS_ASSIGN_OR_RETURN(CValue array_idx_cv,
                         GenerateIR_Expr(array_subscript_expr->getIdx(), loc));
    auto array_int_type =
        std::dynamic_pointer_cast<CIntType>(array_idx_cv.type());
    if (array_int_type == nullptr) {
      return absl::InvalidArgumentError(
          ErrorMessage(loc, "Array index must be an integer"));
    }

    std::shared_ptr<CType> pointee_type = result_pointer_type->GetPointeeType();

    XLSCC_CHECK(*lhs_cv.type() == *pointee_type, loc);

    xls::BValue array_slice_in = base_cv.rvalue();

    XLSCC_CHECK(array_slice_in.GetType()->IsArray(), loc);

    // Out of bounds slices wrap around
    const int64_t array_slice_in_size =
        array_slice_in.GetType()->AsArrayOrDie()->size();

    xls::BValue sliced_array = context().fb->ArraySlice(
        array_slice_in, array_idx_cv.rvalue(), array_slice_in_size, loc);

    XLSCC_CHECK(sliced_array.GetType()->IsArray(), loc);

    return CValue(sliced_array, std::make_shared<CArrayType>(
                                    pointee_type, array_slice_in_size));
  }

  if (auto ref_type = dynamic_cast<const CReferenceType*>(lhs_cv.type().get());
      ref_type != nullptr) {
    XLSCC_CHECK_NE(lhs_cv.lvalue(), nullptr, loc);
    XLSCC_CHECK_NE(lhs_cv.lvalue()->leaf(), nullptr, loc);
    XLS_ASSIGN_OR_RETURN(lhs_cv, GenerateIR_Expr(lhs_cv.lvalue()->leaf(), loc));
  }

  XLS_ASSIGN_OR_RETURN(shared_ptr<CType> resolved_type,
                       ResolveTypeInstance(result_type));

  XLS_ASSIGN_OR_RETURN(xls::BValue lhs_cvc,
                       GenTypeConvert(lhs_cv, result_type, loc));
  CValue lhs_cvcv(lhs_cvc, lhs_cv.type());

  if (auto result_int_type = std::dynamic_pointer_cast<CIntType>(result_type)) {
    switch (clang_op) {
      case clang::UnaryOperatorKind::UO_Minus:
        return CValue(
            context().fb->AddUnOp(xls::Op::kNeg, lhs_cvcv.rvalue(), loc),
            result_type);
      case clang::UnaryOperatorKind::UO_Plus:
        return lhs_cvcv;
      case clang::UnaryOperatorKind::UO_LNot:
      case clang::UnaryOperatorKind::UO_Not:
        return CValue(
            context().fb->AddUnOp(xls::Op::kNot, lhs_cvcv.rvalue(), loc),
            result_type);
      case clang::UnaryOperatorKind::UO_PreInc:
        return Generate_Synthetic_ByOne(xls::Op::kAdd, true, lhs_cvcv,
                                        uop->getSubExpr(), loc);
      case clang::UnaryOperatorKind::UO_PreDec:
        return Generate_Synthetic_ByOne(xls::Op::kSub, true, lhs_cvcv,
                                        uop->getSubExpr(), loc);
      case clang::UnaryOperatorKind::UO_PostInc:
        return Generate_Synthetic_ByOne(xls::Op::kAdd, false, lhs_cvcv,
                                        uop->getSubExpr(), loc);
      case clang::UnaryOperatorKind::UO_PostDec:
        return Generate_Synthetic_ByOne(xls::Op::kSub, false, lhs_cvcv,
                                        uop->getSubExpr(), loc);
      default:
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unimplemented unary operator %i", clang_op));
    }
  } else if (clang_op == clang::UnaryOperatorKind::UO_Deref &&
             (std::dynamic_pointer_cast<CStructType>(resolved_type) ||
              std::dynamic_pointer_cast<CBitsType>(resolved_type))) {
    // We don't support pointers so we don't care about this.
    // It's needed for *this
    XLSCC_CHECK(clang::isa<clang::CXXThisExpr>(uop->getSubExpr()), loc);
    return lhs_cvcv;
  } else if (auto result_int_type =
                 std::dynamic_pointer_cast<CBoolType>(result_type)) {
    switch (clang_op) {
      case clang::UnaryOperatorKind::UO_LNot:
        return CValue(
            context().fb->AddUnOp(xls::Op::kNot, lhs_cvcv.rvalue(), loc),
            result_type);
      default:
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unimplemented unary operator %i", clang_op));
    }
  } else {
    return absl::UnimplementedError(
        ErrorMessage(loc,
                     "Unary operators on types other than builtin-int "
                     "unimplemented for type %s",
                     std::string(*result_type)));
  }
}

absl::StatusOr<CValue> Translator::Generate_BinaryOp(
    clang::BinaryOperatorKind clang_op, bool is_assignment,
    std::shared_ptr<CType> result_type, const clang::Expr* lhs,
    const clang::Expr* rhs, const xls::SourceInfo& loc) {
  CValue result;
  {
    // Don't reduce operands to logical boolean operators to 1 bit.
    std::shared_ptr<CType> input_type = result_type;
    if (input_type->Is<CBoolType>()) {
      XLS_ASSIGN_OR_RETURN(input_type,
                           TranslateTypeFromClang(lhs->getType(), loc));
    }

    if (clang_op == clang::BinaryOperatorKind::BO_Comma) {
      CValue lhs_cv;
      XLS_ASSIGN_OR_RETURN(lhs_cv, GenerateIR_Expr(lhs, loc));
    }
    XLS_ASSIGN_OR_RETURN(
        xls::Op xls_op,
        XLSOpcodeFromClang(clang_op, *input_type, *result_type, loc));

    CValue rhs_cv;
    {
      // For pointer assignments
      LValueModeGuard lvalue_mode(*this);
      XLS_ASSIGN_OR_RETURN(rhs_cv, GenerateIR_Expr(rhs, loc));
    }

    XLS_ASSIGN_OR_RETURN(xls::BValue rhs_cvc,
                         GenTypeConvert(rhs_cv, input_type, loc));
    CValue rhs_cvcv(rhs_cvc, input_type, /*disable_type_check=*/false,
                    rhs_cv.lvalue());

    result = rhs_cvcv;

    if (xls_op != xls::Op::kIdentity) {
      XLS_ASSIGN_OR_RETURN(CValue lhs_cv, GenerateIR_Expr(lhs, loc));

      if (auto ref_type =
              dynamic_cast<const CReferenceType*>(lhs_cv.type().get());
          ref_type != nullptr) {
        XLSCC_CHECK_NE(lhs_cv.lvalue(), nullptr, loc);
        XLSCC_CHECK_NE(lhs_cv.lvalue()->leaf(), nullptr, loc);
        XLS_ASSIGN_OR_RETURN(lhs_cv,
                             GenerateIR_Expr(lhs_cv.lvalue()->leaf(), loc));
      }

      XLS_ASSIGN_OR_RETURN(xls::BValue lhs_cvc,
                           GenTypeConvert(lhs_cv, input_type, loc));
      CValue lhs_cvcv(lhs_cvc, lhs_cv.type());

      if (xls::IsOpClass<xls::CompareOp>(xls_op)) {
        result = CValue(context().fb->AddCompareOp(xls_op, lhs_cvcv.rvalue(),
                                                   rhs_cvcv.rvalue(), loc),
                        result_type);
      } else if (xls::IsOpClass<xls::ArithOp>(xls_op)) {
        result = CValue(context().fb->AddArithOp(
                            xls_op, lhs_cvcv.rvalue(), rhs_cvcv.rvalue(),
                            result_type->GetBitWidth(), loc),
                        result_type);
      } else if (xls::IsOpClass<xls::BinOp>(xls_op)) {
        result = CValue(context().fb->AddBinOp(xls_op, lhs_cvcv.rvalue(),
                                               rhs_cvcv.rvalue(), loc),
                        result_type);
      } else if (xls::IsOpClass<xls::NaryOp>(xls_op)) {
        result = CValue(
            context().fb->AddNaryOp(
                xls_op,
                std::vector<xls::BValue>{lhs_cvcv.rvalue(), rhs_cvcv.rvalue()},
                loc),
            result_type);

      } else {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Internal error: unknown XLS op class"));
      }
    }
  }

  if (is_assignment) {
    XLS_RETURN_IF_ERROR(Assign(lhs, result, loc));
  }
  return result;
}

absl::StatusOr<CValue> Translator::Generate_TernaryOp(
    std::shared_ptr<CType> result_type, const clang::Expr* cond_expr,
    const clang::Expr* true_expr, const clang::Expr* false_expr,
    const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(CValue sel_val, GenerateIR_Expr(cond_expr, loc));
  XLSCC_CHECK(sel_val.type()->Is<CBoolType>(), loc);
  XLSCC_CHECK(sel_val.rvalue().valid(), loc);

  CValue true_cv, false_cv;

  {
    PushContextGuard context_guard(*this, sel_val.rvalue(), loc);
    XLS_ASSIGN_OR_RETURN(true_cv, GenerateIR_Expr(true_expr, loc));
  }
  {
    PushContextGuard context_guard(*this, context().fb->Not(sel_val.rvalue()),
                                   loc);
    XLS_ASSIGN_OR_RETURN(false_cv, GenerateIR_Expr(false_expr, loc));
  }

  // If one is a reference but not both, then make a reference to the other
  if (true_cv.type()->Is<CReferenceType>() !=
      false_cv.type()->Is<CReferenceType>()) {
    if (!true_cv.type()->Is<CReferenceType>()) {
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<LValue> true_lval,
                           CreateReferenceValue(true_expr, loc));
      true_cv = CValue(xls::BValue(),
                       std::make_shared<CReferenceType>(true_cv.type()),
                       /*disable_type_check=*/false, true_lval);
    }
    if (!false_cv.type()->Is<CReferenceType>()) {
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<LValue> false_lval,
                           CreateReferenceValue(false_expr, loc));
      false_cv = CValue(xls::BValue(),
                        std::make_shared<CReferenceType>(false_cv.type()),
                        /*disable_type_check=*/false, false_lval);
    }
  }

  if (result_type->Is<CPointerType>()) {
    if (context().lvalue_mode) {
      XLSCC_CHECK_NE(true_cv.lvalue(), nullptr, loc);
      XLSCC_CHECK_NE(false_cv.lvalue(), nullptr, loc);
      auto select_lvalue = std::make_shared<LValue>(
          sel_val.rvalue(), true_cv.lvalue(), false_cv.lvalue());
      return CValue(xls::BValue(), result_type, /*disable_type_check=*/false,
                    select_lvalue);
    }
    // RValue mode
    XLS_RETURN_IF_ERROR(
        MinSizeArraySlices(true_cv, false_cv, result_type, loc));
  }

  return Generate_TernaryOp(sel_val.rvalue(), true_cv, false_cv, result_type,
                            loc);
}

absl::StatusOr<CValue> Translator::Generate_TernaryOp(
    xls::BValue cond, CValue true_cv, CValue false_cv,
    std::shared_ptr<CType> result_type, const xls::SourceInfo& loc) {
  if (true_cv.lvalue() != nullptr && false_cv.lvalue() != nullptr) {
    XLS_ASSIGN_OR_RETURN(bool result_has_lval,
                         result_type->ContainsLValues(*this));
    if ((result_type->Is<CInstantiableTypeAlias>() ||
         result_type->Is<CStructType>()) &&
        result_has_lval) {
      // Could be mixed rvalue and lvalue
      return absl::UnimplementedError(
          ErrorMessage(loc, "Ternary on structs with LValues"));
    }
    XLSCC_CHECK(*true_cv.type() == *false_cv.type(), loc);
    return CValue(
        xls::BValue(), true_cv.type(), /*disable_type_check=*/false,
        std::make_shared<LValue>(cond, true_cv.lvalue(), false_cv.lvalue()));
  }
  if (true_cv.lvalue() != nullptr || false_cv.lvalue() != nullptr) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Ternary on mixed LValue type and RValue types"));
  }

  XLS_ASSIGN_OR_RETURN(xls::BValue true_val,
                       GenTypeConvert(true_cv, result_type, loc));
  XLS_ASSIGN_OR_RETURN(xls::BValue false_val,
                       GenTypeConvert(false_cv, result_type, loc));

  xls::BValue ret_val = context().fb->Select(cond, true_val, false_val, loc);
  return CValue(ret_val, result_type);
}

absl::StatusOr<std::shared_ptr<CType>> Translator::ResolveTypeInstance(
    std::shared_ptr<CType> t) {
  auto inst = std::dynamic_pointer_cast<CInstantiableTypeAlias>(t);

  // Check if it's a concrete type or an alias
  if (inst == nullptr) {
    return t;
  }

  // Check if it's already been scanned
  {
    auto found = inst_types_.find(inst);

    if (found != inst_types_.end()) {
      return found->second;
    }
  }

  // Needs to be scanned from AST
  XLS_RETURN_IF_ERROR(
      ScanStruct(clang::dyn_cast<const clang::RecordDecl>(inst->base())));

  auto found = inst_types_.find(inst);
  CHECK(found != inst_types_.end());

  return found->second;
}

absl::StatusOr<std::shared_ptr<CType>> Translator::ResolveTypeInstanceDeeply(
    std::shared_ptr<CType> t) {
  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ret, ResolveTypeInstance(t));

  // Handle structs
  {
    auto ret_struct = std::dynamic_pointer_cast<const CStructType>(ret);

    if (ret_struct != nullptr) {
      std::vector<std::shared_ptr<CField>> fields;
      for (const std::shared_ptr<CField>& field : ret_struct->fields()) {
        XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> field_type,
                             ResolveTypeInstanceDeeply(field->type()));
        fields.push_back(std::make_shared<CField>(field->name(), field->index(),
                                                  field_type));
      }
      return std::make_shared<CStructType>(fields, ret_struct->no_tuple_flag(),
                                           ret_struct->synthetic_int_flag());
    }
  }

  // Handle arrays
  {
    auto ret_array = std::dynamic_pointer_cast<const CArrayType>(ret);

    if (ret_array != nullptr) {
      XLS_ASSIGN_OR_RETURN(
          std::shared_ptr<CType> elem_type,
          ResolveTypeInstanceDeeply(ret_array->GetElementType()));
      return std::make_shared<CArrayType>(elem_type, ret_array->GetSize());
    }
  }

  // Handle channels
  {
    auto ret_channel = std::dynamic_pointer_cast<const CChannelType>(ret);
    if (ret_channel != nullptr) {
      XLS_ASSIGN_OR_RETURN(
          std::shared_ptr<CType> elem_type,
          ResolveTypeInstanceDeeply(ret_channel->GetItemType()));
      return std::make_shared<CChannelType>(
          elem_type, ret_channel->GetMemorySize(), ret_channel->GetOpType());
    }
  }

  return ret;
}

absl::StatusOr<bool> Translator::FunctionIsInSyntheticInt(
    const clang::FunctionDecl* decl) {
  if (clang::isa<clang::CXXMethodDecl>(decl)) {
    auto method_decl = clang::dyn_cast<clang::CXXMethodDecl>(decl);
    auto type_alias = std::make_shared<CInstantiableTypeAlias>(
        static_cast<const clang::NamedDecl*>(method_decl->getParent()));
    auto found = inst_types_.find(type_alias);
    if (found == inst_types_.end()) {
      return false;
    }
    XLSCC_CHECK(found != inst_types_.end(), GetLoc(*decl));
    auto struct_type = dynamic_cast<const CStructType*>(found->second.get());
    CHECK_NE(struct_type, nullptr);
    if (struct_type->synthetic_int_flag()) {
      return true;
    }
  }
  return false;
}

absl::Status Translator::TranslateAddCallerChannelsByCalleeChannel(
    std::shared_ptr<LValue> caller_lval, std::shared_ptr<LValue> callee_lval,
    absl::flat_hash_map<IOChannel*, IOChannel*>*
        caller_channels_by_callee_channel,
    const xls::SourceInfo& loc) {
  XLSCC_CHECK_NE(caller_channels_by_callee_channel, nullptr, loc);

  // Nothing to do
  if (caller_lval == nullptr || callee_lval == nullptr) {
    return absl::OkStatus();
  }

  XLSCC_CHECK_EQ(caller_lval->get_compounds().size(),
                 callee_lval->get_compounds().size(), loc);

  for (const auto& [idx, caller_field_lval] : caller_lval->get_compounds()) {
    std::shared_ptr<LValue> callee_field_lval =
        callee_lval->get_compound_or_null(idx);
    XLS_RETURN_IF_ERROR(TranslateAddCallerChannelsByCalleeChannel(
        caller_field_lval, callee_field_lval, caller_channels_by_callee_channel,
        loc));
  }

  if (caller_lval->channel_leaf() != nullptr ||
      callee_lval->channel_leaf() != nullptr) {
    XLSCC_CHECK(callee_lval->channel_leaf() != nullptr, loc);
    XLSCC_CHECK(caller_lval->channel_leaf() != nullptr, loc);

    XLSCC_CHECK(
        caller_lval->channel_leaf()->generated.has_value() ||
            IOChannelInCurrentFunction(caller_lval->channel_leaf(), loc),
        loc);

    XLSCC_CHECK(!caller_channels_by_callee_channel->contains(
                    callee_lval->channel_leaf()),
                loc);
    (*caller_channels_by_callee_channel)[callee_lval->channel_leaf()] =
        caller_lval->channel_leaf();
  }

  return absl::OkStatus();
}

absl::Status Translator::ValidateLValue(std::shared_ptr<LValue> lval,
                                        const xls::SourceInfo& loc) {
  if (lval == nullptr) {
    return absl::OkStatus();
  }

  if (lval->channel_leaf() != nullptr) {
    if (lval->channel_leaf()->generated.has_value() &&
        IOChannelInCurrentFunction(lval->channel_leaf(), loc)) {
      return absl::InternalError(ErrorMessage(
          loc, "lval %s contains channel %s not present in io_channels",
          lval->debug_string().c_str(),
          lval->channel_leaf()->generated.value()
              ? lval->channel_leaf()->generated.value()->name().data()
              : "(null)"));
    }
  }

  for (const auto& [idx, caller_field_lval] : lval->get_compounds()) {
    XLS_RETURN_IF_ERROR(ValidateLValue(caller_field_lval, loc));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<LValue>> Translator::TranslateLValueChannels(
    std::shared_ptr<LValue> outer_lval,
    const absl::flat_hash_map<IOChannel*, IOChannel*>&
        inner_channels_by_outer_channel,
    const xls::SourceInfo& loc) {
  if (outer_lval == nullptr) {
    return nullptr;
  }

  if (outer_lval->is_select()) {
    XLS_ASSIGN_OR_RETURN(
        std::shared_ptr<LValue> inner_true_lval,
        TranslateLValueChannels(outer_lval->lvalue_true(),
                                inner_channels_by_outer_channel, loc));
    XLS_ASSIGN_OR_RETURN(
        std::shared_ptr<LValue> inner_false_lval,
        TranslateLValueChannels(outer_lval->lvalue_false(),
                                inner_channels_by_outer_channel, loc));
    return std::make_shared<LValue>(outer_lval->cond(), inner_true_lval,
                                    inner_false_lval);
  }

  if (outer_lval->channel_leaf() != nullptr) {
    XLSCC_CHECK(
        inner_channels_by_outer_channel.contains(outer_lval->channel_leaf()),
        loc);
    IOChannel* inner_channel =
        inner_channels_by_outer_channel.at(outer_lval->channel_leaf());
    return std::make_shared<LValue>(inner_channel);
  }

  if (!outer_lval->get_compounds().empty()) {
    absl::flat_hash_map<int64_t, std::shared_ptr<LValue>> compounds;
    for (const auto& [idx, outer_field_lval] : outer_lval->get_compounds()) {
      XLS_ASSIGN_OR_RETURN(
          compounds[idx],
          TranslateLValueChannels(outer_field_lval,
                                  inner_channels_by_outer_channel, loc));
    }
    return std::make_shared<LValue>(compounds);
  }

  return outer_lval;
}

absl::StatusOr<std::string> Translator::GetStringLiteral(
    const clang::Expr* expr, const xls::SourceInfo& loc) {
  expr = GetCastsSubexpr(expr);

  if (auto default_expr = clang::dyn_cast<clang::CXXDefaultArgExpr>(expr)) {
    const clang::Expr* value_expr = GetCastsSubexpr(default_expr->getExpr());

    XLSCC_CHECK(
        clang::dyn_cast<clang::CXXNullPtrLiteralExpr>(value_expr) != nullptr,
        loc);

    return "";
  }

  if (auto string_literal_fmt_expr =
          clang::dyn_cast<clang::StringLiteral>(expr)) {
    return string_literal_fmt_expr->getString().str();
  }

  return absl::InvalidArgumentError(
      ErrorMessage(loc, "Argument must be a string literal"));
}

absl::StatusOr<std::pair<bool, CValue>> Translator::GenerateIR_BuiltInCall(
    const clang::CallExpr* call, const xls::SourceInfo& loc) {
  const clang::FunctionDecl* funcdecl = call->getDirectCallee();
  if (funcdecl->getNameAsString() == "__xlscc_unimplemented") {
    return absl::UnimplementedError(ErrorMessage(loc, "Unimplemented marker"));
  }

  if (funcdecl->getNameAsString() == "__xlscc_pipeline" ||
      funcdecl->getNameAsString() == "__xlscc_unroll" ||
      funcdecl->getNameAsString() == "__xlscc_asap") {
    context().last_intrinsic_call = call;
    return std::make_pair(true, CValue());
  }

  if (funcdecl->getNameAsString() == "__xlscc_fixed_32_32_bits_for_double" ||
      funcdecl->getNameAsString() == "__xlscc_fixed_32_32_bits_for_float") {
    XLSCC_CHECK_EQ(call->getNumArgs(), 1, loc);
    const clang::Expr* value_arg = call->getArg(0);
    XLS_ASSIGN_OR_RETURN(CValue value_cval, GenerateIR_Expr(value_arg, loc));
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> return_type,
                         TranslateTypeFromClang(call->getType(), loc));
    XLSCC_CHECK(value_cval.type()->Is<CFloatType>(), loc);
    xls::BValue sliced = context().fb->BitSlice(value_cval.rvalue(),
                                                /*start=*/128,
                                                /*width=*/64, loc);
    return std::make_pair(true, CValue(sliced, return_type));
  }

  // Tracing handled below
  std::string message_string;

  if (funcdecl->getNameAsString() == "__xlscc_assert" ||
      funcdecl->getNameAsString() == "__xlscc_trace") {
    XLSCC_CHECK_GE(call->getNumArgs(), 1, loc);
    const clang::Expr* fmt_arg = call->getArg(0);
    XLS_ASSIGN_OR_RETURN(message_string, GetStringLiteral(fmt_arg, loc));
  } else {
    // Not a built-in call
    return std::make_pair(false, CValue());
  }

  IOOp op = {.op = OpType::kTrace, .trace_message_string = message_string};
  xls::BValue condition = context().full_condition_bval(loc);

  if (funcdecl->getNameAsString() == "__xlscc_assert") {
    XLSCC_CHECK(call->getNumArgs() == 2 || call->getNumArgs() == 3, loc);
    const clang::Expr* cond_arg = call->getArg(1);
    XLS_ASSIGN_OR_RETURN(CValue cond_cval, GenerateIR_Expr(cond_arg, loc));
    if (!cond_cval.type()->Is<CBoolType>()) {
      return absl::InvalidArgumentError(
          ErrorMessage(loc, "Argument to assert must be bool"));
    }
    op.trace_type = TraceType::kAssert;
    // Assertion value is "fire if true"
    op.ret_value = context().fb->And(context().fb->Not(cond_cval.rvalue()),
                                     condition, loc);
    if (call->getNumArgs() == 3) {
      const clang::Expr* label_arg = call->getArg(2);
      XLS_ASSIGN_OR_RETURN(op.label_string, GetStringLiteral(label_arg, loc));
    }
  } else if (funcdecl->getNameAsString() == "__xlscc_trace") {
    std::vector<xls::BValue> tuple_parts = {condition};
    for (int arg = 1; arg < call->getNumArgs(); ++arg) {
      XLS_ASSIGN_OR_RETURN(CValue arg_cval,
                           GenerateIR_Expr(call->getArg(arg), loc));
      if (!arg_cval.rvalue().valid()) {
        return absl::InvalidArgumentError(
            ErrorMessage(loc,
                         "Argument %i to trace must have R-Value (eg cannot be "
                         "pure pointer etc)",
                         arg));
      }
      XLS_ASSIGN_OR_RETURN(bool arg_contains_lvalues,
                           arg_cval.type()->ContainsLValues(*this));
      if (arg_contains_lvalues) {
        XLS_LOG(WARNING) << ErrorMessage(
            loc, "Only R-Value part of argument %i will be traced", arg);
      }
      tuple_parts.push_back(arg_cval.rvalue());
    }

    op.trace_type = TraceType::kTrace;
    op.ret_value = context().fb->Tuple(tuple_parts, loc);
  } else {
    XLSCC_CHECK("Internal consistency failure" == nullptr, loc);
  }

  XLS_RETURN_IF_ERROR(AddOpToChannel(op, /*channel=*/nullptr, loc).status());
  return std::make_pair(true, CValue());
}

absl::StatusOr<CValue> Translator::GenerateIR_Call(const clang::CallExpr* call,
                                                   const xls::SourceInfo& loc) {
  // __xlscc_... functions
  std::pair<bool, CValue> built_in_call;
  XLS_ASSIGN_OR_RETURN(built_in_call, GenerateIR_BuiltInCall(call, loc));

  if (built_in_call.first) {
    return built_in_call.second;
  }

  const clang::FunctionDecl* funcdecl = call->getDirectCallee();

  CValue this_value_orig;

  const clang::Expr* this_expr = nullptr;
  xls::BValue thisval;
  xls::BValue* pthisval = nullptr;
  std::shared_ptr<LValue> this_lval;

  int skip_args = 0;

  // If true, an extra return value is expected for the modified "this" object
  //  in a method call. This mechanism is similar to that used for mutable
  //  reference parameters. A "this" pointer cannot be used in the usual way,
  //  since BValues are immutable, and pointers are unsupported.
  bool add_this_return = false;

  // Evaluate if "this" argument is necessary (eg for method calls)
  if (auto member_call =
          clang::dyn_cast<const clang::CXXMemberCallExpr>(call)) {
    this_expr = member_call->getImplicitObjectArgument();
    pthisval = &thisval;

    clang::QualType thisQual = member_call->getMethodDecl()->getThisType();
    XLSCC_CHECK(thisQual->isPointerType(), loc);
    add_this_return = !thisQual->getPointeeType().isConstQualified();
  } else if (auto op_call =
                 clang::dyn_cast<const clang::CXXOperatorCallExpr>(call)) {
    if (const clang::CXXMethodDecl* cxx_method =
            clang::dyn_cast<const clang::CXXMethodDecl>(
                op_call->getDirectCallee())) {
      CValue ret;

      // There is a special case here for a certain expression form
      XLS_ASSIGN_OR_RETURN(bool applied,
                           ApplyArrayAssignHack(op_call, loc, &ret));
      if (applied) {
        return ret;
      }

      // this comes as first argument for operators
      this_expr = call->getArg(0);

      clang::QualType thisQual = cxx_method->getThisType();
      XLSCC_CHECK(thisQual->isPointerType(), loc);
      add_this_return = !thisQual->getPointeeType().isConstQualified();
      ++skip_args;
    }
  }

  if (this_expr != nullptr) {
    MaskAssignmentsGuard guard_assignments(*this, /*engage=*/add_this_return);
    MaskMemoryWritesGuard guard_writes(*this, /*engage=*/add_this_return);
    MaskIOOtherThanMemoryWritesGuard guard_io(*this,
                                              /*engage=*/call->isLValue());

    {
      // The Assign() statement below will take care of any assignments
      //  in the expression for "this". Don't do these twice, as it can cause
      //  issues like double-increment https://github.com/google/xls/issues/389
      XLS_ASSIGN_OR_RETURN(this_value_orig, GenerateIR_Expr(this_expr, loc));
    }

    // TODO(seanhaskell): Remove special handling
    //    this_lval = this_value_orig.lvalue();
    if (add_this_return) {
      if (this_value_orig.lvalue() == nullptr && call->isLValue()) {
        XLS_ASSIGN_OR_RETURN(this_lval, CreateReferenceValue(this_expr, loc));
      }
    }

    // TODO(seanhaskell): Remove special this handling
    if (this_value_orig.lvalue() != nullptr) {
      if (this_value_orig.lvalue()->leaf() != nullptr ||
          this_value_orig.lvalue()->is_select()) {
        XLS_ASSIGN_OR_RETURN(this_value_orig,
                             GenerateIR_Expr(this_value_orig.lvalue(), loc));
      }
      this_lval = this_value_orig.lvalue();
    }

    thisval = this_value_orig.rvalue();
    pthisval = &thisval;
  }

  std::vector<const clang::Expr*> args;
  for (int pi = skip_args; pi < call->getNumArgs(); ++pi) {
    args.push_back(call->getArg(pi));
  }

  XLS_ASSIGN_OR_RETURN(
      CValue call_res,
      GenerateIR_Call(funcdecl, args, pthisval, &this_lval, loc));

  if (add_this_return) {
    MaskIOOtherThanMemoryWritesGuard guard(*this);
    XLSCC_CHECK(pthisval, loc);

    XLS_RETURN_IF_ERROR(Assign(this_expr,
                               CValue(thisval, this_value_orig.type(),
                                      /*disable_type_check=*/false, this_lval),
                               loc));
  }

  return call_res;
}

absl::StatusOr<bool> Translator::ApplyArrayAssignHack(
    const clang::CXXOperatorCallExpr* op_call, const xls::SourceInfo& loc,
    CValue* output) {
  // Hack to avoid returning reference object.
  //  xls_int[n] = val
  //  CXXOperatorCallExpr '=' {
  //    MaterializeTemporaryExpr {
  //      CXXOperatorCallExpr '[]' {
  //      }
  //    }
  //  }
  if (!op_call->isAssignmentOp()) {
    return false;
  }
  auto materialize = clang::dyn_cast<const clang::MaterializeTemporaryExpr>(
      op_call->getArg(0));
  if (materialize == nullptr) {
    return false;
  }
  auto sub_op_call = clang::dyn_cast<const clang::CXXOperatorCallExpr>(
      materialize->getSubExpr());
  if (sub_op_call == nullptr) {
    return false;
  }

  if (sub_op_call->getOperator() !=
      clang::OverloadedOperatorKind::OO_Subscript) {
    return false;
  }
  const clang::Expr* ivalue = sub_op_call->getArg(1);
  const clang::Expr* rvalue = op_call->getArg(1);
  const clang::Expr* lvalue = sub_op_call->getArg(0);

  const clang::CXXRecordDecl* stype = lvalue->getType()->getAsCXXRecordDecl();
  if (stype == nullptr) {
    return false;
  }

  std::string set_element_method = "set_element_xlsint";
  if (absl::StrContains(rvalue->getType().getAsString(), "BitElemRef")) {
    set_element_method = "set_element_bitref";
  } else if (absl::StrContains(rvalue->getType().getAsString(), "int")) {
    set_element_method = "set_element_int";
  }
  for (auto method : stype->methods()) {
    if (method->getNameAsString() == set_element_method) {
      auto to_call = dynamic_cast<const clang::FunctionDecl*>(method);

      XLSCC_CHECK(to_call != nullptr, loc);

      XLS_ASSIGN_OR_RETURN(CValue lvalue_initial, GenerateIR_Expr(lvalue, loc));

      xls::BValue this_inout = lvalue_initial.rvalue();
      XLS_ASSIGN_OR_RETURN(
          CValue f_return,
          GenerateIR_Call(to_call, {ivalue, rvalue}, &this_inout,
                          /*this_lval=*/nullptr, loc));
      XLS_RETURN_IF_ERROR(
          Assign(lvalue, CValue(this_inout, lvalue_initial.type()), loc));
      *output = f_return;
      return true;
    }
  }
  // Recognized the pattern, but no set_element_* method to use
  return false;
}

absl::StatusOr<const clang::Stmt*> Translator::GetFunctionBody(
    const clang::FunctionDecl*& funcdecl) {
  const bool trivial = funcdecl->hasTrivialBody() || funcdecl->isTrivial();

  if (!trivial && funcdecl->getBody() == nullptr) {
    return absl::NotFoundError(ErrorMessage(GetLoc(*funcdecl),
                                            "Function %s used but has no body",
                                            funcdecl->getNameAsString()));
  }

  // funcdecl parameters may be different for forward declarations
  const clang::FunctionDecl* definition = nullptr;
  const clang::Stmt* body = funcdecl->getBody(definition);
  if (definition == nullptr) {
    if (!trivial) {
      return absl::NotFoundError(ErrorMessage(
          GetLoc(*funcdecl), "Function %s has no body or definition",
          funcdecl->getNameAsString()));
    }
  } else {
    funcdecl = definition;
  }
  XLSCC_CHECK(body != nullptr || trivial, GetLoc(*funcdecl));
  return body;
}

absl::Status Translator::GetChannelsForExprOrNull(
    const clang::Expr* object, std::vector<ConditionedIOChannel>* output,
    const xls::SourceInfo& loc, xls::BValue condition) {
  MaskSideEffectsGuard guard(*this);

  XLS_ASSIGN_OR_RETURN(CValue cval, GenerateIR_Expr(object, GetLoc(*object)));

  if (cval.lvalue() == nullptr) {
    return absl::OkStatus();
  }

  return GetChannelsForLValue(cval.lvalue(), output, loc, condition);
}

absl::Status Translator::GetChannelsForLValue(
    const std::shared_ptr<LValue>& lvalue,
    std::vector<ConditionedIOChannel>* output, const xls::SourceInfo& loc,
    xls::BValue condition) {
  XLSCC_CHECK_NE(lvalue, nullptr, loc);

  if (lvalue->is_select()) {
    XLSCC_CHECK(lvalue->cond().valid(), loc);

    xls::BValue and_condition =
        condition.valid() ? context().fb->And(condition, lvalue->cond(), loc)
                          : lvalue->cond();
    XLS_RETURN_IF_ERROR(GetChannelsForLValue(lvalue->lvalue_true(), output, loc,
                                             and_condition));

    xls::BValue not_lval_cond = context().fb->Not(lvalue->cond(), loc);
    xls::BValue and_not_condition =
        condition.valid() ? context().fb->And(condition, not_lval_cond, loc)
                          : not_lval_cond;
    XLS_RETURN_IF_ERROR(GetChannelsForLValue(lvalue->lvalue_false(), output,
                                             loc, and_not_condition));
    return absl::OkStatus();
  }

  if (lvalue->channel_leaf() == nullptr) {
    return absl::OkStatus();
  }
  ConditionedIOChannel ret;
  ret.channel = lvalue->channel_leaf();
  ret.condition = condition;
  output->push_back(ret);
  return absl::OkStatus();
}

// this_inout can be nullptr for non-members
absl::StatusOr<CValue> Translator::GenerateIR_Call(
    const clang::FunctionDecl* funcdecl,
    std::vector<const clang::Expr*> expr_args, xls::BValue* this_inout,
    std::shared_ptr<LValue>* this_lval, const xls::SourceInfo& loc) {
  // Ensure callee has been parsed
  XLS_RETURN_IF_ERROR(GetFunctionBody(funcdecl).status());

  GeneratedFunction* func = nullptr;

  auto signature = absl::implicit_cast<const clang::NamedDecl*>(funcdecl);

  auto found = inst_functions_.find(signature);
  if (found != inst_functions_.end()) {
    XLSCC_CHECK(!functions_in_progress_.contains(signature), loc);
    func = found->second.get();
  } else if (functions_in_progress_.contains(funcdecl)) {
    XLSCC_CHECK(!inst_functions_.contains(signature), loc);
    func = functions_in_progress_.at(signature)->generated_function.get();
  } else {
    XLSCC_CHECK(!functions_in_progress_.contains(signature), loc);

    std::unique_ptr<FunctionInProgress> generate_function_header =
        std::make_unique<FunctionInProgress>();
    auto generated_function = std::make_unique<GeneratedFunction>();
    func = generated_function.get();

    // Overwrites generated_function
    XLS_ASSIGN_OR_RETURN(*generate_function_header,
                         GenerateIR_Function_Header(*func, funcdecl));

    generate_function_header->generated_function =
        std::move(generated_function);
    functions_in_progress_[signature] = std::move(generate_function_header);
  }

  XLSCC_CHECK_NE(func, nullptr, loc);

  absl::flat_hash_map<IOChannel*, IOChannel*> caller_channels_by_callee_channel;

  // Needed for IO op translation
  if (this_lval != nullptr) {
    XLS_RETURN_IF_ERROR(TranslateAddCallerChannelsByCalleeChannel(
        *this_lval, func->this_lvalue, &caller_channels_by_callee_channel,
        loc));
  }

  std::vector<xls::BValue> args;
  int expected_returns = 0;

  // Add this if needed
  bool add_this_return = false;
  if (this_inout != nullptr) {
    args.push_back(*this_inout);
    XLSCC_CHECK(this_inout->valid(), loc);

    // "This" is a PointerType, ignore and treat as reference
    auto method = clang::dyn_cast<const clang::CXXMethodDecl>(funcdecl);
    clang::QualType thisQual = method->getThisType();
    XLSCC_CHECK(thisQual->isPointerType(), loc);

    add_this_return = !thisQual->getPointeeType().isConstQualified();
  }

  // Number of return values expected. If >1, the return will be a tuple.
  // (See MakeFunctionReturn()).
  if (add_this_return) {
    ++expected_returns;
  }
  if (!funcdecl->getReturnType()->isVoidType()) {
    ++expected_returns;
  }

  if (expr_args.size() != funcdecl->getNumParams()) {
    return absl::UnimplementedError(ErrorMessage(
        loc,
        "Parameter count mismatch: %i params in FunctionDecl, %i arguments to "
        "call",
        funcdecl->getNumParams(), static_cast<int>(expr_args.size())));
  }

  absl::flat_hash_map<const clang::ParmVarDecl*, bool> will_assign_param;

  // Add other parameters
  for (int pi = 0; pi < funcdecl->getNumParams(); ++pi) {
    const clang::ParmVarDecl* p = funcdecl->getParamDecl(pi);

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));

    const bool will_assign =
        stripped.is_ref && (!stripped.base.isConstQualified());
    will_assign_param[p] = will_assign;

    // Map callee IO channels
    CValue argv;
    {
      MaskAssignmentsGuard guard_assignments(*this, will_assign_param.at(p));
      MaskMemoryWritesGuard guard_writes(*this, will_assign_param.at(p));
      XLS_ASSIGN_OR_RETURN(argv, GenerateIR_Expr(expr_args[pi], loc));
    }
    XLS_ASSIGN_OR_RETURN(bool is_channel, TypeIsChannel(p->getType(), loc));
    if (is_channel) {
      // TODO(seanhaskell): This can be further generalized to allow
      // structs with channels inside
      std::shared_ptr<LValue> callee_lvalue = func->lvalues_by_param.at(p);

      XLSCC_CHECK_NE(argv.lvalue(), nullptr, loc);
      if (argv.lvalue()->is_select()) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Channel select passed as parameter"));
      }

      XLS_RETURN_IF_ERROR(TranslateAddCallerChannelsByCalleeChannel(
          /*caller_lval=*/argv.lvalue(),
          /*callee_lval=*/callee_lvalue, &caller_channels_by_callee_channel,
          loc));

      continue;
    }

    // Const references don't need a return
    if (stripped.is_ref && (!stripped.base.isConstQualified())) {
      ++expected_returns;
    }
    if (argv.lvalue() != nullptr) {
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> stripped_base_type,
                           TranslateTypeFromClang(stripped.base, loc));
      XLS_ASSIGN_OR_RETURN(bool has_lvalues,
                           stripped_base_type->ContainsLValues(*this));
      if (has_lvalues) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Passing parameters containing LValues"));
      }
    }

    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> argt,
                         TranslateTypeFromClang(stripped.base, loc));

    xls::BValue pass_bval = argv.rvalue();
    std::shared_ptr<CType> pass_type = argv.type();

    if (argv.type()->Is<CPointerType>() || argv.type()->Is<CArrayType>()) {
      auto arg_arr_type = argt->As<CArrayType>();

      // Pointer to array
      if (argv.type()->Is<CPointerType>()) {
        if (argv.lvalue() == nullptr) {
          return absl::UnimplementedError(
              ErrorMessage(loc,
                           "Pointer value has no lvalue (unsupported "
                           "construct such as ternary?)"));
        }
        XLS_ASSIGN_OR_RETURN(CValue pass_rval,
                             GenerateIR_Expr(argv.lvalue(), loc));
        pass_bval = pass_rval.rvalue();
      }

      XLSCC_CHECK(pass_bval.valid(), loc);

      int64_t pass_bval_arr_size = ArrayBValueWidth(pass_bval);

      if (pass_bval_arr_size < arg_arr_type->GetSize()) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Array slice out of bounds"));
      }

      if (pass_bval_arr_size != arg_arr_type->GetSize()) {
        pass_bval = context().fb->ArraySlice(
            pass_bval, context().fb->Literal(xls::SBits(0, 32), loc),
            arg_arr_type->GetSize(), loc);
      }

      std::shared_ptr<CType> element_type;

      if (argv.type()->Is<CPointerType>()) {
        auto argv_pointer_type = argv.type()->As<CPointerType>();
        element_type = argv_pointer_type->GetPointeeType();
      } else if (argv.type()->Is<CArrayType>()) {
        auto argv_array_type = argv.type()->As<CArrayType>();
        element_type = argv_array_type->GetElementType();
      } else {
        XLSCC_CHECK_EQ("Internal consistency failure", nullptr, loc);
      }

      pass_type =
          std::make_shared<CArrayType>(element_type, arg_arr_type->GetSize());
    }

    if (pass_type->Is<CReferenceType>()) {
      pass_type = pass_type->As<CReferenceType>()->GetPointeeType();
      XLSCC_CHECK_NE(argv.lvalue(), nullptr, loc);
      if (argv.lvalue()->leaf() != nullptr || argv.lvalue()->is_select()) {
        XLS_ASSIGN_OR_RETURN(argv, GenerateIR_Expr(argv.lvalue(), loc));
      }
      pass_bval = argv.rvalue();
      pass_type = argv.type();
    }

    if (*pass_type != *argt) {
      return absl::InternalError(ErrorMessage(
          loc,
          "Internal error: expression type %s doesn't match "
          "parameter type %s in function %s",
          string(*argv.type()), string(*argt), funcdecl->getNameAsString()));
    }

    XLSCC_CHECK(pass_bval.valid(), loc);
    args.push_back(pass_bval);
  }

  // Translate to external channels
  bool multiple_translations_for_a_channel = false;
  for (const auto& [callee_channel, caller_channel] :
       caller_channels_by_callee_channel) {
    if (caller_channel->generated.has_value() ||
        callee_channel->generated.has_value()) {
      XLSCC_CHECK(caller_channel->generated.has_value(), loc);
      XLSCC_CHECK(callee_channel->generated.has_value(), loc);
      continue;
    }
    if (!external_channels_by_internal_channel_.contains(caller_channel) &&
        io_test_mode_) {
      continue;
    }
    XLSCC_CHECK(external_channels_by_internal_channel_.contains(caller_channel),
                loc);

    // Avoid compiler bug with [] = .at()
    XLSCC_CHECK_EQ(
        external_channels_by_internal_channel_.contains(caller_channel) &&
            external_channels_by_internal_channel_.count(caller_channel),
        1, loc);
    const ChannelBundle& caller_bundle =
        external_channels_by_internal_channel_.find(caller_channel)->second;

    std::pair<const IOChannel*, ChannelBundle> pair(callee_channel,
                                                    caller_bundle);
    if (!ContainsKeyValuePair(external_channels_by_internal_channel_, pair)) {
      external_channels_by_internal_channel_.insert(pair);
    }

    if (external_channels_by_internal_channel_.count(callee_channel) > 1) {
      multiple_translations_for_a_channel = true;
    }
  }

  // Translate external channels, then generate sub-procs
  if (multiple_translations_for_a_channel && !func->sub_procs.empty() &&
      !func->sub_procs.front().generated_func->io_ops.empty()) {
    return absl::UnimplementedError(
        ErrorMessage(loc,
                     "IO ops in pipelined loops in subroutines called "
                     "with multiple different channel arguments"));
  }

  {
    // Inserted after header generation so that parameter marshalling doesn't
    // affect recursion detection
    if (functions_in_call_stack_.contains(signature)) {
      return absl::UnimplementedError(ErrorMessage(loc, "Recursion"));
    }
    functions_in_call_stack_.insert(signature);

    auto functions_called_enclosing_guard = MakeLambdaGuard(
        [this, signature]() { functions_in_call_stack_.erase(signature); });

    // Generate the function body if needed
    if (functions_in_progress_.contains(signature)) {
      FunctionInProgress* function_header =
          functions_in_progress_.at(signature).get();

      XLS_RETURN_IF_ERROR(
          GenerateIR_Function_Body(*func, funcdecl, *function_header));

      inst_functions_[signature] =
          std::move(function_header->generated_function);
      functions_in_progress_.erase(signature);
    }
  }

  // Propagate generated and internal channels up
  for (IOChannel& callee_channel : func->io_channels) {
    IOChannel* callee_channel_ptr = &callee_channel;

    if (!callee_channel_ptr->generated.has_value() &&
        !callee_channel_ptr->internal_to_function) {
      continue;
    }

    // Don't add multiple times
    if (context().sf->generated_caller_channels_by_callee.contains(
            callee_channel_ptr)) {
      IOChannel* to_add = context().sf->generated_caller_channels_by_callee.at(
          callee_channel_ptr);
      XLSCC_CHECK(
          !caller_channels_by_callee_channel.contains(callee_channel_ptr) ||
              to_add ==
                  caller_channels_by_callee_channel.at(callee_channel_ptr),
          loc);

      caller_channels_by_callee_channel[callee_channel_ptr] = to_add;
      continue;
    }
    IOChannel* callee_generated_channel = callee_channel_ptr;
    IOChannel* caller_generated_channel =
        AddChannel(*callee_generated_channel, loc);

    caller_generated_channel->total_ops = 0;
    XLSCC_CHECK(!caller_channels_by_callee_channel.contains(callee_channel_ptr),
                loc);
    caller_channels_by_callee_channel[callee_channel_ptr] =
        caller_generated_channel;
    XLSCC_CHECK(!context().sf->generated_caller_channels_by_callee.contains(
                    callee_channel_ptr),
                loc);
    context().sf->generated_caller_channels_by_callee[callee_channel_ptr] =
        caller_generated_channel;

    if (callee_channel_ptr->internal_to_function) {
      XLSCC_CHECK(
          external_channels_by_internal_channel_.contains(callee_channel_ptr),
          loc);
      XLSCC_CHECK_EQ(
          external_channels_by_internal_channel_.count(callee_channel_ptr), 1,
          loc);
      ChannelBundle callee_bundle =
          external_channels_by_internal_channel_.find(callee_channel_ptr)
              ->second;
      external_channels_by_internal_channel_.insert(
          std::make_pair(caller_generated_channel, callee_bundle));
    }

    // Aggregate pipeline_loops_by_internal_channel
    if (func->pipeline_loops_by_internal_channel.contains(
            callee_generated_channel)) {
      const PipelinedLoopSubProc* sub_proc =
          func->pipeline_loops_by_internal_channel.at(callee_generated_channel);
      context()
          .sf->pipeline_loops_by_internal_channel[caller_generated_channel] =
          sub_proc;
    }
  }

  // Map callee ops. There can be multiple for one channel
  std::multimap<const IOOp*, IOOp*> caller_ops_by_callee_op;

  for (IOOp& callee_op : func->io_ops) {
    XLSCC_CHECK(caller_ops_by_callee_op.find(&callee_op) ==
                    caller_ops_by_callee_op.end(),
                loc);

    IOOp caller_op;

    XLSCC_CHECK(!(callee_op.scheduling_option != IOSchedulingOption::kNone &&
                  !callee_op.after_ops.empty()),
                loc);

    // Translate ops that must be sequenced before first
    for (const IOOp* after_op : callee_op.after_ops) {
      XLSCC_CHECK(caller_ops_by_callee_op.find(after_op) !=
                      caller_ops_by_callee_op.end(),
                  loc);
      auto range = caller_ops_by_callee_op.equal_range(after_op);
      for (auto caller_it = range.first; caller_it != range.second;
           ++caller_it) {
        IOOp* after_caller_op = caller_it->second;
        caller_op.after_ops.push_back(after_caller_op);
      }
    }

    IOChannel* caller_channel =
        (callee_op.channel != nullptr)
            ? caller_channels_by_callee_channel.at(callee_op.channel)
            : nullptr;

    // Add super op
    caller_op.op = callee_op.op;
    caller_op.is_blocking = callee_op.is_blocking;
    caller_op.trace_type = callee_op.trace_type;
    caller_op.trace_message_string = callee_op.trace_message_string;
    caller_op.label_string = callee_op.label_string;
    caller_op.scheduling_option = callee_op.scheduling_option;
    caller_op.sub_op = &callee_op;

    XLS_ASSIGN_OR_RETURN(
        IOOp * caller_op_ptr,
        AddOpToChannel(caller_op, caller_channel, callee_op.op_location,
                       /*mask=*/false));

    if (caller_op_ptr != nullptr) {
      XLSCC_CHECK(caller_op_ptr->op == OpType::kTrace ||
                      caller_op_ptr->channel->generated.has_value() ||
                      IOChannelInCurrentFunction(caller_op_ptr->channel, loc),
                  loc);
    }

    caller_ops_by_callee_op.insert(
        std::pair<const IOOp*, IOOp*>(&callee_op, caller_op_ptr));

    // Count expected IO returns
    ++expected_returns;
  }

  // Pass side effecting parameters to call in the order they are declared
  for (const SideEffectingParameter& side_effecting_param :
       func->side_effecting_parameters) {
    switch (side_effecting_param.type) {
      case SideEffectingParameterType::kIOOp: {
        IOOp* callee_op = side_effecting_param.io_op;
        auto range = caller_ops_by_callee_op.equal_range(callee_op);
        for (auto caller_it = range.first; caller_it != range.second;
             ++caller_it) {
          IOOp* caller_op = caller_it->second;
          xls::BValue args_val;
          if (caller_op == nullptr) {
            // Masked, insert dummy argument
            args_val = context().fb->Literal(
                xls::ZeroOfType(side_effecting_param.xls_io_param_type), loc);
          } else {
            // May be empty for sends
            xls::BValue io_value = caller_op->input_value.rvalue();

            args_val = io_value;
          }
          XLSCC_CHECK(args_val.valid(), loc);
          args.push_back(args_val);
          // Expected return already expected in above loop
        }
        break;
      }
      case SideEffectingParameterType::kStatic: {
        // May already be declared if there are multiple calls to the same
        // static-containing function
        if (!context().variables.contains(side_effecting_param.static_value)) {
          XLS_RETURN_IF_ERROR(DeclareStatic(
              side_effecting_param.static_value,
              func->static_values.at(side_effecting_param.static_value),
              /*init_lvalue=*/nullptr, loc,
              /* check_unique_ids= */ false));
        }
        XLS_ASSIGN_OR_RETURN(
            CValue value,
            GetIdentifier(side_effecting_param.static_value, loc));
        XLSCC_CHECK(value.rvalue().valid(), loc);
        args.push_back(value.rvalue());
        // Count expected static returns
        ++expected_returns;
        break;
      }
      default: {
        return absl::InternalError(
            ErrorMessage(loc, "Unknown type of SideEffectingParameter"));
      }
    }
  }

  // Function with no outputs. Body already generated by here
  if (func->xls_func == nullptr) {
    return CValue();
  }

  xls::BValue raw_return = context().fb->Invoke(args, func->xls_func, loc);
  XLSCC_CHECK(expected_returns == 0 || raw_return.valid(), loc);

  list<xls::BValue> unpacked_returns;
  if (expected_returns == 1) {
    unpacked_returns.emplace_back(raw_return);
  } else {
    for (int r = 0; r < expected_returns; ++r) {
      unpacked_returns.emplace_back(
          GetFunctionReturn(raw_return, r, expected_returns, funcdecl, loc));
    }
  }

  CValue retval;

  // First static outputs from callee
  for (const clang::NamedDecl* namedecl :
       func->GetDeterministicallyOrderedStaticValues()) {
    const ConstValue& initval = func->static_values.at(namedecl);

    XLSCC_CHECK(!unpacked_returns.empty(), loc);
    xls::BValue static_output = unpacked_returns.front();
    unpacked_returns.pop_front();

    // Skip assignment to on reset static, as assignment to globals is an error
    XLS_ASSIGN_OR_RETURN(bool is_on_reset, DeclIsOnReset(namedecl));
    if (is_on_reset) {
      continue;
    }
    XLS_RETURN_IF_ERROR(Assign(namedecl, CValue(static_output, initval.type()),
                               loc,
                               /*force_no_lvalue_assign=*/true));
  }

  // Then this return
  if (add_this_return) {
    XLSCC_CHECK(!unpacked_returns.empty(), loc);
    *this_inout = unpacked_returns.front();
    unpacked_returns.pop_front();
  }

  // Special case for constructor
  if (this_lval != nullptr && *this_lval == nullptr) {
    XLS_ASSIGN_OR_RETURN(
        *this_lval,
        TranslateLValueChannels(func->this_lvalue,
                                caller_channels_by_callee_channel, loc));
  }

  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> return_type,
                       TranslateTypeFromClang(funcdecl->getReturnType(), loc));

  // Then explicit return
  if (funcdecl->getReturnType()->isVoidType()) {
    retval = CValue(xls::BValue(), shared_ptr<CType>(new CVoidType()));
  } else if (func->return_lvalue != nullptr) {
    XLSCC_CHECK(!unpacked_returns.empty(), loc);
    xls::BValue unpacked_value = unpacked_returns.front();
    unpacked_returns.pop_front();

    XLS_ASSIGN_OR_RETURN(retval, Generate_LValue_Return_Call(
                                     func->return_lvalue, unpacked_value,
                                     return_type, this_inout, this_lval,
                                     caller_channels_by_callee_channel, loc));

  } else {
    XLSCC_CHECK(!unpacked_returns.empty(), loc);
    retval = CValue(unpacked_returns.front(), return_type);
    unpacked_returns.pop_front();
  }

  // Then reference parameter returns
  for (int pi = 0; pi < funcdecl->getNumParams(); ++pi) {
    const clang::ParmVarDecl* p = funcdecl->getParamDecl(pi);

    // IO returns are later
    XLS_ASSIGN_OR_RETURN(bool is_channel, TypeIsChannel(p->getType(), loc));
    if (is_channel) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));

    XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                         TranslateTypeFromClang(stripped.base, GetLoc(*p)));

    // Const references don't need a return
    if (will_assign_param.at(p)) {
      XLSCC_CHECK(!unpacked_returns.empty(), loc);

      MaskIOOtherThanMemoryWritesGuard guard(*this);
      LValueModeGuard lvalue_guard(*this);
      context().any_writes_generated = false;
      XLS_RETURN_IF_ERROR(
          Assign(expr_args[pi], CValue(unpacked_returns.front(), ctype), loc));

      if (context().any_writes_generated) {
        XLS_LOG(WARNING) << ErrorMessage(
            loc,
            "Memory write in call-by-reference will generate a read and a "
            "write. Is this intended here?");
      }

      unpacked_returns.pop_front();
    }
  }

  // Callee IO returns
  for (IOOp& callee_op : func->io_ops) {
    auto range = caller_ops_by_callee_op.equal_range(&callee_op);
    for (auto caller_it = range.first; caller_it != range.second; ++caller_it) {
      IOOp* caller_op = caller_it->second;

      XLSCC_CHECK(!unpacked_returns.empty(), loc);

      xls::BValue io_value = unpacked_returns.front();

      // Might be masked
      if (caller_op != nullptr) {
        XLS_ASSIGN_OR_RETURN(
            caller_op->ret_value,
            AddConditionToIOReturn(/*op=*/*caller_op, io_value, loc));
      }

      unpacked_returns.pop_front();
    }
  }

  XLSCC_CHECK(unpacked_returns.empty(), loc);

  return retval;
}

absl::StatusOr<Translator::ResolvedInheritance> Translator::ResolveInheritance(
    std::shared_ptr<CType> sub_type, std::shared_ptr<CType> to_type) {
  auto sub_struct =
      std::dynamic_pointer_cast<const CInstantiableTypeAlias>(sub_type);
  auto to_struct =
      std::dynamic_pointer_cast<const CInstantiableTypeAlias>(to_type);

  if (sub_struct && to_struct) {
    XLS_ASSIGN_OR_RETURN(auto sub_struct_res, ResolveTypeInstance(sub_type));
    auto resolved_struct =
        std::dynamic_pointer_cast<const CStructType>(sub_struct_res);
    if (resolved_struct) {
      std::shared_ptr<CField> base_field =
          resolved_struct->get_field(to_struct->base());

      // Derived to Base
      if (base_field) {
        ResolvedInheritance ret;
        ret.base_field = base_field;
        ret.resolved_struct = resolved_struct;
        ret.base_field_name = to_struct->base();
        return ret;
      }
    }
  }
  return ResolvedInheritance();
}

absl::StatusOr<CValue> Translator::ResolveCast(
    const CValue& sub, const std::shared_ptr<CType>& to_type,
    const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(ResolvedInheritance inheritance,
                       ResolveInheritance(sub.type(), to_type));

  // Are we casting to a derived class?
  if (inheritance.base_field != nullptr) {
    XLSCC_CHECK(inheritance.resolved_struct != nullptr, loc);

    xls::BValue val =
        GetStructFieldXLS(sub.rvalue(), inheritance.base_field->index(),
                          *inheritance.resolved_struct, loc);

    return CValue(val, to_type);
  }

  // Pointer conversions
  if (sub.type()->Is<CPointerType>()) {
    if (to_type->Is<CPointerType>()) {
      return sub;
    }
    if (to_type->Is<CArrayType>()) {
      return GenerateIR_Expr(sub.lvalue(), loc);
    }
    return absl::UnimplementedError(
        ErrorMessage(loc, "Don't know how to convert %s to pointer type",
                     std::string(*sub.type())));
  }
  if (auto ref = dynamic_cast<const CReferenceType*>(sub.type().get());
      ref != nullptr) {
    XLS_ASSIGN_OR_RETURN(ResolvedInheritance ref_inheritance,
                         ResolveInheritance(ref->GetPointeeType(), to_type));

    if (*to_type == *ref->GetPointeeType() || ref_inheritance.resolved_struct) {
      XLSCC_CHECK_NE(sub.lvalue(), nullptr, loc);
      XLS_ASSIGN_OR_RETURN(CValue subc, GenerateIR_Expr(sub.lvalue(), loc));
      return ResolveCast(subc, to_type, loc);
    }
    return absl::UnimplementedError(ErrorMessage(
        loc, "Don't know how to convert reference type %s to type %s",
        std::string(*sub.type()), std::string(*to_type)));
  }

  XLS_ASSIGN_OR_RETURN(xls::BValue subc, GenTypeConvert(sub, to_type, loc));

  return CValue(subc, to_type, /*disable_type_check=*/true, sub.lvalue());
}

absl::Status Translator::FailIfTypeHasDtors(
    const clang::CXXRecordDecl* cxx_record) {
  if (cxx_record->hasUserDeclaredDestructor()) {
    return absl::UnimplementedError(
        ErrorMessage(GetLoc(*cxx_record), "Destructors aren't yet called"));
  }

  // Don't need to recurse, since fields will also be initialized, calling this
  // function

  return absl::OkStatus();
}

absl::StatusOr<std::optional<CValue>> Translator::EvaluateNumericConstExpr(
    const clang::Expr* expr, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                       TranslateTypeFromClang(expr->getType(), loc));

  if (!ctype->Is<CIntType>() && !ctype->Is<CFloatType>()) {
    return std::nullopt;
  }

  XLSCC_CHECK_NE(context().ast_context, nullptr, loc);
  clang::Expr::EvalResult const_eval_result;
  if (!expr->EvaluateAsConstantExpr(const_eval_result,
                                    *context().ast_context)) {
    // Not constexpr
    return std::nullopt;
  }

  if (ctype->Is<CIntType>()) {
    if (!const_eval_result.Val.isInt()) {
      // Not constexpr
      return std::nullopt;
    }
    llvm::APInt api = const_eval_result.Val.getInt();
    // Raw data is in little endian format
    auto api_raw = reinterpret_cast<const uint8_t*>(api.getRawData());
    vector<uint8_t> truncated;
    const int truncated_n = ((ctype->GetBitWidth() + 7) / 8);
    truncated.reserve(truncated_n);
    for (int i = 0; i < truncated_n; ++i) {
      truncated.emplace_back(api_raw[i]);
    }
    // FromBytes() accepts little endian format
    auto lbits = xls::Bits::FromBytes(truncated, ctype->GetBitWidth());
    return CValue(context().fb->Literal(lbits, loc), ctype);
  }

  XLSCC_CHECK(ctype->Is<CFloatType>(), loc);
  if (!const_eval_result.Val.isFloat()) {
    // Not constexpr
    return std::nullopt;
  }

  llvm::APFloat apf = const_eval_result.Val.getFloat();
  const double converted = apf.convertToDouble();
  const int64_t integer_value = static_cast<int64_t>(converted);

  vector<uint8_t> formatted;
  formatted.resize(sizeof(converted) + sizeof(integer_value));

  // Pack double and rounded int64_t versions into one bit vector
  memcpy(&formatted[0], reinterpret_cast<const uint8_t*>(&converted),
         sizeof(converted));
  memcpy(&formatted[sizeof(converted)],
         reinterpret_cast<const uint8_t*>(&integer_value),
         sizeof(integer_value));

  // Add 32.32 fixed
  {
    llvm::APFloat decimal_shift(pow(2, 32));
    bool losesInfo;
    apf.convert(decimal_shift.getSemantics(), llvm::RoundingMode::TowardZero,
                &losesInfo);
    apf.multiply(decimal_shift, llvm::RoundingMode::TowardZero);
    llvm::APSInt shifted(64, false);
    bool exact = false;
    apf.convertToInteger(shifted, llvm::RoundingMode::TowardZero, &exact);
    // Raw data is in little endian format
    auto api_raw = reinterpret_cast<const uint8_t*>(shifted.getRawData());
    for (int i = 0; i < 8; ++i) {
      formatted.push_back(api_raw[i]);
    }
  }

  // double, int64_t, fixed 32.32
  auto lbits = xls::Bits::FromBytes(
      formatted,
      /*bit_count=*/(sizeof(converted) + sizeof(integer_value) + 8) * 8);

  return CValue(context().fb->Literal(lbits, loc), ctype);
}

absl::StatusOr<CValue> Translator::GenerateIR_Expr(const clang::Expr* expr,
                                                   const xls::SourceInfo& loc) {
  // Intercept numeric constexprs
  XLS_ASSIGN_OR_RETURN(std::optional<CValue> numeric_constexpr,
                       EvaluateNumericConstExpr(expr, loc));
  if (numeric_constexpr.has_value()) {
    return numeric_constexpr.value();
  }

  if (auto uop = clang::dyn_cast<const clang::UnaryOperator>(expr)) {
    return Generate_UnaryOp(uop, loc);
  }
  if (auto bop = clang::dyn_cast<const clang::BinaryOperator>(expr)) {
    auto clang_op = bop->getOpcode();
    XLS_ASSIGN_OR_RETURN(
        shared_ptr<CType> result_type,
        TranslateTypeFromClang(bop->getType().getCanonicalType(), loc));
    return Generate_BinaryOp(clang_op, bop->isAssignmentOp(), result_type,
                             bop->getLHS(), bop->getRHS(), loc);
  }
  // Ternary: a ? b : c
  if (auto cond = clang::dyn_cast<const clang::ConditionalOperator>(expr)) {
    XLS_ASSIGN_OR_RETURN(
        shared_ptr<CType> result_type,
        TranslateTypeFromClang(cond->getType().getCanonicalType(), loc));
    return Generate_TernaryOp(result_type, cond->getCond(), cond->getTrueExpr(),
                              cond->getFalseExpr(), loc);
  }
  if (auto call = clang::dyn_cast<const clang::CallExpr>(expr)) {
    XLS_ASSIGN_OR_RETURN(IOOpReturn ret, InterceptIOOp(call, GetLoc(*call)));
    // If this call is an IO op, then return the IO value, rather than
    //  generating the call.
    if (!ret.generate_expr) {
      return ret.value;
    }
    return GenerateIR_Call(call, loc);
  }
  if (auto charlit = clang::dyn_cast<const clang::CharacterLiteral>(expr)) {
    if (charlit->getKind() != clang::CharacterLiteralKind::Ascii) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Unimplemented character literaly type %i",
                       static_cast<int>(charlit->getKind())));
    }
    shared_ptr<CType> ctype(new CIntType(8, true, true));
    return CValue(
        context().fb->Literal(xls::UBits(charlit->getValue(), 8), loc), ctype);
  }
  if (auto bl = clang::dyn_cast<const clang::CXXBoolLiteralExpr>(expr)) {
    xls::BValue val =
        context().fb->Literal(xls::UBits(bl->getValue() ? 1 : 0, 1), loc);
    return CValue(val, shared_ptr<CType>(new CBoolType()));
  }
  // This is just a marker Clang places in the AST to show that a template
  //  parameter was substituted. It wraps the substituted value, like:
  // SubstNonTypeTemplateParmExprClass { replacement = IntegerLiteral }
  if (auto subst =
          clang::dyn_cast<const clang::SubstNonTypeTemplateParmExpr>(expr)) {
    return GenerateIR_Expr(subst->getReplacement(), loc);
  }
  // Similar behavior for all cast styles. Clang already enforced the C++
  //  static type-checking rules by this point.
  if (auto cast = clang::dyn_cast<const clang::CastExpr>(expr)) {
    // For converting this pointer from base to derived
    // Don't generate pointer errors for C++ "this" keyword
    IgnorePointersGuard ignore_pointers(*this);
    if (clang::isa<clang::CXXThisExpr>(cast->getSubExpr())) {
      ignore_pointers.enable();
    }

    XLS_ASSIGN_OR_RETURN(CValue sub, GenerateIR_Expr(cast->getSubExpr(), loc));

    XLS_ASSIGN_OR_RETURN(shared_ptr<CType> to_type,
                         TranslateTypeFromClang(cast->getType(), loc));

    if (to_type->Is<CVoidType>()) {
      return CValue(xls::BValue(), to_type);
    }

    // Sometimes array types are converted to pointer types via ImplicitCast,
    // even nested as in mutable array -> mutable pointer -> const pointer.
    // Since we don't generally support pointers (except for array slicing),
    // this case is short-circuited, and the nested expression is evaluated
    // directly, ignoring the casts.
    {
      // Ignore nested ImplicitCastExprs. This case breaks the logic below.
      auto nested_implicit = cast;

      while (
          clang::isa<clang::ImplicitCastExpr>(nested_implicit->getSubExpr())) {
        nested_implicit =
            clang::cast<const clang::CastExpr>(nested_implicit->getSubExpr());
      }

      auto from_arr_type = std::dynamic_pointer_cast<CArrayType>(sub.type());

      // Avoid decay of array to pointer, pointers are unsupported
      if (from_arr_type && nested_implicit->getType()->isPointerType()) {
        XLS_ASSIGN_OR_RETURN(
            CValue sub, GenerateIR_Expr(nested_implicit->getSubExpr(), loc));

        return sub;
      }
    }

    return ResolveCast(sub, to_type, loc);
  }
  if (clang::isa<clang::CXXThisExpr>(expr)) {
    XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl, GetThisDecl(loc));
    XLS_ASSIGN_OR_RETURN(CValue this_val, GetIdentifier(this_decl, loc));
    return this_val;
  }
  // ExprWithCleanups preserves some metadata from Clang's parsing process,
  //  which I think is meant to be used for IDE editing tools. It is
  //  irrelevant to XLS[cc].
  if (auto cast = clang::dyn_cast<const clang::ExprWithCleanups>(expr)) {
    return GenerateIR_Expr(cast->getSubExpr(), loc);
  }
  // MaterializeTemporaryExpr wraps instantiation of temporary objects
  // We don't support memory management, so this is irrelevant to us.
  if (auto cast =
          clang::dyn_cast<const clang::MaterializeTemporaryExpr>(expr)) {
    return GenerateIR_Expr(cast->getSubExpr(), loc);
  }
  // Occurs in the AST for explicit constructor calls. "Foo a = Foo();"
  if (auto ctor = clang::dyn_cast<const clang::CXXConstructExpr>(expr)) {
    XLS_ASSIGN_OR_RETURN(shared_ptr<CType> octype,
                         TranslateTypeFromClang(ctor->getType(), loc));
    XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype, ResolveTypeInstance(octype));

    // A struct/class is being constructed
    if (ctype->Is<CStructType>()) {
      XLS_RETURN_IF_ERROR(
          FailIfTypeHasDtors(ctor->getType()->getAsCXXRecordDecl()));
      XLS_ASSIGN_OR_RETURN(xls::BValue this_inout,
                           CreateDefaultValue(octype, loc));
      std::vector<const clang::Expr*> args;
      args.reserve(ctor->getNumArgs());
      for (int pi = 0; pi < ctor->getNumArgs(); ++pi) {
        args.push_back(ctor->getArg(pi));
      }
      std::shared_ptr<LValue> this_lval;

      XLS_ASSIGN_OR_RETURN(
          CValue ret, GenerateIR_Call(ctor->getConstructor(), args, &this_inout,
                                      /*this_lval=*/&this_lval, loc));
      XLSCC_CHECK(ret.type()->Is<CVoidType>(), loc);
      return CValue(this_inout, octype, /*disable_type_check=*/false,
                    this_lval);
    }

    // A built-in type is being constructed. Create default value if there's
    //  no constructor parameter
    if (ctor->getNumArgs() == 0) {
      if (error_on_uninitialized_) {
        return absl::InvalidArgumentError(
            ErrorMessage(loc, "Built-in type %s defaulted in constructor '%s'",
                         std::string(*octype),
                         ctor->getConstructor()->getQualifiedNameAsString()));
      }
      XLS_ASSIGN_OR_RETURN(xls::BValue dv, CreateDefaultValue(octype, loc));
      return CValue(dv, octype);
    }
    if (ctor->getNumArgs() == 1) {
      XLS_ASSIGN_OR_RETURN(CValue pv, GenerateIR_Expr(ctor->getArg(0), loc));
      XLS_ASSIGN_OR_RETURN(xls::BValue converted,
                           GenTypeConvert(pv, octype, loc));
      return CValue(converted, octype, /*disable_type_check=*/false,
                    pv.lvalue());
    }
    return absl::UnimplementedError(ErrorMessage(
        loc, "Unsupported constructor argument count %i", ctor->getNumArgs()));
  }
  if (auto* cast = clang::dyn_cast<const clang::ArraySubscriptExpr>(expr)) {
    XLS_ASSIGN_OR_RETURN(CValue arr_val, GenerateIR_Expr(cast->getBase(), loc));
    // Implicit dereference
    if (arr_val.type()->Is<CPointerType>()) {
      XLSCC_CHECK_NE(arr_val.lvalue(), nullptr, loc);
      XLS_ASSIGN_OR_RETURN(arr_val, GenerateIR_Expr(arr_val.lvalue(), loc));
    }
    auto arr_type = arr_val.type()->As<CArrayType>();
    XLS_ASSIGN_OR_RETURN(CValue idx_val, GenerateIR_Expr(cast->getIdx(), loc));
    return CValue(
        context().fb->ArrayIndex(arr_val.rvalue(), {idx_val.rvalue()}, loc),
        arr_type->GetElementType());
  }
  // Access to a struct member, for example: x.foo
  if (auto* cast = clang::dyn_cast<const clang::MemberExpr>(expr)) {
    return GenerateIR_MemberExpr(cast, loc);
  }
  // Wraps another expression in parenthesis: (sub_expr).
  // This is irrelevant to XLS[cc], as the parenthesis already affected
  //  Clang's AST ordering.
  if (auto* cast = clang::dyn_cast<const clang::ParenExpr>(expr)) {
    return GenerateIR_Expr(cast->getSubExpr(), loc);
  }
  // A reference to a declaration using its identifier
  if (const auto* cast = clang::dyn_cast<const clang::DeclRefExpr>(expr)) {
    const clang::NamedDecl* named = cast->getFoundDecl();
    XLS_ASSIGN_OR_RETURN(CValue cval, GetIdentifier(named, loc));
    return cval;
  }
  // Wraps the value of an argument default
  if (auto* arg_expr = clang::dyn_cast<const clang::CXXDefaultArgExpr>(expr)) {
    return GenerateIR_Expr(arg_expr->getExpr(), loc);
  }
  // Wraps certain expressions evaluatable in a constant context
  // I am not sure when exactly Clang chooses to do this.
  if (auto* const_expr = clang::dyn_cast<const clang::ConstantExpr>(expr)) {
    return GenerateIR_Expr(const_expr->getSubExpr(), loc);
  }
  // This occurs inside of an ArrayInitLoopExpr, and wraps a value
  //  that is created by implication, rather than explicitly.
  if (auto* const_expr = clang::dyn_cast<const clang::OpaqueValueExpr>(expr)) {
    return GenerateIR_Expr(const_expr->getSourceExpr(), loc);
  }
  // The case in which I've seen Clang generate this is when a struct is
  //  initialized with an array inside.
  // struct ts { tss vv[4]; };
  if (auto* loop_expr = clang::dyn_cast<const clang::ArrayInitLoopExpr>(expr)) {
    XLS_ASSIGN_OR_RETURN(CValue expr,
                         GenerateIR_Expr(loop_expr->getCommonExpr(), loc));

    auto arr_type = std::dynamic_pointer_cast<CArrayType>(expr.type());
    XLSCC_CHECK(arr_type && (arr_type->GetSize() ==
                             loop_expr->getArraySize().getLimitedValue()),
                loc);

    return expr;
  }
  // An expression "T()" which creates a value-initialized rvalue of type T,
  // which is a non-class type. For example: return int();
  if (auto* scalar_init_expr =
          clang::dyn_cast<const clang::CXXScalarValueInitExpr>(expr)) {
    XLS_ASSIGN_OR_RETURN(
        shared_ptr<CType> ctype,
        TranslateTypeFromClang(scalar_init_expr->getType(), loc));
    XLS_ASSIGN_OR_RETURN(xls::BValue def, CreateDefaultValue(ctype, loc));
    return CValue(def, ctype);
  }
  // Implicitly generated value, as in an incomplete initializer list
  if (auto* implicit_value_init_expr =
          clang::dyn_cast<const clang::ImplicitValueInitExpr>(expr)) {
    XLS_ASSIGN_OR_RETURN(
        shared_ptr<CType> ctype,
        TranslateTypeFromClang(implicit_value_init_expr->getType(), loc));
    XLS_ASSIGN_OR_RETURN(xls::BValue def, CreateDefaultValue(ctype, loc));
    return CValue(def, ctype);
  }
  if (auto* default_init_expr =
          clang::dyn_cast<const clang::CXXDefaultInitExpr>(expr)) {
    return GenerateIR_Expr(default_init_expr->getExpr(), loc);
  }
  if (auto* string_literal_expr =
          clang::dyn_cast<const clang::StringLiteral>(expr)) {
    if (!(string_literal_expr->isOrdinary() || string_literal_expr->isUTF8())) {
      return absl::UnimplementedError("Only 8 bit character strings supported");
    }
    llvm::StringRef strref = string_literal_expr->getString();
    std::string str = strref.str();

    std::shared_ptr<CType> element_type(new CIntType(8, true, true));
    std::shared_ptr<CType> type(new CArrayType(element_type, str.size()));

    std::vector<xls::Value> elements;

    for (char c : str) {
      elements.push_back(xls::Value(xls::SBits(c, 8)));
    }

    XLS_ASSIGN_OR_RETURN(xls::Value arrval, xls::Value::Array(elements));

    return CValue(context().fb->Literal(arrval, loc), type);
  }
  if (auto* init_list = clang::dyn_cast<const clang::InitListExpr>(expr)) {
    XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                         TranslateTypeFromClang(expr->getType(), loc));

    if (expr->getType()->isRecordType()) {
      XLS_RETURN_IF_ERROR(
          FailIfTypeHasDtors(expr->getType()->getAsCXXRecordDecl()));
    }

    return CreateInitListValue(ctype, init_list, loc);
  }
  expr->dump();
  return absl::UnimplementedError(ErrorMessage(
      loc, "Unimplemented expression %s", expr->getStmtClassName()));
}

absl::Status Translator::MinSizeArraySlices(CValue& true_cv, CValue& false_cv,
                                            std::shared_ptr<CType>& result_type,
                                            const xls::SourceInfo& loc) {
  // Array slices are the size of source arrays, and indices just wrap around.
  // Take the smaller size
  if (true_cv.type()->Is<CArrayType>() && false_cv.type()->Is<CArrayType>() &&
      (*true_cv.type()->As<CArrayType>()->GetElementType() ==
       *false_cv.type()->As<CArrayType>()->GetElementType())) {
    int64_t min_size = std::min(true_cv.type()->As<CArrayType>()->GetSize(),
                                false_cv.type()->As<CArrayType>()->GetSize());
    result_type = std::make_shared<CArrayType>(
        true_cv.type()->As<CArrayType>()->GetElementType(), min_size);
    XLSCC_CHECK(true_cv.rvalue().valid(), loc);
    XLSCC_CHECK(false_cv.rvalue().valid(), loc);
    xls::BValue bval_0 = context().fb->Literal(xls::UBits(0, 32), loc);
    true_cv = CValue(
        context().fb->ArraySlice(true_cv.rvalue(), bval_0, min_size, loc),
        result_type);
    false_cv = CValue(
        context().fb->ArraySlice(false_cv.rvalue(), bval_0, min_size, loc),
        result_type);
  } else {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Select on different lvalue types %s vs %s",
        std::string(*true_cv.type()), std::string(*false_cv.type())));
  }
  return absl::OkStatus();
}

absl::StatusOr<CValue> Translator::GenerateIR_Expr(std::shared_ptr<LValue> expr,
                                                   const xls::SourceInfo& loc) {
  if (!expr->is_select()) {
    return GenerateIR_Expr(expr->leaf(), loc);
  }

  XLS_ASSIGN_OR_RETURN(CValue true_cv,
                       GenerateIR_Expr(expr->lvalue_true(), loc));
  XLS_ASSIGN_OR_RETURN(CValue false_cv,
                       GenerateIR_Expr(expr->lvalue_false(), loc));

  std::shared_ptr<CType> result_type = true_cv.type();

  if (*true_cv.type() != *false_cv.type()) {
    XLS_RETURN_IF_ERROR(
        MinSizeArraySlices(true_cv, false_cv, result_type, loc));
  }

  return Generate_TernaryOp(expr->cond(), true_cv, false_cv, result_type, loc);
}

absl::StatusOr<CValue> Translator::GenerateIR_MemberExpr(
    const clang::MemberExpr* expr, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(CValue leftval, GenerateIR_Expr(expr->getBase(), loc));

  if (leftval.type()->Is<CReferenceType>()) {
    XLSCC_CHECK_NE(leftval.lvalue().get(), nullptr, loc);
    XLSCC_CHECK(!leftval.lvalue()->is_null(), loc);
    XLS_ASSIGN_OR_RETURN(leftval, GenerateIR_Expr(leftval.lvalue(), loc));
  }

  XLS_ASSIGN_OR_RETURN(auto itype, ResolveTypeInstance(leftval.type()));

  auto sitype = std::dynamic_pointer_cast<CStructType>(itype);

  if (sitype == nullptr) {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Unimplemented member access on type %s", string(*itype)));
  }

  // Get the value referred to
  clang::ValueDecl* member = expr->getMemberDecl();

  // VarDecl for static values
  if (member->getKind() == clang::ValueDecl::Kind::Var) {
    XLS_ASSIGN_OR_RETURN(
        CValue val,
        TranslateVarDecl(clang::dyn_cast<const clang::VarDecl>(member), loc));
    return val;
  }
  if (member->getKind() == clang::ValueDecl::Kind::EnumConstant) {
    XLS_ASSIGN_OR_RETURN(
        CValue val,
        TranslateEnumConstantDecl(
            clang::dyn_cast<const clang::EnumConstantDecl>(member), loc));
    return val;
  }
  if (member->getKind() != clang::ValueDecl::Kind::Field) {
    // Otherwise only FieldDecl is supported. This is the non-static "foo.bar"
    // form.
    return absl::UnimplementedError(ErrorMessage(
        loc, "Unimplemented member expression %s", member->getDeclKindName()));
  }

  auto field = clang::dyn_cast<clang::FieldDecl>(member);
  const auto& fields_by_name = sitype->fields_by_name();
  auto found_field =
      // Upcast to NamedDecl because we track unique identifiers
      //  with NamedDecl pointers
      fields_by_name.find(absl::implicit_cast<const clang::NamedDecl*>(field));
  if (found_field == fields_by_name.end()) {
    return absl::NotFoundError(
        ErrorMessage(loc, "Member access on unknown field %s in type %s",
                     field->getNameAsString(), string(*itype)));
  }
  const CField& cfield = *found_field->second;
  // Upcast to NamedDecl because we track unique identifiers
  //  with NamedDecl pointers
  XLSCC_CHECK_EQ(cfield.name(),
                 absl::implicit_cast<const clang::NamedDecl*>(field), loc);

  xls::BValue bval;

  // Structs must have rvalue even if they also have lvalue
  if (cfield.type()->Is<CInstantiableTypeAlias>() ||
      cfield.type()->Is<CStructType>()) {
    XLS_ASSIGN_OR_RETURN(bval, CreateDefaultValue(cfield.type(), loc));
  }

  std::shared_ptr<LValue> lval;

  if (leftval.lvalue() != nullptr &&
      leftval.lvalue()->get_compound_or_null(cfield.index()) != nullptr) {
    lval = leftval.lvalue()->get_compound_or_null(cfield.index());
  } else if (leftval.rvalue().valid()) {
    bval = GetStructFieldXLS(leftval.rvalue(), cfield.index(), *sitype, loc);
  }

  XLS_ASSIGN_OR_RETURN(bool type_contains_lval,
                       cfield.type()->ContainsLValues(*this));

  if (type_contains_lval && lval == nullptr) {
    return absl::UnimplementedError(
        ErrorMessage(loc,
                     "Compound lvalue not present, lvalues in nested structs "
                     "not yet implemented"));
  }

  XLSCC_CHECK(type_contains_lval || bval.valid(), loc);

  return CValue(bval, cfield.type(), /*disable_type_check=*/false, lval);
}

absl::StatusOr<xls::BValue> Translator::CreateDefaultValue(
    std::shared_ptr<CType> t, const xls::SourceInfo& loc) {
  if (t->Is<CPointerType>()) {
    return xls::BValue();
  }

  XLS_ASSIGN_OR_RETURN(xls::Value value, CreateDefaultRawValue(t, loc));
  return context().fb->Literal(value, loc);
}

absl::StatusOr<xls::Value> Translator::CreateDefaultRawValue(
    std::shared_ptr<CType> t, const xls::SourceInfo& loc) {
  XLSCC_CHECK(t != nullptr, loc);
  if (t->Is<CIntType>()) {
    return xls::Value(xls::UBits(0, t->As<CIntType>()->width()));
  }
  if (t->Is<CFloatType>()) {
    return xls::Value(xls::UBits(0, t->GetBitWidth()));
  }
  if (t->Is<CBitsType>()) {
    auto it = t->As<CBitsType>();
    return xls::Value(xls::UBits(0, it->GetBitWidth()));
  }
  if (t->Is<CBoolType>()) {
    return xls::Value(xls::UBits(0, 1));
  }
  if (t->Is<CArrayType>()) {
    auto it = t->As<CArrayType>();
    std::vector<xls::Value> element_vals;
    XLS_ASSIGN_OR_RETURN(xls::Value default_elem_val,
                         CreateDefaultRawValue(it->GetElementType(), loc));
    element_vals.resize(it->GetSize(), default_elem_val);
    return xls::Value::ArrayOrDie(element_vals);
  }
  if (t->Is<CStructType>()) {
    auto it = t->As<CStructType>();
    vector<xls::Value> args;
    for (const std::shared_ptr<CField>& field : it->fields()) {
      XLS_ASSIGN_OR_RETURN(xls::Value fval,
                           CreateDefaultRawValue(field->type(), loc));
      args.push_back(fval);
    }
    return MakeStructXLS(args, *it);
  }
  if (t->Is<CInternalTuple>()) {
    auto it = t->As<CInternalTuple>();
    vector<xls::Value> args;
    for (const std::shared_ptr<CType>& field_type : it->fields()) {
      XLS_ASSIGN_OR_RETURN(xls::Value fval,
                           CreateDefaultRawValue(field_type, loc));
      args.push_back(fval);
    }
    return xls::Value::Tuple(args);
  }
  if (t->Is<CInstantiableTypeAlias>()) {
    XLS_ASSIGN_OR_RETURN(auto resolved, ResolveTypeInstance(t));
    return CreateDefaultRawValue(resolved, loc);
  }
  if (t->Is<CPointerType>() || t->Is<CReferenceType>() ||
      t->Is<CChannelType>()) {
    return xls::Value::Tuple({});
  }
  return absl::UnimplementedError(ErrorMessage(
      loc, "Don't know how to make default for type %s", std::string(*t)));
}

absl::StatusOr<CValue> Translator::CreateDefaultCValue(
    const std::shared_ptr<CType>& t, const xls::SourceInfo& loc) {
  std::shared_ptr<LValue> lval;
  XLS_ASSIGN_OR_RETURN(xls::BValue this_bval, CreateDefaultValue(t, loc));
  XLS_ASSIGN_OR_RETURN(bool contains_lvalues, t->ContainsLValues(*this));
  if (contains_lvalues) {
    if (t->Is<CStructType>() || t->Is<CInstantiableTypeAlias>()) {
      std::shared_ptr<CType> struct_type = t;
      if (t->Is<CInstantiableTypeAlias>()) {
        XLS_ASSIGN_OR_RETURN(struct_type, ResolveTypeInstance(t));
      }
      auto struct_type_ptr = struct_type->As<CStructType>();
      absl::flat_hash_map<int64_t, std::shared_ptr<LValue>> compound_lvals;
      for (uint64_t i = 0; i < struct_type_ptr->fields().size(); ++i) {
        std::shared_ptr<CType> field_type =
            struct_type_ptr->fields()[i]->type();

        XLS_ASSIGN_OR_RETURN(bool field_contains_lvalues,
                             field_type->ContainsLValues(*this));
        if (field_contains_lvalues) {
          compound_lvals[i] = std::make_shared<LValue>();
        }
      }

      if (!compound_lvals.empty()) {
        lval = std::make_shared<LValue>(compound_lvals);
      }
    } else if (t->Is<CChannelType>() || t->Is<CReferenceType>() ||
               t->Is<CPointerType>()) {
      lval = std::make_shared<LValue>();
    } else {
      return absl::UnimplementedError(ErrorMessage(
          loc, "Don't know how to create default lvalue for type %s",
          t->debug_string()));
    }
  }
  return CValue(this_bval, t, /*disable_type_check=*/true, lval);
}

absl::StatusOr<CValue> Translator::CreateInitListValue(
    const std::shared_ptr<CType>& t, const clang::InitListExpr* init_list,
    const xls::SourceInfo& loc) {
  if (t->Is<CArrayType>()) {
    auto array_t = t->As<CArrayType>();
    if (array_t->GetSize() < init_list->getNumInits()) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Too many initializers"));
    }
    XLS_ASSIGN_OR_RETURN(Pragma pragma,
                         FindPragmaForLoc(init_list->getBeginLoc()));
    if (pragma.type() != Pragma_ArrayAllowDefaultPad &&
        array_t->GetSize() != init_list->getNumInits() &&
        init_list->getNumInits() != 0 && error_on_uninitialized_) {
      return absl::InvalidArgumentError(
          ErrorMessage(loc, "Wrong number of initializers"));
    }
    std::vector<xls::BValue> element_vals;
    for (int i = 0; i < array_t->GetSize(); ++i) {
      const clang::Expr* this_init;
      if (i < init_list->getNumInits()) {
        this_init = init_list->getInit(i);
      } else {
        this_init = init_list->getArrayFiller();
      }
      xls::BValue this_val;
      if (auto init_list_expr =
              clang::dyn_cast<const clang::InitListExpr>(this_init)) {
        CValue this_cval;
        XLS_ASSIGN_OR_RETURN(this_cval,
                             CreateInitListValue(array_t->GetElementType(),
                                                 init_list_expr, loc));
        this_val = this_cval.rvalue();
      } else {
        XLS_ASSIGN_OR_RETURN(CValue expr_val, GenerateIR_Expr(this_init, loc));
        if (*expr_val.type() != *array_t->GetElementType()) {
          return absl::UnimplementedError(ErrorMessage(
              loc, "Wrong initializer type %s", string(*expr_val.type())));
        }
        this_val = expr_val.rvalue();
      }
      XLSCC_CHECK(this_val.valid(), loc);
      element_vals.push_back(this_val);
    }
    XLS_ASSIGN_OR_RETURN(xls::Type * xls_elem_type,
                         TranslateTypeToXLS(array_t->GetElementType(), loc));
    return CValue(context().fb->Array(element_vals, xls_elem_type, loc), t);
  }
  if (t->Is<CInstantiableTypeAlias>()) {
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> struct_type,
                         ResolveTypeInstance(t));
    auto struct_type_ptr = struct_type->As<CStructType>();
    XLSCC_CHECK_NE(nullptr, struct_type_ptr, loc);
    XLSCC_CHECK_EQ(struct_type_ptr->fields().size(), init_list->getNumInits(),
                   loc);

    PushContextGuard temporary_this_context(*this, loc);

    // Declare "this" so fields can refer to one another
    // Nesting of structs is acyclical, so this should be unique
    const clang::NamedDecl* this_decl = t->As<CInstantiableTypeAlias>()->base();

    {
      CValue this_val;
      XLS_ASSIGN_OR_RETURN(this_val, CreateDefaultCValue(t, loc));
      XLS_RETURN_IF_ERROR(DeclareVariable(this_decl, this_val, loc));
    }

    CHECK_NE(init_list->getNumInits(), 0);

    for (uint64_t i = 0; i < init_list->getNumInits(); ++i) {
      // This list always contains an init for each field, so the indices match
      std::shared_ptr<CType> field_type =
          struct_type_ptr->fields().at(i)->type();
      const clang::NamedDecl* field_name =
          struct_type_ptr->fields().at(i)->name();

      const clang::Expr* this_init =
          init_list->getInit(static_cast<unsigned int>(i));

      // Default inits can be self referencing, rather than referencing
      // the context in which the init list occurs. For example:
      // struct Foo { int a; int b = a; }
      // Otherwise, we want to continue to use the enclosing context:
      // struct Foo { Bar a = {1, 2}; Bar b = {a.x, 3}; };
      OverrideThisDeclGuard this_decl_guard(*this, this_decl,
                                            /*activate_now=*/false);

      if (clang::dyn_cast<const clang::CXXDefaultInitExpr>(this_init) !=
          nullptr) {
        this_decl_guard.activate();
      }

      XLS_ASSIGN_OR_RETURN(CValue value,
                           CreateInitValue(field_type, this_init, loc));

      XLSCC_CHECK(*value.type() == *struct_type_ptr->fields().at(i)->type(),
                  loc);

      {
        UnmaskAndIgnoreSideEffectsGuard unmask_guard(*this);

        XLS_RETURN_IF_ERROR(AssignMember(this_decl, field_name, value, loc));
      }
    }

    XLS_ASSIGN_OR_RETURN(CValue got_val, GetIdentifier(this_decl, loc));

    unique_decl_ids_.erase(this_decl);
    context().variables.erase(this_decl);

    return got_val;
  }
  return absl::UnimplementedError(ErrorMessage(
      loc, "Don't know how to interpret initializer list for type %s",
      string(*t)));
}

absl::StatusOr<xls::Value> Translator::EvaluateNode(xls::Node* node,
                                                    const xls::SourceInfo& loc,
                                                    bool do_check) {
  xls::IrInterpreter visitor({});
  absl::Status status = node->Accept(&visitor);
  if (!status.ok()) {
    auto err = absl::UnavailableError(
        ErrorMessage(loc,
                     "Couldn't evaluate node as a constant. Error from IR "
                     "interpreter was: %s",
                     status.message()));
    if (do_check) {
      XLS_LOG(ERROR) << err.ToString();
    } else {
      return err;
    }
  }
  xls::Value result = visitor.ResolveAsValue(node);
  return result;
}

absl::Status Translator::ShortCircuitBVal(xls::BValue& bval,
                                          const xls::SourceInfo& loc) {
  absl::flat_hash_set<xls::Node*> visited;
  return ShortCircuitNode(bval.node(), bval, nullptr, visited, loc);
}

absl::Status Translator::ShortCircuitNode(
    xls::Node* node, xls::BValue& top_bval, xls::Node* parent,
    absl::flat_hash_set<xls::Node*>& visited, const xls::SourceInfo& loc) {
  if (visited.contains(node)) {
    return absl::OkStatus();
  }

  visited.insert(node);

  // Depth-first to allow multi-step short circuits
  // Index based to avoid modify while iterating
  for (int oi = 0; oi < node->operand_count(); ++oi) {
    xls::Node* op = node->operand(oi);
    XLS_RETURN_IF_ERROR(ShortCircuitNode(op, top_bval, node, visited, loc));
  }

  // Don't duplicate literals
  if (node->Is<xls::Literal>()) {
    return absl::OkStatus();
  }

  absl::StatusOr<xls::Value> const_result =
      EvaluateNode(node, loc, /*do_check=*/false);

  // Try to replace the node with a literal
  if (const_result.ok()) {
    xls::BValue literal_bval =
        context().fb->Literal(const_result.value(), node->loc());

    if (parent != nullptr) {
      XLSCC_CHECK(parent->ReplaceOperand(node, literal_bval.node()), loc);
    } else {
      top_bval = literal_bval;
    }
    return absl::OkStatus();
  }

  if (!((node->op() == xls::Op::kAnd) || (node->op() == xls::Op::kOr))) {
    return absl::OkStatus();
  }

  for (xls::Node* op : node->operands()) {
    // Operands that can be evaluated will already have been turned into
    // literals by the above depth-first literalization
    if (!op->Is<xls::Literal>()) {
      continue;
    }
    xls::Literal* literal_node = op->As<xls::Literal>();

    const xls::Value& const_value = literal_node->value();

    if ((node->op() == xls::Op::kAnd) && (!const_value.IsAllZeros())) {
      continue;
    }
    if ((node->op() == xls::Op::kOr) && (!const_value.IsAllOnes())) {
      continue;
    }

    // Replace the node with its literal operand
    if (parent != nullptr) {
      XLSCC_CHECK(parent->ReplaceOperand(node, op), loc);
    } else {
      top_bval = xls::BValue(op, context().fb);
    }

    return absl::OkStatus();
  }

  return absl::OkStatus();
}

absl::StatusOr<xls::Value> Translator::EvaluateBVal(xls::BValue bval,
                                                    const xls::SourceInfo& loc,
                                                    bool do_check) {
  return EvaluateNode(bval.node(), loc, do_check);
}

absl::StatusOr<ConstValue> Translator::TranslateBValToConstVal(
    const CValue& bvalue, const xls::SourceInfo& loc, bool do_check) {
  if (!bvalue.rvalue().valid()) {
    return ConstValue(xls::Value(), bvalue.type(), /*disable_type_check=*/true);
  }
  XLS_ASSIGN_OR_RETURN(xls::Value const_value,
                       EvaluateBVal(bvalue.rvalue(), loc, do_check));
  return ConstValue(const_value, bvalue.type());
}

absl::Status Translator::GenerateIR_Compound(const clang::Stmt* body,
                                             clang::ASTContext& ctx) {
  if (body == nullptr) {
    // Empty block, nothing to do
    return absl::OkStatus();
  }

  if (clang::isa<clang::CompoundStmt>(body)) {
    for (const clang::Stmt* body_st : body->children()) {
      XLS_RETURN_IF_ERROR(GenerateIR_Stmt(body_st, ctx));
    }
  } else {
    // For single-line bodies
    XLS_RETURN_IF_ERROR(GenerateIR_Stmt(body, ctx));
  }

  return absl::OkStatus();
}

absl::StatusOr<CValue> Translator::GenerateIR_LocalChannel(
    const clang::NamedDecl* namedecl,
    const std::shared_ptr<CChannelType>& channel_type,
    const xls::SourceInfo& loc) {
  if (channel_type->GetMemorySize() > 0) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Internal memories unsupported"));
  }

  std::string ch_name = absl::StrFormat(
      "__internal_%s_%s_%i", context().sf->clang_decl->getNameAsString(),
      namedecl->getNameAsString(), next_local_channel_number_++);
  XLS_ASSIGN_OR_RETURN(xls::Type * item_type_xls,
                       TranslateTypeToXLS(channel_type->GetItemType(), loc));
  XLS_ASSIGN_OR_RETURN(
      xls::Channel * xls_channel,
      package_->CreateStreamingChannel(
          ch_name, xls::ChannelOps::kSendReceive, item_type_xls,
          /*initial_values=*/{}, /*fifo_config=*/xls::FifoConfig{.depth = 0},
          xls::FlowControl::kReadyValid));

  unused_xls_channel_ops_.push_back({xls_channel, /*is_send=*/true});
  unused_xls_channel_ops_.push_back({xls_channel, /*is_send=*/false});

  IOChannel new_channel;
  new_channel.item_type = channel_type->GetItemType();
  new_channel.unique_name = ch_name;
  new_channel.internal_to_function = true;

  IOChannel* new_channel_ptr = AddChannel(new_channel, loc);
  auto lvalue = std::make_shared<LValue>(new_channel_ptr);

  CValue cval(/*rvalue=*/xls::BValue(), channel_type,
              /*disable_type_check=*/true, lvalue);

  ChannelBundle bundle = {.regular = xls_channel};
  external_channels_by_internal_channel_.insert(
      std::make_pair(new_channel_ptr, bundle));

  return cval;
}

absl::Status Translator::GenerateIR_StaticDecl(const clang::VarDecl* vard,
                                               const clang::NamedDecl* namedecl,
                                               const xls::SourceInfo& loc) {
  // Intercept internal channel declarations, eg for ping-pong
  {
    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(vard->getType()));

    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> obj_type,
                         TranslateTypeFromClang(stripped.base, loc));

    if (obj_type->Is<CChannelType>() &&
        clang::dyn_cast<clang::CXXConstructExpr>(vard->getAnyInitializer())) {
      XLS_ASSIGN_OR_RETURN(
          CValue generated,
          GenerateIR_LocalChannel(
              namedecl, std::dynamic_pointer_cast<CChannelType>(obj_type),
              loc));

      return DeclareVariable(namedecl, generated, loc);
    }
  }

  // Not an internal channel
  bool use_on_reset = false;
  ConstValue init;
  CValue translated_without_side_effects;

  {
    PushContextGuard guard(*this, loc);
    context().mask_side_effects = true;
    context().any_side_effects_requested = false;

    // Check for side-effects
    XLS_ASSIGN_OR_RETURN(translated_without_side_effects,
                         TranslateVarDecl(vard, loc));
    if (context().any_side_effects_requested) {
      use_on_reset = true;
    } else {
      // Check for const-evaluatability
      absl::StatusOr<ConstValue> translate_result = TranslateBValToConstVal(
          translated_without_side_effects, loc, /*do_check=*/false);
      if (!translate_result.ok()) {
        use_on_reset = true;
      } else {
        init = translate_result.value();
      }
    }

    if (use_on_reset) {
      XLS_ASSIGN_OR_RETURN(
          xls::Value default_val,
          CreateDefaultRawValue(translated_without_side_effects.type(), loc));
      init = ConstValue(default_val, translated_without_side_effects.type());
    }
  }

  // If there are no side effects and it's const-qualified,
  // then state isn't needed. It can just be a literal.
  if (!use_on_reset && vard->getType().isConstQualified()) {
    XLS_RETURN_IF_ERROR(
        DeclareVariable(namedecl, translated_without_side_effects, loc));
    return absl::OkStatus();
  }

  XLS_RETURN_IF_ERROR(DeclareStatic(
      namedecl, init, translated_without_side_effects.lvalue(), loc));

  if (!use_on_reset) {
    return absl::OkStatus();
  }

  if (translated_without_side_effects.lvalue() != nullptr) {
    return absl::UnimplementedError(
        ErrorMessage(loc,
                     "Initializing statics containing LValues using "
                     "side-effects (eg __xls_on_reset)"));
  }

  // Select using __xlscc_on_reset
  CValue translated_with_side_effects = translated_without_side_effects;

  // First, if there are side-effects, retranslate with side-effects enabled,
  // conditional on __xlscc_on_reset
  XLS_ASSIGN_OR_RETURN(CValue on_reset_val, GetOnReset(loc));
  XLSCC_CHECK(on_reset_val.rvalue().valid(), loc);
  XLSCC_CHECK_EQ(on_reset_val.rvalue().BitCountOrDie(), 1, loc);

  PushContextGuard guard(*this, on_reset_val.rvalue(), loc);
  XLS_ASSIGN_OR_RETURN(translated_with_side_effects,
                       TranslateVarDecl(vard, loc));
  XLSCC_CHECK(translated_with_side_effects.rvalue().valid(), loc);

  // This assignment will generate a select on __xlscc_on_reset
  XLS_RETURN_IF_ERROR(Assign(namedecl, translated_with_side_effects, loc));

  return absl::OkStatus();
}

const clang::Expr* Translator::RemoveParensAndCasts(const clang::Expr* expr) {
  while (true) {
    if (auto as_cast = clang::dyn_cast<clang::ImplicitCastExpr>(expr)) {
      expr = as_cast->getSubExpr();
    } else if (auto as_paren = clang::dyn_cast<clang::ParenExpr>(expr)) {
      expr = as_paren->getSubExpr();
    } else {
      break;
    }
  }

  return expr;
}

const clang::CXXThisExpr* Translator::IsThisExpr(const clang::Expr* expr) {
  const clang::Expr* nested = RemoveParensAndCasts(expr);

  if (auto this_expr = clang::dyn_cast<clang::CXXThisExpr>(nested)) {
    return this_expr;
  }

  if (auto unary_expr = clang::dyn_cast<clang::UnaryOperator>(nested)) {
    if (auto this_expr =
            clang::dyn_cast<clang::CXXThisExpr>(unary_expr->getSubExpr())) {
      return this_expr;
    }
  }

  return nullptr;
}

absl::Status Translator::GenerateIR_ReturnStmt(const clang::ReturnStmt* rts,
                                               clang::ASTContext& ctx,
                                               const xls::SourceInfo& loc) {
  if (context().in_pipelined_for_body) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Returns in pipelined loop body unimplemented"));
  }
  const clang::Expr* rvalue = rts->getRetValue();

  if (rvalue != nullptr) {
    // TODO: Remove special 'this' handling
    if (IsThisExpr(rvalue)) {
      if (context().return_cval.valid()) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Compound LValue returns not yet supported"));
      }

      // context().return_val = context().fb->Tuple({}, loc);
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<LValue> ret_lval,
                           CreateReferenceValue(rvalue, loc));
      context().return_cval = CValue(ret_lval, context().return_type);
      return absl::OkStatus();
    }

    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> rvalue_type,
                         TranslateTypeFromClang(rvalue->getType(), loc));
    XLS_ASSIGN_OR_RETURN(bool return_type_has_lvalues,
                         context().return_type->ContainsLValues(*this));

    CValue return_cval;

    if (!return_type_has_lvalues) {
      XLS_ASSIGN_OR_RETURN(CValue rv, GenerateIR_Expr(rvalue, loc));
      XLS_ASSIGN_OR_RETURN(xls::BValue crv,
                           GenTypeConvert(rv, context().return_type, loc));
      return_cval = CValue(crv, context().return_type);
    } else {
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<LValue> ret_lval,
                           CreateReferenceValue(rvalue, loc));
      return_cval = CValue(ret_lval, context().return_type);
    }

    if (!context().return_cval.valid()) {
      context().return_cval = return_cval;
    } else {
      // This is the normal case, where the last return was conditional
      if (context().last_return_condition.valid()) {
        // If there are multiple returns with the same condition, this will
        // take the first one
        xls::BValue this_cond = context().full_condition.valid()
                                    ? context().full_condition
                                    : context().fb->Literal(xls::UBits(1, 1));

        // Select the previous return instead of this one if
        //  the last return condition is true or this one is false
        // Scenario A (sel_prev_ret_cond = true):
        //  if(something true) return last;  // take this return value
        //  if(something true) return this;
        // Scenario B (sel_prev_ret_cond = false):
        //  if(something false) return last;
        //  if(something true) return this;  // take this return value
        // Scenario C  (sel_prev_ret_cond = true):
        //  if(something true) return last;  // take this return value
        //  if(something false) return this;
        // Scenario D  (sel_prev_ret_cond = false):
        //  if(something false) return last;
        //  if(something false) return this;
        //  return later_on;                 // take this return value
        // Scenario E  (sel_prev_ret_cond = true):
        //  return earnier_on;               // take this return value
        //  if(something true) return last;
        xls::BValue sel_prev_ret_cond = context().fb->Or(
            context().last_return_condition, context().fb->Not(this_cond));

        if (!return_type_has_lvalues) {
          XLSCC_CHECK(context().return_cval.rvalue().valid(), loc);
          xls::BValue return_bval = context().fb->Select(
              sel_prev_ret_cond, context().return_cval.rvalue(),
              return_cval.rvalue(), loc);
          context().return_cval = CValue(return_bval, context().return_type);
        } else {
          XLSCC_CHECK(context().return_cval.lvalue() != nullptr, loc);
          auto return_lval = std::make_shared<LValue>(
              sel_prev_ret_cond, context().return_cval.lvalue(),
              return_cval.lvalue());
          context().return_cval = CValue(return_lval, context().return_type);
        }
      } else {
        // In the case of multiple unconditional returns, take the first one
        // (no-op)
      }
    }

    if (context().full_condition.valid()) {
      context().last_return_condition = context().full_condition;
    } else {
      context().last_return_condition = context().fb->Literal(xls::UBits(1, 1));
    }
  }

  xls::BValue reach_here_cond = context().full_condition_bval(loc);

  if (!context().have_returned_condition.valid()) {
    context().have_returned_condition = reach_here_cond;
  } else {
    context().have_returned_condition =
        context().fb->Or(reach_here_cond, context().have_returned_condition);
  }

  XLS_RETURN_IF_ERROR(and_condition(
      context().fb->Not(context().have_returned_condition, loc), loc));

  return absl::OkStatus();
}

absl::Status Translator::GenerateIR_Stmt(const clang::Stmt* stmt,
                                         clang::ASTContext& ctx) {
  const xls::SourceInfo loc = GetLoc(*stmt);

  if (const clang::Expr* expr = clang::dyn_cast<const clang::Expr>(stmt)) {
    XLS_ASSIGN_OR_RETURN(absl::StatusOr<CValue> rv, GenerateIR_Expr(expr, loc));
    return rv.status();
  }
  if (auto rts = clang::dyn_cast<const clang::ReturnStmt>(stmt)) {
    return GenerateIR_ReturnStmt(rts, ctx, loc);
  }
  if (auto declstmt = clang::dyn_cast<const clang::DeclStmt>(stmt)) {
    for (auto decl : declstmt->decls()) {
      if (clang::isa<clang::TypedefDecl>(decl)) {
        continue;
      }
      if (clang::isa<clang::StaticAssertDecl>(decl)) {
        continue;
      }
      if (clang::isa<clang::EnumDecl>(decl)) {
        continue;
      }
      if (clang::isa<clang::TypeAliasDecl>(decl)) {
        continue;
      }
      if (auto recd = clang::dyn_cast<const clang::RecordDecl>(decl)) {
        XLS_RETURN_IF_ERROR(ScanStruct(recd));
      } else {
        auto vard = clang::dyn_cast<const clang::VarDecl>(decl);
        if (vard == nullptr) {
          return absl::UnimplementedError(ErrorMessage(
              loc, "DeclStmt other than Var (%s)", decl->getDeclKindName()));
        }

        if (vard->isStaticLocal() || vard->isStaticDataMember()) {
          XLS_RETURN_IF_ERROR(GenerateIR_StaticDecl(vard, vard, GetLoc(*vard)));
        } else {
          XLS_ASSIGN_OR_RETURN(CValue translated,
                               TranslateVarDecl(vard, GetLoc(*vard)));
          XLS_RETURN_IF_ERROR(DeclareVariable(vard, translated, loc));
        }
      }
    }
  } else if (const auto* pasm =
                 clang::dyn_cast<const clang::GCCAsmStmt>(stmt)) {
    std::string sasm = pasm->getAsmString()->getString().str();
    vector<xls::BValue> args;

    for (int i = 0; i < pasm->getNumInputs(); ++i) {
      const clang::Expr* expr = pasm->getInputExpr(i);
      if (expr->isIntegerConstantExpr(ctx)) {
        const std::string name = pasm->getInputConstraint(i).str();
        XLS_ASSIGN_OR_RETURN(auto val, EvaluateInt64(*expr, ctx, loc));
        sasm = std::regex_replace(
            sasm, std::regex(absl::StrFormat(R"(\b%s\b)", name)),
            absl::StrCat(val));
      } else {
        XLS_ASSIGN_OR_RETURN(CValue ret, GenerateIR_Expr(expr, loc));
        args.emplace_back(ret.rvalue());
      }
    }

    // Unique function name
    RE2::GlobalReplace(&sasm, "\\(fid\\)",
                       absl::StrFormat("fid%i", next_asm_number_++));
    // Unique IR instruction name
    RE2::GlobalReplace(&sasm, "\\(aid\\)",
                       absl::StrFormat("aid%i", next_asm_number_++));
    // File location
    RE2::GlobalReplace(&sasm, "\\(loc\\)", loc.ToString());

    if (pasm->getNumOutputs() != 1) {
      return absl::UnimplementedError(
          absl::StrFormat("asm must have exactly 1 output"));
    }

    XLS_ASSIGN_OR_RETURN(CValue out_val,
                         GenerateIR_Expr(pasm->getOutputExpr(0), loc));

    // verify_function_only because external channels are defined up-front,
    //  which generates "No receive/send node" errors
    XLS_ASSIGN_OR_RETURN(
        xls::Function * af,
        xls::Parser::ParseFunction(sasm, package_,
                                   /*verify_function_only=*/true));

    // No type conversion in or out: inline IR can do whatever it wants.
    // If you use inline IR, you should know exactly what you're doing.
    xls::BValue fret = context().fb->Invoke(args, af, loc);

    XLS_RETURN_IF_ERROR(
        Assign(pasm->getOutputExpr(0), CValue(fret, out_val.type()), loc));
  } else if (const auto* ifst = clang::dyn_cast<const clang::IfStmt>(stmt)) {
    XLS_ASSIGN_OR_RETURN(CValue cond, GenerateIR_Expr(ifst->getCond(), loc));
    XLSCC_CHECK(cond.type()->Is<CBoolType>(), loc);
    if (ifst->getInit() != nullptr) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Unimplemented C++17 if initializers"));
    }
    if (ifst->getThen() != nullptr) {
      PushContextGuard context_guard(*this, cond.rvalue(), loc);
      XLS_RETURN_IF_ERROR(GenerateIR_Compound(ifst->getThen(), ctx));
    }
    if (ifst->getElse() != nullptr) {
      PushContextGuard context_guard(*this, context().fb->Not(cond.rvalue()),
                                     loc);
      XLS_RETURN_IF_ERROR(GenerateIR_Compound(ifst->getElse(), ctx));
    }
  } else if (auto forst = clang::dyn_cast<const clang::ForStmt>(stmt)) {
    XLS_RETURN_IF_ERROR(GenerateIR_Loop(
        /*always_first_iter=*/false, forst->getInit(), forst->getCond(),
        forst->getInc(), forst->getBody(), GetPresumedLoc(*forst), loc, ctx));
  } else if (auto forst = clang::dyn_cast<const clang::WhileStmt>(stmt)) {
    XLS_RETURN_IF_ERROR(GenerateIR_Loop(/*always_first_iter=*/false,
                                        /*init=*/nullptr, forst->getCond(),
                                        /*inc=*/nullptr, forst->getBody(),
                                        GetPresumedLoc(*forst), loc, ctx));
  } else if (auto dost = clang::dyn_cast<const clang::DoStmt>(stmt)) {
    XLS_RETURN_IF_ERROR(GenerateIR_Loop(/*always_first_iter=*/true,
                                        /*init=*/nullptr, dost->getCond(),
                                        /*inc=*/nullptr, dost->getBody(),
                                        GetPresumedLoc(*dost), loc, ctx));
  } else if (auto switchst = clang::dyn_cast<const clang::SwitchStmt>(stmt)) {
    return GenerateIR_Switch(switchst, ctx, loc);
  } else if (clang::isa<clang::ContinueStmt>(stmt)) {
    // Continue should be used inside of for loop bodies only
    XLSCC_CHECK(context().in_for_body, loc);
    context().relative_continue_condition =
        context().relative_condition_bval(loc);
    // Make the rest of the block no-op
    XLS_RETURN_IF_ERROR(
        and_condition(context().fb->Literal(xls::UBits(0, 1), loc), loc));
  } else if (clang::isa<clang::BreakStmt>(stmt)) {
    if (context().in_for_body) {
      // We are in a for body
      XLSCC_CHECK(!context().in_switch_body, loc);
      context().relative_break_condition =
          context().relative_condition_bval(loc);

      // Make the rest of the block no-op
      XLS_RETURN_IF_ERROR(
          and_condition(context().fb->Literal(xls::UBits(0, 1), loc), loc));
    } else {
      // We are in a switch body
      XLSCC_CHECK(context().in_switch_body, loc);
      // We use the original condition because we only care about
      //  enclosing conditions, such as if(...) { break; }
      //  Not if(...) {return;} break;
      if (context().full_condition_on_enter_block.node() ==
          context().full_switch_cond.node()) {
        context().hit_break = true;
      } else {
        context().relative_break_condition =
            context().relative_condition_bval(loc);

        // Make the rest of the block no-op
        XLS_RETURN_IF_ERROR(
            and_condition(context().fb->Literal(xls::UBits(0, 1), loc), loc));
      }
    }
  } else if (clang::isa<clang::CompoundStmt>(stmt)) {
    PushContextGuard context_guard(*this, loc);
    XLS_RETURN_IF_ERROR(GenerateIR_Compound(stmt, ctx));
  } else if (clang::isa<clang::NullStmt>(stmt)) {
    // Empty line (just ;)
  } else if (auto label_stmt = clang::dyn_cast<const clang::LabelStmt>(stmt)) {
    // Just ignore labels for now
    return GenerateIR_Stmt(label_stmt->getSubStmt(), ctx);
  } else {
    stmt->dump();
    return absl::UnimplementedError(ErrorMessage(
        loc, "Unimplemented construct %s", stmt->getStmtClassName()));
  }
  return absl::OkStatus();
}

int Debug_CountNodes(const xls::Node* node,
                     std::set<const xls::Node*>& visited) {
  if (visited.find(node) != visited.end()) {
    return 0;
  }
  visited.insert(node);

  int ret = 1;
  for (const xls::Node* child : node->operands()) {
    ret += Debug_CountNodes(child, visited);
  }
  return ret;
}

std::string Debug_NodeToInfix(xls::BValue bval) {
  if (bval.node() == nullptr) {
    return "[null]";
  }
  int64_t n_printed = 0;
  return Debug_NodeToInfix(bval.node(), n_printed);
}

std::string Debug_NodeToInfix(const xls::Node* node, int64_t& n_printed) {
  ++n_printed;
  if (n_printed > 100) {
    return "[...]";
  }

  if (node->Is<xls::Literal>()) {
    const xls::Literal* literal = node->As<xls::Literal>();
    if (literal->value().kind() == xls::ValueKind::kBits) {
      return absl::StrFormat("%li", literal->value().bits().ToInt64().value());
    }
  }
  if (node->Is<xls::Param>()) {
    const xls::Param* param = node->As<xls::Param>();
    return param->GetName();
  }
  if (node->Is<xls::UnOp>()) {
    const xls::UnOp* op = node->As<xls::UnOp>();
    if (op->op() == xls::Op::kNot) {
      return absl::StrFormat("!%s",
                             Debug_NodeToInfix(op->operand(0), n_printed));
    }
  }
  if (node->op() == xls::Op::kSGt) {
    return absl::StrFormat("(%s > %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kSLt) {
    return absl::StrFormat("(%s < %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kSLe) {
    return absl::StrFormat("(%s <= %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kEq) {
    return absl::StrFormat("(%s == %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kAnd) {
    return absl::StrFormat("(%s & %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kOr) {
    return absl::StrFormat("(%s | %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kAdd) {
    return absl::StrFormat("(%s + %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kSignExt) {
    return absl::StrFormat("%s",
                           Debug_NodeToInfix(node->operand(0), n_printed));
  }
  if (node->op() == xls::Op::kSel) {
    return absl::StrFormat("(%s ? %s : %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(2), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }

  return absl::StrFormat("[unsupported %s / %s]", node->GetName(),
                         typeid(*node).name());
}

std::string Debug_VariablesChangedBetween(const TranslationContext& before,
                                          const TranslationContext& after) {
  std::ostringstream ostr;

  for (const auto& [key, value] : before.variables) {
    if (after.variables.contains(key)) {
      ostr << "changed " << key->getNameAsString() << ": "
           << before.variables.at(key).debug_string() << " -> "
           << after.variables.at(key).debug_string() << "\n";
    } else {
      ostr << "only after " << key->getNameAsString() << ": "
           << after.variables.at(key).debug_string() << "\n";
    }
  }

  for (const auto& [key, value] : before.variables) {
    if (!after.variables.contains(key)) {
      ostr << "only before " << key->getNameAsString() << ": "
           << before.variables.at(key).debug_string() << "\n";
    }
  }

  return ostr.str();
}

absl::StatusOr<Z3_lbool> Translator::CheckAssumptions(
    absl::Span<xls::Node*> positive_nodes,
    absl::Span<xls::Node*> negative_nodes, Z3_solver& solver,
    xls::solvers::z3::IrTranslator& z3_translator) {
  Z3_context ctx = z3_translator.ctx();
  xls::solvers::z3::ScopedErrorHandler seh(ctx);

  std::vector<Z3_ast> z3_nodes_asserted;
  for (xls::Node* node : positive_nodes) {
    CHECK_EQ(node->BitCountOrDie(), 1);
    Z3_ast z3_node = z3_translator.GetTranslation(node);
    z3_nodes_asserted.push_back(
        xls::solvers::z3::BitVectorToBoolean(ctx, z3_node));
  }
  for (xls::Node* node : negative_nodes) {
    CHECK_EQ(node->BitCountOrDie(), 1);
    Z3_ast z3_node = z3_translator.GetTranslation(node);
    Z3_ast z3_node_not = Z3_mk_bvnot(z3_translator.ctx(), z3_node);
    z3_nodes_asserted.push_back(
        xls::solvers::z3::BitVectorToBoolean(ctx, z3_node_not));
  }

  if (z3_rlimit_ >= 0) {
    z3_translator.SetRlimit(z3_rlimit_);
  }

  Z3_lbool satisfiable = Z3_solver_check_assumptions(
      ctx, solver, static_cast<unsigned int>(z3_nodes_asserted.size()),
      z3_nodes_asserted.data());

  if (seh.status().ok()) {
    return satisfiable;
  }
  return seh.status();
}

absl::StatusOr<bool> Translator::BitMustBe(bool assert_value, xls::BValue& bval,
                                           Z3_solver& solver, Z3_context ctx,
                                           const xls::SourceInfo& loc) {
  // Invalid is interpreted as literal 1
  if (!bval.valid()) {
    return assert_value;
  }

  // Simplify break logic in easy ways;
  // Z3 fails to solve some cases without this.

  XLS_RETURN_IF_ERROR(ShortCircuitBVal(bval, loc));

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<xls::solvers::z3::IrTranslator> z3_translator,
      xls::solvers::z3::IrTranslator::CreateAndTranslate(
          /*ctx=*/ctx,
          /*source=*/bval.node(),
          /*allow_unsupported=*/false));

  absl::Span<xls::Node*> positive_assumptions, negative_assumptions;
  xls::Node* assumptions[] = {bval.node()};

  if (assert_value) {
    positive_assumptions = absl::Span<xls::Node*>();
    negative_assumptions = absl::Span<xls::Node*>(assumptions);
  } else {
    positive_assumptions = absl::Span<xls::Node*>(assumptions);
    negative_assumptions = absl::Span<xls::Node*>();
  }

  XLS_ASSIGN_OR_RETURN(
      Z3_lbool result,
      CheckAssumptions(positive_assumptions, negative_assumptions, solver,
                       *z3_translator));

  // No combination of variables can satisfy !break condition.
  return result == Z3_L_FALSE;
}

void GeneratedFunction::SortNamesDeterministically(
    std::vector<const clang::NamedDecl*>& names) const {
  std::sort(names.begin(), names.end(),
            [this](const clang::NamedDecl* a, const clang::NamedDecl* b) {
              return declaration_order_by_name_.at(a) <
                     declaration_order_by_name_.at(b);
            });
}

std::vector<const clang::NamedDecl*>
GeneratedFunction::GetDeterministicallyOrderedStaticValues() const {
  std::vector<const clang::NamedDecl*> ret;
  for (const auto& [decl, _] : static_values) {
    ret.push_back(decl);
  }
  SortNamesDeterministically(ret);
  return ret;
}

absl::Status Translator::CheckInitIntervalValidity(int initiation_interval_arg,
                                                   const xls::SourceInfo& loc) {
  if (initiation_interval_arg != 1) {
    std::string message = ErrorMessage(
        loc,
        "Only initiation interval 1 supported, %i requested, defaulting to 1",
        initiation_interval_arg);
    if (error_on_init_interval_) {
      return absl::UnimplementedError(message);
    }
    XLS_LOG(WARNING) << message;
  }
  return absl::OkStatus();
}

// First, flatten the statements in the switch
// It follows a strange pattern where
// case X: foo(); bar(); break;
// Has the form:
// case X: { foo(); } bar(); break;
// And even:
// case X: case Y: bar(); break;
// Has the form:
// case X: { case Y: } bar(); break;
static void FlattenCaseOrDefault(
    const clang::Stmt* stmt, clang::ASTContext& ctx,
    std::vector<const clang::Stmt*>& flat_statements) {
  flat_statements.push_back(stmt);
  if (auto case_it = clang::dyn_cast<const clang::CaseStmt>(stmt)) {
    FlattenCaseOrDefault(case_it->getSubStmt(), ctx, flat_statements);
  } else if (auto default_it =
                 clang::dyn_cast<const clang::DefaultStmt>(stmt)) {
    FlattenCaseOrDefault(default_it->getSubStmt(), ctx, flat_statements);
  }
}

absl::Status Translator::GenerateIR_Switch(const clang::SwitchStmt* switchst,
                                           clang::ASTContext& ctx,
                                           const xls::SourceInfo& loc) {
  PushContextGuard switch_guard(*this, loc);
  context().in_switch_body = true;
  context().in_for_body = false;

  if (switchst->getInit() != nullptr) {
    return absl::UnimplementedError(ErrorMessage(loc, "Switch init"));
  }
  XLS_ASSIGN_OR_RETURN(CValue switch_val,
                       GenerateIR_Expr(switchst->getCond(), loc));
  if (!switch_val.type()->Is<CIntType>()) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Switch on non-integers"));
  }

  // (See comment for FlattenCaseOrDefault())
  std::vector<const clang::Stmt*> flat_statements;
  auto body = switchst->getBody();
  for (const clang::Stmt* child : body->children()) {
    FlattenCaseOrDefault(child, ctx, flat_statements);
  }

  // Scan all cases to create default condition
  std::vector<xls::BValue> case_conds;
  for (const clang::Stmt* stmt : flat_statements) {
    if (auto case_it = clang::dyn_cast<const clang::CaseStmt>(stmt)) {
      const xls::SourceInfo loc = GetLoc(*case_it);
      XLS_ASSIGN_OR_RETURN(CValue case_val,
                           GenerateIR_Expr(case_it->getLHS(), loc));
      auto case_int_type = std::dynamic_pointer_cast<CIntType>(case_val.type());
      XLSCC_CHECK(case_int_type, loc);
      if (*switch_val.type() != *case_int_type) {
        return absl::UnimplementedError(ErrorMessage(loc, ""));
      }
      if (case_it->getRHS() != nullptr) {
        return absl::UnimplementedError(ErrorMessage(loc, "Case RHS"));
      }
      xls::BValue case_condition =
          context().fb->Eq(switch_val.rvalue(), case_val.rvalue(), loc);
      case_conds.emplace_back(case_condition);
    }
  }

  xls::BValue accum_cond;

  // for indexing into case_conds
  int case_idx = 0;
  for (const clang::Stmt* stmt : flat_statements) {
    const xls::SourceInfo loc = GetLoc(*stmt);
    xls::BValue cond;

    if (clang::isa<clang::CaseStmt>(stmt)) {
      cond = case_conds[case_idx++];
    } else if (clang::isa<clang::DefaultStmt>(stmt)) {
      cond = (case_conds.empty())
                 ? context().fb->Literal(xls::UBits(1, 1), loc)
                 : context().fb->Not(context().fb->Or(case_conds, loc), loc);
    } else {
      // For anything other than a case or break, translate it through
      //  the default path. case and break  can still occur inside of
      //  CompoundStmts, and will be processed in GenerateIR_Stmt().

      // No condition = false
      xls::BValue and_accum = accum_cond.valid()
                                  ? accum_cond
                                  : context().fb->Literal(xls::UBits(0, 1));
      // Break goes through this path, sets hit_break
      auto ocond = context().full_condition;
      PushContextGuard stmt_guard(*this, and_accum, loc);
      context().hit_break = false;
      context().full_switch_cond = ocond;
      // Variables declared should be visible in the outer context
      context().propagate_declarations = true;

      XLS_RETURN_IF_ERROR(GenerateIR_Compound(stmt, ctx));

      if (context().hit_break) {
        accum_cond = xls::BValue();
      }
      continue;
    }

    // This was a case or default
    if (accum_cond.valid()) {
      accum_cond = context().fb->Or(cond, accum_cond, loc);
    } else {
      accum_cond = cond;
    }
  }

  XLSCC_CHECK(case_idx == case_conds.size(), loc);
  return absl::OkStatus();
}

absl::StatusOr<int64_t> Translator::EvaluateInt64(
    const clang::Expr& expr, const class clang::ASTContext& ctx,
    const xls::SourceInfo& loc) {
  clang::Expr::EvalResult result;
  if (!expr.EvaluateAsInt(result, ctx, clang::Expr::SE_NoSideEffects,
                          /*InConstantContext=*/false)) {
    return absl::InvalidArgumentError(
        ErrorMessage(loc, "Failed to evaluate expression as int"));
  }
  const clang::APValue& val = result.Val;
  const llvm::APSInt& aps = val.getInt();

  return aps.getExtValue();
}

absl::StatusOr<bool> Translator::EvaluateBool(
    const clang::Expr& expr, const class clang::ASTContext& ctx,
    const xls::SourceInfo& loc) {
  bool result;
  if (!expr.EvaluateAsBooleanCondition(result, ctx,
                                       /*InConstantContext=*/false)) {
    return absl::InvalidArgumentError(
        ErrorMessage(loc, "Failed to evaluate expression as bool"));
  }
  return result;
}

absl::StatusOr<std::shared_ptr<CType>> Translator::TranslateTypeFromClang(
    clang::QualType t, const xls::SourceInfo& loc) {
  const clang::Type* type = t.getTypePtr();

  XLS_ASSIGN_OR_RETURN(bool channel, TypeIsChannel(t, loc));
  if (channel) {
    XLSCC_CHECK_NE(context().ast_context, nullptr, loc);
    XLS_ASSIGN_OR_RETURN(auto channel_type,
                         GetChannelType(t, *context().ast_context, loc));
    return channel_type;
  }

  if (auto builtin = clang::dyn_cast<const clang::BuiltinType>(type)) {
    if (builtin->isVoidType()) {
      return shared_ptr<CType>(new CVoidType());
    }
    if (builtin->isNullPtrType()) {
      return absl::UnimplementedError(ErrorMessage(loc, "nullptr"));
    }
    if (builtin->isFloatingType()) {
      switch (builtin->getKind()) {
        case clang::BuiltinType::Kind::Float:
          return shared_ptr<CType>(new CFloatType(/*double_precision=*/false));
        case clang::BuiltinType::Kind::Double:
          return shared_ptr<CType>(new CFloatType(/*double_precision=*/true));
        default:
          return absl::UnimplementedError(absl::StrFormat(
              "Unsupported BuiltIn type %i", builtin->getKind()));
      }
    }
    if (!builtin->isInteger()) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "BuiltIn type other than integer"));
    }
    switch (builtin->getKind()) {
      case clang::BuiltinType::Kind::Short:
        return shared_ptr<CType>(new CIntType(16, true));
      case clang::BuiltinType::Kind::UShort:
        return shared_ptr<CType>(new CIntType(16, false));
      case clang::BuiltinType::Kind::Int:
        return shared_ptr<CType>(new CIntType(32, true));
      case clang::BuiltinType::Kind::Long:
      case clang::BuiltinType::Kind::LongLong:
        return shared_ptr<CType>(new CIntType(64, true));
      case clang::BuiltinType::Kind::UInt:
        return shared_ptr<CType>(new CIntType(32, false));
      case clang::BuiltinType::Kind::ULong:
      case clang::BuiltinType::Kind::ULongLong:
        return shared_ptr<CType>(new CIntType(64, false));
      case clang::BuiltinType::Kind::Bool:
        return shared_ptr<CType>(new CBoolType());
      case clang::BuiltinType::Kind::SChar:
        return shared_ptr<CType>(new CIntType(8, true, false));
      case clang::BuiltinType::Kind::Char_S:  // These depend on the target
        return shared_ptr<CType>(new CIntType(8, true, true));
      case clang::BuiltinType::Kind::UChar:
        return shared_ptr<CType>(new CIntType(8, false, false));
      case clang::BuiltinType::Kind::Char_U:
        return shared_ptr<CType>(new CIntType(8, false, true));
      default:
        return absl::UnimplementedError(
            absl::StrFormat("Unsupported BuiltIn type %i", builtin->getKind()));
    }
  } else if (auto enum_type = clang::dyn_cast<const clang::EnumType>(type)) {
    clang::EnumDecl* decl = enum_type->getDecl();
    int width = decl->getNumPositiveBits() + decl->getNumNegativeBits();
    bool is_signed = decl->getNumNegativeBits() > 0;
    absl::btree_map<std::string, int64_t> variants_by_name;
    for (auto variant : decl->decls()) {
      auto variant_decl =
          clang::dyn_cast<const clang::EnumConstantDecl>(variant);
      auto value = variant_decl->getInitVal();
      variants_by_name.insert(
          {variant_decl->getNameAsString(), value.getExtValue()});
    }

    return shared_ptr<CType>(new CEnumType(decl->getNameAsString(), width,
                                           is_signed,
                                           std::move(variants_by_name)));
  } else if (type->getTypeClass() ==
             clang::Type::TypeClass::TemplateSpecialization) {
    // Up-cast to avoid multiple inheritance of getAsRecordDecl()
    std::shared_ptr<CInstantiableTypeAlias> ret(
        new CInstantiableTypeAlias(type->getAsRecordDecl()));
    return ret;
  } else if (auto record = clang::dyn_cast<const clang::RecordType>(type)) {
    clang::RecordDecl* decl = record->getDecl();

    if (clang::isa<clang::RecordDecl>(decl)) {
      return std::shared_ptr<CType>(new CInstantiableTypeAlias(
          decl->getTypeForDecl()->getAsRecordDecl()));
    }
    return absl::UnimplementedError(ErrorMessage(
        loc, "Unsupported recordtype kind %s in translate: %s",
        // getDeclKindName() is inherited from multiple base classes, so
        //  it is necessary to up-cast before calling it to avoid an error.
        absl::implicit_cast<const clang::Decl*>(decl)->getDeclKindName(),
        decl->getNameAsString()));
  } else if (auto array =
                 clang::dyn_cast<const clang::ConstantArrayType>(type)) {
    XLS_ASSIGN_OR_RETURN(shared_ptr<CType> type,
                         TranslateTypeFromClang(array->getElementType(), loc));
    return std::shared_ptr<CType>(
        new CArrayType(type, array->getSize().getZExtValue()));
  } else if (auto td = clang::dyn_cast<const clang::TypedefType>(type)) {
    return TranslateTypeFromClang(td->getDecl()->getUnderlyingType(), loc);
  } else if (auto elab = clang::dyn_cast<const clang::ElaboratedType>(type)) {
    return TranslateTypeFromClang(elab->getCanonicalTypeInternal(), loc);
  } else if (auto aut = clang::dyn_cast<const clang::AutoType>(type)) {
    return TranslateTypeFromClang(
        aut->getContainedDeducedType()->getDeducedType(), loc);
  } else if (auto subst =
                 clang::dyn_cast<const clang::SubstTemplateTypeParmType>(
                     type)) {
    return TranslateTypeFromClang(subst->getReplacementType(), loc);
  } else if (auto dec = clang::dyn_cast<const clang::DecayedType>(type)) {
    // No pointer support
    return TranslateTypeFromClang(dec->getOriginalType(), loc);
  } else if (auto lval =
                 clang::dyn_cast<const clang::LValueReferenceType>(type)) {
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> pointee_type,
                         TranslateTypeFromClang(lval->getPointeeType(), loc));
    return std::shared_ptr<CType>(new CReferenceType(pointee_type));
  } else if (auto lval =
                 clang::dyn_cast<const clang::RValueReferenceType>(type)) {
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> pointee_type,
                         TranslateTypeFromClang(lval->getPointeeType(), loc));
    return std::shared_ptr<CType>(new CReferenceType(pointee_type));
  } else if (auto lval = clang::dyn_cast<const clang::ParenType>(type)) {
    return TranslateTypeFromClang(lval->desugar(), loc);
  } else if (auto lval = clang::dyn_cast<const clang::PointerType>(type)) {
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> pointee_type,
                         TranslateTypeFromClang(lval->getPointeeType(), loc));
    if (context().ignore_pointers) {
      return pointee_type;
    }
    return std::shared_ptr<CType>(new CPointerType(pointee_type));
  }
  return absl::UnimplementedError(
      ErrorMessage(loc, "Unsupported type class in translate: %s",
                   type->getTypeClassName()));
}

absl::StatusOr<xls::Type*> Translator::TranslateTypeToXLS(
    std::shared_ptr<CType> t, const xls::SourceInfo& loc) {
  if (t->Is<CIntType>()) {
    auto it = t->As<CIntType>();
    return package_->GetBitsType(it->width());
  }
  if (t->Is<CFloatType>()) {
    return package_->GetBitsType(t->GetBitWidth());
  }
  if (t->Is<CBitsType>()) {
    auto it = t->As<CBitsType>();
    return package_->GetBitsType(it->GetBitWidth());
  }
  if (t->Is<CBoolType>()) {
    return package_->GetBitsType(1);
  }
  if (t->Is<CInstantiableTypeAlias>()) {
    XLS_ASSIGN_OR_RETURN(auto ctype, ResolveTypeInstance(t));
    return TranslateTypeToXLS(ctype, loc);
  }
  if (t->Is<CStructType>()) {
    auto it = t->As<CStructType>();
    std::vector<xls::Type*> members;
    for (auto it2 = it->fields().rbegin(); it2 != it->fields().rend(); it2++) {
      std::shared_ptr<CField> field = *it2;
      xls::Type* ft = nullptr;
      if (field->type()->Is<CPointerType>() ||
          field->type()->Is<CChannelType>()) {
        ft = package_->GetTupleType({});
      } else {
        XLS_ASSIGN_OR_RETURN(ft, TranslateTypeToXLS(field->type(), loc));
      }
      members.push_back(ft);
    }
    return GetStructXLSType(members, *it, loc);
  }
  if (t->Is<CInternalTuple>()) {
    auto it = t->As<CInternalTuple>();
    std::vector<xls::Type*> tuple_types;
    for (const std::shared_ptr<CType>& field_type : it->fields()) {
      XLS_ASSIGN_OR_RETURN(xls::Type * xls_type,
                           TranslateTypeToXLS(field_type, loc));
      tuple_types.push_back(xls_type);
    }
    return package_->GetTupleType(tuple_types);
  }
  if (t->Is<CArrayType>()) {
    auto it = t->As<CArrayType>();
    XLS_ASSIGN_OR_RETURN(auto xls_elem_type,
                         TranslateTypeToXLS(it->GetElementType(), loc));
    return package_->GetArrayType(it->GetSize(), xls_elem_type);
  }
  if (t->Is<CReferenceType>() || t->Is<CPointerType>() ||
      t->Is<CChannelType>()) {
    return package_->GetTupleType({});
  }
  auto& r = *t;
  return absl::UnimplementedError(
      ErrorMessage(loc, "Unsupported CType subclass in TranslateTypeToXLS: %s",
                   typeid(r).name()));
}

absl::StatusOr<Translator::StrippedType> Translator::StripTypeQualifiers(
    clang::QualType t) {
  StrippedType ret = StrippedType(t, false);

  {
    const clang::Type* type = ret.base.getTypePtr();
    if (clang::isa<clang::ReferenceType>(type)) {
      ret = StrippedType(type->getPointeeType(), true);
    } else if (auto dec = clang::dyn_cast<const clang::DecayedType>(type)) {
      ret = StrippedType(dec->getOriginalType(), true);
    }
  }

  const bool wasConst = ret.base.isConstQualified();

  const clang::Type* type = ret.base.getTypePtr();
  if (auto dec = clang::dyn_cast<const clang::ElaboratedType>(type)) {
    ret = StrippedType(dec->desugar(), ret.is_ref);
  }

  if (wasConst) {
    ret.base.addConst();
  }

  return ret;
}

absl::Status Translator::ScanFile(
    std::string_view source_filename,
    absl::Span<std::string_view> command_line_args) {
  CHECK_NE(parser_.get(), nullptr);
  return parser_->ScanFile(source_filename, command_line_args);
}

absl::StatusOr<std::string> Translator::GetEntryFunctionName() const {
  CHECK_NE(parser_.get(), nullptr);
  return parser_->GetEntryFunctionName();
}

absl::Status Translator::SelectTop(std::string_view top_function_name,
                                   std::string_view top_class_name) {
  CHECK_NE(parser_.get(), nullptr);
  return parser_->SelectTop(top_function_name, top_class_name);
}

absl::StatusOr<GeneratedFunction*> Translator::GenerateIR_Top_Function(
    xls::Package* package,
    const absl::flat_hash_map<const clang::NamedDecl*, ChannelBundle>&
        top_channel_injections,
    bool force_static, bool member_references_become_channels,
    int default_init_interval) {
  const clang::FunctionDecl* top_function = nullptr;

  CHECK_NE(parser_.get(), nullptr);
  XLS_ASSIGN_OR_RETURN(top_function, parser_->GetTopFunction());

  package_ = package;
  default_init_interval_ = default_init_interval;

  auto func_init = std::make_unique<GeneratedFunction>();

  auto signature = absl::implicit_cast<const clang::NamedDecl*>(top_function);

  // Detect recursion of top function
  functions_in_call_stack_.insert(signature);

  inst_functions_[signature] = std::make_unique<GeneratedFunction>();
  GeneratedFunction& sf = *inst_functions_[signature];

  XLS_ASSIGN_OR_RETURN(FunctionInProgress header,
                       GenerateIR_Function_Header(
                           sf, top_function,
                           /*name_override=*/top_function->getNameAsString(),
                           force_static, member_references_become_channels));

  for (const auto& [decl, bundle] : top_channel_injections) {
    XLSCC_CHECK(sf.lvalues_by_param.contains(decl), GetLoc(*top_function));
    std::shared_ptr<LValue> lval = sf.lvalues_by_param.at(decl);
    if (lval->channel_leaf() == nullptr) {
      return absl::UnimplementedError(
          ErrorMessage(GetLoc(*top_function),
                       "Compound top level parameters containing channels"));
    }
    IOChannel* channel = lval->channel_leaf();

    std::pair<const IOChannel*, ChannelBundle> pair(channel, bundle);
    if (!ContainsKeyValuePair(external_channels_by_internal_channel_, pair)) {
      external_channels_by_internal_channel_.insert(pair);
    }
  }

  XLS_RETURN_IF_ERROR(GenerateIR_Function_Body(sf, top_function, header));

  if (sf.xls_func == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Top function %s has no outputs at %s", top_function->getNameAsString(),
        LocString(*top_function)));
  }

  return &sf;
}

absl::StatusOr<bool> Translator::ExprIsChannel(const clang::Expr* object,
                                               const xls::SourceInfo& loc) {
  // Avoid "this", as it's a pointer
  if (auto cast = clang::dyn_cast<const clang::CastExpr>(object)) {
    if (clang::isa<clang::CXXThisExpr>(cast->getSubExpr())) {
      return false;
    }
  }
  if (clang::isa<clang::CXXThisExpr>(object)) {
    return false;
  }
  return TypeIsChannel(object->getType(), loc);
}

absl::Status Translator::InlineAllInvokes(xls::Package* package) {
  std::unique_ptr<xls::OptimizationCompoundPass> pipeline =
      xls::CreateOptimizationPassPipeline();
  xls::OptimizationPassOptions options;
  xls::PassResults results;

  // This pass wants a delay estimator
  options.skip_passes = {"bdd_cse"};

  XLS_RETURN_IF_ERROR(pipeline->Run(package, options, &results).status());
  return absl::OkStatus();
}

absl::StatusOr<xls::BValue> Translator::GenTypeConvert(
    CValue const& in, std::shared_ptr<CType> out_type,
    const xls::SourceInfo& loc) {
  XLSCC_CHECK_NE(in.type().get(), nullptr, loc);
  XLSCC_CHECK_NE(out_type.get(), nullptr, loc);
  if (*in.type() == *out_type) {
    return in.rvalue();
  }
  if (out_type->Is<CStructType>()) {
    return in.rvalue();
  }
  if (out_type->Is<CVoidType>()) {
    return xls::BValue();
  }
  if (out_type->Is<CArrayType>()) {
    return in.rvalue();
  }
  if (out_type->Is<CBitsType>()) {
    auto out_bits_type = out_type->As<CBitsType>();
    XLS_ASSIGN_OR_RETURN(auto conv_type, ResolveTypeInstance(in.type()));
    if (!conv_type->Is<CBitsType>()) {
      return absl::UnimplementedError(ErrorMessage(
          loc, "Cannot convert type %s to bits", std::string(*in.type())));
    }
    if (conv_type->GetBitWidth() != out_bits_type->GetBitWidth()) {
      return absl::UnimplementedError(absl::StrFormat(
          "No implicit bit width conversions for __xls_bits: from %s to %s at "
          "%s",
          std::string(*in.type()), std::string(*out_type), LocString(loc)));
    }
    return in.rvalue();
  }
  if (out_type->Is<CBoolType>()) {
    return GenBoolConvert(in, loc);
  }
  if (out_type->Is<CFloatType>()) {
    if (in.type()->Is<CFloatType>()) {
      return in.rvalue();
    }
    return absl::UnimplementedError(ErrorMessage(
        loc, "Cannot convert type %s to float", std::string(*in.type())));
  }
  if (out_type->Is<CIntType>()) {
    if (in.type()->Is<CFloatType>()) {
      if (out_type->GetBitWidth() != 64 ||
          !out_type->As<CIntType>()->is_signed()) {
        return absl::UnimplementedError(
            ErrorMessage(loc,
                         "Only know how to convert floats to 64 bit signed "
                         "integers, asked to convert to %s",
                         out_type->debug_string()));
      }
      XLSCC_CHECK(in.rvalue().valid(), loc);

      // Double value comes first
      const size_t offset = sizeof(double) * 8;
      return context().fb->BitSlice(in.rvalue(), offset,
                                    out_type->GetBitWidth(), loc);
    }
    if (!(in.type()->Is<CBoolType>() || in.type()->Is<CIntType>())) {
      return absl::UnimplementedError(ErrorMessage(
          loc, "Cannot convert type %s to int", std::string(*in.type())));
    }
    int expr_width = in.type()->GetBitWidth();
    xls::BValue return_value = in.rvalue();
    if (expr_width == out_type->GetBitWidth()) {
      return return_value;
    }
    if (expr_width < out_type->GetBitWidth()) {
      auto p_in_int = std::dynamic_pointer_cast<const CIntType>(in.type());
      if ((!in.type()->Is<CBoolType>()) &&
          (p_in_int != nullptr && p_in_int->is_signed())) {
        return context().fb->SignExtend(return_value, out_type->GetBitWidth(),
                                        loc);
      }
      return context().fb->ZeroExtend(return_value, out_type->GetBitWidth(),
                                      loc);
    }
    return context().fb->BitSlice(return_value, 0, out_type->GetBitWidth(),
                                  loc);
  }
  if (out_type->Is<CInstantiableTypeAlias>()) {
    XLS_ASSIGN_OR_RETURN(auto t, ResolveTypeInstance(out_type));
    return GenTypeConvert(in, t, loc);
  }
  return absl::UnimplementedError(
      ErrorMessage(loc, "Don't know how to convert %s to type %s",
                   in.debug_string().c_str(), std::string(*out_type)));
}

absl::StatusOr<xls::BValue> Translator::GenBoolConvert(
    CValue const& in, const xls::SourceInfo& loc) {
  if (!(in.type()->Is<CBoolType>() || in.type()->Is<CIntType>())) {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Cannot convert type %s to bool", std::string(*in.type())));
  }
  XLSCC_CHECK_GT(in.type()->GetBitWidth(), 0, loc);
  if (in.type()->GetBitWidth() == 1) {
    return in.rvalue();
  }
  xls::BValue const0 =
      context().fb->Literal(xls::UBits(0, in.type()->GetBitWidth()), loc);
  return context().fb->Ne(in.rvalue(), const0, loc);
}

std::string Translator::LocString(const xls::SourceInfo& loc) {
  XLSCC_CHECK_NE(parser_.get(), nullptr, loc);
  return parser_->LocString(loc);
}

xls::SourceInfo Translator::GetLoc(const clang::Stmt& stmt) {
  CHECK_NE(parser_.get(), nullptr);
  if (context().sf != nullptr && context().sf->in_synthetic_int) {
    return xls::SourceInfo();
  }
  return parser_->GetLoc(stmt);
}

xls::SourceInfo Translator::GetLoc(const clang::Decl& decl) {
  CHECK_NE(parser_.get(), nullptr);
  if (context().sf != nullptr && context().sf->in_synthetic_int) {
    return xls::SourceInfo();
  }
  return parser_->GetLoc(decl);
}

clang::PresumedLoc Translator::GetPresumedLoc(const clang::Stmt& stmt) {
  CHECK_NE(parser_.get(), nullptr);
  return parser_->GetPresumedLoc(stmt);
}

clang::PresumedLoc Translator::GetPresumedLoc(const clang::Decl& decl) {
  CHECK_NE(parser_.get(), nullptr);
  return parser_->GetPresumedLoc(decl);
}

absl::StatusOr<const clang::CallExpr*> Translator::FindIntrinsicCall(
    const clang::PresumedLoc& target_loc) {
  // NOTE: Semantics should be the same as CCParser::FindPragmaForLoc()!

  xls::SourceInfo xls_loc = parser_->GetLoc(target_loc);

  const clang::CallExpr* ret = context().last_intrinsic_call;

  // No intrinsics yet in this scope
  if (ret == nullptr) {
    return nullptr;
  }

  const clang::PresumedLoc intrinsic_loc = GetPresumedLoc(*ret);

  // Not in the same file at all (macros?)

  if (target_loc.getFilename() != intrinsic_loc.getFilename()) {
    return absl::UnimplementedError(ErrorMessage(
        xls_loc, "Intrinsic call in this scope, but in another file (macro?)"));
  }

  XLSCC_CHECK_GE(target_loc.getLine(), intrinsic_loc.getLine(), xls_loc);

  // Does not apply if more than 2 lines behind
  if (target_loc.getLine() > (intrinsic_loc.getLine() + 2)) {
    return nullptr;
  }

  // Applies if it's the same line or the one before
  if (target_loc.getLine() <= (intrinsic_loc.getLine() + 1)) {
    return ret;
  }

  XLSCC_CHECK_GE(target_loc.getLine(), intrinsic_loc.getLine() + 2, xls_loc);

  // Must be a label in between if it's two lines before
  XLS_ASSIGN_OR_RETURN(Pragma pragma_before,
                       FindPragmaForLoc(target_loc, /*ignore_label=*/false));

  if (pragma_before.type() != PragmaType::Pragma_Label) {
    return nullptr;
  }

  return ret;
}

absl::StatusOr<Pragma> Translator::FindPragmaForLoc(
    const clang::SourceLocation& loc, bool ignore_label) {
  CHECK_NE(parser_.get(), nullptr);
  return parser_->FindPragmaForLoc(loc, ignore_label);
}

absl::StatusOr<Pragma> Translator::FindPragmaForLoc(
    const clang::PresumedLoc& ploc, bool ignore_label) {
  CHECK_NE(parser_.get(), nullptr);
  return parser_->FindPragmaForLoc(ploc, ignore_label);
}

}  // namespace xlscc
