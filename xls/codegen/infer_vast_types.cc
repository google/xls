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

#include "xls/codegen/infer_vast_types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/fold_vast_constants.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace verilog {
namespace {

// Builds a map of the return types of supported system functions. Many system
// functions are "unsupported" currently, because VAST doesn't support "real"
// values, for example. The passed-in `file` is only relevant to serve as the
// owner of the created type objects.
absl::flat_hash_map<std::string, DataType*> BuildSystemFunctionReturnTypes(
    VerilogFile* file) {
  DataType* int_type = file->IntegerType(SourceInfo());
  DataType* scalar_type = file->ScalarType(SourceInfo());
  return {{"bits", int_type},        {"clog2", int_type},
          {"countbits", int_type},   {"countones", int_type},
          {"onehot", scalar_type},   {"onehot0", scalar_type},
          {"isunknown", scalar_type}};
}

// A utility that helps build a map of inferred types.
class TypeInferenceVisitor {
 public:
  // Creates a visitor that will populate the given `types` map for whatever
  // entities it traverses.
  TypeInferenceVisitor(
      absl::flat_hash_map<Expression*, DataType*>* types,
      absl::flat_hash_map<std::string, DataType*> system_function_return_types)
      : types_(types),
        system_function_return_types_(std::move(system_function_return_types)) {
  }

  // Traverses and populates `types_` for all expressions in the given file.
  absl::Status TraverseFile(VerilogFile* file) {
    for (verilog::FileMember member : file->members()) {
      if (std::holds_alternative<verilog::Module*>(member)) {
        XLS_RETURN_IF_ERROR(TraverseModule(std::get<verilog::Module*>(member)));
      }
    }
    return absl::OkStatus();
  }

  // Traverses and populates `types_` for all expressions in the given module.
  absl::Status TraverseModule(Module* module) {
    for (verilog::ModuleMember member : module->top()->members()) {
      if (std::holds_alternative<verilog::VerilogFunction*>(member)) {
        XLS_RETURN_IF_ERROR(
            TraverseFunction(std::get<verilog::VerilogFunction*>(member)));
      } else if (std::holds_alternative<verilog::Typedef*>(member)) {
        XLS_RETURN_IF_ERROR(
            TraverseTypedef(std::get<verilog::Typedef*>(member)));
      } else if (std::holds_alternative<verilog::Parameter*>(member)) {
        XLS_RETURN_IF_ERROR(
            TraverseParameter(std::get<verilog::Parameter*>(member)));
      }
    }
    return absl::OkStatus();
  }

  // Traverses and populates `types_` recursively for the given expression.
  // Invoked for independent expression trees only, not for every subexpression.
  // This function uses other helper functions on subexpressions.
  absl::Status TraverseExpression(
      Expression* expr,
      std::optional<DataType*> external_context_type = std::nullopt) {
    XLS_ASSIGN_OR_RETURN(DataType * largest_type, InferType(expr));
    if (external_context_type.has_value()) {
      // Note: according to 11.8.1, the signedness of the LHS has no bearing on
      // the RHS, hence we pass false for the flag here.
      XLS_ASSIGN_OR_RETURN(largest_type,
                           LargestType(*external_context_type, largest_type,
                                       /*reconcile_signedness=*/false));
    }
    ApplyInferredTypeRecursively(largest_type, expr);
    // The context type must win for the output of `expr` itself. In a case like
    // `uint16_ans = (uint16_a + uint16_b + 0) >> 1`, we allow the 0 to virally
    // promote the internals of RHS, while reflecting the fact that the top node
    // must be downcast to fit the variable.
    if (external_context_type.has_value()) {
      (*types_)[expr] = MaybeFoldConstants(*external_context_type);
    }
    return absl::OkStatus();
  }

 private:
  absl::Status TraverseFunction(VerilogFunction* function) {
    for (Statement* statement : function->statement_block()->statements()) {
      XLS_RETURN_IF_ERROR(TraverseStatement(function, statement));
    }
    return absl::OkStatus();
  }

  absl::Status TraverseStatement(VerilogFunction* function,
                                 Statement* statement) {
    if (auto* return_statement = dynamic_cast<ReturnStatement*>(statement);
        return_statement) {
      return TraverseExpression(
          return_statement->expr(),
          /*external_context_type=*/function->return_value_def()->data_type());
    }
    return absl::OkStatus();
  }

  absl::Status TraverseDataType(DataType* data_type) {
    if (auto* bit_vector_type = dynamic_cast<BitVectorType*>(data_type);
        bit_vector_type && !bit_vector_type->size_expr()->IsLiteral()) {
      return TraverseExpression(bit_vector_type->size_expr());
    }
    if (auto* array_type = dynamic_cast<ArrayTypeBase*>(data_type);
        array_type) {
      XLS_RETURN_IF_ERROR(TraverseDataType(array_type->element_type()));
      for (Expression* dim : array_type->dims()) {
        if (!dim->IsLiteral()) {
          XLS_RETURN_IF_ERROR(TraverseExpression(dim));
        }
      }
    }
    if (auto* enum_def = dynamic_cast<Enum*>(data_type); enum_def) {
      return TraverseEnum(enum_def);
    }
    if (auto* struct_def = dynamic_cast<Struct*>(data_type); struct_def) {
      return TraverseStruct(struct_def);
    }
    return absl::OkStatus();
  }

  absl::Status TraverseStruct(Struct* struct_def) {
    for (Def* def : struct_def->members()) {
      XLS_RETURN_IF_ERROR(TraverseDataType(def->data_type()));
    }
    return absl::OkStatus();
  }

  absl::Status TraverseTypedef(Typedef* type_def) {
    return TraverseDataType(type_def->data_type());
  }

  absl::Status TraverseParameter(Parameter* parameter) {
    std::optional<DataType*> data_type = std::nullopt;
    if (parameter->def()) {
      data_type = parameter->def()->data_type();
      XLS_RETURN_IF_ERROR(TraverseDataType(*data_type));
    }
    XLS_RETURN_IF_ERROR(
        TraverseExpression(parameter->rhs(),
                           /*external_context_type=*/data_type));
    if (!parameter->def()) {
      // For parameters of the form `parameter foo = some_expr;`, without a type
      // on the LHS, the RHS decides the type, and there has to be an RHS. We
      // store this type in the separate `auto_parameter_types_` map, because
      // the normal type map can only have exprs as keys (and clients of
      // inference only care about expr types).
      const auto it = types_->find(parameter->rhs());
      if (it == types_->end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("No type could be inferred for untyped parameter: ",
                         parameter->Emit(nullptr)));
      }
      auto_parameter_types_.emplace(parameter, it->second);
    }
    return absl::OkStatus();
  }

  absl::Status TraverseEnum(Enum* enum_def) {
    XLS_RETURN_IF_ERROR(TraverseDataType(enum_def->BaseType()));
    for (EnumMember* member : enum_def->members()) {
      if (member->rhs() != nullptr) {
        XLS_RETURN_IF_ERROR(TraverseExpression(
            member->rhs(), /*external_context_type=*/enum_def->BaseType()));
      }
    }
    return absl::OkStatus();
  }

  // These correspond to the operator rows in table 11-21 in the spec, in order.
  enum class OperatorSizeTreatment : uint8_t {
    kContextDependentBinary,
    kContextDependentUnary,
    kScalarBinaryWithMaxSizeOperands,
    kScalarBinaryWithSelfDeterminedOperands,
    kScalarUnaryWithSelfDeterminedOperand,
    kContextDependentBinaryWithSelfDeterminedRhs
  };

  OperatorSizeTreatment GetOperatorSizeTreatment(Operator* op) {
    switch (op->kind()) {
      case OperatorKind::kAdd:
      case OperatorKind::kSub:
      case OperatorKind::kMul:
      case OperatorKind::kDiv:
      case OperatorKind::kMod:
      case OperatorKind::kBitwiseAnd:
      case OperatorKind::kBitwiseOr:
      case OperatorKind::kBitwiseXor:
        return OperatorSizeTreatment::kContextDependentBinary;
      case OperatorKind::kNegate:
      case OperatorKind::kBitwiseNot:
        return OperatorSizeTreatment::kContextDependentUnary;
      case OperatorKind::kEq:
      case OperatorKind::kNe:
      case OperatorKind::kEqX:
      case OperatorKind::kNeX:
      case OperatorKind::kGe:
      case OperatorKind::kGt:
      case OperatorKind::kLe:
      case OperatorKind::kLt:
        return OperatorSizeTreatment::kScalarBinaryWithMaxSizeOperands;
      case OperatorKind::kLogicalAnd:
      case OperatorKind::kLogicalOr:
        return OperatorSizeTreatment::kScalarBinaryWithSelfDeterminedOperands;
      case OperatorKind::kAndReduce:
      case OperatorKind::kOrReduce:
      case OperatorKind::kXorReduce:
      case OperatorKind::kLogicalNot:
        return OperatorSizeTreatment::kScalarUnaryWithSelfDeterminedOperand;
      case OperatorKind::kShll:
      case OperatorKind::kShrl:
      case OperatorKind::kShra:
      case OperatorKind::kPower:
        return OperatorSizeTreatment::
            kContextDependentBinaryWithSelfDeterminedRhs;
    }
  }

  // Infers the type of `expr` according to the applicable rules, but ignoring
  // the containing context, which must be accounted for by the caller. Runs a
  // whole separate `TraverseExpression` flow on any independent subtrees
  // encountered. For example, function call args are independent subtrees.
  absl::StatusOr<DataType*> InferType(Expression* expr) {
    if (auto* op = dynamic_cast<Operator*>(expr); op) {
      DataType* lhs_type = nullptr;
      DataType* rhs_type = nullptr;
      switch (GetOperatorSizeTreatment(op)) {
        case OperatorSizeTreatment::kContextDependentBinary: {
          // These operators follow the normal context rule that the operands
          // and the expression itself are the size of the largest operand.
          XLS_ASSIGN_OR_RETURN(
              lhs_type, InferType(dynamic_cast<BinaryInfix*>(expr)->lhs()));
          XLS_ASSIGN_OR_RETURN(
              rhs_type, InferType(dynamic_cast<BinaryInfix*>(expr)->rhs()));
          return LargestType(lhs_type, rhs_type);
        }

        case OperatorSizeTreatment::kContextDependentUnary: {
          return InferType(dynamic_cast<Unary*>(expr)->arg());
        }

        case OperatorSizeTreatment::kScalarBinaryWithMaxSizeOperands: {
          // A comparison forms an independent subtree of scalar width whose
          // operands are all the size of the largest of them. A comparison
          // operand can overflow even if the overall result is assigned to a
          // variable that fits that operand.
          XLS_ASSIGN_OR_RETURN(
              lhs_type, InferType(dynamic_cast<BinaryInfix*>(expr)->lhs()));
          XLS_ASSIGN_OR_RETURN(
              rhs_type, InferType(dynamic_cast<BinaryInfix*>(expr)->rhs()));
          XLS_ASSIGN_OR_RETURN(DataType * largest_type,
                               LargestType(lhs_type, rhs_type));
          ApplyInferredTypeRecursively(largest_type,
                                       dynamic_cast<BinaryInfix*>(expr)->lhs());
          ApplyInferredTypeRecursively(largest_type,
                                       dynamic_cast<BinaryInfix*>(expr)->rhs());
          return expr->file()->ScalarType(expr->loc());
        }

        case OperatorSizeTreatment::kScalarBinaryWithSelfDeterminedOperands: {
          XLS_RETURN_IF_ERROR(
              TraverseExpression(dynamic_cast<BinaryInfix*>(expr)->lhs()));
          XLS_RETURN_IF_ERROR(
              TraverseExpression(dynamic_cast<BinaryInfix*>(expr)->rhs()));
          return expr->file()->ScalarType(expr->loc());
        }

        case OperatorSizeTreatment::kScalarUnaryWithSelfDeterminedOperand: {
          XLS_RETURN_IF_ERROR(
              TraverseExpression(dynamic_cast<Unary*>(expr)->arg()));
          return expr->file()->ScalarType(expr->loc());
        }

        case OperatorSizeTreatment::
            kContextDependentBinaryWithSelfDeterminedRhs: {
          // These operators have a self-determined RHS, with only the LHS
          // having any bearing on the expression itself.
          XLS_RETURN_IF_ERROR(
              TraverseExpression(dynamic_cast<BinaryInfix*>(expr)->rhs()));
          return InferType(dynamic_cast<BinaryInfix*>(expr)->lhs());
        }
      }
    }
    if (auto* ref = dynamic_cast<ParameterRef*>(expr); ref) {
      if (ref->parameter()->def() != nullptr) {
        return ref->parameter()->def()->data_type();
      }
      const auto it = auto_parameter_types_.find(ref->parameter());
      if (it != auto_parameter_types_.end()) {
        return it->second;
      }
      return ref->file()->IntegerType(ref->loc());
    }
    if (auto* ref = dynamic_cast<EnumMemberRef*>(expr); ref) {
      return ref->enum_def();
    }
    if (auto* ref = dynamic_cast<LogicRef*>(expr); ref) {
      return ref->def()->data_type();
    }
    if (auto* literal = dynamic_cast<Literal*>(expr); literal) {
      if (literal->effective_bit_count() == 32) {
        return expr->file()->Make<IntegerType>(
            expr->loc(), literal->is_declared_as_signed());
      }
      return expr->file()->BitVectorType(literal->effective_bit_count(),
                                         expr->loc(),
                                         literal->is_declared_as_signed());
    }
    if (auto* concat = dynamic_cast<Concat*>(expr); concat) {
      int64_t bit_count = 0;
      for (Expression* arg : concat->args()) {
        XLS_RETURN_IF_ERROR(TraverseExpression(arg));
        const auto it = types_->find(arg);
        if (it == types_->end()) {
          return absl::InvalidArgumentError(
              absl::StrCat("No type was inferred for concat argument: ",
                           arg->Emit(nullptr)));
        }
        XLS_ASSIGN_OR_RETURN(int64_t arg_bit_count,
                             EvaluateBitCount(it->second));
        bit_count += arg_bit_count;
      }
      return expr->file()->BitVectorType(bit_count, expr->loc(),
                                         /*signed=*/false);
    }
    if (auto* ternary = dynamic_cast<Ternary*>(expr); ternary) {
      XLS_RETURN_IF_ERROR(TraverseExpression(ternary->test()));
      XLS_ASSIGN_OR_RETURN(DataType * consequent_type,
                           InferType(ternary->consequent()));
      XLS_ASSIGN_OR_RETURN(DataType * alternate_type,
                           InferType(ternary->alternate()));
      return LargestType(consequent_type, alternate_type);
    }
    if (auto* call = dynamic_cast<VerilogFunctionCall*>(expr); call) {
      absl::Span<Expression* const> args = call->args();
      absl::Span<Def* const> formal_args = call->func()->arguments();
      if (args.size() != formal_args.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Expected %d arguments but have %d for invocation of %s",
            formal_args.size(), args.size(), call->func()->name()));
      }
      for (int i = 0; i < args.size(); i++) {
        XLS_RETURN_IF_ERROR(TraverseExpression(
            args[i],
            /*external_context_type=*/formal_args[i]->data_type()));
      }
      return call->func()->return_value_def()->data_type();
    }
    if (auto* call = dynamic_cast<SystemFunctionCall*>(expr); call) {
      const auto it = system_function_return_types_.find(call->name());
      if (it == system_function_return_types_.end()) {
        return absl::NotFoundError(absl::StrCat(
            "No return type specified for system function: ", call->name()));
      }
      std::optional<std::vector<Expression*>> args = call->args();
      if (args.has_value()) {
        for (Expression* arg : *args) {
          // We could add knowledge of the formal argument types of system
          // functions, and pass them as the external context type here.
          // However, it seems they are generally more flexible than
          // user-defined functions, e.g. clog2 can take an integer or any bit
          // vector.
          XLS_RETURN_IF_ERROR(TraverseExpression(arg));
        }
      }
      return it->second;
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Unhandled expression type: ", expr->Emit(nullptr)));
  }

  // Populates `types_` with the given inferred `data_type` for `expr` and any
  // descendants that the type logically applies to. It doesn't logically apply,
  // for example, to the children of a concat, even though they have bearing on
  // the type of the concat. On the other hand, it does apply to a logical
  // operator descendant of `expr` that claimed in InferType that it wanted a
  // scalar size. This is only called for independent expressions.
  void ApplyInferredTypeRecursively(DataType* data_type, Expression* expr) {
    types_->emplace(expr, data_type);
    if (auto* op = dynamic_cast<Operator*>(expr); op) {
      switch (GetOperatorSizeTreatment(op)) {
        case OperatorSizeTreatment::kContextDependentBinary: {
          ApplyInferredTypeRecursively(data_type,
                                       dynamic_cast<BinaryInfix*>(expr)->lhs());
          ApplyInferredTypeRecursively(data_type,
                                       dynamic_cast<BinaryInfix*>(expr)->rhs());
          break;
        }

        case OperatorSizeTreatment::kContextDependentUnary: {
          ApplyInferredTypeRecursively(data_type,
                                       dynamic_cast<Unary*>(expr)->arg());
          break;
        }

        case OperatorSizeTreatment::
            kContextDependentBinaryWithSelfDeterminedRhs: {
          ApplyInferredTypeRecursively(data_type,
                                       dynamic_cast<BinaryInfix*>(expr)->lhs());
          break;
        }

        default:
          break;
      }
      return;
    }
    if (auto* ternary = dynamic_cast<Ternary*>(expr); ternary) {
      ApplyInferredTypeRecursively(data_type, ternary->consequent());
      ApplyInferredTypeRecursively(data_type, ternary->alternate());
    }
  }

  absl::StatusOr<DataType*> LargestType(DataType* a, DataType* b,
                                        bool reconcile_signedness = true) {
    DataType* folded_a = MaybeFoldConstants(a);
    DataType* folded_b = MaybeFoldConstants(b);
    absl::StatusOr<int64_t> maybe_a_bit_count = folded_a->FlatBitCountAsInt64();
    absl::StatusOr<int64_t> maybe_b_bit_count = folded_b->FlatBitCountAsInt64();
    // In cases like `parameter logic[$clog2(32767):0] = ...`, we don't have the
    // ability to infer one of the types, and it's unlikely to matter.
    if (!maybe_b_bit_count.ok()) {
      return folded_a;
    }
    if (!maybe_a_bit_count.ok()) {
      return folded_b;
    }
    int64_t a_bit_count = *maybe_a_bit_count;
    int64_t b_bit_count = *maybe_b_bit_count;
    bool b_int = dynamic_cast<IntegerType*>(b) != nullptr;
    int64_t result_bit_count;
    DataType* result;
    // Prefer the larger type, but if they are equivalent:
    // 1. Prefer the integer type, if any, as it's more precise as to intent.
    // 2. Prefer the RHS in a case of sign mismatch without reconciliation.
    if (a_bit_count > b_bit_count ||
        (a_bit_count == b_bit_count && a->is_signed() == b->is_signed() &&
         !b_int)) {
      result = folded_a;
      result_bit_count = a_bit_count;
    } else {
      result = folded_b;
      result_bit_count = b_bit_count;
    }
    // Don't propagate user-defined types where the user didn't use them.
    if (dynamic_cast<Struct*>(result) || dynamic_cast<TypedefType*>(result)) {
      result = result->file()->BitVectorType(result_bit_count, result->loc());
    }
    // According to 11.8.1, if any operand is unsigned, the result is unsigned.
    if (reconcile_signedness && result->is_signed() &&
        (a->is_signed() ^ b->is_signed())) {
      return UnsignedEquivalent(result);
    }
    return result;
  }

  absl::StatusOr<DataType*> UnsignedEquivalent(DataType* data_type) {
    if (dynamic_cast<IntegerType*>(data_type) != nullptr) {
      return data_type->file()->Make<IntegerType>(data_type->loc(),
                                                  /*is_signed=*/false);
    }
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, EvaluateBitCount(data_type));
    return data_type->file()->BitVectorType(bit_count, data_type->loc());
  }

  DataType* MaybeFoldConstants(DataType* data_type) {
    if (!data_type->FlatBitCountAsInt64().ok()) {
      absl::StatusOr<DataType*> folded_type =
          FoldVastConstants(data_type, *types_);
      if (folded_type.ok()) {
        return *folded_type;
      }
      VLOG(2) << "Could not fold: " << data_type->Emit(nullptr)
              << ", status: " << folded_type.status();
    }
    return data_type;
  }

  absl::StatusOr<int64_t> EvaluateBitCount(DataType* data_type) {
    absl::StatusOr<int64_t> direct_answer = data_type->FlatBitCountAsInt64();
    if (direct_answer.ok()) {
      return direct_answer;
    }
    absl::StatusOr<DataType*> folded_type =
        FoldVastConstants(data_type, *types_);
    if (folded_type.ok()) {
      return (*folded_type)->FlatBitCountAsInt64();
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "Could not evaluate bit count for type: ", data_type->Emit(nullptr)));
  }

  absl::flat_hash_map<Expression*, DataType*>* types_;
  absl::flat_hash_map<Parameter*, DataType*> auto_parameter_types_;
  const absl::flat_hash_map<std::string, DataType*>
      system_function_return_types_;
};

}  // namespace

absl::StatusOr<absl::flat_hash_map<Expression*, DataType*>> InferVastTypes(
    VerilogFile* file) {
  absl::flat_hash_map<Expression*, DataType*> types;
  auto visitor = std::make_unique<TypeInferenceVisitor>(
      &types, BuildSystemFunctionReturnTypes(file));
  XLS_RETURN_IF_ERROR(visitor->TraverseFile(file));
  return types;
}

absl::StatusOr<absl::flat_hash_map<Expression*, DataType*>> InferVastTypes(
    absl::Span<VerilogFile* const> corpus) {
  absl::flat_hash_map<Expression*, DataType*> types;
  std::unique_ptr<TypeInferenceVisitor> visitor;
  for (VerilogFile* file : corpus) {
    if (visitor == nullptr) {
      visitor = std::make_unique<TypeInferenceVisitor>(
          &types, BuildSystemFunctionReturnTypes(file));
    }
    XLS_RETURN_IF_ERROR(visitor->TraverseFile(file));
  }
  return types;
}

absl::StatusOr<absl::flat_hash_map<Expression*, DataType*>> InferVastTypes(
    Expression* expr) {
  absl::flat_hash_map<Expression*, DataType*> types;
  auto visitor = std::make_unique<TypeInferenceVisitor>(
      &types, BuildSystemFunctionReturnTypes(expr->file()));
  XLS_RETURN_IF_ERROR(visitor->TraverseExpression(expr));
  return types;
}

}  // namespace verilog
}  // namespace xls
