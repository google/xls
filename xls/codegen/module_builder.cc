// Copyright 2020 The XLS Authors
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

#include "xls/codegen/module_builder.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/lint_annotate.h"
#include "xls/codegen/node_expressions.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"

namespace xls {
namespace verilog {

namespace {

// Returns the bounds of the potentially-nested array type as a vector of
// int64. Ordering of the vector is outer-most bound to inner-most. For example,
// given array type 'bits[32][4][5]' yields {5, 4, 32}.
std::vector<int64> NestedArrayBounds(ArrayType* type) {
  std::vector<int64> bounds;
  Type* t = type;
  while (t->IsArray()) {
    bounds.push_back(t->AsArrayOrDie()->size());
    t = t->AsArrayOrDie()->element_type();
  }
  return bounds;
}

// Creates UnpackedArrayBounds corresponding to the given array type. If
// use_system_verilog is true, the bound are expressed using "sizes" (e.g.,
// "[3][4][5]") other wise it is expressed using ranges (e.g.,
// "[0:2][0:3][0:4]").
std::vector<UnpackedArrayBound> MakeUnpackedArrayBounds(
    ArrayType* type, VerilogFile* file, bool use_system_verilog) {
  std::vector<UnpackedArrayBound> bounds;
  for (int64 size : NestedArrayBounds(type)) {
    if (use_system_verilog) {
      bounds.push_back(UnpackedArrayBound(file->PlainLiteral(size)));
    } else {
      bounds.push_back(UnpackedArrayBound(
          std::make_pair(file->PlainLiteral(0), file->PlainLiteral(size - 1))));
    }
  }
  return bounds;
}

// Returns the width of the element of the potentially nested array type. For
// example, given array type 'bits[32][4][5]' yields 32.
int64 NestedElementWidth(ArrayType* type) {
  Type* t = type;
  while (t->IsArray()) {
    t = t->AsArrayOrDie()->element_type();
  }
  return t->GetFlatBitCount();
}

// Flattens a value into a single bits-typed expression. Tuples and arrays are
// represented as a concatenation of their elements.
absl::StatusOr<Expression*> FlattenValueToExpression(const Value& value,
                                                     VerilogFile* file) {
  XLS_RET_CHECK_GT(value.GetFlatBitCount(), 0);
  if (value.IsBits()) {
    return file->Literal(value.bits());
  }
  // Compound types are represented as a concatentation of their elements.
  std::vector<Expression*> elements;
  for (const Value& element : value.elements()) {
    if (element.GetFlatBitCount() > 0) {
      XLS_ASSIGN_OR_RETURN(Expression * element_expr,
                           FlattenValueToExpression(element, file));
      elements.push_back(element_expr);
    }
  }
  return file->Concat(elements);
}

// Returns the given array value as an array assignment pattern. For example,
// the array:
//
//   [bits[8]:42, bits[8]:10, bits[8]:2]
//
// would produce:
//
//   '{8'h42, 8'h10, 8'h2}
absl::StatusOr<ArrayAssignmentPattern*> ValueToArrayAssignmentPattern(
    const Value& value, VerilogFile* file) {
  XLS_RET_CHECK(value.IsArray());
  std::vector<Expression*> pieces;
  for (const Value& element : value.elements()) {
    Expression* element_expr;
    if (element.IsArray()) {
      XLS_ASSIGN_OR_RETURN(element_expr,
                           ValueToArrayAssignmentPattern(element, file));
    } else {
      XLS_ASSIGN_OR_RETURN(element_expr,
                           FlattenValueToExpression(element, file));
    }
    pieces.push_back(element_expr);
  }
  return file->Make<ArrayAssignmentPattern>(pieces);
}

}  // namespace

absl::Status ModuleBuilder::AddAssignment(
    Type* xls_type, Expression* lhs, Expression* rhs,
    std::function<void(Expression*, Expression*)> add_assignment) {
  return AddAssignmentToGeneratedExpression(
      xls_type, lhs, /*inputs=*/{rhs}, /*gen_rhs_expr=*/
      [](absl::Span<Expression* const> inputs) { return inputs[0]; },
      add_assignment, /*sv_array_expr=*/true);
}

absl::Status ModuleBuilder::AddAssignmentToGeneratedExpression(
    Type* xls_type, Expression* lhs, absl::Span<Expression* const> inputs,
    std::function<Expression*(absl::Span<Expression* const>)> gen_rhs_expr,
    std::function<void(Expression*, Expression*)> add_assignment,
    bool sv_array_expr) {
  // Assign arrays element by element unless using SystemVerilog AND
  // sv_array_expr is true.
  if (xls_type != nullptr && xls_type->IsArray() &&
      !(use_system_verilog_ && sv_array_expr)) {
    ArrayType* array_type = xls_type->AsArrayOrDie();
    for (int64 i = 0; i < array_type->size(); ++i) {
      std::vector<Expression*> input_elements;
      for (Expression* input : inputs) {
        input_elements.push_back(
            file_->Index(input->AsIndexableExpressionOrDie(), i));
      }
      XLS_RETURN_IF_ERROR(AddAssignmentToGeneratedExpression(
          array_type->element_type(),
          file_->Index(lhs->AsIndexableExpressionOrDie(), i), input_elements,
          gen_rhs_expr, add_assignment, sv_array_expr));
    }
    return absl::OkStatus();
  }
  add_assignment(lhs, gen_rhs_expr(inputs));
  return absl::OkStatus();
}

absl::Status ModuleBuilder::AddAssignmentFromValue(
    Expression* lhs, const Value& value,
    std::function<void(Expression*, Expression*)> add_assignment) {
  if (value.IsArray()) {
    if (use_system_verilog_) {
      // If using system verilog emit using an array assignment pattern like so:
      //   logic [4:0] foo [0:4][0:1] = '{'{5'h0, 5'h1}, '{..}, ...}
      XLS_ASSIGN_OR_RETURN(Expression * rhs,
                           ValueToArrayAssignmentPattern(value, file_));
      add_assignment(lhs, rhs);
    } else {
      for (int64 i = 0; i < value.size(); ++i) {
        XLS_RETURN_IF_ERROR(AddAssignmentFromValue(
            file_->Index(lhs->AsIndexableExpressionOrDie(), i),
            value.element(i), add_assignment));
      }
    }
  } else {
    XLS_ASSIGN_OR_RETURN(Expression * flattened_expr,
                         FlattenValueToExpression(value, file_));
    add_assignment(lhs, flattened_expr);
  }
  return absl::OkStatus();
}

ModuleBuilder::ModuleBuilder(absl::string_view name, VerilogFile* file,
                             bool use_system_verilog)
    : module_name_(SanitizeIdentifier(name)),
      file_(file),
      use_system_verilog_(use_system_verilog) {
  module_ = file_->AddModule(module_name_);
  functions_section_ = module_->Add<ModuleSection>(file_);
  constants_section_ = module_->Add<ModuleSection>(file_);
  input_section_ = module_->Add<ModuleSection>(file_);
  declaration_and_assignment_section_ = module_->Add<ModuleSection>(file_);
  output_section_ = module_->Add<ModuleSection>(file_);

  NewDeclarationAndAssignmentSections();
}

void ModuleBuilder::NewDeclarationAndAssignmentSections() {
  declaration_subsections_.push_back(
      declaration_and_assignment_section_->Add<ModuleSection>(file_));
  assignment_subsections_.push_back(
      declaration_and_assignment_section_->Add<ModuleSection>(file_));
}

LogicRef* ModuleBuilder::DeclareUnpackedArrayWire(absl::string_view name,
                                                  ArrayType* array_type,
                                                  ModuleSection* section) {
  return file_->Make<LogicRef>(section->Add<UnpackedArrayWireDef>(
      name, file_->PlainLiteral(NestedElementWidth(array_type)),
      MakeUnpackedArrayBounds(array_type, file_, use_system_verilog_)));
}

LogicRef* ModuleBuilder::DeclareUnpackedArrayReg(absl::string_view name,
                                                 ArrayType* array_type,
                                                 ModuleSection* section) {
  return file_->Make<LogicRef>(section->Add<UnpackedArrayRegDef>(
      name, file_->PlainLiteral(NestedElementWidth(array_type)),
      MakeUnpackedArrayBounds(array_type, file_, use_system_verilog_)));
}

absl::Status ModuleBuilder::AssignFromSlice(
    Expression* lhs, Expression* rhs, Type* xls_type, int64 slice_start,
    std::function<void(Expression*, Expression*)> add_assignment) {
  if (xls_type->IsArray()) {
    ArrayType* array_type = xls_type->AsArrayOrDie();
    for (int64 i = 0; i < array_type->size(); ++i) {
      XLS_RETURN_IF_ERROR(
          AssignFromSlice(file_->Index(lhs->AsIndexableExpressionOrDie(), i),
                          rhs, array_type->element_type(),
                          slice_start + GetFlatBitIndexOfElement(array_type, i),
                          add_assignment));
    }
  } else {
    add_assignment(
        lhs, file_->Slice(rhs->AsIndexableExpressionOrDie(),
                          /*hi=*/slice_start + xls_type->GetFlatBitCount() - 1,
                          /*lo=*/slice_start));
  }
  return absl::OkStatus();
}

absl::StatusOr<LogicRef*> ModuleBuilder::AddInputPort(absl::string_view name,
                                                      Type* type) {
  LogicRef* port = module_->AddPort(Direction::kInput, SanitizeIdentifier(name),
                                    type->GetFlatBitCount());
  if (!type->IsArray()) {
    return port;
  }
  // All inputs are flattened so unflatten arrays with a sequence of
  // assignments.
  LogicRef* ar = DeclareUnpackedArrayWire(
      absl::StrCat(SanitizeIdentifier(name), "_unflattened"),
      type->AsArrayOrDie(), input_section());
  XLS_RETURN_IF_ERROR(AssignFromSlice(
      ar, port, type->AsArrayOrDie(), 0, [&](Expression* lhs, Expression* rhs) {
        input_section()->Add<ContinuousAssignment>(lhs, rhs);
      }));
  return ar;
}

LogicRef* ModuleBuilder::AddInputPort(absl::string_view name, int64 bit_count) {
  return module_->AddPort(Direction::kInput, SanitizeIdentifier(name),
                          bit_count);
}

absl::Status ModuleBuilder::AddOutputPort(absl::string_view name, Type* type,
                                          Expression* value) {
  LogicRef* output_port = module_->AddPort(
      Direction::kOutput, SanitizeIdentifier(name), type->GetFlatBitCount());
  if (type->IsArray()) {
    // The output is flattened so flatten arrays with a sequence of assignments.
    XLS_RET_CHECK(value->IsIndexableExpression());
    output_section()->Add<ContinuousAssignment>(
        output_port, FlattenArray(value->AsIndexableExpressionOrDie(),
                                  type->AsArrayOrDie(), file_));
  } else {
    output_section()->Add<ContinuousAssignment>(output_port, value);
  }
  return absl::OkStatus();
}

absl::Status ModuleBuilder::AddOutputPort(absl::string_view name,
                                          int64 bit_count, Expression* value) {
  LogicRef* output_port =
      module_->AddPort(Direction::kOutput, SanitizeIdentifier(name), bit_count);
  output_section()->Add<ContinuousAssignment>(output_port, value);
  return absl::OkStatus();
}

absl::StatusOr<LogicRef*> ModuleBuilder::DeclareModuleConstant(
    absl::string_view name, const Value& value) {
  // To generate XLS types we need a package.
  // TODO(meheff): There should be a way of generating a Type for a value
  // without instantiating a package.
  Package p("TypeGenerator");
  Type* type = p.GetTypeForValue(value);
  LogicRef* ref;
  if (type->IsArray()) {
    ref = DeclareUnpackedArrayWire(SanitizeIdentifier(name),
                                   type->AsArrayOrDie(), constants_section());
  } else {
    ref = module_->AddWire(SanitizeIdentifier(name), type->GetFlatBitCount(),
                           constants_section());
  }
  XLS_RETURN_IF_ERROR(
      AddAssignmentFromValue(ref, value, [&](Expression* lhs, Expression* rhs) {
        constants_section()->Add<ContinuousAssignment>(lhs, rhs);
      }));
  return ref;
}

LogicRef* ModuleBuilder::DeclareVariable(absl::string_view name, Type* type) {
  if (type->IsArray()) {
    return DeclareUnpackedArrayWire(
        SanitizeIdentifier(name), type->AsArrayOrDie(), declaration_section());
  }
  return module_->AddWire(SanitizeIdentifier(name), type->GetFlatBitCount(),
                          declaration_section());
}

LogicRef* ModuleBuilder::DeclareVariable(absl::string_view name,
                                         int64 bit_count) {
  return module_->AddWire(SanitizeIdentifier(name), bit_count,
                          declaration_section());
}

bool ModuleBuilder::CanEmitAsInlineExpression(
    Node* node, absl::optional<absl::Span<Node* const>> users_of_expression) {
  if (node->GetType()->IsArray()) {
    // TODO(meheff): With system verilog we can do array assignment.
    return false;
  }
  absl::Span<Node* const> users =
      users_of_expression.has_value() ? *users_of_expression : node->users();
  for (Node* user : users) {
    for (int64 i = 0; i < user->operand_count(); ++i) {
      if (user->operand(i) == node && OperandMustBeNamedReference(user, i)) {
        return false;
      }
    }
  }
  // To sidestep Verilog's jolly bit-width inference rules, emit arithmetic
  // expressions as assignments. This gives the results of these expression
  // explicit bit-widths.
  switch (node->op()) {
    case Op::kAdd:
    case Op::kSub:
    case Op::kSMul:
    case Op::kUMul:
    case Op::kSDiv:
    case Op::kUDiv:
      return false;
    default:
      break;
  }
  return true;
}

absl::StatusOr<Expression*> ModuleBuilder::EmitAsInlineExpression(
    Node* node, absl::Span<Expression* const> inputs) {
  if (MustEmitAsFunction(node)) {
    XLS_ASSIGN_OR_RETURN(VerilogFunction * func, DefineFunction(node));
    return file_->Make<VerilogFunctionCall>(func, inputs);
  }
  return NodeToExpression(node, inputs, file_);
}

// Emits a copy and update of an array as a sequence of assignments.
// Specifically, 'rhs' is copied to 'lhs' with the element at the indices
// 'indices' replaced with 'update_value'.  Examples of emitted verilog:
//
//   // lhs: bits[32][3] = array_update(rhs, value, indices=[1]):
//   assign lhs[0] = rhs[0];
//   assign lhs[1] = value;
//   assign lhs[2] = rhs[1];
//
//   // lhs: bits[32][3] = array_update(rhs, value, indices=[x]):
//   assign lhs[0] = x == 0 ? value : rhs[0];
//   assign lhs[1] = x == 1 ? value : rhs[1];
//   assign lhs[2] = x == 2 ? value : rhs[2];
//
//   // lhs: bits[32][3][2] = array_update(rhs, value, indices=[x, 1]):
//   assign lhs[0][0] = rhs[0][0];
//   assign lhs[0][1] = x == 0 ? value : rhs[0][1];
//   assign lhs[0][2] = rhs[0][2];
//   assign lhs[1][0] = rhs[1][0];
//   assign lhs[1][1] = x == 1 ? value : rhs[0][1];
//   assign lhs[1][2] = rhs[0][2];
//
//   // lhs: bits[32][3][2] = array_update(rhs, value, indices=[x, y]):
//   assign lhs[0][0] = (x == 0 && y == 0) ? value : rhs[0][0];
//   assign lhs[0][1] = (x == 0 && y == 1) ? value : rhs[0][1];
//   assign lhs[0][2] = (x == 0 && y == 2) ? value : rhs[0][2];
//   assign lhs[1][0] = (x == 1 && y == 0) ? value : rhs[1][0];
//   assign lhs[1][1] = (x == 1 && y == 1) ? value : rhs[1][1];
//   assign lhs[1][2] = (x == 1 && y == 2) ? value : rhs[1][2];
//
//   // lhs: bits[32][3][2] = array_update(rhs, value, indices=[x]):
//   assign lhs[0][0] = x == 0 ? value[0] : rhs[0][0];
//   assign lhs[0][1] = x == 0 ? value[1] : rhs[0][1];
//   assign lhs[0][2] = x == 0 ? value[2] : rhs[0][2];
//   assign lhs[1][0] = x == 1 ? value[0] : rhs[1][0];
//   assign lhs[1][1] = x == 1 ? value[1] : rhs[1][1];
//   assign lhs[1][2] = x == 1 ? value[2] : rhs[1][2];
//
// EmitArrayCopyAndUpdate recursively constructs the assignments. 'index_match'
// is the index match expression (or explicit true/false value) used in ternary
// expression to select element(s) from the update value or the rhs.
absl::Status ModuleBuilder::EmitArrayCopyAndUpdate(
    IndexableExpression* lhs, IndexableExpression* rhs,
    Expression* update_value, absl::Span<Expression* const> indices,
    IndexMatch index_match, Type* xls_type) {
  auto is_statically_true = [](const IndexMatch& im) {
    return absl::holds_alternative<bool>(im) && absl::get<bool>(im);
  };
  auto is_statically_false = [](const IndexMatch& im) {
    return absl::holds_alternative<bool>(im) && !absl::get<bool>(im);
  };
  auto combine_index_matches = [&](const IndexMatch& a,
                                   const IndexMatch& b) -> IndexMatch {
    if (is_statically_false(a) || is_statically_false(b)) {
      return false;
    }
    if (is_statically_true(a)) {
      return b;
    }
    if (is_statically_true(b)) {
      return a;
    }
    return file_->LogicalAnd(absl::get<Expression*>(a),
                             absl::get<Expression*>(b));
  };

  if (indices.empty()) {
    if (is_statically_true(index_match)) {
      // Indices definitely *do* match the subarray/element being replaced with
      // update value. Assign from update value exclusively. E.g.:
      //   assign lhs[i][j] = update_value[j]
      return AddAssignment(
          /*xls_type=*/xls_type,
          /*lhs=*/lhs,
          /*rhs=*/update_value,
          /*add_assignment=*/
          [&](Expression* lhs, Expression* rhs) {
            assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
          });
    } else if (is_statically_false(index_match)) {
      // Indices definitely do *NOT* match the subarray/element being replaced
      // with update value. Assign from rhs exclusively. E.g.:
      //   assign lhs[i][j] = rhs[j]
      return AddAssignment(
          /*xls_type=*/xls_type,
          /*lhs=*/lhs,
          /*rhs=*/rhs,
          /*add_assignment=*/
          [&](Expression* lhs, Expression* rhs) {
            assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
          });
    } else {
      // Indices may or may not match the subarray/element being replaced with
      // update value. Use a ternary expression to pick from rhs or update
      // value. E.g:
      //   assign lhs[i][j] = (i == idx) ? update_value[j] : rhs[j]
      auto gen_ternary = [&](absl::Span<Expression* const> inputs) {
        return file_->Ternary(absl::get<Expression*>(index_match), inputs[0],
                              inputs[1]);
      };

      // Emit a continuous assignment with a ternary select. The ternary
      // operation supports array types in SystemVerilog so sv_array_expr is
      // true.
      return AddAssignmentToGeneratedExpression(
          xls_type, lhs, /*inputs=*/{update_value, rhs}, gen_ternary,
          /*add_assignment=*/
          [&](Expression* lhs, Expression* rhs) {
            assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
          },
          /*sv_array_expr=*/true);
    }
  }

  // Iterate through array elements and recurse.
  ArrayType* array_type = xls_type->AsArrayOrDie();
  Expression* index = indices.front();
  for (int64 i = 0; i < array_type->size(); ++i) {
    // Compute the current index match expression for this index element.
    IndexMatch current_index_match;
    if (index->IsLiteral()) {
      // Index element is a literal. The condition is statically known (true or
      // false).
      current_index_match = index->IsLiteralWithValue(i);
    } else {
      // Index element is a not literal. The condition is not statically known.
      current_index_match = file_->Equals(index, file_->PlainLiteral(i));
    }
    XLS_RETURN_IF_ERROR(EmitArrayCopyAndUpdate(
        file_->Index(lhs, i), file_->Index(rhs, i), update_value,
        indices.subspan(1),
        combine_index_matches(current_index_match, index_match),
        array_type->element_type()));
  }

  return absl::OkStatus();
}

absl::StatusOr<LogicRef*> ModuleBuilder::EmitAsAssignment(
    absl::string_view name, Node* node, absl::Span<Expression* const> inputs) {
  LogicRef* ref = DeclareVariable(name, node->GetType());

  // TODO(meheff): Arrays should not be special cased here. Instead each op
  // should be expressed using a generator which takes an span of input
  // expressions which is passed to AddAssignmentToGeneratedExpression to handle
  // all types uniformly.
  if (node->GetType()->IsArray()) {
    // Array-shaped operations are handled specially. XLS arrays are represented
    // as unpacked arrays in Verilog/SystemVerilog and unpacked arrays must be
    // assigned element-by-element in Verilog.
    ArrayType* array_type = node->GetType()->AsArrayOrDie();
    switch (node->op()) {
      case Op::kArray: {
        for (int64 i = 0; i < inputs.size(); ++i) {
          XLS_RETURN_IF_ERROR(AddAssignment(
              array_type->element_type(),
              file_->Index(ref, file_->PlainLiteral(i)), inputs[i],
              [&](Expression* lhs, Expression* rhs) {
                assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
              }));
        }
        break;
      }
      case Op::kArrayIndex: {
        IndexableExpression* element = inputs[0]->AsIndexableExpressionOrDie();
        for (Expression* index : inputs.subspan(1)) {
          // TODO(meheff): Handle out-of-bounds index.
          element = file_->Index(element, index);
        }
        XLS_RETURN_IF_ERROR(AddAssignment(
            array_type, ref, element, [&](Expression* lhs, Expression* rhs) {
              assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
            }));
        break;
      }
      case Op::kArrayUpdate:
        XLS_RETURN_IF_ERROR(EmitArrayCopyAndUpdate(
            /*lhs=*/ref,
            /*rhs=*/inputs[0]->AsIndexableExpressionOrDie(),
            /*update_value=*/inputs[1],
            /*indices=*/inputs.subspan(2),
            /*index_match=*/true,
            /*xls_type=*/array_type));
        break;
      case Op::kArrayConcat: {
        if (inputs.size() != node->operands().size()) {
          return absl::InternalError(absl::StrFormat(
              "EmitAsAssignment %s has %d operands, but VAST inputs has %d",
              node->ToString(), node->operands().size(), inputs.size()));
        }

        int64 result_index = 0;
        for (int64 i = 0; i < node->operand_count(); ++i) {
          Node* operand = node->operand(i);

          if (!operand->GetType()->IsArray()) {
            return absl::InternalError(absl::StrFormat(
                "EmitAsAssignment %s has operand %s expected to be array",
                node->ToString(), operand->ToString()));
          }

          Expression* input = inputs.at(i);
          ArrayType* input_type = operand->GetType()->AsArrayOrDie();
          int64 input_size = input_type->size();

          for (int64 j = 0; j < input_size; ++j) {
            XLS_RETURN_IF_ERROR(AddAssignment(
                input_type->element_type(),
                file_->Index(ref, file_->PlainLiteral(result_index)),
                file_->Index(input->AsIndexableExpressionOrDie(),
                             file_->PlainLiteral(j)),
                [&](Expression* lhs, Expression* rhs) {
                  assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
                }));

            ++result_index;
          }
        }
        break;
      }
      case Op::kTupleIndex:
        XLS_RETURN_IF_ERROR(AssignFromSlice(
            ref, inputs[0], array_type,
            GetFlatBitIndexOfElement(
                node->operand(0)->GetType()->AsTupleOrDie(),
                node->As<TupleIndex>()->index()),
            [&](Expression* lhs, Expression* rhs) {
              assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
            }));
        break;
      case Op::kSel: {
        Select* sel = node->As<Select>();
        Expression* selector = inputs[0];
        // Vector of cases including the default case.
        absl::Span<Expression* const> cases = inputs.subspan(1);

        // Selects an element from the set of cases 'inputs' according to the
        // semantics of the select instruction. 'inputs' is the set of all cases
        // including the optional default case which appears last.
        auto select_element = [&](absl::Span<Expression* const> inputs) {
          absl::Span<Expression* const> cases =
              inputs.subspan(0, sel->cases().size());
          Expression* default_expr =
              sel->default_value().has_value() ? inputs.back() : nullptr;
          Expression* result = default_expr;
          for (int64 i = cases.size() - 1; i >= 0; --i) {
            if (result == nullptr) {
              result = cases[i];
            } else {
              result = file_->Ternary(
                  file_->Equals(
                      selector,
                      file_->Literal(
                          i,
                          /*bit_count=*/sel->selector()->BitCountOrDie())),
                  cases[i], result);
            }
          }
          return result;
        };
        XLS_RETURN_IF_ERROR(AddAssignmentToGeneratedExpression(
            array_type, /*lhs=*/ref, /*inputs=*/cases,
            /*gen_rhs_expr=*/select_element,
            /*add_assignment=*/
            [&](Expression* lhs, Expression* rhs) {
              assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
            },
            /*sv_array_expr=*/true));
        break;
      }
      case Op::kOneHotSel: {
        IndexableExpression* selector = inputs[0]->AsIndexableExpressionOrDie();
        // Determine the element type of the potentially-multidimensional
        // array. This is the type of the inputs passed into the expression
        // generator ohs_element.
        Type* element_type = array_type->element_type();
        while (element_type->IsArray()) {
          element_type = element_type->AsArrayOrDie()->element_type();
        }
        int64 element_width = element_type->GetFlatBitCount();
        absl::Span<Expression* const> cases = inputs.subspan(1);
        // Generate a one-hot-select operation of the given inputs.
        auto ohs_element = [&](absl::Span<Expression* const> inputs) {
          Expression* result = nullptr;
          for (int64 i = 0; i < inputs.size(); ++i) {
            Expression* masked_input =
                element_width == 1
                    ? file_->BitwiseAnd(inputs[i], file_->Index(selector, i))
                    : file_->BitwiseAnd(inputs[i],
                                        file_->Concat(
                                            /*replication=*/element_width,
                                            {file_->Index(selector, i)}));
            result = result == nullptr ? masked_input
                                       : file_->BitwiseOr(result, masked_input);
          }
          return result;
        };
        // sv_array_expr is false because the generated one-hot-select
        // expression which consists of ANDs and ORs is invalid for array
        // inputs.
        XLS_RETURN_IF_ERROR(AddAssignmentToGeneratedExpression(
            array_type, /*lhs=*/ref, /*inputs=*/cases,
            ohs_element, /*add_assignment=*/
            [&](Expression* lhs, Expression* rhs) {
              assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
            },
            /*sv_array_expr=*/false));
        break;
      }
      default:
        return absl::UnimplementedError(
            absl::StrCat("Unsupported array-shaped op: ", node->ToString()));
    }
  } else {
    XLS_ASSIGN_OR_RETURN(Expression * expr,
                         EmitAsInlineExpression(node, inputs));
    XLS_RETURN_IF_ERROR(Assign(ref, expr, node->GetType()));
  }
  return ref;
}

absl::Status ModuleBuilder::Assign(LogicRef* lhs, Expression* rhs, Type* type) {
  XLS_RETURN_IF_ERROR(
      AddAssignment(type, lhs, rhs, [&](Expression* lhs, Expression* rhs) {
        assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
      }));
  return absl::OkStatus();
}

absl::StatusOr<ModuleBuilder::Register> ModuleBuilder::DeclareRegister(
    absl::string_view name, Type* type, Expression* next,
    absl::optional<Expression*> reset_value) {
  LogicRef* reg;
  if (type->IsArray()) {
    // Currently, an array register requires SystemVerilog because there is an
    // array assignment in the always flop block.
    reg = DeclareUnpackedArrayReg(SanitizeIdentifier(name),
                                  type->AsArrayOrDie(), declaration_section());
  } else {
    reg = module_->AddReg(SanitizeIdentifier(name), type->GetFlatBitCount(),
                          /*init=*/absl::nullopt, declaration_section());
  }
  return Register{
      .ref = reg,
      .next = next,
      .reset_value = reset_value.has_value() ? reset_value.value() : nullptr,
      .xls_type = type};
}

absl::StatusOr<ModuleBuilder::Register> ModuleBuilder::DeclareRegister(
    absl::string_view name, int64 bit_count, Expression* next,
    absl::optional<Expression*> reset_value) {
  return Register{
      .ref = module_->AddReg(SanitizeIdentifier(name), bit_count,
                             /*init=*/absl::nullopt, declaration_section()),
      .next = next,
      .reset_value = reset_value.has_value() ? reset_value.value() : nullptr,
      .xls_type = nullptr};
}

absl::Status ModuleBuilder::AssignRegisters(
    LogicRef* clk, absl::Span<const Register> registers,
    Expression* load_enable, absl::optional<Reset> rst) {
  // Construct an always_ff block.
  std::vector<SensitivityListElement> sensitivity_list;
  sensitivity_list.push_back(file_->Make<PosEdge>(clk));
  if (rst.has_value()) {
    if (rst->asynchronous) {
      if (rst->active_low) {
        sensitivity_list.push_back(file_->Make<NegEdge>(rst->signal));
      } else {
        sensitivity_list.push_back(file_->Make<PosEdge>(rst->signal));
      }
    }
  }
  AlwaysBase* always;
  if (use_system_verilog_) {
    always = assignment_section()->Add<AlwaysFf>(file_, sensitivity_list);
  } else {
    always = assignment_section()->Add<Always>(file_, sensitivity_list);
  }
  // assignment_block is the block in which the foo <= foo_next assignments
  // go. It can either be conditional (if there is a reset signal) or
  // unconditional.
  StatementBlock* assignment_block = always->statements();
  if (rst.has_value()) {
    // Registers have a reset signal. Conditionally assign the registers based
    // on whether the reset signal is asserted.
    Expression* rst_condition;
    if (rst->active_low) {
      rst_condition = file_->LogicalNot(rst->signal);
    } else {
      rst_condition = rst->signal;
    }
    Conditional* conditional =
        always->statements()->Add<Conditional>(file_, rst_condition);
    for (const Register& reg : registers) {
      XLS_RET_CHECK_NE(reg.reset_value, nullptr);
      XLS_RETURN_IF_ERROR(AddAssignment(
          reg.xls_type, reg.ref, reg.reset_value,
          [&](Expression* lhs, Expression* rhs) {
            conditional->consequent()->Add<NonblockingAssignment>(lhs, rhs);
          }));
    }
    assignment_block = conditional->AddAlternate();
  }
  // Assign registers to the next value for the non-reset case (either no
  // reset signal or reset signal is not asserted).
  for (const Register& reg : registers) {
    XLS_RETURN_IF_ERROR(AddAssignment(
        reg.xls_type, reg.ref, reg.next, [&](Expression* lhs, Expression* rhs) {
          assignment_block->Add<NonblockingAssignment>(
              lhs, load_enable == nullptr
                       ? rhs
                       : file_->Ternary(load_enable, rhs, lhs));
        }));
  }
  return absl::OkStatus();
}

bool ModuleBuilder::MustEmitAsFunction(Node* node) {
  switch (node->op()) {
    case Op::kSMul:
    case Op::kUMul:
    case Op::kDynamicBitSlice:
      return true;
    default:
      return false;
  }
}

std::string ModuleBuilder::VerilogFunctionName(Node* node) {
  switch (node->op()) {
    case Op::kSMul:
    case Op::kUMul:
      // Multiplies may be mixed width so include result and operand widths in
      // the name.
      return absl::StrFormat(
          "%s%db_%db_x_%db", OpToString(node->op()), node->BitCountOrDie(),
          node->operand(0)->BitCountOrDie(), node->operand(1)->BitCountOrDie());
    case Op::kDynamicBitSlice:
      return absl::StrFormat(
          "%s_w%d_%db_%db", OpToString(node->op()), node->BitCountOrDie(),
          node->operand(0)->BitCountOrDie(), node->operand(1)->BitCountOrDie());
    default:
      XLS_LOG(FATAL) << "Cannot emit node as function: " << node->ToString();
  }
}

namespace {

// Defines and returns a function which implements the given DynamicBitSlice
// node.
VerilogFunction* DefineDynamicBitSliceFunction(Node* node,
                                               absl::string_view function_name,
                                               ModuleSection* section) {
  XLS_CHECK_EQ(node->op(), Op::kDynamicBitSlice);
  VerilogFile* file = section->file();
  VerilogFunction* func =
      section->Add<VerilogFunction>(function_name, node->BitCountOrDie(), file);
  XLS_CHECK_EQ(node->operand_count(), 2);
  DynamicBitSlice* slice = node->As<DynamicBitSlice>();
  Expression* operand =
      func->AddArgument("operand", node->operand(0)->BitCountOrDie());
  Expression* start =
      func->AddArgument("start", node->operand(1)->BitCountOrDie());
  int64 width = slice->width();

  LogicRef* zexted_operand = func->AddRegDef(
      "zexted_operand",
      file->PlainLiteral(node->operand(0)->BitCountOrDie() + width),
      /*init=*/UninitializedSentinel(), /*is_signed=*/false);

  Expression* zeros = file->Literal(0, width);
  Expression* op_width = file->Literal(
      slice->operand(0)->BitCountOrDie(),
      Bits::MinBitCountUnsigned(slice->operand(0)->BitCountOrDie()));
  // If start of slice is greater than or equal to operand width, result is
  // completely out of bounds and set to all zeros.
  Expression* out_of_bounds = file->GreaterThanEquals(start, op_width);
  // Pad with width zeros
  func->AddStatement<BlockingAssignment>(zexted_operand,
                                         file->Concat({zeros, operand}));
  Expression* sliced_operand = file->DynamicSlice(zexted_operand, start, width);
  func->AddStatement<BlockingAssignment>(
      func->return_value_ref(),
      file->Ternary(out_of_bounds, zeros, sliced_operand));
  return func;
}

// Defines and returns a function which implements the given SMul node.
VerilogFunction* DefineSmulFunction(Node* node, absl::string_view function_name,
                                    ModuleSection* section) {
  XLS_CHECK_EQ(node->op(), Op::kSMul);
  VerilogFile* file = section->file();

  ScopedLintDisable lint_disable(section, {Lint::kSignedType, Lint::kMultiply});

  VerilogFunction* func =
      section->Add<VerilogFunction>(function_name, node->BitCountOrDie(), file);
  XLS_CHECK_EQ(node->operand_count(), 2);
  Expression* lhs = func->AddArgument("lhs", node->operand(0)->BitCountOrDie());
  Expression* rhs = func->AddArgument("rhs", node->operand(1)->BitCountOrDie());
  // The code conservatively assigns signed-casted inputs to temporary
  // variables, uses them in the multiply expression which is assigned to
  // another signed temporary. Finally, this is unsign-casted and assigned to
  // the return value of the function. These shenanigans ensure no surprising
  // sign/zero extensions of any values.
  LogicRef* signed_lhs = func->AddRegDef(
      "signed_lhs", file->PlainLiteral(node->operand(0)->BitCountOrDie()),
      /*init=*/UninitializedSentinel(), /*is_signed=*/true);
  LogicRef* signed_rhs = func->AddRegDef(
      "signed_rhs", file->PlainLiteral(node->operand(1)->BitCountOrDie()),
      /*init=*/UninitializedSentinel(), /*is_signed=*/true);
  LogicRef* signed_result = func->AddRegDef(
      "signed_result", file->PlainLiteral(node->BitCountOrDie()),
      /*init=*/UninitializedSentinel(), /*is_signed=*/true);
  func->AddStatement<BlockingAssignment>(signed_lhs,
                                         file->Make<SignedCast>(lhs));
  func->AddStatement<BlockingAssignment>(signed_rhs,
                                         file->Make<SignedCast>(rhs));
  func->AddStatement<BlockingAssignment>(signed_result,
                                         file->Mul(signed_lhs, signed_rhs));
  func->AddStatement<BlockingAssignment>(
      func->return_value_ref(), file->Make<UnsignedCast>(signed_result));

  return func;
}

// Defines and returns a function which implements the given UMul node.
VerilogFunction* DefineUmulFunction(Node* node, absl::string_view function_name,
                                    ModuleSection* section) {
  XLS_CHECK_EQ(node->op(), Op::kUMul);
  VerilogFile* file = section->file();

  ScopedLintDisable lint_disable(section, {Lint::kMultiply});

  VerilogFunction* func =
      section->Add<VerilogFunction>(function_name, node->BitCountOrDie(), file);
  XLS_CHECK_EQ(node->operand_count(), 2);
  Expression* lhs = func->AddArgument("lhs", node->operand(0)->BitCountOrDie());
  Expression* rhs = func->AddArgument("rhs", node->operand(1)->BitCountOrDie());
  func->AddStatement<BlockingAssignment>(func->return_value_ref(),
                                         file->Mul(lhs, rhs));

  return func;
}

}  // namespace

absl::StatusOr<VerilogFunction*> ModuleBuilder::DefineFunction(Node* node) {
  std::string function_name = VerilogFunctionName(node);
  if (node_functions_.contains(function_name)) {
    return node_functions_.at(function_name);
  }
  VerilogFunction* func;
  switch (node->op()) {
    case Op::kSMul:
      func = DefineSmulFunction(node, function_name, functions_section_);
      break;
    case Op::kUMul:
      func = DefineUmulFunction(node, function_name, functions_section_);
      break;
    case Op::kDynamicBitSlice:
      func = DefineDynamicBitSliceFunction(node, function_name,
                                           functions_section_);
      break;
    default:
      XLS_LOG(FATAL) << "Cannot define node as function: " << node->ToString();
  }
  node_functions_[function_name] = func;
  return func;
}

}  // namespace verilog
}  // namespace xls
