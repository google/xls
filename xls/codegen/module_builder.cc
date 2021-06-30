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
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/lint_annotate.h"
#include "xls/codegen/node_expressions.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "re2/re2.h"

namespace xls {
namespace verilog {

namespace {

// Returns the bounds of the potentially-nested array type as a vector of
// int64_t. Ordering of the vector is outer-most bound to inner-most. For
// example, given array type 'bits[32][4][5]' yields {5, 4, 32}.
std::vector<int64_t> NestedArrayBounds(ArrayType* type) {
  std::vector<int64_t> bounds;
  Type* t = type;
  while (t->IsArray()) {
    bounds.push_back(t->AsArrayOrDie()->size());
    t = t->AsArrayOrDie()->element_type();
  }
  return bounds;
}

// Returns the width of the element of the potentially nested array type. For
// example, given array type 'bits[32][4][5]' yields 32.
int64_t NestedElementWidth(ArrayType* type) {
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
  // Compound types are represented as a concatenation of their elements.
  std::vector<Value> value_elements;
  if (value.IsArray()) {
    for (int64_t i = value.size() - 1; i >= 0; --i) {
      value_elements.push_back(value.element(i));
    }
  } else {
    XLS_RET_CHECK(value.IsTuple());
    for (const Value& element : value.elements()) {
      value_elements.push_back(element);
    }
  }
  std::vector<Expression*> elements;
  for (const Value& element : value_elements) {
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
    for (int64_t i = 0; i < array_type->size(); ++i) {
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
      for (int64_t i = 0; i < value.size(); ++i) {
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
                             bool use_system_verilog,
                             absl::optional<absl::string_view> clk_name,
                             absl::optional<ResetProto> rst_proto)
    : module_name_(SanitizeIdentifier(name)),
      file_(file),
      package_("__ModuleBuilder_type_generator"),
      use_system_verilog_(use_system_verilog) {
  module_ = file_->AddModule(module_name_);
  functions_section_ = module_->Add<ModuleSection>();
  constants_section_ = module_->Add<ModuleSection>();
  input_section_ = module_->Add<ModuleSection>();
  declaration_and_assignment_section_ = module_->Add<ModuleSection>();
  assert_section_ = module_->Add<ModuleSection>();
  cover_section_ = module_->Add<ModuleSection>();
  output_section_ = module_->Add<ModuleSection>();

  NewDeclarationAndAssignmentSections();

  if (clk_name.has_value()) {
    clk_ = AddInputPort(clk_name.value(), /*bit_count=*/1);
  }

  if (rst_proto.has_value()) {
    rst_ = Reset();
    rst_->signal = AddInputPort(rst_proto->name(), /*bit_count=*/1);
    rst_->asynchronous = rst_proto->asynchronous();
    rst_->active_low = rst_proto->active_low();
  }
}

void ModuleBuilder::NewDeclarationAndAssignmentSections() {
  declaration_subsections_.push_back(
      declaration_and_assignment_section_->Add<ModuleSection>());
  assignment_subsections_.push_back(
      declaration_and_assignment_section_->Add<ModuleSection>());
}

absl::Status ModuleBuilder::AssignFromSlice(
    Expression* lhs, Expression* rhs, Type* xls_type, int64_t slice_start,
    std::function<void(Expression*, Expression*)> add_assignment) {
  if (xls_type->IsArray()) {
    ArrayType* array_type = xls_type->AsArrayOrDie();
    for (int64_t i = 0; i < array_type->size(); ++i) {
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
  LogicRef* port =
      AddInputPort(SanitizeIdentifier(name), type->GetFlatBitCount());
  if (!type->IsArray()) {
    return port;
  }
  // All inputs are flattened so unflatten arrays with a sequence of
  // assignments.
  ArrayType* array_type = type->AsArrayOrDie();
  LogicRef* ar =
      module_->AddWire(absl::StrCat(SanitizeIdentifier(name), "_unflattened"),
                       file_->UnpackedArrayType(NestedElementWidth(array_type),
                                                NestedArrayBounds(array_type)),
                       input_section());
  XLS_RETURN_IF_ERROR(AssignFromSlice(
      ar, port, type->AsArrayOrDie(), 0, [&](Expression* lhs, Expression* rhs) {
        input_section()->Add<ContinuousAssignment>(lhs, rhs);
      }));
  return ar;
}

LogicRef* ModuleBuilder::AddInputPort(absl::string_view name,
                                      int64_t bit_count) {
  return module_->AddInput(SanitizeIdentifier(name),
                           file_->BitVectorType(bit_count));
}

absl::Status ModuleBuilder::AddOutputPort(absl::string_view name, Type* type,
                                          Expression* value) {
  LogicRef* output_port = module_->AddOutput(
      SanitizeIdentifier(name), file_->BitVectorType(type->GetFlatBitCount()));

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
                                          int64_t bit_count,
                                          Expression* value) {
  LogicRef* output_port = module_->AddOutput(SanitizeIdentifier(name),
                                             file_->BitVectorType(bit_count));
  output_section()->Add<ContinuousAssignment>(output_port, value);
  return absl::OkStatus();
}

absl::StatusOr<LogicRef*> ModuleBuilder::DeclareModuleConstant(
    absl::string_view name, const Value& value) {
  Type* type = package_.GetTypeForValue(value);
  LogicRef* ref;
  if (type->IsArray()) {
    ArrayType* array_type = type->AsArrayOrDie();
    ref = module_->AddWire(
        SanitizeIdentifier(name),
        file_->UnpackedArrayType(NestedElementWidth(array_type),
                                 NestedArrayBounds(array_type)),
        constants_section());
  } else {
    ref = module_->AddWire(SanitizeIdentifier(name),
                           file_->BitVectorType(type->GetFlatBitCount()),
                           constants_section());
  }
  XLS_RETURN_IF_ERROR(
      AddAssignmentFromValue(ref, value, [&](Expression* lhs, Expression* rhs) {
        constants_section()->Add<ContinuousAssignment>(lhs, rhs);
      }));
  return ref;
}

LogicRef* ModuleBuilder::DeclareVariable(absl::string_view name, Type* type) {
  DataType* data_type;
  if (type->IsArray()) {
    ArrayType* array_type = type->AsArrayOrDie();
    data_type = file_->UnpackedArrayType(NestedElementWidth(array_type),
                                         NestedArrayBounds(array_type));
  } else {
    data_type = file_->BitVectorType(type->GetFlatBitCount());
  }
  return module_->AddWire(SanitizeIdentifier(name), data_type,
                          declaration_section());
}

LogicRef* ModuleBuilder::DeclareVariable(absl::string_view name,
                                         int64_t bit_count) {
  return module_->AddWire(SanitizeIdentifier(name),
                          file_->BitVectorType(bit_count),
                          declaration_section());
}

bool ModuleBuilder::CanEmitAsInlineExpression(
    Node* node, absl::optional<absl::Span<Node* const>> users_of_expression) {
  if (node->GetType()->IsArray()) {
    // TODO(meheff): With system verilog we can do array assignment.
    return false;
  }

  std::vector<Node*> users_vec;
  absl::Span<Node* const> users;
  if (users_of_expression.has_value()) {
    users = *users_of_expression;
  } else {
    users_vec.insert(users_vec.begin(), node->users().begin(),
                     node->users().end());
    users = users_vec;
  }
  for (Node* user : users) {
    for (int64_t i = 0; i < user->operand_count(); ++i) {
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
    Expression* update_value,
    absl::Span<const ModuleBuilder::IndexType> indices, IndexMatch index_match,
    Type* xls_type) {
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
  IndexType index_type = indices.front();
  int64_t index_bit_count = index_type.xls_type->bit_count();
  for (int64_t i = 0; i < array_type->size(); ++i) {
    // Compute the current index match expression for this index element.
    IndexMatch current_index_match;
    if (index_type.expression->IsLiteral()) {
      // Index element is a literal. The condition is statically known (true or
      // false).
      current_index_match = index_type.expression->IsLiteralWithValue(i);
    } else if (Bits::MinBitCountUnsigned(i) > index_bit_count) {
      // Index is not wide enough to hold 'i'.
      current_index_match = false;
    } else {
      // Index element is a not literal. The condition is not statically known.
      current_index_match = file_->Equals(
          index_type.expression, file_->Literal(UBits(i, index_bit_count)));
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
        for (int64_t i = 0; i < inputs.size(); ++i) {
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
        XLS_ASSIGN_OR_RETURN(
            IndexableExpression * rhs,
            ArrayIndexExpression(inputs[0]->AsIndexableExpressionOrDie(),
                                 inputs.subspan(1), node->As<ArrayIndex>()));
        XLS_RETURN_IF_ERROR(AddAssignment(
            array_type, ref, rhs, [&](Expression* lhs, Expression* rhs) {
              assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
            }));
        break;
      }
      case Op::kArraySlice: {
        ArraySlice* slice = node->As<ArraySlice>();
        IndexableExpression* input_array =
            inputs[0]->AsIndexableExpressionOrDie();
        int64_t input_array_size =
            slice->array()->GetType()->AsArrayOrDie()->size();

        // If the start value is too narrow to hold the maximum index value of
        // the input array, zero-extend it to avoid overflows in the necessary
        // comparison and arithmetic operations below.
        int64_t min_index_width =
            Bits::MinBitCountUnsigned(input_array_size - 1);

        Expression* start_expr = inputs[1];
        int64_t start_width = slice->start()->BitCountOrDie();
        if (start_width < min_index_width) {
          // Zero-extend start to `min_index_width` bits.
          start_expr = file_->Concat(
              {file_->Literal(0, min_index_width - start_width), start_expr});
          start_width = min_index_width;
        }

        Expression* max_index_expr =
            file_->Literal(input_array_size - 1, start_width);
        for (int64_t i = 0; i < array_type->size(); i++) {
          // The index for iteration `i` is out of bounds if the following
          // condition is true:
          //   start + i > $INPUT_ARRAY_SIZE - 1
          // However, the expression `start + i` might overflow so instead
          // equivalently compute as:
          //   start > $INPUT_ARRAY_SIZE - 1 - i
          // Check that `$INPUT_ARRAY_SIZE - 1 - i` is non-negative to avoid
          // underflow. This is possible if the input array is narrower than the
          // output array (slice is wider than its input)
          Expression* element;
          if (input_array_size - 1 - i < 0) {
            // Index is definitely out of bounds.
            element = file_->Index(input_array, max_index_expr);
          } else {
            // Index might be out of bounds.
            Expression* oob_condition = file_->GreaterThan(
                start_expr,
                file_->Literal(input_array_size - 1 - i, start_width));
            element = file_->Index(
                input_array,
                file_->Ternary(
                    oob_condition, max_index_expr,
                    file_->Add(start_expr, file_->Literal(i, start_width))));
          }
          assignment_section()->Add<ContinuousAssignment>(
              file_->Index(ref, file_->PlainLiteral(i)), element);
        }
        break;
      }
      case Op::kArrayUpdate: {
        // Gather the index expression values and types together. The index
        // operands of array update start at 2.
        std::vector<IndexType> index_types;
        for (int64_t i = 2; i < node->operand_count(); ++i) {
          index_types.push_back(
              IndexType{inputs[i], node->operand(i)->GetType()->AsBitsOrDie()});
        }
        XLS_RETURN_IF_ERROR(EmitArrayCopyAndUpdate(
            /*lhs=*/ref,
            /*rhs=*/inputs[0]->AsIndexableExpressionOrDie(),
            /*update_value=*/inputs[1],
            /*indices=*/index_types,
            /*index_match=*/true,
            /*xls_type=*/array_type));
        break;
      }
      case Op::kArrayConcat: {
        if (inputs.size() != node->operands().size()) {
          return absl::InternalError(absl::StrFormat(
              "EmitAsAssignment %s has %d operands, but VAST inputs has %d",
              node->ToString(), node->operands().size(), inputs.size()));
        }

        int64_t result_index = 0;
        for (int64_t i = 0; i < node->operand_count(); ++i) {
          Node* operand = node->operand(i);

          if (!operand->GetType()->IsArray()) {
            return absl::InternalError(absl::StrFormat(
                "EmitAsAssignment %s has operand %s expected to be array",
                node->ToString(), operand->ToString()));
          }

          Expression* input = inputs.at(i);
          ArrayType* input_type = operand->GetType()->AsArrayOrDie();
          int64_t input_size = input_type->size();

          for (int64_t j = 0; j < input_size; ++j) {
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
          for (int64_t i = cases.size() - 1; i >= 0; --i) {
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
        int64_t element_width = element_type->GetFlatBitCount();
        absl::Span<Expression* const> cases = inputs.subspan(1);
        // Generate a one-hot-select operation of the given inputs.
        auto ohs_element = [&](absl::Span<Expression* const> inputs) {
          Expression* result = nullptr;
          for (int64_t i = 0; i < inputs.size(); ++i) {
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

absl::StatusOr<std::string> ModuleBuilder::GenerateAssertString(
    absl::string_view fmt_string, xls::Assert* asrt, Expression* condition) {
  RE2 re(R"(({\w+}))");
  std::string placeholder;
  absl::string_view piece(fmt_string);
  // Using std::set for ordered emission of strings in error message.
  const std::set<std::string> kSupportedPlaceholders = {
      "{message}", "{condition}", "{label}", "{clk}", "{rst}"};
  while (RE2::FindAndConsume(&piece, re, &placeholder)) {
    if (kSupportedPlaceholders.find(placeholder) ==
        kSupportedPlaceholders.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid placeholder '%s' in assert format string. "
          "Supported placeholders: %s",
          placeholder,
          absl::StrJoin(std::vector<std::string>(kSupportedPlaceholders.begin(),
                                                 kSupportedPlaceholders.end()),
                        ", ")));
    }
  }

  std::string assert_str(fmt_string);
  absl::StrReplaceAll({{"{message}", asrt->message()}}, &assert_str);
  absl::StrReplaceAll({{"{condition}", condition->Emit()}}, &assert_str);
  if (absl::StrContains(assert_str, "{label}")) {
    if (!asrt->label().has_value()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Assert format string has '{label}' placeholder, "
                          "but assert operation has no label."));
    }
    absl::StrReplaceAll({{"{label}", asrt->label().value()}}, &assert_str);
  }
  if (absl::StrContains(assert_str, "{clk}")) {
    if (clk_ == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Assert format string has '{clk}' placeholder, "
                          "but block has no clock signal."));
    }
    absl::StrReplaceAll({{"{clk}", clk_->GetName()}}, &assert_str);
  }
  if (absl::StrContains(assert_str, "{rst}")) {
    if (!rst_.has_value()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Assert format string has '{rst}' placeholder, "
                          "but block has no reset signal."));
    }
    absl::StrReplaceAll({{"{rst}", rst_->signal->GetName()}}, &assert_str);
  }
  return assert_str;
}

absl::Status ModuleBuilder::EmitAssert(
    xls::Assert* asrt, Expression* condition,
    absl::optional<absl::string_view> fmt_string) {
  if (fmt_string.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        std::string assert_str,
        GenerateAssertString(fmt_string.value(), asrt, condition));
    assert_section_->Add<RawStatement>(assert_str + ";");
    return absl::OkStatus();
  }

  if (!use_system_verilog_) {
    // Asserts are a SystemVerilog only feature.
    // TODO(meheff): 2021/02/27 We should raise an error here or possibly emit a
    // construct like: if (!condition) $display("Assert failed ...");
    XLS_LOG(WARNING) << "Asserts are only supported in SystemVerilog.";
    return absl::OkStatus();
  }
  if (assert_always_comb_ == nullptr) {
    // Lazily create the always_comb block.
    assert_always_comb_ = assert_section_->Add<AlwaysComb>();
  }

  // Guard the assert with $isunknown to avoid triggering the assert condition
  // prior to inputs being driven in the testbench. For example:
  //
  //   assert($isunknown(cond) || cond) else $fatal("Error message");
  //
  // TODO(meheff): Figure out a better way of handling this, perhaps by
  // adjusting our testbench architecture to drive inputs immediately for
  // combinational blocks and asserting only on rising clock edge for
  // non-combinational blocks.
  assert_always_comb_->statements()->Add<Assert>(
      file_->LogicalOr(file_->Make<SystemFunctionCall>(
                           "isunknown", std::vector<Expression*>({condition})),
                       condition),
      asrt->message());
  return absl::OkStatus();
}

absl::Status ModuleBuilder::EmitCover(xls::Cover* cover,
                                      Expression* condition) {
  if (!use_system_verilog_) {
    // Coverpoints are a SystemVerilog only feature.
    XLS_LOG(WARNING) << "Coverpoints are only supported in SystemVerilog.";
    return absl::OkStatus();
  }
  if (clk_ == nullptr) {
    return absl::InvalidArgumentError(
        "Coverpoints require a clock to be present in the module.");
  }
  if (cover_always_comb_ == nullptr) {
    cover_always_comb_ = cover_section_->Add<AlwaysComb>();
  }
  cover_always_comb_->statements()->Add<Cover>(clk_, condition, cover->label());
  return absl::OkStatus();
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
    Expression* reset_value) {
  if (clk_ == nullptr) {
    return absl::InvalidArgumentError("Clock signal required for register.");
  }
  if (!rst_.has_value() && reset_value != nullptr) {
    return absl::InvalidArgumentError(
        "Block has no reset signal, but register has reset value.");
  }

  LogicRef* reg;
  if (type->IsArray()) {
    // Currently, an array register requires SystemVerilog because there is an
    // array assignment in the always flop block.
    ArrayType* array_type = type->AsArrayOrDie();
    reg =
        module_->AddReg(SanitizeIdentifier(name),
                        file_->UnpackedArrayType(NestedElementWidth(array_type),
                                                 NestedArrayBounds(array_type)),
                        /*init=*/nullptr, declaration_section());
  } else {
    reg = module_->AddReg(SanitizeIdentifier(name),
                          file_->BitVectorType(type->GetFlatBitCount()),
                          /*init=*/nullptr, declaration_section());
  }
  return Register{.ref = reg,
                  .next = next,
                  .reset_value = reset_value,
                  .load_enable = nullptr,
                  .xls_type = type};
}

absl::StatusOr<ModuleBuilder::Register> ModuleBuilder::DeclareRegister(
    absl::string_view name, int64_t bit_count, Expression* next,
    Expression* reset_value) {
  if (clk_ == nullptr) {
    return absl::InvalidArgumentError("Clock signal required for register.");
  }
  if (!rst_.has_value() && reset_value != nullptr) {
    return absl::InvalidArgumentError(
        "Block has no reset signal, but register has reset value.");
  }

  return Register{.ref = module_->AddReg(
                      SanitizeIdentifier(name), file_->BitVectorType(bit_count),
                      /*init=*/nullptr, declaration_section()),
                  .next = next,
                  .reset_value = reset_value,
                  .load_enable = nullptr,
                  .xls_type = nullptr};
}

absl::Status ModuleBuilder::AssignRegisters(
    absl::Span<const Register> registers) {
  XLS_RET_CHECK(clk_ != nullptr);

  // Construct an always_ff block.
  std::vector<SensitivityListElement> sensitivity_list;
  sensitivity_list.push_back(file_->Make<PosEdge>(clk_));
  if (rst_.has_value()) {
    if (rst_->asynchronous) {
      if (rst_->active_low) {
        sensitivity_list.push_back(file_->Make<NegEdge>(rst_->signal));
      } else {
        sensitivity_list.push_back(file_->Make<PosEdge>(rst_->signal));
      }
    }
  }
  AlwaysBase* always;
  if (use_system_verilog_) {
    always = assignment_section()->Add<AlwaysFf>(sensitivity_list);
  } else {
    always = assignment_section()->Add<Always>(sensitivity_list);
  }
  // assignment_block is the block in which the foo <= foo_next assignments
  // go. It can either be conditional (if there is a reset signal) or
  // unconditional.
  StatementBlock* assignment_block = always->statements();
  if (rst_.has_value() &&
      std::any_of(registers.begin(), registers.end(),
                  [](const Register& r) { return r.reset_value != nullptr; })) {
    // Registers have a reset signal. Conditionally assign the registers based
    // on whether the reset signal is asserted.
    Expression* rst_condition;
    if (rst_->active_low) {
      rst_condition = file_->LogicalNot(rst_->signal);
    } else {
      rst_condition = rst_->signal;
    }
    Conditional* conditional =
        always->statements()->Add<Conditional>(rst_condition);
    for (const Register& reg : registers) {
      if (reg.reset_value == nullptr) {
        // Not all registers may have reset values.
        continue;
      }
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
    XLS_RET_CHECK(reg.next != nullptr);
    XLS_RETURN_IF_ERROR(AddAssignment(
        reg.xls_type, reg.ref, reg.next, [&](Expression* lhs, Expression* rhs) {
          assignment_block->Add<NonblockingAssignment>(
              lhs, reg.load_enable == nullptr
                       ? rhs
                       : file_->Ternary(reg.load_enable, rhs, lhs));
        }));
  }
  return absl::OkStatus();
}

bool ModuleBuilder::MustEmitAsFunction(Node* node) {
  switch (node->op()) {
    case Op::kSMul:
    case Op::kUMul:
    case Op::kDynamicBitSlice:
    case Op::kBitSliceUpdate:
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
    case Op::kBitSliceUpdate:
      return absl::StrFormat(
          "%s_w%d_%db_%db", OpToString(node->op()), node->BitCountOrDie(),
          node->operand(1)->BitCountOrDie(), node->operand(2)->BitCountOrDie());
    default:
      XLS_LOG(FATAL) << "Cannot emit node as function: " << node->ToString();
  }
}

namespace {

// Defines and returns a function which implements the given DynamicBitSlice
// node.
VerilogFunction* DefineDynamicBitSliceFunction(DynamicBitSlice* slice,
                                               absl::string_view function_name,
                                               ModuleSection* section) {
  VerilogFile* file = section->file();
  VerilogFunction* func = section->Add<VerilogFunction>(
      function_name, file->BitVectorType(slice->BitCountOrDie()));
  Expression* operand = func->AddArgument(
      "operand", file->BitVectorType(slice->to_slice()->BitCountOrDie()));
  Expression* start = func->AddArgument(
      "start", file->BitVectorType(slice->start()->BitCountOrDie()));
  int64_t width = slice->width();

  LogicRef* zexted_operand = func->AddRegDef(
      "zexted_operand",
      file->BitVectorType(slice->to_slice()->BitCountOrDie() + width),
      /*init=*/nullptr);

  Expression* zeros = file->Literal(0, width);
  Expression* op_width = file->Literal(
      slice->operand(0)->BitCountOrDie(),
      Bits::MinBitCountUnsigned(slice->to_slice()->BitCountOrDie()));
  // If start of slice is greater than or equal to operand width, result is
  // completely out of bounds and set to all zeros.
  Expression* out_of_bounds = file->GreaterThanEquals(start, op_width);
  // Pad with width zeros
  func->AddStatement<BlockingAssignment>(zexted_operand,
                                         file->Concat({zeros, operand}));
  Expression* sliced_operand = file->PartSelect(zexted_operand, start, width);
  func->AddStatement<BlockingAssignment>(
      func->return_value_ref(),
      file->Ternary(out_of_bounds, zeros, sliced_operand));
  return func;
}

// Defines and returns a function which implements the given BitSliceUpdate
// node.
VerilogFunction* DefineBitSliceUpdateFunction(BitSliceUpdate* update,
                                              absl::string_view function_name,
                                              ModuleSection* section) {
  VerilogFile* file = section->file();
  int64_t to_update_width = update->to_update()->BitCountOrDie();
  int64_t start_width = update->start()->BitCountOrDie();
  int64_t update_value_width = update->update_value()->BitCountOrDie();

  // We purposefully avoid using scalars here, because they cannot be sliced.
  VerilogFunction* func = section->Add<VerilogFunction>(
      function_name, file->BitVectorTypeNoScalar(update->BitCountOrDie()));
  IndexableExpression* to_update = func->AddArgument(
      "to_update", file->BitVectorTypeNoScalar(to_update_width));
  IndexableExpression* start =
      func->AddArgument("start", file->BitVectorTypeNoScalar(start_width));
  IndexableExpression* update_value = func->AddArgument(
      "update_value", file->BitVectorTypeNoScalar(update_value_width));

  Expression* adjusted_update_value;
  if (update_value_width > to_update_width) {
    // Update value is the wider than the value to be updated. Slice update
    // value to match the width.
    adjusted_update_value =
        file->Slice(update_value, file->PlainLiteral(to_update_width - 1),
                    file->PlainLiteral(0));
  } else if (update_value_width < to_update_width) {
    // Update value is the narrower than the value to be updated. Zero-extend
    // update value to match the width.
    adjusted_update_value =
        file->Concat({file->Literal(Bits(to_update_width - update_value_width)),
                      update_value});
  } else {
    // Update value is the same width as the value to be updated.
    adjusted_update_value = update_value;
  }

  // Create a mask for zeroing the bits of the updated bits of the value to
  // update.
  //
  //            update value width
  //                     |
  //             +-------+------+
  //             V              V
  // mask:   111110000000000000001111111111
  //                            ^         ^
  //                          start       0
  //
  // updated_value = update_value << start | mask & to_update
  Bits all_ones = bits_ops::ZeroExtend(
      Bits::AllOnes(std::min(update_value_width, to_update_width)),
      to_update_width);
  Expression* mask =
      file->BitwiseNot(file->Shll(file->Literal(all_ones), start));
  Expression* updated_value =
      file->BitwiseOr(file->Shll(adjusted_update_value, start),
                      file->BitwiseAnd(mask, to_update));

  if (Bits::MinBitCountUnsigned(to_update_width) > start_width) {
    // Start value is not wide enough to encode the width of the value to
    // update. No need to protect against overshifting.
    func->AddStatement<BlockingAssignment>(func->return_value_ref(),
                                           updated_value);
  } else {
    // Start value is wide enough to encode the width of the value to
    // update. Protect against overshifting by selecting the unchanged value to
    // update if start is greater than or equal to width.
    func->AddStatement<BlockingAssignment>(
        func->return_value_ref(),
        file->Ternary(
            file->GreaterThanEquals(
                start, file->Literal(UBits(to_update_width, start_width))),
            to_update, updated_value));
  }
  return func;
}

// Defines and returns a function which implements the given SMul node.
VerilogFunction* DefineSmulFunction(Node* node, absl::string_view function_name,
                                    ModuleSection* section) {
  XLS_CHECK_EQ(node->op(), Op::kSMul);
  VerilogFile* file = section->file();

  ScopedLintDisable lint_disable(section, {Lint::kSignedType, Lint::kMultiply});

  VerilogFunction* func = section->Add<VerilogFunction>(
      function_name, file->BitVectorType(node->BitCountOrDie()));
  XLS_CHECK_EQ(node->operand_count(), 2);
  Expression* lhs = func->AddArgument(
      "lhs", file->BitVectorType(node->operand(0)->BitCountOrDie()));
  Expression* rhs = func->AddArgument(
      "rhs", file->BitVectorType(node->operand(1)->BitCountOrDie()));
  // The code conservatively assigns signed-casted inputs to temporary
  // variables, uses them in the multiply expression which is assigned to
  // another signed temporary. Finally, this is unsign-casted and assigned to
  // the return value of the function. These shenanigans ensure no surprising
  // sign/zero extensions of any values.
  LogicRef* signed_lhs =
      func->AddRegDef("signed_lhs",
                      file->BitVectorType(node->operand(0)->BitCountOrDie(),
                                          /*is_signed=*/true),
                      /*init=*/nullptr);
  LogicRef* signed_rhs =
      func->AddRegDef("signed_rhs",
                      file->BitVectorType(node->operand(1)->BitCountOrDie(),
                                          /*is_signed=*/true),
                      /*init=*/nullptr);
  LogicRef* signed_result =
      func->AddRegDef("signed_result",
                      file->BitVectorType(node->BitCountOrDie(),
                                          /*is_signed=*/true),
                      /*init=*/nullptr);
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

  VerilogFunction* func = section->Add<VerilogFunction>(
      function_name, file->BitVectorType(node->BitCountOrDie()));
  XLS_CHECK_EQ(node->operand_count(), 2);
  Expression* lhs = func->AddArgument(
      "lhs", file->BitVectorType(node->operand(0)->BitCountOrDie()));
  Expression* rhs = func->AddArgument(
      "rhs", file->BitVectorType(node->operand(1)->BitCountOrDie()));
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
      func = DefineDynamicBitSliceFunction(node->As<DynamicBitSlice>(),
                                           function_name, functions_section_);
      break;
    case Op::kBitSliceUpdate:
      func = DefineBitSliceUpdateFunction(node->As<BitSliceUpdate>(),
                                          function_name, functions_section_);
      break;
    default:
      XLS_LOG(FATAL) << "Cannot define node as function: " << node->ToString();
  }
  node_functions_[function_name] = func;
  return func;
}

}  // namespace verilog
}  // namespace xls
