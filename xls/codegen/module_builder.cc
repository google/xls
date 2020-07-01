// Copyright 2020 Google LLC
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
xabsl::StatusOr<Expression*> FlattenValueToExpression(const Value& value,
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
xabsl::StatusOr<ArrayAssignmentPattern*> ValueToArrayAssignmentPattern(
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
    Expression* lhs, Expression* rhs, Type* xls_type,
    std::function<void(Expression*, Expression*)> add_assignment_statement) {
  // Array assignment is only supported in SystemVerilog. In Verilog, arrays
  // must be assigned element-by-element.
  if (!use_system_verilog_ && xls_type != nullptr && xls_type->IsArray()) {
    ArrayType* array_type = xls_type->AsArrayOrDie();
    for (int64 i = 0; i < array_type->size(); ++i) {
      XLS_RETURN_IF_ERROR(
          AddAssignment(file_->Index(lhs->AsIndexableExpressionOrDie(), i),
                        file_->Index(rhs->AsIndexableExpressionOrDie(), i),
                        array_type->element_type(), add_assignment_statement));
    }
  } else {
    add_assignment_statement(lhs, rhs);
  }
  return absl::OkStatus();
}

absl::Status ModuleBuilder::AddAssignmentFromValue(
    Expression* lhs, const Value& value,
    std::function<void(Expression*, Expression*)> add_assignment_statement) {
  if (value.IsArray()) {
    if (use_system_verilog_) {
      // If using system verilog emit using an array assignment pattern like so:
      //   logic [4:0] foo [0:4][0:1] = '{'{5'h0, 5'h1}, '{..}, ...}
      XLS_ASSIGN_OR_RETURN(Expression * rhs,
                           ValueToArrayAssignmentPattern(value, file_));
      add_assignment_statement(lhs, rhs);
    } else {
      for (int64 i = 0; i < value.size(); ++i) {
        XLS_RETURN_IF_ERROR(AddAssignmentFromValue(
            file_->Index(lhs->AsIndexableExpressionOrDie(), i),
            value.element(i), add_assignment_statement));
      }
    }
  } else {
    XLS_ASSIGN_OR_RETURN(Expression * flattened_expr,
                         FlattenValueToExpression(value, file_));
    add_assignment_statement(lhs, flattened_expr);
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
    std::function<void(Expression*, Expression*)> add_assignment_statement) {
  if (xls_type->IsArray()) {
    ArrayType* array_type = xls_type->AsArrayOrDie();
    for (int64 i = 0; i < array_type->size(); ++i) {
      XLS_RETURN_IF_ERROR(
          AssignFromSlice(file_->Index(lhs->AsIndexableExpressionOrDie(), i),
                          rhs, array_type->element_type(),
                          slice_start + GetFlatBitIndexOfElement(array_type, i),
                          add_assignment_statement));
    }
  } else {
    add_assignment_statement(
        lhs, file_->Slice(rhs->AsIndexableExpressionOrDie(),
                          /*hi=*/slice_start + xls_type->GetFlatBitCount() - 1,
                          /*lo=*/slice_start));
  }
  return absl::OkStatus();
}

xabsl::StatusOr<LogicRef*> ModuleBuilder::AddInputPort(absl::string_view name,
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

xabsl::StatusOr<LogicRef*> ModuleBuilder::DeclareModuleConstant(
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

xabsl::StatusOr<Expression*> ModuleBuilder::EmitAsInlineExpression(
    Node* node, absl::Span<Expression* const> inputs) {
  if (MustEmitAsFunction(node)) {
    XLS_ASSIGN_OR_RETURN(VerilogFunction * func, DefineFunction(node));
    return file_->Make<VerilogFunctionCall>(func, inputs);
  }
  return NodeToExpression(node, inputs, file_);
}

absl::Status ModuleBuilder::EmitArrayUpdateElement(Expression* lhs,
                                                   BinaryInfix* condition,
                                                   Expression* new_value,
                                                   Expression* original_value,
                                                   Type* element_type) {
  if (element_type->IsArray()) {
    ArrayType* array_type = element_type->AsArrayOrDie();
    for (int64 idx = 0; idx < array_type->size(); ++idx) {
      XLS_RETURN_IF_ERROR(EmitArrayUpdateElement(
          /*lhs=*/file_->Index(lhs->AsIndexableExpressionOrDie(), idx),
          /*condition=*/condition,
          /*new_value=*/
          file_->Index(new_value->AsIndexableExpressionOrDie(), idx),
          /*original_value=*/
          file_->Index(original_value->AsIndexableExpressionOrDie(), idx),
          /*element_type=*/array_type->element_type()));
    }
    return absl::OkStatus();
  }

  if (!element_type->IsBits() && !element_type->IsTuple()) {
    return absl::UnimplementedError(absl::StrFormat(
        "EmitArrayUpdateHelper cannot handle elements of type %s",
        element_type->ToString()));
  }

  XLS_RETURN_IF_ERROR(AddAssignment(
      /*lhs=*/lhs,
      /*rhs=*/
      file_->Ternary(condition, new_value, original_value),
      /*xls_type=*/element_type,
      /*add_assignment_statement=*/
      [&](Expression* lhs, Expression* rhs) {
        assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
      }));

  return absl::OkStatus();
}

xabsl::StatusOr<LogicRef*> ModuleBuilder::EmitAsAssignment(
    absl::string_view name, Node* node, absl::Span<Expression* const> inputs) {
  LogicRef* ref = DeclareVariable(name, node->GetType());
  if (node->GetType()->IsArray()) {
    // Array-shaped operations are handled specially. XLS arrays are represented
    // as unpacked arrays in Verilog/SystemVerilog and unpacked arrays must be
    // assigned element-by-element in Verilog.
    ArrayType* array_type = node->GetType()->AsArrayOrDie();
    switch (node->op()) {
      case Op::kArray: {
        for (int64 i = 0; i < inputs.size(); ++i) {
          XLS_RETURN_IF_ERROR(AddAssignment(
              file_->Index(ref, file_->PlainLiteral(i)), inputs[i],
              array_type->element_type(),
              [&](Expression* lhs, Expression* rhs) {
                assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
              }));
        }
        break;
      }
      case Op::kArrayIndex:
        XLS_RETURN_IF_ERROR(AddAssignment(
            ref,
            file_->Index(inputs[0]->AsIndexableExpressionOrDie(), inputs[1]),
            array_type, [&](Expression* lhs, Expression* rhs) {
              assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
            }));
        break;
      case Op::kArrayUpdate:
        for (int64 i = 0; i < array_type->size(); ++i) {
          XLS_RETURN_IF_ERROR(EmitArrayUpdateElement(
              /*lhs=*/file_->Index(ref, i),
              /*condition=*/file_->Equals(inputs[1], file_->PlainLiteral(i)),
              /*new_value=*/inputs[2],
              /*original_value=*/
              file_->Index(inputs[0]->AsIndexableExpressionOrDie(), i),
              /*element_type=*/array_type->element_type()));
        }
        break;
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
      AddAssignment(lhs, rhs, type, [&](Expression* lhs, Expression* rhs) {
        assignment_section()->Add<ContinuousAssignment>(lhs, rhs);
      }));
  return absl::OkStatus();
}

xabsl::StatusOr<ModuleBuilder::Register> ModuleBuilder::DeclareRegister(
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

xabsl::StatusOr<ModuleBuilder::Register> ModuleBuilder::DeclareRegister(
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
          reg.ref, reg.reset_value, reg.xls_type,
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
        reg.ref, reg.next, reg.xls_type, [&](Expression* lhs, Expression* rhs) {
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
    default:
      XLS_LOG(FATAL) << "Cannot emit node as function: " << node->ToString();
  }
}

namespace {

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

xabsl::StatusOr<VerilogFunction*> ModuleBuilder::DefineFunction(Node* node) {
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
    default:
      XLS_LOG(FATAL) << "Cannot define node as function: " << node->ToString();
  }
  node_functions_[function_name] = func;
  return func;
}

}  // namespace verilog
}  // namespace xls
