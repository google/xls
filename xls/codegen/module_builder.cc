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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/lint_annotate.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/node_expressions.h"
#include "xls/codegen/node_representation.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_function.h"
#include "xls/passes/bdd_query_engine.h"

namespace xls {
namespace verilog {

namespace {

// Returns the name and polarity (ifdef/ifndef) of the macro used to guard
// simulation-only constructs.
std::pair<std::string, ConditionalDirectiveKind> SimulationMacroNameAndPolarity(
    const CodegenOptions& options) {
  if (absl::StartsWith(options.simulation_macro_name(), "!")) {
    return {
        std::string{absl::StripPrefix(options.simulation_macro_name(), "!")},
        ConditionalDirectiveKind::kIfndef};
  }
  return {options.simulation_macro_name(), ConditionalDirectiveKind::kIfdef};
}

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
    return file->Literal(value.bits(), SourceInfo());
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
  if (elements.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Empty expression for value %s.", value.ToString()));
  }
  if (elements.size() == 1) {
    return elements[0];
  }
  return file->Concat(elements, SourceInfo());
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
  return file->Make<ArrayAssignmentPattern>(SourceInfo(), pieces);
}

// Defines and returns a function which implements the given PrioritySelect
// node.
absl::StatusOr<VerilogFunction*> DefinePrioritySelectFunction(
    Node* selector, Type* tpe, int64_t num_cases, const SourceInfo& loc,
    std::string_view function_name, ModuleSection* section,
    SelectorProperties selector_properties, const CodegenOptions& options) {
  VerilogFile* file = section->file();

  VerilogFunction* func = section->Add<VerilogFunction>(
      loc, function_name, file->BitVectorType(tpe->GetFlatBitCount(), loc));
  Expression* selector_expression = func->AddArgument(
      "sel", file->BitVectorType(selector->BitCountOrDie(), loc), loc);

  std::vector<Expression*> cases;
  cases.reserve(num_cases);
  for (size_t i = 0; i < num_cases; ++i) {
    cases.push_back(func->AddArgument(
        absl::StrCat("case", i),
        file->BitVectorType(tpe->GetFlatBitCount(), loc), loc));
  }

  Expression* default_value = func->AddArgument(
      "default_value", file->BitVectorType(tpe->GetFlatBitCount(), loc), loc);

  CaseType case_type(selector->BitCountOrDie() > 1 ? CaseKeyword::kCasez
                                                   : CaseKeyword::kCase);
  const FourValueBit case_label_top_bits = FourValueBit::kHighZ;
  FourValueBit case_label_bottom_bits = FourValueBit::kZero;
  if (options.use_system_verilog()) {
    // If using system verilog, always use the "unique" keyword.
    case_type.modifier = CaseModifier::kUnique;
  }

  Case* case_statement =
      func->AddStatement<Case>(loc, selector_expression, case_type);
  // Make a ternary vector that looks like ???...?1000...0.
  // Each label will be a sliding window into this larger vector.
  // If the selector is known to be one hot, the window will look like
  // 000...01000...0 instead and the "unique" keyword will be used instead.
  std::vector<FourValueBit> ternary_vector;
  ternary_vector.reserve((cases.size() * 2) - 1);
  std::fill_n(std::back_inserter(ternary_vector), cases.size() - 1,
              case_label_top_bits);
  ternary_vector.push_back(FourValueBit::kOne);
  std::fill_n(std::back_inserter(ternary_vector), cases.size() - 1,
              case_label_bottom_bits);
  absl::Span<FourValueBit const> ternary_span = ternary_vector;

  for (size_t i = 0; i < cases.size(); ++i) {
    absl::StatusOr<Expression*> label_expression =
        file->Make<FourValueBinaryLiteral>(
            loc, ternary_span.subspan(i, cases.size()));
    CHECK_OK(label_expression.status());
    StatementBlock* block =
        case_statement->AddCaseArm(label_expression.value());
    block->Add<BlockingAssignment>(loc, func->return_value_ref(), cases[i]);
  }
  Expression* zero_label = file->Literal(0, selector->BitCountOrDie(), loc,
                                         FormatPreference::kBinary);
  Expression* x_literal;
  if (options.use_system_verilog()) {
    // Use 'X when generating SystemVerilog.
    x_literal = file->Make<XLiteral>(loc);
  } else {
    // Verilog doesn't support 'X, so use the less desirable 16'dxxxx format.
    x_literal = file->Make<XSentinel>(loc, tpe->GetFlatBitCount());
  }
  if (selector_properties.never_zero) {
    // If the selector cannot be zero, throw an error if we see zero.
    // We still explicitly propagate X (like the default case below) for
    // synthesis.
    // TODO: github/xls#1481 - this should really be an assert (or assume)
    // predicated on selector being valid, but it's a bit tricky to do. For now,
    // it seems better to have an overactive error rather than silently
    // propagate X.
    Expression* error_message =
        file->Make<QuotedString>(loc, "Zero selector not allowed.");
    StatementBlock* case_block = case_statement->AddCaseArm(zero_label);
    auto [macro_name, polarity] = SimulationMacroNameAndPolarity(options);
    MacroStatementBlock* ifdef_block =
        case_block
            ->Add<StatementConditionalDirective>(loc, polarity, macro_name)
            ->consequent();
    ifdef_block->Add<SystemTaskCall>(
        loc, "error", std::initializer_list<Expression*>{error_message});
    case_block->Add<Comment>(loc, "Never taken, propagate X");
    case_block->Add<BlockingAssignment>(loc, func->return_value_ref(),
                                        x_literal);
  } else {
    case_statement->AddCaseArm(zero_label)
        ->Add<BlockingAssignment>(loc, func->return_value_ref(), default_value);
  }

  // Add a default case that propagates X.
  StatementBlock* case_block = case_statement->AddCaseArm(DefaultSentinel());
  case_block->Add<Comment>(loc, "Propagate X");
  case_block->Add<BlockingAssignment>(loc, func->return_value_ref(), x_literal);

  return func;
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
      !(options_.use_system_verilog() && sv_array_expr)) {
    ArrayType* array_type = xls_type->AsArrayOrDie();
    for (int64_t i = 0; i < array_type->size(); ++i) {
      std::vector<Expression*> input_elements;
      for (Expression* input : inputs) {
        input_elements.push_back(
            file_->Index(input->AsIndexableExpressionOrDie(), i, SourceInfo()));
      }
      XLS_RETURN_IF_ERROR(AddAssignmentToGeneratedExpression(
          array_type->element_type(),
          file_->Index(lhs->AsIndexableExpressionOrDie(), i, SourceInfo()),
          input_elements, gen_rhs_expr, add_assignment, sv_array_expr));
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
    if (options_.use_system_verilog()) {
      // If using system verilog emit using an array assignment pattern like so:
      //   logic [4:0] foo [0:4][0:1] = '{'{5'h0, 5'h1}, '{..}, ...}
      XLS_ASSIGN_OR_RETURN(Expression * rhs,
                           ValueToArrayAssignmentPattern(value, file_));
      add_assignment(lhs, rhs);
    } else {
      for (int64_t i = 0; i < value.size(); ++i) {
        XLS_RETURN_IF_ERROR(AddAssignmentFromValue(
            file_->Index(lhs->AsIndexableExpressionOrDie(), i, SourceInfo()),
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

ModuleBuilder::ModuleBuilder(std::string_view name, VerilogFile* file,
                             CodegenOptions options,
                             std::optional<std::string_view> clk_name,
                             std::optional<ResetProto> rst_proto)
    : module_name_(SanitizeIdentifier(name)),
      file_(file),
      package_("__ModuleBuilder_type_generator"),
      options_(std::move(options)),
      query_engine_(std::nullopt) {
  module_ = file_->AddModule(module_name_, SourceInfo());
  functions_section_ = module_->Add<ModuleSection>(SourceInfo());
  constants_section_ = module_->Add<ModuleSection>(SourceInfo());
  input_section_ = module_->Add<ModuleSection>(SourceInfo());
  declaration_and_assignment_section_ =
      module_->Add<ModuleSection>(SourceInfo());
  instantiation_section_ = module_->Add<ModuleSection>(SourceInfo());
  assert_section_ = module_->Add<ModuleSection>(SourceInfo());
  cover_section_ = module_->Add<ModuleSection>(SourceInfo());
  output_section_ = module_->Add<ModuleSection>(SourceInfo());
  trace_section_ = module_->Add<ModuleSection>(SourceInfo());

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
      declaration_and_assignment_section_->Add<ModuleSection>(SourceInfo()));
  assignment_subsections_.push_back(
      declaration_and_assignment_section_->Add<ModuleSection>(SourceInfo()));
}

absl::Status ModuleBuilder::AssignFromSlice(
    Expression* lhs, Expression* rhs, Type* xls_type, int64_t slice_start,
    std::function<void(Expression*, Expression*)> add_assignment) {
  if (xls_type->IsArray()) {
    ArrayType* array_type = xls_type->AsArrayOrDie();
    for (int64_t i = 0; i < array_type->size(); ++i) {
      XLS_RETURN_IF_ERROR(AssignFromSlice(
          file_->Index(lhs->AsIndexableExpressionOrDie(), i, SourceInfo()), rhs,
          array_type->element_type(),
          slice_start + GetFlatBitIndexOfElement(array_type, i),
          add_assignment));
    }
  } else {
    add_assignment(
        lhs, file_->Slice(rhs->AsIndexableExpressionOrDie(),
                          /*hi=*/slice_start + xls_type->GetFlatBitCount() - 1,
                          /*lo=*/slice_start, SourceInfo()));
  }
  return absl::OkStatus();
}

absl::StatusOr<LogicRef*> ModuleBuilder::AddInputPort(
    std::string_view name, Type* type,
    std::optional<std::string_view> sv_type) {
  LogicRef* port =
      AddInputPort(SanitizeIdentifier(name), type->GetFlatBitCount(), sv_type);
  if (!type->IsArray()) {
    return port;
  }
  // All inputs are flattened so unflatten arrays with a sequence of
  // assignments.
  ArrayType* array_type = type->AsArrayOrDie();
  LogicRef* ar = module_->AddWire(
      absl::StrCat(SanitizeIdentifier(name), "_unflattened"),
      file_->UnpackedArrayType(NestedElementWidth(array_type),
                               NestedArrayBounds(array_type), SourceInfo()),
      SourceInfo(), input_section());
  XLS_RETURN_IF_ERROR(AssignFromSlice(
      ar, port, type->AsArrayOrDie(), 0, [&](Expression* lhs, Expression* rhs) {
        input_section()->Add<ContinuousAssignment>(SourceInfo(), lhs, rhs);
      }));
  return ar;
}

LogicRef* ModuleBuilder::AddInputPort(std::string_view name, int64_t bit_count,
                                      std::optional<std::string_view> sv_type) {
  auto* raw_bits_type = file_->BitVectorType(bit_count, SourceInfo());
  if (sv_type && options_.emit_sv_types()) {
    return module_->AddInput(
        SanitizeIdentifier(name),
        file_->ExternType(raw_bits_type, *sv_type, SourceInfo()), SourceInfo());
  }
  return module_->AddInput(SanitizeIdentifier(name), raw_bits_type,
                           SourceInfo());
}

absl::Status ModuleBuilder::AddOutputPort(
    std::string_view name, Type* type, Expression* value,
    std::optional<std::string_view> sv_type) {
  LogicRef* output_port;
  DataType* bits_type =
      file_->BitVectorType(type->GetFlatBitCount(), SourceInfo());
  if (sv_type && options_.emit_sv_types()) {
    output_port = module_->AddOutput(
        SanitizeIdentifier(name),
        file_->ExternType(bits_type, *sv_type, SourceInfo()), SourceInfo());
  } else {
    output_port =
        module_->AddOutput(SanitizeIdentifier(name), bits_type, SourceInfo());
  }

  if (type->IsArray()) {
    // The output is flattened so flatten arrays with a sequence of assignments.
    XLS_RET_CHECK(value->IsIndexableExpression());
    output_section()->Add<ContinuousAssignment>(
        SourceInfo(), output_port,
        FlattenArray(value->AsIndexableExpressionOrDie(), type->AsArrayOrDie(),
                     file_, SourceInfo()));
  } else {
    output_section()->Add<ContinuousAssignment>(SourceInfo(), output_port,
                                                value);
  }
  return absl::OkStatus();
}

absl::Status ModuleBuilder::AddOutputPort(
    std::string_view name, int64_t bit_count, Expression* value,
    std::optional<std::string_view> sv_type) {
  LogicRef* output_port;
  DataType* bits_type = file_->BitVectorType(bit_count, SourceInfo());
  if (sv_type && options_.emit_sv_types()) {
    output_port = module_->AddOutput(
        SanitizeIdentifier(name),
        file_->ExternType(bits_type, *sv_type, SourceInfo()), SourceInfo());
  } else {
    output_port =
        module_->AddOutput(SanitizeIdentifier(name), bits_type, SourceInfo());
  }
  output_section()->Add<ContinuousAssignment>(SourceInfo(), output_port, value);
  return absl::OkStatus();
}

absl::StatusOr<LogicRef*> ModuleBuilder::DeclareModuleConstant(
    std::string_view name, const Value& value) {
  Type* type = package_.GetTypeForValue(value);
  DataType* data_type;
  if (type->IsArray()) {
    ArrayType* array_type = type->AsArrayOrDie();
    data_type =
        file_->UnpackedArrayType(NestedElementWidth(array_type),
                                 NestedArrayBounds(array_type), SourceInfo());
  } else {
    data_type = file_->BitVectorType(type->GetFlatBitCount(), SourceInfo());
  }
  // Verilator does not like declaration of arrays and assignments in the same
  // line so declare and assign separately.
  if (!value.IsArray()) {
    XLS_ASSIGN_OR_RETURN(Expression * rhs,
                         FlattenValueToExpression(value, file_));
    // Add wire with init.
    return module_->AddWire(SanitizeIdentifier(name), data_type, rhs,
                            SourceInfo(), constants_section());
  }
  LogicRef* ref = module_->AddWire(SanitizeIdentifier(name), data_type,
                                   SourceInfo(), constants_section());
  XLS_RETURN_IF_ERROR(
      AddAssignmentFromValue(ref, value, [&](Expression* lhs, Expression* rhs) {
        constants_section()->Add<ContinuousAssignment>(SourceInfo(), lhs, rhs);
      }));
  return ref;
}

LogicRef* ModuleBuilder::DeclareVariable(std::string_view name, Type* type) {
  DataType* data_type;
  if (type->IsArray()) {
    ArrayType* array_type = type->AsArrayOrDie();
    data_type =
        file_->UnpackedArrayType(NestedElementWidth(array_type),
                                 NestedArrayBounds(array_type), SourceInfo());
  } else {
    data_type = file_->BitVectorType(type->GetFlatBitCount(), SourceInfo());
  }
  return module_->AddWire(SanitizeIdentifier(name), data_type, SourceInfo(),
                          declaration_section());
}

LogicRef* ModuleBuilder::DeclareVariable(std::string_view name,
                                         int64_t bit_count) {
  return module_->AddWire(SanitizeIdentifier(name),
                          file_->BitVectorType(bit_count, SourceInfo()),
                          SourceInfo(), declaration_section());
}

bool ModuleBuilder::CanEmitAsInlineExpression(
    Node* node, std::optional<absl::Span<Node* const>> users_of_expression) {
  if (node->GetType()->IsArray()) {
    // TODO(meheff): With system verilog we can do array assignment.
    return false;
  }

  if (node->Is<Gate>()) {
    // Gate instructions may be emitted using a format string and should not be
    // emitted inline in a larger expression.
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
    case Op::kSMulp:
    case Op::kUMulp:
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
    return file_->Make<VerilogFunctionCall>(node->loc(), func, inputs);
  }
  return NodeToExpression(node, inputs, file_, options_);
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
    return std::holds_alternative<bool>(im) && std::get<bool>(im);
  };
  auto is_statically_false = [](const IndexMatch& im) {
    return std::holds_alternative<bool>(im) && !std::get<bool>(im);
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
    return file_->LogicalAnd(std::get<Expression*>(a), std::get<Expression*>(b),
                             SourceInfo());
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
            assignment_section()->Add<ContinuousAssignment>(SourceInfo(), lhs,
                                                            rhs);
          });
    }
    if (is_statically_false(index_match)) {
      // Indices definitely do *NOT* match the subarray/element being replaced
      // with update value. Assign from rhs exclusively. E.g.:
      //   assign lhs[i][j] = rhs[j]
      return AddAssignment(
          /*xls_type=*/xls_type,
          /*lhs=*/lhs,
          /*rhs=*/rhs,
          /*add_assignment=*/
          [&](Expression* lhs, Expression* rhs) {
            assignment_section()->Add<ContinuousAssignment>(SourceInfo(), lhs,
                                                            rhs);
          });
    } else {
      // Indices may or may not match the subarray/element being replaced with
      // update value. Use a ternary expression to pick from rhs or update
      // value. E.g:
      //   assign lhs[i][j] = (i == idx) ? update_value[j] : rhs[j]
      auto gen_ternary = [&](absl::Span<Expression* const> inputs) {
        return file_->Ternary(std::get<Expression*>(index_match), inputs[0],
                              inputs[1], SourceInfo());
      };

      // Emit a continuous assignment with a ternary select. The ternary
      // operation supports array types in SystemVerilog so sv_array_expr is
      // true.
      return AddAssignmentToGeneratedExpression(
          xls_type, lhs, /*inputs=*/{update_value, rhs}, gen_ternary,
          /*add_assignment=*/
          [&](Expression* lhs, Expression* rhs) {
            assignment_section()->Add<ContinuousAssignment>(SourceInfo(), lhs,
                                                            rhs);
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
      current_index_match =
          file_->Equals(index_type.expression,
                        file_->Literal(UBits(i, index_bit_count), SourceInfo()),
                        SourceInfo());
    }
    XLS_RETURN_IF_ERROR(EmitArrayCopyAndUpdate(
        file_->Index(lhs, i, SourceInfo()), file_->Index(rhs, i, SourceInfo()),
        update_value, indices.subspan(1),
        combine_index_matches(current_index_match, index_match),
        array_type->element_type()));
  }

  return absl::OkStatus();
}

absl::StatusOr<SelectorProperties> ModuleBuilder::GetSelectorProperties(
    Node* selector) {
  XLS_RET_CHECK(selector->GetType()->IsBits());
  bool never_zero = false;
  if (!query_engine_.has_value()) {
    query_engine_ = BddQueryEngine(BddFunction::kDefaultPathLimit);
    XLS_RETURN_IF_ERROR(
        query_engine_->Populate(selector->function_base()).status());
  }
  if (query_engine_.has_value()) {
    never_zero = query_engine_->AtLeastOneBitTrue(selector);
  }
  return SelectorProperties{.never_zero = never_zero};
}

absl::StatusOr<LogicRef*> ModuleBuilder::EmitAsAssignment(
    std::string_view name, Node* node, absl::Span<Expression* const> inputs) {
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
              file_->Index(ref, file_->PlainLiteral(i, node->loc()),
                           node->loc()),
              inputs[i], [&](Expression* lhs, Expression* rhs) {
                assignment_section()->Add<ContinuousAssignment>(node->loc(),
                                                                lhs, rhs);
              }));
        }
        break;
      }
      case Op::kArrayIndex: {
        XLS_ASSIGN_OR_RETURN(
            IndexableExpression * rhs,
            ArrayIndexExpression(inputs[0]->AsIndexableExpressionOrDie(),
                                 inputs.subspan(1), node->As<ArrayIndex>(),
                                 options_));
        XLS_RETURN_IF_ERROR(AddAssignment(
            array_type, ref, rhs, [&](Expression* lhs, Expression* rhs) {
              assignment_section()->Add<ContinuousAssignment>(node->loc(), lhs,
                                                              rhs);
            }));
        break;
      }
      case Op::kArraySlice: {
        ArraySlice* slice = node->As<ArraySlice>();
        IndexableExpression* input_array =
            inputs[0]->AsIndexableExpressionOrDie();
        int64_t input_array_size =
            slice->array()->GetType()->AsArrayOrDie()->size();
        Type* array_element_type =
            slice->array()->GetType()->AsArrayOrDie()->element_type();

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
              {file_->Literal(0, min_index_width - start_width, node->loc()),
               start_expr},
              node->loc());
          start_width = min_index_width;
        }

        Expression* max_index_expr =
            file_->Literal(input_array_size - 1, start_width, node->loc());
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
            element = file_->Index(input_array, max_index_expr, node->loc());
          } else {
            // Index might be out of bounds.
            Expression* oob_condition =
                file_->GreaterThan(start_expr,
                                   file_->Literal(input_array_size - 1 - i,
                                                  start_width, node->loc()),
                                   node->loc());
            element = file_->Index(
                input_array,
                file_->Ternary(
                    oob_condition, max_index_expr,
                    file_->Add(start_expr,
                               file_->Literal(i, start_width, node->loc()),
                               node->loc()),
                    node->loc()),
                node->loc());
          }
          XLS_RETURN_IF_ERROR(AddAssignment(
              array_element_type,
              file_->Index(ref, file_->PlainLiteral(i, node->loc()),
                           node->loc()),
              element, [&](Expression* lhs, Expression* rhs) {
                assignment_section()->Add<ContinuousAssignment>(node->loc(),
                                                                lhs, rhs);
              }));
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
                file_->Index(ref,
                             file_->PlainLiteral(result_index, node->loc()),
                             node->loc()),
                file_->Index(input->AsIndexableExpressionOrDie(),
                             file_->PlainLiteral(j, node->loc()), node->loc()),
                [&](Expression* lhs, Expression* rhs) {
                  assignment_section()->Add<ContinuousAssignment>(node->loc(),
                                                                  lhs, rhs);
                }));

            ++result_index;
          }
        }
        break;
      }
      case Op::kTupleIndex:
        XLS_RETURN_IF_ERROR(
            AssignFromSlice(ref, inputs[0], array_type,
                            GetFlatBitIndexOfElement(
                                node->operand(0)->GetType()->AsTupleOrDie(),
                                node->As<TupleIndex>()->index()),
                            [&](Expression* lhs, Expression* rhs) {
                              assignment_section()->Add<ContinuousAssignment>(
                                  node->loc(), lhs, rhs);
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
                          /*bit_count=*/sel->selector()->BitCountOrDie(),
                          node->loc()),
                      node->loc()),
                  cases[i], result, node->loc());
            }
          }
          return result;
        };
        XLS_RETURN_IF_ERROR(AddAssignmentToGeneratedExpression(
            array_type, /*lhs=*/ref, /*inputs=*/cases,
            /*gen_rhs_expr=*/select_element,
            /*add_assignment=*/
            [&](Expression* lhs, Expression* rhs) {
              assignment_section()->Add<ContinuousAssignment>(node->loc(), lhs,
                                                              rhs);
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
                    ? file_->BitwiseAnd(inputs[i],
                                        file_->Index(selector, i, node->loc()),
                                        node->loc())
                    : file_->BitwiseAnd(
                          inputs[i],
                          file_->Concat(
                              /*replication=*/element_width,
                              {file_->Index(selector, i, node->loc())},
                              node->loc()),
                          node->loc());
            result = result == nullptr
                         ? masked_input
                         : file_->BitwiseOr(result, masked_input, node->loc());
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
              assignment_section()->Add<ContinuousAssignment>(node->loc(), lhs,
                                                              rhs);
            },
            /*sv_array_expr=*/false));
        break;
      }
      case Op::kPrioritySel: {
        Expression* selector_expression = inputs[0];
        // Determine the element type of the potentially-multidimensional
        // array. This is the type of the inputs passed into the expression
        // generator ohs_element.
        Type* element_type = array_type->element_type();
        while (element_type->IsArray()) {
          element_type = element_type->AsArrayOrDie()->element_type();
        }
        absl::Span<Expression* const> cases_and_default = inputs.subspan(1);
        if (cases_and_default.size() == 2) {
          // Selects an element from the set of cases 'inputs' according to the
          // semantics of the select instruction. 'inputs' is the set of all
          // cases including the optional default case which appears last.
          auto priority_sel_element =
              [&](absl::Span<Expression* const> inputs) {
                Expression* selected_expr = inputs[0];
                Expression* default_expr = inputs[1];
                return file_->Ternary(selector_expression, selected_expr,
                                      default_expr, node->loc());
              };
          XLS_RETURN_IF_ERROR(AddAssignmentToGeneratedExpression(
              array_type, /*lhs=*/ref, /*inputs=*/cases_and_default,
              /*gen_rhs_expr=*/priority_sel_element,
              /*add_assignment=*/
              [&](Expression* lhs, Expression* rhs) {
                assignment_section()->Add<ContinuousAssignment>(node->loc(),
                                                                lhs, rhs);
              },
              /*sv_array_expr=*/true));
          break;
        }
        XLS_ASSIGN_OR_RETURN(std::string function_name,
                             VerilogFunctionName(node));
        absl::StrAppend(&function_name, "_element",
                        element_type->GetFlatBitCount());
        VerilogFunction* func;
        if (node_functions_.contains(function_name)) {
          func = node_functions_.at(function_name);
        } else {
          XLS_ASSIGN_OR_RETURN(
              SelectorProperties selector_properties,
              GetSelectorProperties(node->As<PrioritySelect>()->selector()));
          XLS_ASSIGN_OR_RETURN(
              func,
              DefinePrioritySelectFunction(
                  node->As<PrioritySelect>()->selector(), /*tpe=*/element_type,
                  /*num_cases=*/node->As<PrioritySelect>()->cases().size(),
                  /*loc=*/node->loc(), function_name, functions_section_,
                  selector_properties, options_));
          node_functions_[function_name] = func;
        }
        auto priority_sel_element = [&](absl::Span<Expression* const> inputs) {
          std::vector<Expression*> selector_and_inputs{selector_expression};
          selector_and_inputs.insert(selector_and_inputs.end(), inputs.begin(),
                                     inputs.end());
          // We emit priority selects as function invocations.
          return file_->Make<VerilogFunctionCall>(node->loc(), func,
                                                  selector_and_inputs);
        };
        XLS_RETURN_IF_ERROR(AddAssignmentToGeneratedExpression(
            array_type, /*lhs=*/ref, /*inputs=*/cases_and_default,
            priority_sel_element,
            /*add_assignment=*/
            [&](Expression* lhs, Expression* rhs) {
              assignment_section()->Add<ContinuousAssignment>(node->loc(), lhs,
                                                              rhs);
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

absl::StatusOr<NodeRepresentation> ModuleBuilder::EmitAssert(
    xls::Assert* asrt, Expression* condition) {
  if (!options_.use_system_verilog()) {
    // Asserts are a SystemVerilog only feature.
    // TODO(meheff): 2021/02/27 We should raise an error here or possibly emit a
    // construct like: if (!condition) $display("Assert failed ...");
    LOG(WARNING) << "Asserts are only supported in SystemVerilog.";
    return UnrepresentedSentinel();
  }

  if (asrt->label().has_value()) {
    if (asrt->label().value() != SanitizeIdentifier(asrt->label().value())) {
      return absl::InvalidArgumentError(
          "Assert label must be a valid SystemVerilog identifier.");
    }
  }

  Expression* disable_iff;
  if (reset().has_value()) {
    // Disable the assert when in reset.
    disable_iff = reset()->active_low
                      ? static_cast<Expression*>(
                            file_->LogicalNot(reset()->signal, asrt->loc()))
                      : static_cast<Expression*>(reset()->signal);
  } else {
    // As a backup if no reset is available, disable the assert if the signal is
    // unknown. This is not great but avoids unconditionally triggering the
    // assert at the start of simulation.
    disable_iff = file_->Make<SystemFunctionCall>(
        asrt->loc(), "isunknown", std::vector<Expression*>({condition}));
  }

  if (clock() == nullptr) {
    return assert_section()->Add<DeferredImmediateAssertion>(
        asrt->loc(), condition, disable_iff,
        asrt->label().has_value() ? asrt->label().value() : "",
        asrt->message());
  }

  // Sample the disable_iff signal under the following conditions
  //  1. reset exists and is synchronous.
  //  2. reset does not exist and a clock exists.
  if (!reset().has_value() || !reset()->asynchronous) {
    disable_iff = file_->Make<SystemFunctionCall>(
        asrt->loc(), "sampled", std::vector<Expression*>({disable_iff}));
  }

  return assert_section()->Add<ConcurrentAssertion>(
      asrt->loc(), condition,
      /*clocking_event=*/file_->Make<PosEdge>(asrt->loc(), clock()),
      disable_iff, asrt->label().has_value() ? asrt->label().value() : "",
      asrt->message());
}

absl::StatusOr<Display*> ModuleBuilder::EmitTrace(
    xls::Trace* trace, Expression* condition,
    absl::Span<Expression* const> trace_args) {
  auto [macro_name, polarity] = SimulationMacroNameAndPolarity(options_);
  ModuleConditionalDirective* directive =
      trace_section_->Add<ModuleConditionalDirective>(SourceInfo(), polarity,
                                                      macro_name);

  StructuredProcedure* trace_always;
  std::vector<SensitivityListElement> sensitivity_list;
  if (options_.use_system_verilog()) {
    if (clk_ != nullptr) {
      // Even though this is purely behavioral and not synthesizable, use
      // always_ff as a stylistic choice.
      trace_always = directive->consequent()->Add<AlwaysFf>(
          trace->loc(), std::initializer_list<SensitivityListElement>{
                            file_->Make<PosEdge>(trace->loc(), clk_)});

    } else {
      // When targeting SystemVerilog with no clock, we use `always_comb` and
      // have no need for a sensitivity list.
      trace_always = directive->consequent()->Add<AlwaysComb>(trace->loc());
    }
  } else {
    if (clk_ != nullptr) {
      // When targeting Verilog with a clock, we use `always` triggered on the
      // posedge.
      trace_always = directive->consequent()->Add<Always>(
          trace->loc(), std::initializer_list<SensitivityListElement>{
                            file_->Make<PosEdge>(trace->loc(), clk_)});
    } else {
      // When targeting Verilog with no clock, we use `always` and need to
      // populate the sensitivity list with the implicit event expression.
      trace_always = directive->consequent()->Add<Always>(
          trace->loc(), std::initializer_list<SensitivityListElement>{
                            ImplicitEventExpression{}});
    }
  }

  Conditional* trace_if =
      trace_always->statements()->Add<Conditional>(trace->loc(), condition);

  Expression* format_arg = file_->Make<QuotedString>(
      trace->loc(), StepsToVerilogFormatString(trace->format()));

  std::vector<Expression*> display_args = {format_arg};
  for (Expression* arg : trace_args) {
    display_args.push_back(arg);
  }

  return trace_if->consequent()->Add<Display>(trace->loc(), display_args);
}

absl::StatusOr<IndexableExpression*> ModuleBuilder::EmitGate(
    xls::Gate* gate, Expression* condition, Expression* data) {
  // Only bits-typed or tuple-typed data supported.
  // TODO(https://github.com/google/xls/issues/463) 2021/07/20 Add support for
  // array types.
  if (!gate->GetType()->IsBits() && !gate->GetType()->IsTuple()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Gate operation only supported for bits and tuple types, has type: %s",
        gate->GetType()->ToString()));
  }

  LogicRef* ref = DeclareVariable(gate->GetName(), gate->GetType());

  // Emit the gate as an AND of the (potentially replicated) condition and the
  // data. For example:
  //
  //   wire gated_data [31:0];
  //   assign gated_data = {32{condition}} & data;
  //
  Expression* gate_expr;
  if (gate->GetType()->GetFlatBitCount() == 1) {
    // Data is a single bit. Just AND with the condition.
    gate_expr = file_->BitwiseAnd(condition, data, gate->loc());
  } else {
    // Data is wider than a single bit. Replicate the condition to match the
    // width of the data.
    gate_expr =
        file_->BitwiseAnd(file_->Concat(gate->GetType()->GetFlatBitCount(),
                                        {condition}, gate->loc()),
                          data, gate->loc());
  }
  XLS_RETURN_IF_ERROR(Assign(ref, gate_expr, gate->GetType()));
  return ref;
}

absl::StatusOr<NodeRepresentation> ModuleBuilder::EmitCover(
    xls::Cover* cover, Expression* condition) {
  if (!options_.use_system_verilog()) {
    // Coverpoints are a SystemVerilog only feature.
    LOG(WARNING) << "Coverpoints are only supported in SystemVerilog.";
    return UnrepresentedSentinel();
  }
  if (clk_ == nullptr) {
    return absl::InvalidArgumentError(
        "Coverpoints require a clock to be present in the module.");
  }
  return cover_section()->Add<Cover>(cover->loc(), clk_, condition,
                                     cover->label());
}

absl::Status ModuleBuilder::Assign(LogicRef* lhs, Expression* rhs, Type* type) {
  XLS_RETURN_IF_ERROR(
      AddAssignment(type, lhs, rhs, [&](Expression* lhs, Expression* rhs) {
        assignment_section()->Add<ContinuousAssignment>(SourceInfo(), lhs, rhs);
      }));
  return absl::OkStatus();
}

absl::StatusOr<ModuleBuilder::Register> ModuleBuilder::DeclareRegister(
    std::string_view name, Type* type, Expression* next,
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
    reg = module_->AddReg(
        SanitizeIdentifier(name),
        file_->UnpackedArrayType(NestedElementWidth(array_type),
                                 NestedArrayBounds(array_type), SourceInfo()),
        SourceInfo(),
        /*init=*/nullptr, declaration_section());
  } else {
    reg = module_->AddReg(
        SanitizeIdentifier(name),
        file_->BitVectorType(type->GetFlatBitCount(), SourceInfo()),
        SourceInfo(), /*init=*/nullptr, declaration_section());
  }
  return Register{.ref = reg,
                  .next = next,
                  .reset_value = reset_value,
                  .load_enable = nullptr,
                  .xls_type = type};
}

absl::StatusOr<ModuleBuilder::Register> ModuleBuilder::DeclareRegister(
    std::string_view name, int64_t bit_count, Expression* next,
    Expression* reset_value) {
  if (clk_ == nullptr) {
    return absl::InvalidArgumentError("Clock signal required for register.");
  }
  if (!rst_.has_value() && reset_value != nullptr) {
    return absl::InvalidArgumentError(
        "Block has no reset signal, but register has reset value.");
  }

  return Register{.ref = module_->AddReg(
                      SanitizeIdentifier(name),
                      file_->BitVectorType(bit_count, SourceInfo()),
                      SourceInfo(), /*init=*/nullptr, declaration_section()),
                  .next = next,
                  .reset_value = reset_value,
                  .load_enable = nullptr,
                  .xls_type = nullptr};
}

absl::Status ModuleBuilder::AssignRegisters(
    absl::Span<const Register> registers) {
  XLS_RET_CHECK(clk_ != nullptr);

  if (registers.empty()) {
    return absl::OkStatus();
  }

  // All registers must either all have reset values or none have reset values,
  // or incorrect Verilog may be generated. The condition resulting in incorrect
  // Verilog requires that the not-reset register uses the reset signal as a
  // logic input. For example (`a` is a register with a reset, `b` uses the
  // reset signal as a logic input):
  //
  // always_ff @(posedge clk) begin
  //   if (rst) begin
  //     a <= ...;
  //   end else begin
  //     a <= ...;
  //     b <= rst || ...;
  // end
  //
  // In this case, b will not be assigned the correct value if `rst` is enabled
  // because of the guarding if. This check avoids the potential issue.
  for (const Register& reg : registers) {
    if ((reg.reset_value == nullptr) !=
        (registers.front().reset_value == nullptr)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "All registers passed to AssignRegisters must either have a reset or "
          "none have a reset. Registers %s (%s) and %s (%s) differ.",
          reg.ref->GetName(),
          reg.reset_value == nullptr ? "no reset" : "has reset",
          registers.front().ref->GetName(),
          registers.front().reset_value == nullptr ? "no reset" : "has reset"));
    }
  }

  // Construct an always_ff block.
  std::vector<SensitivityListElement> sensitivity_list;
  sensitivity_list.push_back(file_->Make<PosEdge>(SourceInfo(), clk_));
  if (rst_.has_value()) {
    if (rst_->asynchronous) {
      if (rst_->active_low) {
        sensitivity_list.push_back(
            file_->Make<NegEdge>(SourceInfo(), rst_->signal));
      } else {
        sensitivity_list.push_back(
            file_->Make<PosEdge>(SourceInfo(), rst_->signal));
      }
    }
  }
  AlwaysBase* always;
  if (options_.use_system_verilog()) {
    always =
        assignment_section()->Add<AlwaysFf>(SourceInfo(), sensitivity_list);
  } else {
    always = assignment_section()->Add<Always>(SourceInfo(), sensitivity_list);
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
      rst_condition = file_->LogicalNot(rst_->signal, SourceInfo());
    } else {
      rst_condition = rst_->signal;
    }
    Conditional* conditional =
        always->statements()->Add<Conditional>(SourceInfo(), rst_condition);
    for (const Register& reg : registers) {
      if (reg.reset_value == nullptr) {
        // Not all registers may have reset values.
        continue;
      }
      XLS_RETURN_IF_ERROR(
          AddAssignment(reg.xls_type, reg.ref, reg.reset_value,
                        [&](Expression* lhs, Expression* rhs) {
                          conditional->consequent()->Add<NonblockingAssignment>(
                              SourceInfo(), lhs, rhs);
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
              SourceInfo(), lhs,
              reg.load_enable == nullptr
                  ? rhs
                  : file_->Ternary(reg.load_enable, rhs, lhs, SourceInfo()));
        }));
  }
  return absl::OkStatus();
}

bool ModuleBuilder::MustEmitAsFunction(Node* node) {
  switch (node->op()) {
    case Op::kSMul:
    case Op::kUMul:
    case Op::kSMulp:
    case Op::kUMulp:
    case Op::kDynamicBitSlice:
    case Op::kBitSliceUpdate:
    case Op::kSDiv:
    case Op::kUDiv:
      return true;
    case Op::kPrioritySel:
      return node->As<PrioritySelect>()->cases().size() > 1;
    default:
      return false;
  }
}

absl::StatusOr<std::string> ModuleBuilder::VerilogFunctionName(Node* node) {
  switch (node->op()) {
    case Op::kSMul:
    case Op::kUMul:
      // Multiplies may be mixed width so include result and operand widths in
      // the name.
      return absl::StrFormat(
          "%s%db_%db_x_%db", OpToString(node->op()), node->BitCountOrDie(),
          node->operand(0)->BitCountOrDie(), node->operand(1)->BitCountOrDie());
    case Op::kSMulp:
    case Op::kUMulp:
      return absl::StrFormat("%s%db_%db_x_%db", OpToString(node->op()),
                             node->As<PartialProductOp>()->width(),
                             node->operand(0)->BitCountOrDie(),
                             node->operand(1)->BitCountOrDie());
    case Op::kDynamicBitSlice:
      return absl::StrFormat(
          "%s_w%d_%db_%db", OpToString(node->op()), node->BitCountOrDie(),
          node->operand(0)->BitCountOrDie(), node->operand(1)->BitCountOrDie());
    case Op::kBitSliceUpdate:
      return absl::StrFormat(
          "%s_w%d_%db_%db", OpToString(node->op()), node->BitCountOrDie(),
          node->operand(1)->BitCountOrDie(), node->operand(2)->BitCountOrDie());
    case Op::kPrioritySel: {
      XLS_ASSIGN_OR_RETURN(
          SelectorProperties selector_properties,
          GetSelectorProperties(node->As<PrioritySelect>()->selector()));
      std::string_view never_zero_str;
      if (selector_properties.never_zero) {
        never_zero_str = "_snz";  // selector never zero
      }
      return absl::StrFormat("%s_%db_%dway%s", OpToString(node->op()),
                             node->GetType()->GetFlatBitCount(),
                             node->operand(0)->BitCountOrDie(), never_zero_str);
    }
    case Op::kSDiv:
    case Op::kUDiv:
      CHECK_EQ(node->BitCountOrDie(), node->operand(0)->BitCountOrDie());
      CHECK_EQ(node->BitCountOrDie(), node->operand(1)->BitCountOrDie());
      return absl::StrFormat("%s_%db", OpToString(node->op()),
                             node->BitCountOrDie());
    default:
      LOG(FATAL) << "Cannot emit node as function: " << node->ToString();
  }
}

namespace {

// Defines and returns a function which implements the given DynamicBitSlice
// node.
VerilogFunction* DefineDynamicBitSliceFunction(DynamicBitSlice* slice,
                                               std::string_view function_name,
                                               ModuleSection* section) {
  VerilogFile* file = section->file();
  VerilogFunction* func = section->Add<VerilogFunction>(
      slice->loc(), function_name,
      file->BitVectorType(slice->BitCountOrDie(), slice->loc()));
  Expression* operand = func->AddArgument(
      "operand",
      file->BitVectorType(slice->to_slice()->BitCountOrDie(), slice->loc()),
      slice->loc());
  Expression* start = func->AddArgument(
      "start",
      file->BitVectorType(slice->start()->BitCountOrDie(), slice->loc()),
      slice->loc());
  int64_t width = slice->width();

  LogicRef* zexted_operand = func->AddRegDef(
      slice->loc(), "zexted_operand",
      file->BitVectorType(slice->to_slice()->BitCountOrDie() + width,
                          slice->loc()),
      /*init=*/nullptr);

  Expression* zeros = file->Literal(0, width, slice->loc());
  Expression* op_width = file->Literal(
      slice->operand(0)->BitCountOrDie(),
      Bits::MinBitCountUnsigned(slice->to_slice()->BitCountOrDie()),
      slice->loc());
  // If start of slice is greater than or equal to operand width, result is
  // completely out of bounds and set to all zeros.
  Expression* out_of_bounds =
      file->GreaterThanEquals(start, op_width, slice->loc());
  // Pad with width zeros
  func->AddStatement<BlockingAssignment>(
      slice->loc(), zexted_operand,
      file->Concat({zeros, operand}, slice->loc()));
  Expression* sliced_operand =
      file->PartSelect(zexted_operand, start, width, slice->loc());
  func->AddStatement<BlockingAssignment>(
      slice->loc(), func->return_value_ref(),
      file->Ternary(out_of_bounds, zeros, sliced_operand, slice->loc()));
  return func;
}

// Defines and returns a function which implements the given BitSliceUpdate
// node.
VerilogFunction* DefineBitSliceUpdateFunction(BitSliceUpdate* update,
                                              std::string_view function_name,
                                              ModuleSection* section) {
  VerilogFile* file = section->file();
  int64_t to_update_width = update->to_update()->BitCountOrDie();
  int64_t start_width = update->start()->BitCountOrDie();
  int64_t update_value_width = update->update_value()->BitCountOrDie();

  // We purposefully avoid using scalars here, because they cannot be sliced.
  VerilogFunction* func = section->Add<VerilogFunction>(
      update->loc(), function_name,
      file->BitVectorTypeNoScalar(update->BitCountOrDie(), update->loc()));
  IndexableExpression* to_update = func->AddArgument(
      "to_update", file->BitVectorTypeNoScalar(to_update_width, update->loc()),
      update->loc());
  IndexableExpression* start = func->AddArgument(
      "start", file->BitVectorTypeNoScalar(start_width, update->loc()),
      update->loc());
  IndexableExpression* update_value = func->AddArgument(
      "update_value",
      file->BitVectorTypeNoScalar(update_value_width, update->loc()),
      update->loc());

  Expression* adjusted_update_value;
  if (update_value_width > to_update_width) {
    // Update value is the wider than the value to be updated. Slice update
    // value to match the width.
    adjusted_update_value = file->Slice(
        update_value, file->PlainLiteral(to_update_width - 1, update->loc()),
        file->PlainLiteral(0, update->loc()), update->loc());
  } else if (update_value_width < to_update_width) {
    // Update value is the narrower than the value to be updated. Zero-extend
    // update value to match the width.
    adjusted_update_value =
        file->Concat({file->Literal(Bits(to_update_width - update_value_width),
                                    update->loc()),
                      update_value},
                     update->loc());
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
  Expression* mask = file->BitwiseNot(
      file->Shll(file->Literal(all_ones, update->loc()), start, update->loc()),
      update->loc());
  Expression* updated_value = file->BitwiseOr(
      file->Shll(adjusted_update_value, start, update->loc()),
      file->BitwiseAnd(mask, to_update, update->loc()), update->loc());

  if (Bits::MinBitCountUnsigned(to_update_width) > start_width) {
    // Start value is not wide enough to encode the width of the value to
    // update. No need to protect against overshifting.
    func->AddStatement<BlockingAssignment>(
        update->loc(), func->return_value_ref(), updated_value);
  } else {
    // Start value is wide enough to encode the width of the value to
    // update. Protect against overshifting by selecting the unchanged value to
    // update if start is greater than or equal to width.
    func->AddStatement<BlockingAssignment>(
        update->loc(), func->return_value_ref(),
        file->Ternary(file->GreaterThanEquals(
                          start,
                          file->Literal(UBits(to_update_width, start_width),
                                        update->loc()),
                          update->loc()),
                      to_update, updated_value, update->loc()));
  }
  return func;
}

// Defines and returns a function which implements the given SMul node.
VerilogFunction* DefineSmulFunction(Node* node, std::string_view function_name,
                                    ModuleSection* section) {
  CHECK_EQ(node->op(), Op::kSMul);
  VerilogFile* file = section->file();

  ScopedLintDisable lint_disable(section, {Lint::kSignedType, Lint::kMultiply});

  VerilogFunction* func = section->Add<VerilogFunction>(
      node->loc(), function_name,
      file->BitVectorType(node->BitCountOrDie(), node->loc()));
  CHECK_EQ(node->operand_count(), 2);
  Expression* lhs = func->AddArgument(
      "lhs",
      file->BitVectorType(node->operand(0)->BitCountOrDie(), node->loc()),
      node->loc());
  Expression* rhs = func->AddArgument(
      "rhs",
      file->BitVectorType(node->operand(1)->BitCountOrDie(), node->loc()),
      node->loc());
  // The code conservatively assigns signed-casted inputs to temporary
  // variables, uses them in the multiply expression which is assigned to
  // another signed temporary. Finally, this is unsign-casted and assigned to
  // the return value of the function. These shenanigans ensure no surprising
  // sign/zero extensions of any values.
  LogicRef* signed_lhs = func->AddRegDef(
      node->loc(), "signed_lhs",
      file->BitVectorType(node->operand(0)->BitCountOrDie(), node->loc(),
                          /*is_signed=*/true),
      /*init=*/nullptr);
  LogicRef* signed_rhs = func->AddRegDef(
      node->loc(), "signed_rhs",
      file->BitVectorType(node->operand(1)->BitCountOrDie(), node->loc(),
                          /*is_signed=*/true),
      /*init=*/nullptr);
  LogicRef* signed_result =
      func->AddRegDef(node->loc(), "signed_result",
                      file->BitVectorType(node->BitCountOrDie(), node->loc(),
                                          /*is_signed=*/true),
                      /*init=*/nullptr);
  func->AddStatement<BlockingAssignment>(
      node->loc(), signed_lhs, file->Make<SignedCast>(node->loc(), lhs));
  func->AddStatement<BlockingAssignment>(
      node->loc(), signed_rhs, file->Make<SignedCast>(node->loc(), rhs));
  func->AddStatement<BlockingAssignment>(
      node->loc(), signed_result,
      file->Mul(signed_lhs, signed_rhs, node->loc()));
  func->AddStatement<BlockingAssignment>(
      node->loc(), func->return_value_ref(),
      file->Make<UnsignedCast>(node->loc(), signed_result));
  return func;
}

// Defines and returns a function which implements the given UMul node.
VerilogFunction* DefineUmulFunction(Node* node, std::string_view function_name,
                                    ModuleSection* section) {
  CHECK_EQ(node->op(), Op::kUMul);
  VerilogFile* file = section->file();

  ScopedLintDisable lint_disable(section, {Lint::kMultiply});

  VerilogFunction* func = section->Add<VerilogFunction>(
      node->loc(), function_name,
      file->BitVectorType(node->BitCountOrDie(), node->loc()));
  CHECK_EQ(node->operand_count(), 2);
  Expression* lhs = func->AddArgument(
      "lhs",
      file->BitVectorType(node->operand(0)->BitCountOrDie(), node->loc()),
      node->loc());
  Expression* rhs = func->AddArgument(
      "rhs",
      file->BitVectorType(node->operand(1)->BitCountOrDie(), node->loc()),
      node->loc());
  func->AddStatement<BlockingAssignment>(node->loc(), func->return_value_ref(),
                                         file->Mul(lhs, rhs, node->loc()));

  return func;
}

// Defines and returns a function which implements the given SMulp node.
absl::StatusOr<VerilogFunction*> DefineSmulpFunction(
    Node* node, std::string_view function_name, ModuleSection* section) {
  CHECK_EQ(node->op(), Op::kSMulp);
  CHECK_EQ(node->operand_count(), 2);
  int64_t width = node->As<PartialProductOp>()->width();

  VerilogFile* file = section->file();

  ScopedLintDisable lint_disable(section, {Lint::kMultiply});

  VerilogFunction* func = section->Add<VerilogFunction>(
      node->loc(), function_name, file->BitVectorType(width * 2, node->loc()));
  Expression* lhs = func->AddArgument(
      "lhs",
      file->BitVectorType(node->operand(0)->BitCountOrDie(), node->loc()),
      node->loc());
  Expression* rhs = func->AddArgument(
      "rhs",
      file->BitVectorType(node->operand(1)->BitCountOrDie(), node->loc()),
      node->loc());

  // The code conservatively assigns signed-casted inputs to temporary
  // variables, uses them in the multiply expression which is assigned to
  // another signed temporary. Finally, this is unsign-casted and assigned to
  // the return value of the function. These shenanigans ensure no surprising
  // sign/zero extensions of any values.
  LogicRef* signed_lhs = func->AddRegDef(
      node->loc(), "signed_lhs",
      file->BitVectorType(node->operand(0)->BitCountOrDie(), node->loc(),
                          /*is_signed=*/true),
      /*init=*/nullptr);
  LogicRef* signed_rhs = func->AddRegDef(
      node->loc(), "signed_rhs",
      file->BitVectorType(node->operand(1)->BitCountOrDie(), node->loc(),
                          /*is_signed=*/true),
      /*init=*/nullptr);
  func->AddStatement<BlockingAssignment>(
      node->loc(), signed_lhs, file->Make<SignedCast>(node->loc(), lhs));
  func->AddStatement<BlockingAssignment>(
      node->loc(), signed_rhs, file->Make<SignedCast>(node->loc(), rhs));

  LogicRef* signed_result =
      func->AddRegDef(node->loc(), "signed_result",
                      file->BitVectorType(width, node->loc(),
                                          /*is_signed=*/true),
                      /*init=*/nullptr);
  func->AddStatement<BlockingAssignment>(
      node->loc(), signed_result,
      file->Mul(signed_lhs, signed_rhs, node->loc()));

  Bits offset_bits = MulpOffsetForSimulation(
      node->GetType()->AsTupleOrDie()->element_type(0)->GetFlatBitCount(),
      /*shift_size=*/4);
  Literal* offset = file->Literal(offset_bits, node->loc());
  func->AddStatement<BlockingAssignment>(
      node->loc(), func->return_value_ref(),
      file->Concat({offset, file->Sub(file->Make<UnsignedCast>(node->loc(),
                                                               signed_result),
                                      offset, node->loc())},
                   node->loc()));

  return func;
}

// Defines and returns a function which implements the given UMulp node.
absl::StatusOr<VerilogFunction*> DefineUmulpFunction(
    Node* node, std::string_view function_name, ModuleSection* section) {
  CHECK_EQ(node->op(), Op::kUMulp);
  CHECK_EQ(node->operand_count(), 2);
  int64_t width = node->As<PartialProductOp>()->width();

  VerilogFile* file = section->file();

  ScopedLintDisable lint_disable(section, {Lint::kMultiply});

  VerilogFunction* func = section->Add<VerilogFunction>(
      node->loc(), function_name, file->BitVectorType(width * 2, node->loc()));
  Expression* lhs = func->AddArgument(
      "lhs",
      file->BitVectorType(node->operand(0)->BitCountOrDie(), node->loc()),
      node->loc());
  Expression* rhs = func->AddArgument(
      "rhs",
      file->BitVectorType(node->operand(1)->BitCountOrDie(), node->loc()),
      node->loc());

  LogicRef* result = func->AddRegDef(node->loc(), "result",
                                     file->BitVectorType(width, node->loc(),
                                                         /*is_signed=*/false),
                                     /*init=*/nullptr);
  func->AddStatement<BlockingAssignment>(node->loc(), result,
                                         file->Mul(lhs, rhs, node->loc()));

  Bits offset_bits = MulpOffsetForSimulation(
      node->GetType()->AsTupleOrDie()->element_type(0)->GetFlatBitCount(),
      /*shift_size=*/4);
  Literal* offset = file->Literal(offset_bits, node->loc());
  func->AddStatement<BlockingAssignment>(
      node->loc(), func->return_value_ref(),
      file->Concat({offset, file->Sub(result, offset, node->loc())},
                   node->loc()));

  return func;
}

// Defines and returns a function which implements the given Udiv node.
VerilogFunction* DefineUDivFunction(Node* node, std::string_view function_name,
                                    ModuleSection* section) {
  CHECK_EQ(node->op(), Op::kUDiv);
  VerilogFile* file = section->file();

  VerilogFunction* func = section->Add<VerilogFunction>(
      node->loc(), function_name,
      file->BitVectorType(node->BitCountOrDie(), node->loc()));
  CHECK_EQ(node->operand_count(), 2);
  Expression* lhs = func->AddArgument(
      "lhs",
      file->BitVectorType(node->operand(0)->BitCountOrDie(), node->loc()),
      node->loc());
  Expression* rhs = func->AddArgument(
      "rhs",
      file->BitVectorType(node->operand(1)->BitCountOrDie(), node->loc()),
      node->loc());
  Expression* rhs_is_zero = file->Equals(
      rhs,
      file->Literal(UBits(0, node->operand(1)->BitCountOrDie()), node->loc()),
      node->loc());
  Expression* all_ones =
      file->Literal(Bits::AllOnes(node->BitCountOrDie()), node->loc());
  func->AddStatement<BlockingAssignment>(
      node->loc(), func->return_value_ref(),
      file->Ternary(rhs_is_zero, all_ones, file->Div(lhs, rhs, node->loc()),
                    node->loc()));
  return func;
}

// Defines and returns a function which implements the given SDiv node.
VerilogFunction* DefineSDivFunction(Node* node, std::string_view function_name,
                                    ModuleSection* section) {
  CHECK_EQ(node->op(), Op::kSDiv);
  VerilogFile* file = section->file();

  VerilogFunction* func = section->Add<VerilogFunction>(
      node->loc(), function_name,
      file->BitVectorType(node->BitCountOrDie(), node->loc()));
  CHECK_EQ(node->operand_count(), 2);
  IndexableExpression* lhs = func->AddArgument(
      "lhs",
      file->BitVectorType(node->operand(0)->BitCountOrDie(), node->loc()),
      node->loc());
  IndexableExpression* rhs = func->AddArgument(
      "rhs",
      file->BitVectorType(node->operand(1)->BitCountOrDie(), node->loc()),
      node->loc());
  Expression* rhs_is_zero = file->Equals(
      rhs,
      file->Literal(UBits(0, node->operand(1)->BitCountOrDie()), node->loc()),
      node->loc());
  Expression* max_positive =
      file->Literal(Bits::MaxSigned(node->BitCountOrDie()), node->loc());
  Expression* min_negative =
      file->Literal(Bits::MinSigned(node->BitCountOrDie()), node->loc());
  Expression* lhs_is_negative =
      file->Index(lhs, node->operand(0)->BitCountOrDie() - 1, node->loc());
  Expression* div_by_zero_result =
      file->Ternary(lhs_is_negative, min_negative, max_positive, node->loc());

  // Wrap the expression in $unsigned to prevent the signed property from
  // leaking out into the rest of the expression.
  Expression* quotient = file->Make<UnsignedCast>(
      node->loc(),
      file->Div(file->Make<SignedCast>(node->loc(), lhs),
                file->Make<SignedCast>(node->loc(), rhs), node->loc()));
  // The divide overflows in the case of `min / -1`. Should return min value in
  // this case.
  Expression* overflow_condition = file->LogicalAnd(
      file->Equals(
          lhs,
          file->Literal(Bits::MinSigned(node->operand(0)->BitCountOrDie()),
                        node->loc()),
          node->loc()),
      file->Equals(rhs,
                   file->Literal(SBits(-1, node->operand(1)->BitCountOrDie()),
                                 node->loc()),
                   node->loc()),
      node->loc());
  Expression* overflow_protected_quotient = file->Ternary(
      overflow_condition,
      file->Literal(Bits::MinSigned(node->BitCountOrDie()), node->loc()),
      quotient, node->loc());

  func->AddStatement<BlockingAssignment>(
      node->loc(), func->return_value_ref(),
      file->Ternary(rhs_is_zero, div_by_zero_result,
                    overflow_protected_quotient, node->loc()));
  return func;
}

}  // namespace

absl::StatusOr<VerilogFunction*> ModuleBuilder::DefineFunction(Node* node) {
  XLS_ASSIGN_OR_RETURN(std::string function_name, VerilogFunctionName(node));
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
    case Op::kSMulp: {
      XLS_ASSIGN_OR_RETURN(
          func, DefineSmulpFunction(node, function_name, functions_section_));
      break;
    }
    case Op::kUMulp: {
      XLS_ASSIGN_OR_RETURN(
          func, DefineUmulpFunction(node, function_name, functions_section_));
      break;
    }
    case Op::kDynamicBitSlice:
      func = DefineDynamicBitSliceFunction(node->As<DynamicBitSlice>(),
                                           function_name, functions_section_);
      break;
    case Op::kBitSliceUpdate:
      func = DefineBitSliceUpdateFunction(node->As<BitSliceUpdate>(),
                                          function_name, functions_section_);
      break;
    case Op::kPrioritySel: {
      XLS_ASSIGN_OR_RETURN(
          SelectorProperties selector_properties,
          GetSelectorProperties(node->As<PrioritySelect>()->selector()));
      XLS_ASSIGN_OR_RETURN(
          func,
          DefinePrioritySelectFunction(
              node->As<PrioritySelect>()->selector(), /*tpe=*/node->GetType(),
              /*num_cases=*/node->As<PrioritySelect>()->cases().size(),
              /*loc=*/node->loc(), function_name, functions_section_,
              selector_properties, options_));
      break;
    }
    case Op::kUDiv:
      func = DefineUDivFunction(node, function_name, functions_section_);
      break;
    case Op::kSDiv:
      func = DefineSDivFunction(node, function_name, functions_section_);
      break;
    default:
      LOG(FATAL) << "Cannot define node as function: " << node->ToString();
  }
  node_functions_[function_name] = func;
  return func;
}

}  // namespace verilog
}  // namespace xls
