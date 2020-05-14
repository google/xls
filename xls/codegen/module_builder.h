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

#ifndef THIRD_PARTY_XLS_CODEGEN_MODULE_BUILDER_H_
#define THIRD_PARTY_XLS_CODEGEN_MODULE_BUILDER_H_

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/node.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace verilog {

// An abstraction wrapping a VAST module which assists with lowering of XLS IR
// into Verilog. Key functionality:
//   (1) Handles mapping of XLS types to Verilog types.
//   (2) Hides Verilog vs. SystemVerilog differences and enables targeting
//       either from same code.
//   (3) Imposes common organization to the module.
class ModuleBuilder {
 public:
  ModuleBuilder(absl::string_view name, VerilogFile* file,
                bool use_system_verilog);

  // Returns the underlying module being constructed.
  Module* module() { return module_; }

  // Add an input port of the given XLS type to the module.
  xabsl::StatusOr<LogicRef*> AddInputPort(absl::string_view name, Type* type);

  // Add an input port of the given width.
  LogicRef* AddInputPort(absl::string_view name, int64 bit_count);

  // Add an output port of the given XLS type to the module. The output is
  // assigned the given value.
  absl::Status AddOutputPort(absl::string_view name, Type* type,
                             Expression* value);

  // Add an output port of the given width to the module. The output is assigned
  // the given value.
  absl::Status AddOutputPort(absl::string_view name, int64 bit_count,
                             Expression* value);

  // Returns whether the given node can be emitted as an inline expression in
  // Verilog (the alternative is to assign the expression of node to a temporary
  // variable). Generally operations on bits-typed values can be emitted inline
  // (for example, Op::kAnd). Operations on compound types such as arrays and
  // tuples may require declaration of temporary variables and one or more
  // assignment statements. Also, the users of a node may also force the node to
  // be emitted as a temporary variable, if, for example, the emitted code for
  // the user indexes into the node's value. 'users_of_expression', if given,
  // are the users 'node' in the emitted Verilog to consider when determining
  // whether node can be emitted inline. If not specified all users of node are
  // considered.
  bool CanEmitAsInlineExpression(Node* node,
                                 absl::optional<absl::Span<Node* const>>
                                     users_of_expression = absl::nullopt);

  // Returns the given node as a Verilog expression. 'inputs' contains the
  // operand expressions for the node.
  xabsl::StatusOr<Expression*> EmitAsInlineExpression(
      Node* node, absl::Span<Expression* const> inputs);

  // Emits the node as one or more assignments to a newly declared variable with
  // the given name. 'inputs' contains the operand expressions for the
  // node. Returns a reference to the declared variable.
  xabsl::StatusOr<LogicRef*> EmitAsAssignment(
      absl::string_view name, Node* node, absl::Span<Expression* const> inputs);

  // Declares a variable with the given name and XLS type. Returns a reference
  // to the variable.
  LogicRef* DeclareVariable(absl::string_view name, Type* type);

  // Assigns the rhs to the lhs using continuous assignment where both sides
  // have the given XLS type. The emitted verilog may require multiple
  // assignment statements for compound types such as arrays.
  absl::Status Assign(LogicRef* lhs, Expression* rhs, Type* type);

  // Declares variable with the given name and assigns the given value to
  // it. Returns a reference to the variable.
  xabsl::StatusOr<LogicRef*> DeclareModuleConstant(absl::string_view name,
                                                   const Value& Value);

  // Data structure describing a register (collection of flops).
  struct Register {
    // Reference to the declared logic/reg variable holding the register value.
    LogicRef* ref;

    // The expression to assign to the register at each clock.
    Expression* next;

    // The register value upon reset. Should be non-null iff AssignRegisters is
    // called with non-null Reset argument.
    Expression* reset_value;

    // Optional XLS type of this register. Can be null.
    Type* xls_type;
  };

  // Declares a register of the given XLS type. Arguments:
  //   name: name of the declared Verilog register.
  //   type: XLS type of the register.
  //   next: The expression to assign to the register each clock.
  //   reset_value: The value of the register on reset. Should be non-null iff
  //     the corresponding AssignRegisters call includes a non-null Reset
  //     argument.
  //
  // Declared registers must be passed to a subsequent AssignRegisters call for
  // assignment within an always block.
  xabsl::StatusOr<Register> DeclareRegister(
      absl::string_view name, Type* type, Expression* next,
      absl::optional<Expression*> reset_value = absl::nullopt);

  // As above, but declares a register of a given bit width.
  xabsl::StatusOr<Register> DeclareRegister(
      absl::string_view name, int64 bit_count, Expression* next,
      absl::optional<Expression*> reset_value = absl::nullopt);

  // Construct an always block to assign values to the registers. Arguments:
  //   clk: Clock signal to use for registers.
  //   registers: Registers to assign within this block.
  //   load_enable: Optional load enable signal. The register is loaded only if
  //     this signal is asserted.
  //   rst: Optional reset signal.
  absl::Status AssignRegisters(LogicRef* clk,
                               absl::Span<const Register> registers,
                               Expression* load_enable = nullptr,
                               absl::optional<Reset> rst = absl::nullopt);

  // For organization (not functionality) the module is divided into several
  // sections. The emitted module has the following structure:
  //
  //   module foo(
  //     ...
  //   );
  //      { functions_section }
  //        // definitions of functions used in module.
  //      { constants_section }
  //        // declarations of module-level constants.
  //      { input_section }
  //        // converts potentially flattened input values to
  //        // module-internal form (e.g. unpacked array).
  //      { declarations_sections_[0] }
  //        // declarations of module variables.
  //      { assignments_sections_[0] }
  //        // assignments to module variables and always_ff sections.
  //      { declarations_sections_[1] } // Optional
  //      { assignments_sections_[1] }  // Optional
  //        ...
  //      { output_section }
  //        // assigns the output port(s) including any flattening.
  //   endmodule
  //
  // The declarations and assignment sections appear as a pair and more than one
  // instance of this pair of sections can be added to the module by calling
  // NewDeclarationAndAssignmentSections.

  // Creates new declaration and assignment sections immediately after the
  // current declaration and assignment sections.
  void NewDeclarationAndAssignmentSections();

  // Methods to returns one of the various sections in the module.
  ModuleSection* declaration_section() const {
    return declaration_subsections_.back();
  }
  ModuleSection* assignment_section() const {
    return assignment_subsections_.back();
  }
  ModuleSection* functions_section() const { return functions_section_; }
  ModuleSection* constants_section() const { return constants_section_; }
  ModuleSection* input_section() const { return input_section_; }
  ModuleSection* output_section() const { return output_section_; }

 private:
  // Declares an unpacked array wire/reg variable of the given XLS array type in
  // the given ModuleSection.
  LogicRef* DeclareUnpackedArrayWire(absl::string_view name,
                                     ArrayType* array_type,
                                     ModuleSection* section);
  LogicRef* DeclareUnpackedArrayReg(absl::string_view name,
                                    ArrayType* array_type,
                                    ModuleSection* section);

  // Assigns 'rhs' to 'lhs'. Depending upon the type this may require multiple
  // assignment statements (e.g., for array assignments in Verilog). The
  // function add_assignment_statement should add a single assignment
  // statement. This function argument enables customization of the type of
  // assignment (continuous, blocking, or non-blocking) as well as the location
  // where the assignment statements are added.
  absl::Status AddAssignment(
      Expression* lhs, Expression* rhs, Type* xls_type,
      std::function<void(Expression*, Expression*)> add_assignment_statement);

  // Assigns the arbitrarily-typed Value 'value' to 'lhs'. Depending upon the
  // type this may require multiple assignment statements. The function
  // add_assignment_statement should add a single assignment statement.
  absl::Status AddAssignmentFromValue(
      Expression* lhs, const Value& value,
      std::function<void(Expression*, Expression*)> add_assignment_statement);

  // Extracts a slice from the bits-typed 'rhs' and assigns it to 'lhs' in
  // unflattened form.  Depending upon the type this may require multiple
  // assignment statements. The function add_assignment_statement should add a
  // single assignment statement.
  absl::Status AssignFromSlice(
      Expression* lhs, Expression* rhs, Type* xls_type, int64 slice_start,
      std::function<void(Expression*, Expression*)> add_assignment_statement);

  // Returns true if the node must be emitted as a function.
  bool MustEmitAsFunction(Node* node);

  // Returns the name of the function which implements node. The function name
  // should encapsulate all metainformation about the node (opcode, bitwidth,
  // etc) because a function definition may be reused to implement multiple
  // identical nodes (for example, two different 32-bit multiplies may map to
  // the same function).
  std::string VerilogFunctionName(Node* node);

  // Defines a function which implements the given node. If a function already
  // exists which implements this node then the existing function is returned.
  xabsl::StatusOr<VerilogFunction*> DefineFunction(Node* node);

  std::string module_name_;
  VerilogFile* file_;

  // True if SystemVerilog constructs can be used. Otherwise the emitted code is
  // strictly Verilog.
  bool use_system_verilog_;

  Module* module_;
  ModuleSection* functions_section_;
  ModuleSection* constants_section_;
  ModuleSection* input_section_;
  ModuleSection* declaration_and_assignment_section_;
  std::vector<ModuleSection*> declaration_subsections_;
  std::vector<ModuleSection*> assignment_subsections_;
  ModuleSection* output_section_;

  // Verilog functions defined inside the module. Map is indexed by the function
  // name.
  absl::flat_hash_map<std::string, VerilogFunction*> node_functions_;
};

}  // namespace verilog
}  // namespace xls

#endif  // THIRD_PARTY_XLS_CODEGEN_MODULE_BUILDER_H_
