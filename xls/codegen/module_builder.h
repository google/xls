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

#ifndef XLS_CODEGEN_MODULE_BUILDER_H_
#define XLS_CODEGEN_MODULE_BUILDER_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/node_representation.h"
#include "xls/codegen/vast/vast.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_query_engine.h"

namespace xls {
namespace verilog {
// Properties for selectors that are relevant for codegen.
struct SelectorProperties {
  bool never_zero;
};

// An abstraction wrapping a VAST module which assists with lowering of XLS IR
// into Verilog. Key functionality:
//   (1) Handles mapping of XLS types to Verilog types.
//   (2) Hides Verilog vs. SystemVerilog differences and enables targeting
//       either from same code.
//   (3) Imposes common organization to the module.
class ModuleBuilder {
 public:
  ModuleBuilder(std::string_view name, VerilogFile* file,
                CodegenOptions options,
                std::optional<std::string_view> clk_name = std::nullopt,
                std::optional<ResetProto> rst = std::nullopt);

  // Returns the underlying module being constructed.
  Module* module() { return module_; }

  // Add an input port of the given XLS type to the module.
  absl::StatusOr<LogicRef*> AddInputPort(
      std::string_view name, Type* type,
      std::optional<std::string_view> sv_type = std::nullopt);

  // Add an input port of the given width.
  LogicRef* AddInputPort(
      std::string_view name, int64_t bit_count,
      std::optional<std::string_view> sv_type = std::nullopt);

  // Add an output port of the given XLS type to the module. The output is
  // assigned the given value.
  absl::Status AddOutputPort(
      std::string_view name, Type* type, Expression* value,
      std::optional<std::string_view> sv_type = std::nullopt);

  // Add an output port of the given width to the module. The output is assigned
  // the given value.
  absl::Status AddOutputPort(
      std::string_view name, int64_t bit_count, Expression* value,
      std::optional<std::string_view> sv_type = std::nullopt);

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
                                 std::optional<absl::Span<Node* const>>
                                     users_of_expression = std::nullopt);

  // Returns the given node as a Verilog expression. 'inputs' contains the
  // operand expressions for the node.
  absl::StatusOr<Expression*> EmitAsInlineExpression(
      Node* node, absl::Span<Expression* const> inputs);

  // Emits the node as one or more assignments to a newly declared variable with
  // the given name. 'inputs' contains the operand expressions for the
  // node. Returns a reference to the declared variable.
  absl::StatusOr<LogicRef*> EmitAsAssignment(
      std::string_view name, Node* node, absl::Span<Expression* const> inputs);

  // Emit an XLS assert operation as a SystemVerilog assert statement. If
  // SystemVerilog is not enabled then this is a nop.
  absl::StatusOr<NodeRepresentation> EmitAssert(xls::Assert* asrt,
                                                Expression* condition);

  // Emit an IR trace operation as a Verilog $display statement.
  absl::StatusOr<Display*> EmitTrace(xls::Trace* trace, Expression* condition,
                                     absl::Span<Expression* const> trace_args);

  // Emit an XLS cover statement as a SystemVerilog `cover property` statement.
  // If SystemVerilog is not enabled, then this is a nop. Note that the emitted
  // statement will have no effect unless a clock is present in the module.
  absl::StatusOr<NodeRepresentation> EmitCover(xls::Cover* cover,
                                               Expression* condition);

  // Emit a gate operation.
  absl::StatusOr<IndexableExpression*> EmitGate(xls::Gate* gate,
                                                Expression* condition,
                                                Expression* data);

  // Declares a variable with the given name and XLS type. Returns a reference
  // to the variable.
  LogicRef* DeclareVariable(std::string_view name, Type* type);

  // Declares a flat variable with the given name and number of bits. Returns a
  // reference to the variable.
  LogicRef* DeclareVariable(std::string_view name, int64_t bit_count);

  // Assigns the rhs to the lhs using continuous assignment where both sides
  // have the given XLS type. The emitted verilog may require multiple
  // assignment statements for compound types such as arrays.
  absl::Status Assign(LogicRef* lhs, Expression* rhs, Type* type);

  // Declares variable with the given name and assigns the given value to
  // it. Returns a reference to the variable.
  absl::StatusOr<LogicRef*> DeclareModuleConstant(std::string_view name,
                                                  const Value& Value);

  // Data structure describing a register (collection of flops).
  struct Register {
    // Reference to the declared logic/reg variable holding the register value.
    LogicRef* ref;

    // The expression to assign to the register at each clock.
    Expression* next;

    // The register value upon reset. Should be non-null iff the module has a
    // reset signal.
    Expression* reset_value;

    // The load enable signal for the register. Can be null.
    Expression* load_enable;

    // Optional XLS type of this register. Can be null.
    Type* xls_type;
  };

  // Declares a register of the given XLS type. Arguments:
  //   name: name of the declared Verilog register.
  //   type: XLS type of the register.
  //   next: The expression to assign to the register each clock. If not
  //     specified then the returned Register's 'next' field must be filled in
  //     prior to calling AssignRegisters.
  //   reset_value: The value of the register on reset. Should be non-null iff
  //     the module has a reset signal.
  //
  // Declared registers must be passed to a subsequent AssignRegisters call for
  // assignment within an always block.
  absl::StatusOr<Register> DeclareRegister(std::string_view name, Type* type,
                                           Expression* next,
                                           Expression* reset_value = nullptr);

  // As above, but declares a register of a given bit width.
  absl::StatusOr<Register> DeclareRegister(std::string_view name,
                                           int64_t bit_count, Expression* next,
                                           Expression* reset_value = nullptr);

  // Construct an always block to assign values to the registers.
  absl::Status AssignRegisters(absl::Span<const Register> registers);

  // For organization (not functionality) the module is divided into several
  // sections. The emitted module has the following structure:
  //
  //   { includes_section }
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
  //      { instantiation, assert, cover, etc. sections }
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

  ModuleSection* include_section() const { return includes_section_; }
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
  ModuleSection* instantiation_section() const {
    return instantiation_section_;
  }
  ModuleSection* assert_section() const { return assert_section_; }
  ModuleSection* cover_section() const { return cover_section_; }
  ModuleSection* output_section() const { return output_section_; }
  ModuleSection* trace_section() const { return trace_section_; }

  // Return clock signal. Is null if the module does not have a clock.
  LogicRef* clock() const { return clk_; }

  // Returns reset signal and reset metadata.
  const std::optional<Reset>& reset() const { return rst_; }

  VerilogFile* file() const { return file_; }

 private:
  // Assigns 'rhs' to 'lhs'. Depending upon the type this may require multiple
  // assignment statements (e.g., for array assignments in Verilog). The
  // function add_assignment should add a single assignment
  // statement. This function argument enables customization of the type of
  // assignment (continuous, blocking, or non-blocking) as well as the location
  // where the assignment statements are added.
  absl::Status AddAssignment(
      Type* xls_type, Expression* lhs, Expression* rhs,
      std::function<void(Expression*, Expression*)> add_assignment);

  // Assigns an expression generated by 'gen_rhs_expr' to 'lhs'. Depending upon
  // the type being assigned this may require multiple assignment
  // statements. For example, arrays in Verilog must be assigned
  // element-by-element. Arguments:
  //
  //   xls_type: The XLS type of the LHS.
  //   lhs: The LHS of the assignment.
  //   inputs: The input expressions used to generate the RHS of the assignment.
  //   gen_rhs_expr: The function used to generate the RHS of the
  //     assignment. The arguments to this function are given by 'inputs' or the
  //     element-wise decomposition of 'inputs'.
  //   add_assignment: A function which emits a single assignment of rhs (second
  //     argument) to lhs (first argument).
  //   sv_array_expr: Whether the expression generated by gen_rhs_expr works on
  //     array-types.
  absl::Status AddAssignmentToGeneratedExpression(
      Type* xls_type, Expression* lhs, absl::Span<Expression* const> inputs,
      std::function<Expression*(absl::Span<Expression* const>)> gen_rhs_expr,
      std::function<void(Expression*, Expression*)> add_assignment,
      bool sv_array_expr);

  // Emits a copy and update of an array as a sequence of assignments. See
  // method definition for details and examples.
  using IndexMatch = std::variant<bool, Expression*>;
  // Bundles an array index with its XLS type.
  struct IndexType {
    Expression* expression;
    BitsType* xls_type;
  };
  absl::Status EmitArrayCopyAndUpdate(IndexableExpression* lhs,
                                      IndexableExpression* rhs,
                                      Expression* update_value,
                                      absl::Span<const IndexType> indices,
                                      IndexMatch index_match, Type* xls_type);

  // Assigns the arbitrarily-typed Value 'value' to 'lhs'. Depending upon the
  // type this may require multiple assignment statements. The function
  // add_assignment should add a single assignment statement.
  absl::Status AddAssignmentFromValue(
      Expression* lhs, const Value& value,
      std::function<void(Expression*, Expression*)> add_assignment);

  // Extracts a slice from the bits-typed 'rhs' and assigns it to 'lhs' in
  // unflattened form.  Depending upon the type this may require multiple
  // assignment statements. The function add_assignment should add a
  // single assignment statement.
  absl::Status AssignFromSlice(
      Expression* lhs, Expression* rhs, Type* xls_type, int64_t slice_start,
      std::function<void(Expression*, Expression*)> add_assignment);

  // Returns true if the node must be emitted as a function.
  bool MustEmitAsFunction(Node* node);

  // Returns the name of the function which implements node. The function name
  // should encapsulate all metainformation about the node (opcode, bitwidth,
  // etc) because a function definition may be reused to implement multiple
  // identical nodes (for example, two different 32-bit multiplies may map to
  // the same function).
  absl::StatusOr<std::string> VerilogFunctionName(Node* node);

  // Defines a function which implements the given node. If a function already
  // exists which implements this node then the existing function is returned.
  absl::StatusOr<VerilogFunction*> DefineFunction(Node* node);

  // Uses query_engine_ to compute selector properties. If query_engine_ is not
  // populated, it will populate first.
  absl::StatusOr<SelectorProperties> GetSelectorProperties(Node* selector);

  std::string module_name_;
  VerilogFile* file_;

  // A dummy package is required to generate types from Values.
  Package package_;

  CodegenOptions options_;

  LogicRef* clk_ = nullptr;
  std::optional<Reset> rst_;

  Module* module_;
  ModuleSection* includes_section_;
  ModuleSection* functions_section_;
  ModuleSection* constants_section_;
  ModuleSection* input_section_;
  ModuleSection* declaration_and_assignment_section_;
  std::vector<ModuleSection*> declaration_subsections_;
  std::vector<ModuleSection*> assignment_subsections_;
  ModuleSection* instantiation_section_;
  ModuleSection* assert_section_;
  ModuleSection* cover_section_;
  ModuleSection* trace_section_;
  ModuleSection* output_section_;

  // Verilog functions defined inside the module. Map is indexed by the function
  // name.
  absl::flat_hash_map<std::string, VerilogFunction*> node_functions_;

  std::optional<BddQueryEngine> query_engine_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_MODULE_BUILDER_H_
