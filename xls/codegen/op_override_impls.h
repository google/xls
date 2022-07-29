// Copyright 2022 The XLS Authors
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

#ifndef XLS_CODEGEN_OP_OVERRIDE_IMPLS_H_
#define XLS_CODEGEN_OP_OVERRIDE_IMPLS_H_

#include <array>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/module_builder.h"
#include "xls/codegen/op_override.h"
#include "xls/ir/node.h"

namespace xls::verilog {

class OpOverrideAssignment : public OpOverride {
 public:
  explicit OpOverrideAssignment(absl::string_view fmt_string)
      : assignment_format_string_(fmt_string), placeholder_aliases_({}) {}
  explicit OpOverrideAssignment(
      absl::string_view fmt_string,
      absl::flat_hash_map<std::string, std::string> placeholder_aliases)
      : assignment_format_string_(fmt_string),
        placeholder_aliases_(std::move(placeholder_aliases)) {}

  std::unique_ptr<OpOverride> Clone() const override;
  absl::StatusOr<NodeRepresentation> Emit(
      Node* node, absl::string_view name,
      absl::Span<NodeRepresentation const> inputs, ModuleBuilder& mb) override;

 private:
  std::string assignment_format_string_;
  absl::flat_hash_map<std::string, std::string> placeholder_aliases_;
};

// Uses format string to override gates via an assignment.
//
// Format string supports the following placeholders:
//
//  {condition} : Identifier (or expression) of the condition of the assert.
//  {input}     : The identifier (or expression) for the data input of the gate
//                operation.
//  {output}    : The identifier of the gate operation.
//  {width}     : The bit width of the gate operation.
//
// For example, consider a format string which instantiates a particular
// custom AND gate for gating:
//
//    'my_and gated_{output} [{width}-1:0] (.Z({output}), .A({condition}),
//    .B({input}))'
//
// And the IR gate operations is:
//
//    the_result: bits[32] = gate(the_cond, the_data)
//
// This results in the following emitted Verilog:
//
//    my_and gated_the_result [32-1:0] (.Z(the_result), .A(the cond),
//    .B(the_data));
//
// To ensure valid Verilog, the instantiated template must declare a value
// named {output} (e.g., `the_result` in the example).
//
// If no format value is given, then a logical AND with the condition value is
// generated. For example:
//
//   wire the_result [31:0];
//   assign the_result = {32{the_cond}} & the_data;
class OpOverrideGateAssignment : public OpOverrideAssignment {
 public:
  explicit OpOverrideGateAssignment(absl::string_view fmt_string);
};

// Uses format string to override assert statement codegen.
//
// Format string supports the following placeholders:
//
//  {message}   : Message of the assert operation.
//  {condition} : Condition of the assert.
//  {label}     : Label of the assert operation. Returns error if the operation
//                has no label.
//  {clk}       : Name of the clock signal. Returns error if no clock is
//                specified.
//  {rst}       : Name of the reset signal. Returns error if no reset is
//                specified.
//
// For example, the format string:
//
//    '{label}: `MY_ASSERT({condition}, "{message}")'
//
// Might result in the following in the emitted Verilog:
//
//    my_label: `MY_ASSERT(foo < 8'h42, "Oh noes!");
class OpOverrideAssertion : public OpOverride {
 public:
  explicit OpOverrideAssertion(absl::string_view fmt_string)
      : assertion_format_string_(fmt_string) {}

  std::unique_ptr<OpOverride> Clone() const override;
  absl::StatusOr<NodeRepresentation> Emit(
      Node* node, absl::string_view name,
      absl::Span<NodeRepresentation const> inputs, ModuleBuilder& mb) override;

 private:
  std::string assertion_format_string_;
};

class OpOverrideInstantiation : public OpOverride {
 public:
  explicit OpOverrideInstantiation(absl::string_view fmt_string)
      : instantiation_format_string_(fmt_string) {}

  std::unique_ptr<OpOverride> Clone() const override;
  absl::StatusOr<NodeRepresentation> Emit(
      Node* node, absl::string_view name,
      absl::Span<NodeRepresentation const> inputs, ModuleBuilder& mb) override;

 private:
  std::string instantiation_format_string_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_OP_OVERRIDE_IMPLS_H_
