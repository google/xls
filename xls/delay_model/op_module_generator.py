# Lint as: python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates single-op modules for delay characterization."""

import random
import re
import textwrap

from typing import Sequence, Optional, Tuple

from google.protobuf import text_format
from xls.codegen import module_signature_pb2
from xls.codegen.python import module_signature as module_signature_mod
from xls.codegen.python import pipeline_generator as pipeline_generator_mod
from xls.ir.python import ir_parser as ir_parser_mod


def _random_bits_value(width: int) -> str:
  """Returns a random value of the given bit-width as a hexadecimal string."""
  return '0x%x' % random.getrandbits(width)


def _random_array_value(element_width: int, num_elements: int) -> str:
  """Returns a random array value as a string with hexadecimal elements."""
  elements = [_random_bits_value(element_width) for _ in range(num_elements)]
  return '[' + ', '.join(elements) + ']'


def _generate_literal(literal_type: str) -> str:
  """Returns a literal value of the given type."""
  random.seed(0)
  m = re.match(r'bits\[(\d+)\]$', literal_type)
  if m:
    return _random_bits_value(int(m.group(1)))
  else:
    m = re.match(r'bits\[(\d+)\]\[(\d+)\]$', literal_type)
    if not m:
      raise ValueError(f'Invalid or unsupported type {literal_type}')
    return _random_array_value(int(m.group(1)), int(m.group(2)))


def generate_ir_package(op: str,
                        output_type: str,
                        operand_types: Sequence[str],
                        attributes: Sequence[Tuple[str, str]] = (),
                        literal_operand: Optional[int] = None) -> str:
  """Generates an IR package containing a operation of the given op.

  The IR package contains a single function which has an operation with the
  given op. The parameters of the function are the operands of the
  operation. Example output:

    package add_characterization

    fn main(op0: bits[8], op1: bits[8]) -> bits[8] {
      ret add.1: bits[8] = add(op0, op1)
    }

  Arguments:
    op: The op of the operation. For example: "add".
    output_type: The type of the output of the operation. For example:
      "bits[32]".
    operand_types: The types of the output of the operation. For example:
      ("bits[32]", "bits[16]").
    attributes: Attributes to include in the operation mnemonic. For example,
      "new_bit_count" in extend operatoins.
    literal_operand: Optionally specifies that the given operand number should
      be substituted with a randomly generated literal instead of a function
      parameter.

  Returns:
    The text of the IR package.
  """
  params = [
      f'op{i}: {operand_types[i]}' for i in range(len(operand_types))
      if i != literal_operand
  ]
  if literal_operand is None:
    operands = [f'op{i}' for i in range(len(operand_types))]
    ir_text = textwrap.dedent("""\
    package {op}_characterization

    fn main({params}) -> {output_type} {{
      ret {op}.1: {output_type} = {op}({operands}{attributes})
    }}""").format(
        op=op,
        output_type=output_type,
        params=', '.join(params),
        operands=', '.join(operands),
        attributes=''.join(f', {k}={v}' for k, v in attributes))
  else:
    literal_type = operand_types[literal_operand]
    literal_value = _generate_literal(literal_type)
    operands = ('literal.1' if i == literal_operand else f'op{i}'
                for i in range(len(operand_types)))
    ir_text = textwrap.dedent("""\
    package {op}_characterization

    fn main({params}) -> {output_type} {{
      literal.1: {literal_type} = literal(value={literal_value})
      ret {op}.2: {output_type} = {op}({operands}{attributes})
    }}""").format(
        op=op,
        output_type=output_type,
        params=', '.join(params),
        operands=', '.join(operands),
        literal_type=literal_type,
        literal_value=literal_value,
        attributes=''.join(f', {k}={v}' for k, v in attributes))

  # Verify the IR parses and verifies.
  ir_parser_mod.Parser.parse_package(ir_text)
  return ir_text


def generate_verilog_module(
    module_name: str,
    op: str,
    output_type: str,
    operand_types: Sequence[str],
    attributes: Sequence[Tuple[str, str]] = (),
    literal_operand: Optional[int] = None
) -> module_signature_mod.ModuleGeneratorResult:
  """Generates a verilog module with the given properties.

  Most arguments are passed directly to generate_ir_package.

  Arguments:
    module_name: The name of the generated Verilog module.
    op: The op of the operation. For example: "add".
    output_type: The type of the output of the operation. For example:
      "bits[32]".
    operand_types: The types of the output of the operation. For example:
      ("bits[32]", "bits[16]").
    attributes: Attributes to include in the operation mnemonic. For example,
      "new_bit_count" in extend operatoins.
    literal_operand: Optionally specifies that the given operand number should
      be substituted with a randomly generated literal instead of a function
      parameter.

  Returns:
    The module signature and text of the Verilog module.
  """
  ir_text = generate_ir_package(op, output_type, operand_types, attributes,
                                literal_operand)
  package = ir_parser_mod.Parser.parse_package(ir_text)
  module_generator_result = pipeline_generator_mod.generate_pipelined_module_with_n_stages(
      package, 1, module_name)
  return module_generator_result


def generate_parallel_module(modules: Sequence[
    module_signature_mod.ModuleGeneratorResult], module_name: str) -> str:
  """Generates a module composed of instantiated instances of the given modules.

  Each module in 'modules' is instantiated exactly once in a enclosing,
  composite module. Inputs to each instantiation are provided by inputs to the
  enclosing module. For example, if given two modules, add8_module and
  add16_module, the generated module might look like:

    module add8_module(
      input wire clk,
      input wire [7:0] op0,
      input wire [7:0] op1,
      output wire [7:0] out
    );
    // contents of module elided...
    endmodule

    module add16_module(
      input wire clk,
      input wire [15:0] op0,
      input wire [15:0] op1,
      output wire [15:0] out
    );
    // contents of module elided...
    endmodule

    module foo(
      input wire clk,
      input wire [7:0] add8_module_op0,
      input wire [7:0] add8_module_op1,
      output wire [7:0] add8_module_out,
      input wire [15:0] add16_module_op0,
      input wire [15:0] add16_module_op1,
      output wire [15:0] add16_module_out,
    );
    add8_module add8_module_inst(
      .clk(clk),
      .op0(add8_module_op0),
      .op1(add8_module_op1),
      .out(add8_module_out)
    );
    add16_module add16_module_inst(
      .clk(clk),
      .op0(add16_module_op0),
      .op1(add16_module_op1),
      .out(add16_module_out)
    );
  endmodule


  Arguments:
    modules: Modules to include instantiate.
    module_name: Name of the module containing the instantiated input modules.

  Returns:
    Verilog text containing the composite module and component modules.
  """
  module_protos = [
      text_format.Parse(m.signature.as_text_proto(),
                        module_signature_pb2.ModuleSignatureProto())
      for m in modules
  ]
  ports = ['input wire clk']
  for module in module_protos:
    for data_port in module.data_ports:
      width_str = f'[{data_port.width - 1}:0]'
      signal_name = f'{module.module_name}_{data_port.name}'
      if data_port.direction == module_signature_pb2.DIRECTION_INPUT:
        ports.append(f'input wire {width_str} {signal_name}')
      elif data_port.direction == module_signature_pb2.DIRECTION_OUTPUT:
        ports.append(f'output wire {width_str} {signal_name}')
  header = """module {module_name}(\n{ports}\n);""".format(
      module_name=module_name, ports=',\n'.join(f'  {p}' for p in ports))
  instantiations = []
  for module in module_protos:
    connections = ['.clk(clk)']
    for data_port in module.data_ports:
      connections.append(
          f'.{data_port.name}({module.module_name}_{data_port.name})')
    instantiations.append('  {name} {name}_inst(\n{connections}\n  );'.format(
        name=module.module_name,
        connections=',\n'.join(f'    {c}' for c in connections)))
  return '{modules}\n\n{header}\n{instantiations}\nendmodule\n'.format(
      modules='\n\n'.join(m.verilog_text for m in modules),
      header=header,
      instantiations='\n'.join(instantiations))
