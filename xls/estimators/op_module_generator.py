# Copyright 2020 The XLS Authors
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

import dataclasses
import random
import re
import subprocess
import tempfile
import textwrap
from typing import Optional, Sequence, Tuple

from google.protobuf import text_format
from xls.codegen import module_signature_pb2
from xls.common import runfiles


CODEGEN_MAIN_PATH = runfiles.get_path('xls/tools/codegen_main')
PARSE_IR_PATH = runfiles.get_path('xls/dev_tools/parse_ir')


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
  m = re.search(r'bits\[(\d+)\]$', literal_type)
  if m:
    return _random_bits_value(int(m.group(1)))
  else:
    m = re.search(r'bits\[(\d+)\]\[(\d+)\]$', literal_type)
    if not m:
      raise ValueError(f'Invalid or unsupported type {literal_type}')
    return _random_array_value(int(m.group(1)), int(m.group(2)))


def generate_ir_package(
    op: str,
    output_type: str,
    operand_types: Sequence[str],
    attributes: Sequence[Tuple[str, str]] = (),
    literal_operand: Optional[int] = None,
    repeated_operand: Optional[int] = None,
) -> str:
  """Generates an IR package containing a operation of the given op.

  The IR package contains a single function which has an operation with the
  given op. The parameters of the function are the operands of the
  operation. Example output:

    package add_characterization

    fn main(op0: bits[8], op1: bits[8]) -> bits[8] {
      ret result: bits[8] = add(op0, op1)
    }

  Arguments:
    op: The op of the operation. For example: "add".
    output_type: The type of the output of the operation. For example:
      "bits[32]".
    operand_types: The types of the operands of the operation. For example:
      ("bits[32]", "bits[16]").
    attributes: Attributes to include in the operation mnemonic. For example,
      "new_bit_count" in extend operations.
    literal_operand: Optionally specifies that the given operand number should
      be substituted with a randomly generated literal instead of a function
      parameter.
    repeated_operand: Optionally specifies that the given operand number should
      be substituted with the previous operand.

  Returns:
    The text of the IR package.
  """
  params = [
      f'op{i}: {operand_types[i]}'
      for i in range(len(operand_types))
      if (i != literal_operand and i != repeated_operand)
  ]
  # Some ops have named operands which appear in the argument list as
  # attributes. For example, the 'indices' attributes of array_index:
  #
  #   array_index: bits[32] = array_index(a, indices=[i, j])
  #
  # Extract these out as a separate element in the argument list.
  if repeated_operand:
    args = [
        'op' + str(i - 1 if i == repeated_operand else i)
        for i in range(len(operand_types))
    ]
  else:
    args = [f'op{i}' for i in range(len(operand_types))]
  if op == 'array_index':
    indices = args[1:]
    args = args[0:1]
    args.append('indices=[%s]' % ', '.join(indices))
  elif op == 'array_update':
    indices = args[2:]
    args = args[0:2]
    args.append('indices=[%s]' % ', '.join(indices))
  elif op == 'sel':
    # Determine number of selector bits.
    selector_type = operand_types[0]
    m = re.search(r'bits\[(\d+)\]$', selector_type)
    if not m:
      raise ValueError(
          'Invalid or unsupported type forsel op selector {selector_type}'
      )
    num_selector_bits = int(m.group(1))

    # Set default operand as necessary.
    num_addressable_cases = 2**num_selector_bits
    if num_addressable_cases > len(args) - 1:
      cases = args[1:-1]
      default = args[-1]
      args = args[0:1]
      args.append('cases=[%s]' % ', '.join(cases))
      args.append('default=%s' % default)
    else:
      cases = args[1:]
      args = args[0:1]
      args.append('cases=[%s]' % ', '.join(cases))
  elif op == 'one_hot_sel':
    cases = args[1:]
    args = args[0:1]
    args.append('cases=[%s]' % ', '.join(cases))
  elif op == 'priority_sel':
    cases = args[1:-1]
    default = args[-1]
    args = args[0:1]
    args.append('cases=[%s]' % ', '.join(cases))
    args.append('default=%s' % default)

  args.extend(f'{k}={v}' for k, v in attributes)

  # Special case for kUMulp / kSMulp (dual results)
  if op == 'umulp' or op == 'smulp':
    output_type = f'({output_type}, {output_type})'

  if literal_operand is None:
    ir_text = textwrap.dedent("""\
    package {op}_characterization

    top fn main({params}) -> {output_type} {{
      ret result: {output_type} = {op}({args})
    }}""").format(
        op=op,
        output_type=output_type,
        params=', '.join(params),
        args=', '.join(args),
    )
  else:
    literal_type = operand_types[literal_operand]
    literal_value = _generate_literal(literal_type)
    ir_text = textwrap.dedent("""\
    package {op}_characterization

    top fn main({params}) -> {output_type} {{
      op{literal_operand}: {literal_type} = literal(value={literal_value})
      ret result: {output_type} = {op}({args})
    }}""").format(
        op=op,
        output_type=output_type,
        params=', '.join(params),
        literal_type=literal_type,
        literal_value=literal_value,
        literal_operand=literal_operand,
        args=', '.join(args),
    )

  # Verify the IR parses and verifies.
  with tempfile.NamedTemporaryFile(suffix='.ir', mode='w') as ir_file:
    ir_file.write(ir_text)
    ir_file.flush()
    parse_ir = subprocess.Popen(
        [PARSE_IR_PATH, ir_file.name],
        stderr=subprocess.PIPE,
    )
    _, stderr = parse_ir.communicate()
    if parse_ir.returncode != 0:
      raise ValueError(stderr)
  return ir_text


@dataclasses.dataclass(frozen=True)
class ModuleGeneratorResult:
  signature: module_signature_pb2.ModuleSignatureProto
  verilog_text: str


def generate_verilog_module(
    module_name: str,
    ir_text: str,
) -> ModuleGeneratorResult:
  """Generates a verilog module for the given IR.

  Arguments:
    module_name: The name of the generated Verilog module.
    ir_text: The XLS IR from which to generate Verilog.

  Returns:
    The module signature and Verilog text (as ModuleGeneratorResult).
  """
  with tempfile.NamedTemporaryFile(
      prefix=f'{module_name}', suffix='.ir', mode='w'
  ) as ir_file, tempfile.NamedTemporaryFile(
      prefix=f'{module_name}', suffix='.txtpb', mode='r'
  ) as module_signature_file, tempfile.NamedTemporaryFile(
      prefix=f'{module_name}', suffix='.v', mode='r'
  ) as verilog_file:
    ir_file.write(ir_text)
    ir_file.flush()
    subprocess.check_call([
        CODEGEN_MAIN_PATH,
        '--generator=pipeline',
        '--pipeline_stages=1',
        '--delay_model=unit',
        '--nouse_system_verilog',
        f'--module_name={module_name}',
        f'--output_signature_path={module_signature_file.name}',
        f'--output_verilog_path={verilog_file.name}',
        ir_file.name,
    ])
    module_signature = module_signature_file.read()
    verilog_text = verilog_file.read()
    return ModuleGeneratorResult(
        text_format.Parse(
            module_signature, module_signature_pb2.ModuleSignatureProto()
        ),
        verilog_text,
    )


def generate_parallel_module(
    modules: Sequence[ModuleGeneratorResult], module_name: str
) -> str:
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
  module_protos = [m.signature for m in modules]
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
      module_name=module_name, ports=',\n'.join(f'  {p}' for p in ports)
  )
  instantiations = []
  for module in module_protos:
    connections = ['.clk(clk)']
    for data_port in module.data_ports:
      connections.append(
          f'.{data_port.name}({module.module_name}_{data_port.name})'
      )
    instantiations.append(
        '  {name} {name}_inst(\n{connections}\n  );'.format(
            name=module.module_name,
            connections=',\n'.join(f'    {c}' for c in connections),
        )
    )
  return '{modules}\n\n{header}\n{instantiations}\nendmodule\n'.format(
      modules='\n\n'.join(m.verilog_text for m in modules),
      header=header,
      instantiations='\n'.join(instantiations),
  )
