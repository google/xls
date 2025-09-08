# Copyright 2025 The XLS Authors
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


import dataclasses
import subprocess
from google.protobuf import text_format
from absl.testing import absltest
from xls.codegen import codegen_residual_data_pb2
from xls.codegen import module_signature_pb2
from xls.common import runfiles
from xls.common import test_base


BLOCK_TO_VERILOG_MAIN_PATH = runfiles.get_path(
    'xls/tools/block_to_verilog_main'
)


@dataclasses.dataclass
class NodeMetadata:
  node_name: str
  node_id: int
  emitted_inline: bool = False


BLOCK_IR = '''package add

#[signature("""module_name: "my_function" data_ports { direction: PORT_DIRECTION_INPUT name: "a" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          data_ports { direction: PORT_DIRECTION_INPUT name: "b" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          data_ports { direction: PORT_DIRECTION_OUTPUT name: "out" width: 32 type { type_enum: BITS bit_count: 32 } } 
                                          fixed_latency { latency: 0 } """)]

top block my_function(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=6)
  b: bits[32] = input_port(name=b, id=7)
  neg_a: bits[32] = neg(a, id=8)
  not_b: bits[32] = not(b, id=9)
  id_not_b: bits[32] = identity(not_b, id=10)
  sum: bits[32] = add(neg_a, id_not_b, id=11)
  out: () = output_port(sum, name=out, id=12)
}
'''

INLINE_OR_IR = '''package inline_or

#[signature("""module_name: "inline_or" data_ports { direction: PORT_DIRECTION_INPUT name: "a" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          data_ports { direction: PORT_DIRECTION_INPUT name: "b" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          data_ports { direction: PORT_DIRECTION_OUTPUT name: "out" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          fixed_latency { latency: 0 } """)]

top block inline_or(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=1)
  b: bits[32] = input_port(name=b, id=2)
  and_ab: bits[32] = and(a, b, id=3)
  xor_ab: bits[32] = xor(a, b, id=4)
  not_a: bits[32] = not(a, id=5)
  nand_notab: bits[32] = nand(not_a, b, id=6)
  combined: bits[32] = or(and_ab, xor_ab, not_a, nand_notab, id=7)
  out: () = output_port(combined, name=out, id=8)
}
'''

INSTANTIATION_IR = '''package inst

block sub_block(in: bits[32], out: bits[32]) {
  in: bits[32] = input_port(name=in, id=1)
  out: () = output_port(in, name=out, id=2)
}

// Signature for the top block module interface.
#[signature("""module_name: "my_function" data_ports { direction: PORT_DIRECTION_INPUT name: "a" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          data_ports { direction: PORT_DIRECTION_OUTPUT name: "out" width: 32 type { type_enum: BITS bit_count: 32 } }
                                          fixed_latency { latency: 0 } """)]
top block my_function(a: bits[32], out: bits[32]) {
  instantiation my_inst(block=sub_block, kind=block)
  a: bits[32] = input_port(name=a, id=3)
  a_in: () = instantiation_input(a, instantiation=my_inst, port_name=in, id=4)
  y: bits[32] = instantiation_output(instantiation=my_inst, port_name=out, id=5)
  out: () = output_port(y, name=out, id=6)
}
'''


class BlockToVerilogMainTest(absltest.TestCase):

  def write_residual_data(
      self, block_name: str, node_specs: list[NodeMetadata], filename: str
  ):
    """Writes residual data for a block from a list of NodeMetadata."""
    ref = codegen_residual_data_pb2.CodegenResidualData()
    b = ref.blocks.add(block_name=block_name)
    for spec in node_specs:
      b.nodes.add(
          node_name=spec.node_name,
          node_id=spec.node_id,
          emitted_inline=spec.emitted_inline,
      )
    path = test_base.create_named_output_text_file(filename)
    with open(path, 'w') as f:
      f.write(text_format.MessageToString(ref))
    return path

  def find_line_number(self, text: str, substring: str) -> int:
    """Returns line number of first occurrence of substring, or fails."""
    for i, line in enumerate(text.splitlines(), start=1):
      if substring in line:
        return i
    self.fail(f"Substring '{substring}' not found in provided text")

  def run_with_inline_specs(self, ir_file, block_name: str, node_specs) -> str:
    """Writes residual using `node_specs` and returns Verilog."""
    ref_inline = self.write_residual_data(
        block_name, node_specs, 'residual_data.textproto'
    )
    verilog_text = subprocess.check_output([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--alsologtostderr',
        '--generator=combinational',
        '--reference_residual_data_path=' + ref_inline,
        ir_file.full_path,
    ]).decode('utf-8')
    return verilog_text

  def test_block_ir_generates_verilog(self):
    # Use a simple combinational block IR directly.
    block_ir_file = self.create_tempfile(content=BLOCK_IR)
    verilog_path = test_base.create_named_output_text_file('my_function.v')
    signature_path = test_base.create_named_output_text_file(
        'my_function.sig.textproto'
    )
    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--alsologtostderr',
        '--output_signature_path=' + signature_path,
        '--output_verilog_path=' + verilog_path,
        block_ir_file.full_path,
    ])

    with open(verilog_path, 'r') as f:
      verilog = f.read()
      self.assertIn('module my_function(', verilog)
      self.assertIn('endmodule', verilog)
      self.assertIn('input wire', verilog)
      self.assertIn('output wire', verilog)

    with open(signature_path, 'r') as f:
      sig_proto = text_format.Parse(
          f.read(), module_signature_pb2.ModuleSignatureProto()
      )
      self.assertEqual(sig_proto.module_name, 'my_function')
      self.assertLen(sig_proto.data_ports, 3)
      names = [p.name for p in sig_proto.data_ports]
      self.assertEqual(names, ['a', 'b', 'out'])
      dirs = [p.direction for p in sig_proto.data_ports]
      # 1=input, 2=output
      self.assertEqual(dirs, [1, 1, 2])

  def test_block_ir_with_instantiation(self):
    ir_file = self.create_tempfile(content=INSTANTIATION_IR)
    verilog_path = test_base.create_named_output_text_file('inst_block.v')
    residual_path = test_base.create_named_output_text_file(
        'inst_residual.textproto'
    )

    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--alsologtostderr',
        '--output_residual_data_path=' + residual_path,
        '--output_verilog_path=' + verilog_path,
        ir_file.full_path,
    ])
    with open(verilog_path, 'r') as f:
      verilog = f.read()
      self.assertIn('module my_function(', verilog)
      self.assertIn('module sub_block', verilog)
      self.assertIn('sub_block my_inst', verilog)

    # Verify residual data contains entries for both sub_block and top block.
    with open(residual_path, 'r') as f:
      residual = text_format.Parse(
          f.read(), codegen_residual_data_pb2.CodegenResidualData()
      )
    block_names = {b.block_name for b in residual.blocks}
    self.assertIn('sub_block', block_names)
    self.assertIn('my_function', block_names)

  def test_block_ir_with_specified_order(self):
    block_ir_file = self.create_tempfile(content=BLOCK_IR)

    # Provide name/id pairs corresponding to BLOCK_IR ids.
    ref1_path = self.write_residual_data(
        'my_function',
        [
            NodeMetadata('neg_a', 8),
            NodeMetadata('not_b', 9),
            NodeMetadata('id_not_b', 10),
        ],
        'ref_block_order1.textproto',
    )
    verilog1_path = test_base.create_named_output_text_file('ordered1.v')

    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--separate_lines',
        '--generator=combinational',
        '--output_verilog_path=' + verilog1_path,
        '--reference_residual_data_path=' + ref1_path,
        block_ir_file.full_path,
    ])
    with open(verilog1_path, 'r') as f:
      v1 = f.read()
      # Check the module is emitted and matches expected top name.
      self.assertIn('module my_function(', v1)
      self.assertLess(
          self.find_line_number(v1, 'assign neg_a ='),
          self.find_line_number(v1, 'assign not_b ='),
      )
      self.assertLess(
          self.find_line_number(v1, 'assign not_b ='),
          self.find_line_number(v1, 'assign id_not_b ='),
      )

    ref2_path = self.write_residual_data(
        'my_function',
        [
            NodeMetadata('not_b', 9),
            NodeMetadata('neg_a', 8),
            NodeMetadata('id_not_b', 10),
        ],
        'ref_block_order2.textproto',
    )
    verilog2_path = test_base.create_named_output_text_file('ordered2.v')

    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--separate_lines',
        '--generator=combinational',
        '--output_verilog_path=' + verilog2_path,
        '--reference_residual_data_path=' + ref2_path,
        block_ir_file.full_path,
    ])
    with open(verilog2_path, 'r') as f:
      v2 = f.read()
      self.assertIn('module my_function(', v2)
      self.assertLess(
          self.find_line_number(v2, 'assign not_b ='),
          self.find_line_number(v2, 'assign neg_a ='),
      )
      self.assertLess(
          self.find_line_number(v2, 'assign neg_a ='),
          self.find_line_number(v2, 'assign id_not_b ='),
      )

  def test_block_ir_residual_roundtrip(self):
    # Emit residual in first invocation, consume as reference in second.
    block_ir_file = self.create_tempfile(content=BLOCK_IR)

    residual_path = test_base.create_named_output_text_file(
        'residual_roundtrip.textproto'
    )
    verilog1_path = test_base.create_named_output_text_file('roundtrip1.v')

    # First run: write Verilog and residual data.
    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--alsologtostderr',
        '--generator=combinational',
        '--output_verilog_path=' + verilog1_path,
        '--output_residual_data_path=' + residual_path,
        block_ir_file.full_path,
    ])

    # Verify residual contents and node count.
    with open(residual_path, 'r') as f:
      residual = text_format.Parse(
          f.read(), codegen_residual_data_pb2.CodegenResidualData()
      )
      self.assertLen(residual.blocks, 1)

    # Second run: consume the residual as reference and ensure identical output.
    verilog2_path = test_base.create_named_output_text_file('roundtrip2.v')
    subprocess.check_call([
        BLOCK_TO_VERILOG_MAIN_PATH,
        '--alsologtostderr',
        '--generator=combinational',
        '--output_verilog_path=' + verilog2_path,
        '--reference_residual_data_path=' + residual_path,
        block_ir_file.full_path,
    ])

    with open(verilog1_path, 'r') as f1, open(verilog2_path, 'r') as f2:
      self.assertEqual(f1.read(), f2.read())

  def test_pipeline_generator_rejects_reference_residual(self):
    # Using reference residual data with pipeline generator should fail.
    block_ir_file = self.create_tempfile(content=BLOCK_IR)

    ref_path = self.write_residual_data(
        'my_function',
        [
            NodeMetadata('neg_a', 8),
            NodeMetadata('not_b', 9),
            NodeMetadata('id_not_b', 10),
        ],
        'ref_pipeline_order.textproto',
    )

    proc = subprocess.run(
        [
            BLOCK_TO_VERILOG_MAIN_PATH,
            '--generator=pipeline',
            '--reference_residual_data_path=' + ref_path,
            block_ir_file.full_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    self.assertNotEqual(proc.returncode, 0)
    self.assertIn('not supported when generating pipelines', proc.stderr)

  def test_inline_all_expressions(self):
    ir_file = self.create_tempfile(content=INLINE_OR_IR)
    v = self.run_with_inline_specs(
        ir_file,
        'inline_or',
        [
            NodeMetadata('and_ab', 3, True),
            NodeMetadata('xor_ab', 4, True),
            NodeMetadata('not_a', 5, True),
            NodeMetadata('nand_notab', 6, True),
        ],
    )
    self.assertIn('assign combined = a & b | a ^ b | ~a | ~(~a & b)', v)

  def test_inline_some_expressions(self):
    ir_file = self.create_tempfile(content=INLINE_OR_IR)
    v = self.run_with_inline_specs(
        ir_file,
        'inline_or',
        [
            NodeMetadata('and_ab', 3, True),
            NodeMetadata('xor_ab', 4, False),
            NodeMetadata('not_a', 5, True),
            NodeMetadata('nand_notab', 6, False),
        ],
    )
    self.assertIn('assign xor_ab = a ^ b', v)
    self.assertIn('assign nand_notab = ~(~a & b)', v)
    self.assertIn('assign combined = a & b | xor_ab | ~a | nand_notab', v)

  def test_inline_alternate_subset(self):
    ir_file = self.create_tempfile(content=INLINE_OR_IR)
    v = self.run_with_inline_specs(
        ir_file,
        'inline_or',
        [
            NodeMetadata('and_ab', 3, False),
            NodeMetadata('xor_ab', 4, True),
            NodeMetadata('not_a', 5, False),
            NodeMetadata('nand_notab', 6, True),
        ],
    )
    self.assertIn('assign and_ab = a & b', v)
    self.assertIn('assign not_a = ~a', v)
    self.assertIn('assign combined = and_ab | a ^ b | not_a | ~(not_a & b)', v)

  def test_inline_none(self):
    ir_file = self.create_tempfile(content=INLINE_OR_IR)
    v = self.run_with_inline_specs(
        ir_file,
        'inline_or',
        [
            NodeMetadata('and_ab', 3, False),
            NodeMetadata('xor_ab', 4, False),
            NodeMetadata('not_a', 5, False),
            NodeMetadata('nand_notab', 6, False),
        ],
    )
    self.assertIn('assign and_ab = a & b', v)
    self.assertIn('assign xor_ab = a ^ b', v)
    self.assertIn('assign not_a = ~a', v)
    self.assertIn('assign nand_notab = ~(not_a & b)', v)
    self.assertIn('assign combined = and_ab | xor_ab | not_a | nand_notab', v)


if __name__ == '__main__':
  absltest.main()
