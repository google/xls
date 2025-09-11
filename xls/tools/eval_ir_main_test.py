#
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

import ctypes
import struct
import subprocess

from google.protobuf import text_format
from absl.testing import parameterized
from xls.common import runfiles
from xls.common import test_base
from xls.ir import evaluator_result_pb2
from xls.ir import xls_value_pb2
from xls.tools import node_coverage_stats_pb2

EVAL_IR_MAIN_PATH = runfiles.get_path('xls/tools/eval_ir_main')

ADD_IR = """package foo

top fn foo(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
"""

TUPLE_IR = """package foo

top fn foo(x: (bits[8], bits[32])) -> ((bits[8], bits[32])) {
  ret tuple.1: ((bits[8], bits[32])) = tuple(x)
}
"""

FAIL_IR = """package foo
top fn foo(x: bits[32], y:bits[32]) -> bits[64] {
  ret smul.1: bits[64] = smul(x, y)
}
"""


def _value_32_bits(v: int) -> xls_value_pb2.ValueProto:
  return xls_value_pb2.ValueProto(
      bits=xls_value_pb2.ValueProto.Bits(
          bit_count=32, data=struct.pack('<i', v)
      )
  )


def parameterized_proc_backends(func):
  return parameterized.named_parameters(
      ('jit', ['--use_llvm_jit']),
      ('interpreter', ['--nouse_llvm_jit']),
  )(func)


class EvalMainTest(parameterized.TestCase):

  def test_one_input_jit(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--input=bits[32]:0x42; bits[32]:0x123',
        '--use_llvm_jit=true',
        ir_file.full_path,
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[32]:0x165')

  def test_one_input_nojit(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--input=bits[32]:0x42; bits[32]:0x123',
        '--use_llvm_jit=false',
        ir_file.full_path,
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[32]:0x165')

  def test_one_input_jit_with_vlog(self):
    # Checks that enabling vlog doesn't crash.
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '-v=5',
        '--input=bits[32]:0x42; bits[32]:0x123',
        '--use_llvm_jit=true',
        ir_file.full_path,
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[32]:0x165')

  def test_one_input_nojit_with_vlog(self):
    # Checks that enabling vlog doesn't crash.
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '-v=5',
        '--input=bits[32]:0x42; bits[32]:0x123',
        '--use_llvm_jit=false',
        ir_file.full_path,
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[32]:0x165')

  def test_input_missing_arg(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    comp = subprocess.run(
        [EVAL_IR_MAIN_PATH, '--input=bits[32]:0x42', ir_file.full_path],
        stderr=subprocess.PIPE,
        check=False,
    )
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn(
        "Arg list to 'foo' has the wrong size: 1 vs expected 2",
        comp.stderr.decode('utf-8'),
    )

  def test_one_input_with_expected(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--input=bits[32]:0x42; bits[32]:0x123',
        '--expected=bits[32]:0x165',
        ir_file.full_path,
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[32]:0x165')

  def test_one_input_with_failed_expected(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    comp = subprocess.run(
        [
            EVAL_IR_MAIN_PATH,
            '--input=bits[32]:0x42; bits[32]:0x123',
            '--expected=bits[32]:0x123',
            ir_file.full_path,
        ],
        stderr=subprocess.PIPE,
        check=False,
    )
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn(
        'Miscompare for input[0] "bits[32]:0x42; bits[32]:0x123"',
        comp.stderr.decode('utf-8'),
    )

  def test_input_file(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    input_file = self.create_tempfile(
        content='\n'.join(
            ('bits[32]:0x42; bits[32]:0x123', 'bits[32]:0x10; bits[32]:0xf0f')
        )
    )
    results = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--input_file=' + input_file.full_path,
        ir_file.full_path,
    ])
    self.assertSequenceEqual(
        ('bits[32]:0x165', 'bits[32]:0xf1f'),
        results.decode('utf-8').strip().split('\n'),
    )

  def test_input_file_extra_whitespace(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    # Empty lines and extra whitespace in the arg file should be ignored.
    input_file = self.create_tempfile(
        content='\n'.join((
            'bits[32]:0x42; bits[32]:0x123',
            '',
            'bits[32]:0x10; bits[32]:0xf0f',
            '',
        ))
    )
    results = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--input_file=' + input_file.full_path,
        ir_file.full_path,
    ])
    self.assertSequenceEqual(
        ('bits[32]:0x165', 'bits[32]:0xf1f'),
        results.decode('utf-8').strip().split('\n'),
    )

  def test_input_file_with_expected_file(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    input_file = self.create_tempfile(
        content='\n'.join(
            ('bits[32]:0x42; bits[32]:0x123', 'bits[32]:0x10; bits[32]:0xf0f')
        )
    )
    expected_file = self.create_tempfile(
        content='\n'.join(('bits[32]:0x165', 'bits[32]:0xf1f'))
    )
    results = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--input_file=' + input_file.full_path,
        '--expected_file=' + expected_file.full_path,
        ir_file.full_path,
    ])
    self.assertSequenceEqual(
        ('bits[32]:0x165', 'bits[32]:0xf1f'),
        results.decode('utf-8').strip().split('\n'),
    )

  def test_input_file_with_failed_expected_file(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    input_file = self.create_tempfile(
        content='\n'.join(
            ('bits[32]:0x42; bits[32]:0x123', 'bits[32]:0x10; bits[32]:0x00')
        )
    )
    expected_file = self.create_tempfile(
        content='\n'.join(('bits[32]:0x165', 'bits[32]:0xf1f'))
    )
    comp = subprocess.run(
        [
            EVAL_IR_MAIN_PATH,
            '--input_file=' + input_file.full_path,
            '--expected_file=' + expected_file.full_path,
            ir_file.full_path,
        ],
        stderr=subprocess.PIPE,
        check=False,
    )
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn(
        'Miscompare for input[1] "bits[32]:0x10; bits[32]:0x0"',
        comp.stderr.decode('utf-8'),
    )

  def test_empty_input_file(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    input_file = self.create_tempfile(content='')
    results = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--input_file=' + input_file.full_path,
        ir_file.full_path,
    ])
    self.assertEqual(results.decode('utf-8'), '')

  def test_tuple_in_out(self):
    ir_file = self.create_tempfile(content=TUPLE_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--input=(bits[8]:0x42, bits[32]:0x123)',
        ir_file.full_path,
    ])
    self.assertEqual(
        result.decode('utf-8').strip(), '((bits[8]:0x42, bits[32]:0x123))'
    )

  def test_random_inputs(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output(
        [EVAL_IR_MAIN_PATH, '--random_inputs=42', ir_file.full_path]
    )
    # There should be 42 results.
    self.assertLen(result.decode('utf-8').strip().split('\n'), 42)
    # And with overwhelming probability they should all be different.
    self.assertLen(set(result.decode('utf-8').strip().split('\n')), 42)

  def test_output_results_proto_with_trace(self):
    trace_ir = """package foo

top fn foo(x: bits[8]) -> bits[8] {
  tkn: token = literal(value=token)
  on: bits[1] = literal(value=1)
  trace.1: token = trace(tkn, on, format="x is {:x}", data_operands=[x])
  ret identity.2: bits[8] = identity(x)
}
"""
    ir_file = self.create_tempfile(content=trace_ir)
    out = self.create_tempfile()
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--nouse_llvm_jit',
        '--input=bits[8]:0x2a',
        f'--output_results_proto={out.full_path}',
        ir_file.full_path,
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[8]:0x2a')
    # Parse textproto and verify trace message
    results_proto = evaluator_result_pb2.EvaluatorResultsProto()
    text_format.Parse(out.read_text(), results_proto)
    self.assertLen(results_proto.results, 1)
    # Verify result matches bits[8]:0x2a
    res = results_proto.results[0].result
    self.assertTrue(res.HasField('bits'))
    self.assertEqual(res.bits.bit_count, 8)
    self.assertEqual(res.bits.data, struct.pack('b', 0x2A))
    self.assertGreaterEqual(len(results_proto.results[0].events.trace_msgs), 1)
    self.assertEqual(
        results_proto.results[0].events.trace_msgs[0].message, 'x is 2a'
    )

  def test_trace_to_stderr(self):
    trace_ir = """package foo

top fn foo(x: bits[8]) -> bits[8] {
  tkn: token = literal(value=token)
  on: bits[1] = literal(value=1)
  trace.1: token = trace(tkn, on, format="x is {:x}", data_operands=[x])
  ret identity.2: bits[8] = identity(x)
}
"""
    ir_file = self.create_tempfile(content=trace_ir)
    comp = subprocess.run(
        [
            EVAL_IR_MAIN_PATH,
            '--nouse_llvm_jit',
            '--input=bits[8]:0x2a',
            '--trace_to_stderr',
            ir_file.full_path,
        ],
        stderr=subprocess.PIPE,
        check=True,
    )
    self.assertIn('x is 2a', comp.stderr.decode('utf-8'))

  def test_jit_result_injection(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--input=bits[32]:0x42; bits[32]:0x123',
        '--test_only_inject_jit_result=bits[32]:0x22',
        '--use_llvm_jit=true',
        ir_file.full_path,
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[32]:0x22')

  def test_test_llvm_jit_no_mismatch(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    comp = subprocess.run(
        [
            EVAL_IR_MAIN_PATH,
            '--input=bits[32]:0x42; bits[32]:0x123',
            '--test_llvm_jit',
            ir_file.full_path,
        ],
        check=False,
    )
    self.assertEqual(comp.returncode, 0)

  def test_test_llvm_jit_mismatch(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    comp = subprocess.run(
        [
            EVAL_IR_MAIN_PATH,
            '--input=bits[32]:0x42; bits[32]:0x123',
            '--test_llvm_jit',
            '--test_only_inject_jit_result=bits[32]:0x22',
            ir_file.full_path,
        ],
        stderr=subprocess.PIPE,
        check=False,
    )
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn(
        'Miscompare for input[0] "bits[32]:0x42; bits[32]:0x123"',
        comp.stderr.decode('utf-8'),
    )

  def test_validator(self):
    # We want to ensure that the output is negative and odd, so the inputs
    # must have different signs and must both be odd.
    input_validator = """fn validator(x: s32, y:s32) -> bool {
  let same_sign = (x < s32:0 && y < s32:0) || (x > s32:0 && y > s32:0);
  let x_odd = (x & s32:1) as bool;
  let y_odd = (y & s32:1) as bool;
  let both_odd = x_odd & y_odd;
  !same_sign && both_odd
}"""

    ir_file = self.create_tempfile(content=FAIL_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--input_validator_expr={}'.format(input_validator),
        '--random_inputs=1024',
        ir_file.full_path,
    ])
    products = result.decode('utf-8').split()
    for product in products:
      self.assertStartsWith(product, 'bits[64]:0x')
      value = ctypes.c_longlong(int(product[len('bits[64]:') :], 16)).value
      self.assertLess(value, 0, f'value should be negative: {product}')
      self.assertTrue(value % 2, f'value should be odd: {product}')

  def test_validator_fails(self):
    input_validator = """fn validator(x: s32, y:s32) -> bool { false }"""

    ir_file = self.create_tempfile(content=FAIL_IR)
    comp = subprocess.run(
        [
            EVAL_IR_MAIN_PATH,
            '--input_validator_expr={}'.format(input_validator),
            '--random_inputs=1024',
            ir_file.full_path,
        ],
        stderr=subprocess.PIPE,
        check=False,
    )
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn('Unable to generate valid input', comp.stderr.decode('utf-8'))

  @parameterized_proc_backends
  def test_coverage(self, backend):
    ir_file = self.create_tempfile(content=ADD_IR)
    cov = self.create_tempfile()
    subprocess.run(
        [
            EVAL_IR_MAIN_PATH,
            ir_file.full_path,
            '--input',
            'bits[32]:0x5; bits[32]:0xC',
            '--expected=bits[32]:0x11',
            '--alsologtostderr',
            f'--output_node_coverage_stats_proto={cov.full_path}',
        ]
        + backend,
        check=True,
    )
    node_coverage = node_coverage_stats_pb2.NodeCoverageStatsProto.FromString(
        cov.read_bytes()
    )
    node_stats = node_coverage_stats_pb2.NodeCoverageStatsProto.NodeStats
    node_coverage.nodes.sort(key=lambda n: n.node_id)
    self.assertSequenceEqual(
        list(node_coverage.nodes),
        [
            node_stats(
                node_id=1,
                node_text='add.1: bits[32] = add(x, y, id=1)',
                set_bits=_value_32_bits(0x11),
                unset_bit_count=30,
                total_bit_count=32,
            ),
            node_stats(
                node_id=4,
                node_text='x: bits[32] = param(name=x, id=4)',
                set_bits=_value_32_bits(0x5),
                unset_bit_count=30,
                total_bit_count=32,
            ),
            node_stats(
                node_id=5,
                node_text='y: bits[32] = param(name=y, id=5)',
                set_bits=_value_32_bits(0xC),
                unset_bit_count=30,
                total_bit_count=32,
            ),
        ],
    )


if __name__ == '__main__':
  test_base.main()
