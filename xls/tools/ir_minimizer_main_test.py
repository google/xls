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

import os
import stat
import subprocess

from xls.common import runfiles
from absl.testing import absltest

IR_MINIMIZER_MAIN_PATH = runfiles.get_path('xls/tools/ir_minimizer_main')

ADD_IR = """package foo

top fn foo(x: bits[32], y: bits[32]) -> bits[32] {
  not.1: bits[32] = not(x, id=1)
  add.2: bits[32] = add(not.1, y, id=2)
  ret not.3: bits[32] = not(add.2, id=3)
}
"""


class IrMinimizerMainTest(absltest.TestCase):

  def _write_sh_script(self, path, commands):
    with open(path, 'w') as f:
      f.write('#!/bin/sh -e\n')
      f.write('\n'.join(commands))
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR)

  def test_minimize_add_no_remove_params(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, ['/bin/grep add $1'])
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH, '--test_executable=' + test_sh_file.full_path,
        '--can_remove_params=false', ir_file.full_path
    ])
    self.assertEqual(
        minimized_ir.decode('utf-8'), """package foo

top fn foo(x: bits[32], y: bits[32]) -> bits[32] {
  literal.9: bits[32] = literal(value=0, id=9)
  ret add.2: bits[32] = add(literal.9, literal.9, id=2)
}
""")

  def test_minimize_add_remove_params(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, ['/bin/grep add $1'])
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH, '--test_executable=' + test_sh_file.full_path,
        '--can_remove_params', ir_file.full_path
    ])
    self.assertEqual(
        minimized_ir.decode('utf-8'), """package foo

top fn foo() -> bits[32] {
  literal.11: bits[32] = literal(value=0, id=11)
  ret add.2: bits[32] = add(literal.11, literal.11, id=2)
}
""")

  def test_no_reduction_possible(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    # Shell script is run with -e so if any of the greps fail then the script
    # fails.
    self._write_sh_script(test_sh_file.full_path, [
        '/bin/grep not.1.*x $1', '/bin/grep add.2.*not.1.*y $1',
        '/bin/grep not.3.*add.2 $1'
    ])
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH, '--test_executable=' + test_sh_file.full_path,
        '--can_remove_params', ir_file.full_path
    ])
    self.assertEqual(minimized_ir.decode('utf-8'), ADD_IR)

  def test_simplify_and_unbox_array(self):
    input_ir = """package foo

top fn foo(x: bits[32], y: bits[32]) -> bits[32][3] {
  not: bits[32] = not(x, id=3)
  ret a: bits[32][3] = array(x, y, not)
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    # The test script only checks to see if a not(x) instruction is in the IR.
    self._write_sh_script(test_sh_file.full_path, ['/bin/grep not.*x $1'])
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH, '--test_executable=' + test_sh_file.full_path,
        ir_file.full_path
    ])
    # The array operation should have been stripped from the function.
    self.assertIn('array(', input_ir)
    self.assertNotIn('array(', minimized_ir.decode('utf-8'))

  def test_simplify_tuple(self):
    input_ir = """package foo

top fn foo(x: bits[32], y: bits[32], z: bits[32]) -> (bits[32], (bits[32], bits[32]), bits[32]) {
  tmp: (bits[32], bits[32]) = tuple(y, x)
  ret a: (bits[32], (bits[32], bits[32]), bits[32]) = tuple(y, tmp, z)
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    # The test script only checks to see if a tuple(... x ...) instruction is in
    # the IR. A single element tuple containing x should remain.
    self._write_sh_script(test_sh_file.full_path, ['/bin/grep "tuple(.*x" $1'])
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH, '--test_executable=' + test_sh_file.full_path,
        ir_file.full_path
    ])
    self.assertIn('ret tuple.7: (bits[32]) = tuple(x, id=7)',
                  minimized_ir.decode('utf-8'))

  def test_simplify_array(self):
    input_ir = """package foo

top fn foo() -> bits[32][3] {
  ret a: bits[32][3] = literal(value=[0, 0, 0], id=3)
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path,
                          [r'/bin/grep "bits\[32\]\[[123]\]" $1'])
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH,
        '--test_executable=' + test_sh_file.full_path,
        ir_file.full_path,
    ],
                                           encoding='utf-8')
    self.assertRegex(
        minimized_ir,
        r'ret \w+.\d+: bits\[32\]\[1\] = literal\(value=\[0\], id=\d+\)')

  def test_proc(self):
    input_ir = '''package foo

chan input(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan output(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

top proc foo(tkn: token, foo: bits[32], bar: bits[32], baz: bits[32], init={1, 2, 3}) {
  receive.1: (token, bits[32]) = receive(tkn, channel_id=0)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tkn, baz, channel_id=1)
  after_all.5: token = after_all(tuple_index.2, send.4)
  next (after_all.5, tuple_index.3, foo, bar)
}
'''
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, [r'/usr/bin/env'])  # = true
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH,
        '--test_executable=' + test_sh_file.full_path,
        '--can_remove_params',
        ir_file.full_path,
    ],
                                           encoding='utf-8')
    self.assertIn('proc foo', minimized_ir)

  def test_proc_remove_sends(self):
    input_ir = '''package foo

chan input(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan output(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

top proc foo(tkn: token, foo: bits[32], bar: bits[32], baz: bits[32], init={1, 2, 3}) {
  receive.1: (token, bits[32]) = receive(tkn, channel_id=0)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tkn, baz, channel_id=1)
  after_all.5: token = after_all(tuple_index.2, send.4)
  next (after_all.5, tuple_index.3, foo, bar)
}
'''
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, [r'/usr/bin/env'])  # = true
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH,
        '--test_executable=' + test_sh_file.full_path,
        '--can_remove_params',
        '--can_remove_sends',
        ir_file.full_path,
    ],
                                           encoding='utf-8')
    self.assertIn('receive', minimized_ir)
    self.assertNotIn('send', minimized_ir)

  def test_remove_receives(self):
    input_ir = '''package foo

chan input(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan output(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

top proc foo(tkn: token, foo: bits[32], bar: bits[32], baz: bits[32], init={1, 2, 3}) {
  receive.1: (token, bits[32]) = receive(tkn, channel_id=0)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tkn, baz, channel_id=1)
  after_all.5: token = after_all(tuple_index.2, send.4)
  next (after_all.5, tuple_index.3, foo, bar)
}
'''
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, [r'/usr/bin/env'])  # = true
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH,
        '--test_executable=' + test_sh_file.full_path,
        '--can_remove_params',
        '--can_remove_receives',
        ir_file.full_path,
    ],
                                           encoding='utf-8')
    self.assertNotIn('receive', minimized_ir)
    self.assertIn('send', minimized_ir)

  def test_proc_remove_sends_and_receives(self):
    input_ir = '''package foo

chan input(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan output(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

top proc foo(tkn: token, foo: bits[32], bar: bits[32], baz: bits[32], init={1, 2, 3}) {
  receive.1: (token, bits[32]) = receive(tkn, channel_id=0)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tkn, baz, channel_id=1)
  after_all.5: token = after_all(tuple_index.2, send.4)
  next (after_all.5, tuple_index.3, foo, bar)
}
'''
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, [r'/usr/bin/env'])  # = true
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH,
        '--test_executable=' + test_sh_file.full_path,
        '--can_remove_params',
        '--can_remove_receives',
        '--can_remove_sends',
        ir_file.full_path,
    ],
                                           encoding='utf-8')
    self.assertNotIn('receive', minimized_ir)
    self.assertNotIn('send', minimized_ir)

  def test_proc_preserve_channels(self):
    input_ir = '''package foo

chan input(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan output(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

top proc foo(tkn: token, foo: bits[32], bar: bits[32], baz: bits[32], init={1, 2, 3}) {
  receive.1: (token, bits[32]) = receive(tkn, channel_id=0)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tkn, baz, channel_id=1)
  after_all.5: token = after_all(tuple_index.2, send.4)
  next (after_all.5, tuple_index.3, foo, bar)
}
'''
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, [r'/usr/bin/env'])  # = true
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH,
        '--test_executable=' + test_sh_file.full_path,
        '--can_remove_params',
        '--can_remove_receives',
        '--can_remove_sends',
        '--preserve_channels=input',
        ir_file.full_path,
    ],
                                           encoding='utf-8')
    self.assertIn('chan input', minimized_ir)
    self.assertNotIn('chan output', minimized_ir)

  def test_verify_return_code(self):
    # If the test script never successfully runs, then ir_minimizer_main should
    # return nonzero.
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, ['exit 1'])
    with self.assertRaises(subprocess.CalledProcessError):
      subprocess.check_call([
          IR_MINIMIZER_MAIN_PATH, '--test_executable=' + test_sh_file.full_path,
          '--can_remove_params', ir_file.full_path
      ])

  def test_minimize_jit_mismatch(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH, '--test_llvm_jit',
        '--use_optimization_pipeline', '--input=bits[32]:0x42; bits[32]:0x123',
        '--test_only_inject_jit_result=bits[32]:0x22', ir_file.full_path
    ],
                                           stderr=subprocess.PIPE)
    # The minimizer should reduce the test case to just a literal.
    self.assertIn('ret literal', minimized_ir.decode('utf-8'))

  def test_minimize_jit_mismatch_but_no_mismatch(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    comp = subprocess.run([
        IR_MINIMIZER_MAIN_PATH, '--test_llvm_jit',
        '--use_optimization_pipeline', '--input=bits[32]:0x42; bits[32]:0x123',
        ir_file.full_path
    ],
                          stderr=subprocess.PIPE,
                          check=False)
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn('main function provided does not fail',
                  comp.stderr.decode('utf-8'))

  def test_remove_userless_sideeffecting_op(self):
    input_ir = """package foo

top fn foo(x: bits[32], y: bits[1]) -> bits[32] {
  gate_node: bits[32] = gate(y, x)
  ret test_node: bits[32] = identity(x)
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    # The test script only checks to see if `test_node` is in the IR.
    self._write_sh_script(test_sh_file.full_path, ['/bin/grep test_node $1'])
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH, '--test_executable=' + test_sh_file.full_path,
        ir_file.full_path
    ])
    self.assertNotIn('gate_node', minimized_ir.decode('utf-8'))

if __name__ == '__main__':
  absltest.main()
