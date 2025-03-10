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
import re
import stat
import subprocess
from typing import Optional

from absl.testing import absltest
from xls.common import runfiles

IR_MINIMIZER_MAIN_PATH = runfiles.get_path('xls/dev_tools/ir_minimizer_main')

ADD_IR = """package foo

top fn foo(x: bits[32], y: bits[32]) -> bits[32] {
  mynot: bits[32] = not(x)
  myadd: bits[32] = add(mynot, y)
  ret result: bits[32] = not(myadd)
}
"""

ARRAY_IR = """package foo

fn bar(x: bits[8][8]) -> bits[8][4] {
  literal.2: bits[32] = literal(value=0, id=2)
  literal.4: bits[32] = literal(value=1, id=4)
  literal.7: bits[32] = literal(value=2, id=7)
  literal.9: bits[32] = literal(value=3, id=9)
  literal.12: bits[32] = literal(value=4, id=12)
  literal.14: bits[32] = literal(value=5, id=14)
  literal.17: bits[32] = literal(value=6, id=17)
  literal.19: bits[32] = literal(value=7, id=19)
  array_index.3: bits[8] = array_index(x, indices=[literal.2], id=3)
  array_index.5: bits[8] = array_index(x, indices=[literal.4], id=5)
  array_index.8: bits[8] = array_index(x, indices=[literal.7], id=8)
  array_index.10: bits[8] = array_index(x, indices=[literal.9], id=10)
  array_index.13: bits[8] = array_index(x, indices=[literal.12], id=13)
  array_index.15: bits[8] = array_index(x, indices=[literal.14], id=15)
  array_index.18: bits[8] = array_index(x, indices=[literal.17], id=18)
  array_index.20: bits[8] = array_index(x, indices=[literal.19], id=20)
  add.6: bits[8] = add(array_index.3, array_index.5, id=6)
  add.11: bits[8] = add(array_index.8, array_index.10, id=11)
  add.16: bits[8] = add(array_index.13, array_index.15, id=16)
  add.21: bits[8] = add(array_index.18, array_index.20, id=21)
  ret array.22: bits[8][4] = array(add.6, add.11, add.16, add.21, id=22)
}

top fn foo(x: bits[8], y: bits[8]) -> bits[8][4] {
  array.25: bits[8][8] = array(x, y, x, y, x, y, x, y, id=25)
  ret invoke.26: bits[8][4] = invoke(array.25, to_apply=bar, id=26)
}
"""

INVOKE_TWO = """package foo

fn bar(x: bits[32]) -> bits[1] {
    ret and_reduce.1: bits[1] = and_reduce(x, id=1)
}

fn baz(x: bits[32]) -> bits[1] {
    ret or_reduce.3: bits[1] = or_reduce(x, id=3)
}

top fn foo(x: bits[32], y: bits[32]) -> bits[1] {
  invoke.6: bits[1] = invoke(x, to_apply=bar, id=6)
  invoke.7: bits[1] = invoke(y, to_apply=baz, id=7)
  ret and.8: bits[1] = and(invoke.6, invoke.7, id=8)
}
"""

INVOKE_TWO_DEEP = """package foo

fn mul(a: bits[8], b: bits[8]) -> bits[8] {
  ret umul.3: bits[8] = umul(a, b, id=3)
}

fn muladd(a: bits[8], b: bits[8], c: bits[8]) -> bits[8] {
  invoke.7: bits[8] = invoke(a, b, to_apply=mul, id=7)
  ret add.8: bits[8] = add(invoke.7, c, id=8)
}

top fn muladdadd(a: bits[8], b: bits[8], c: bits[8], d: bits[8]) -> bits[8] {
  invoke.13: bits[8] = invoke(a, b, c, to_apply=muladd, id=13)
  ret add.14: bits[8] = add(invoke.13, d, id=14)
}
"""

INVOKE_MAP = """package foo

fn bar(x: bits[32]) -> bits[1] {
    ret and_reduce.1: bits[1] = and_reduce(x, id=1)
}

top fn foo(x: bits[32][8]) -> bits[1][8] {
    ret map.3: bits[1][8] = map(x, to_apply=bar, id=3)
}
"""

PROC = """package testit

file_number 0 "/tmp/testit.x"

chan testit__output(bits[32], id=0, kind=streaming, ops=send_only, flow_control=ready_valid, strictness=proven_mutually_exclusive)

top proc __testit__main_0_next(__state: bits[32], init={3}) {
  __token: token = literal(value=token, id=1)
  literal.4: bits[32] = literal(value=3, id=4, pos=[(0,9,46)])
  umul.5: bits[32] = umul(__state, literal.4, id=5, pos=[(0,9,40)])
  literal.7: bits[32] = literal(value=1, id=7, pos=[(0,10,18)])
  tok: token = send(__token, umul.5, channel=testit__output, id=6)
  literal.3: bits[1] = literal(value=1, id=3)
  add.8: bits[32] = add(__state, literal.7, id=8, pos=[(0,10,12)])
  next (add.8)
}
"""

PROC_2 = """package multi_proc

file_number 0 "xls/jit/multi_proc.x"

chan multi_proc__bytes_src(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=proven_mutually_exclusive)
chan multi_proc__bytes_result(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, strictness=proven_mutually_exclusive)
chan multi_proc__send_double_pipe(bits[32], id=2, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive)
chan multi_proc__send_quad_pipe(bits[32], id=3, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive)
chan multi_proc__recv_double_pipe(bits[32], id=4, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive)
chan multi_proc__recv_quad_pipe(bits[32], id=5, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive)

fn __multi_proc__double_it(n: bits[32]) -> bits[32] {
  ret add.2: bits[32] = add(n, n, id=2, pos=[(0,17,32)])
}

fn __multi_proc__proc_double.init() -> () {
  ret tuple.3: () = tuple(id=3, pos=[(0,25,11)])
}

fn __multi_proc__proc_quad.init() -> () {
  ret tuple.4: () = tuple(id=4, pos=[(0,40,11)])
}

top proc __multi_proc__proc_ten_0_next(__state: (), init={()}) {
  tok: token = after_all(id=8)
  receive.9: (token, bits[32]) = receive(tok, channel=multi_proc__bytes_src, id=9)
  tok__1: token = tuple_index(receive.9, index=0, id=11, pos=[(0,74,13)])
  v: bits[32] = tuple_index(receive.9, index=1, id=12, pos=[(0,74,18)])
  tok__2: token = send(tok__1, v, channel=multi_proc__send_quad_pipe, id=13)
  receive.14: (token, bits[32]) = receive(tok__2, channel=multi_proc__recv_quad_pipe, id=14)
  tok__3: token = tuple_index(receive.14, index=0, id=16, pos=[(0,76,13)])
  tok__4: token = send(tok__3, v, channel=multi_proc__send_double_pipe, id=19)
  qv: bits[32] = tuple_index(receive.14, index=1, id=17, pos=[(0,76,18)])
  receive.20: (token, bits[32]) = receive(tok__4, channel=multi_proc__recv_double_pipe, id=20)
  ev: bits[32] = invoke(qv, to_apply=__multi_proc__double_it, id=18, pos=[(0,77,26)])
  dv: bits[32] = tuple_index(receive.20, index=1, id=23, pos=[(0,79,18)])
  tok__5: token = tuple_index(receive.20, index=0, id=22, pos=[(0,79,13)])
  add.24: bits[32] = add(ev, dv, id=24, pos=[(0,81,35)])
  __token: token = literal(value=token, id=5)
  literal.7: bits[1] = literal(value=1, id=7)
  tuple_index.10: token = tuple_index(receive.9, index=0, id=10)
  tuple_index.15: token = tuple_index(receive.14, index=0, id=15)
  tuple_index.21: token = tuple_index(receive.20, index=0, id=21)
  send.25: token = send(tok__5, add.24, channel=multi_proc__bytes_result, id=25)
  tuple.26: () = tuple(id=26, pos=[(0,72,15)])
  next_state: () = next_value(param=__state, value=tuple.26, id=33)
}

proc __multi_proc__proc_ten__proc_double_0_next(__state: (), init={()}) {
  tok: token = after_all(id=30)
  receive.31: (token, bits[32]) = receive(tok, channel=multi_proc__send_double_pipe, id=31)
  v: bits[32] = tuple_index(receive.31, index=1, id=34, pos=[(0,29,18)])
  tok__1: token = tuple_index(receive.31, index=0, id=62, pos=[(0,29,13)])
  invoke.35: bits[32] = invoke(v, to_apply=__multi_proc__double_it, id=35, pos=[(0,30,41)])
  __token: token = literal(value=token, id=27)
  literal.29: bits[1] = literal(value=1, id=29)
  tuple_index.32: token = tuple_index(receive.31, index=0, id=32)
  send.36: token = send(tok__1, invoke.35, channel=multi_proc__recv_double_pipe, id=36)
  tuple.37: () = tuple(id=37, pos=[(0,27,15)])
  next (tuple.37)
}

proc __multi_proc__proc_ten__proc_quad_0_next(__state: (), init={()}) {
  tok: token = after_all(id=41)
  receive.42: (token, bits[32]) = receive(tok, channel=multi_proc__send_quad_pipe, id=42)
  v: bits[32] = tuple_index(receive.42, index=1, id=45, pos=[(0,44,18)])
  invoke.46: bits[32] = invoke(v, to_apply=__multi_proc__double_it, id=46, pos=[(0,45,51)])
  tok__1: token = tuple_index(receive.42, index=0, id=44, pos=[(0,44,13)])
  invoke.47: bits[32] = invoke(invoke.46, to_apply=__multi_proc__double_it, id=47, pos=[(0,45,41)])
  __token: token = literal(value=token, id=38)
  literal.40: bits[1] = literal(value=1, id=40)
  tuple_index.43: token = tuple_index(receive.42, index=0, id=43)
  send.48: token = send(tok__1, invoke.47, channel=multi_proc__recv_quad_pipe, id=48)
  tuple.49: () = tuple(id=49, pos=[(0,42,15)])
  next (tuple.49)
}
"""


def function_count(ir: str) -> int:
  return len(re.findall(R'^\s+(top\s+)?fn\s+', ir, re.M))


def proc_count(ir: str) -> int:
  return len(re.findall(R'^\s+(top\s+)?proc\s+', ir, re.M))


def node_count(ir: str, op: Optional[str] = None) -> int:
  count = 0
  for m in re.finditer(
      R'^\s+(ret\s+)?[a-z0-9_\.]+:.*=\s*([a-z_]+)\(.*\)\s*$', ir, re.M
  ):
    if op is None or m.group(2) == op:
      count += 1
  return count


class IrMinimizerMainTest(absltest.TestCase):

  def _maybe_record_property(self, name, value):
    if callable(getattr(self, 'recordProperty', None)):
      self.recordProperty(name, value)

  def _write_sh_script(self, path, commands):
    with open(path, 'w') as f:
      all_cmds = ['#!/bin/sh -e'] + commands
      self._maybe_record_property('test_script', '\n'.join(all_cmds))
      f.write('\n'.join(all_cmds))
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR)

  def test_minimize_extract_things(self):
    ir_file = self.create_tempfile(content=PROC)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(
        test_sh_file.full_path, ["/usr/bin/env grep 'add' $1"]
    )
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params=true',
            '--can_remove_sends=true',
            '--can_remove_receives=true',
            '--can_extract_segments=true',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self._maybe_record_property('output', minimized_ir)
    self.assertEqual(function_count(minimized_ir), 1)
    self.assertEqual(node_count(minimized_ir), 2)
    self.assertIn('ret add', minimized_ir)

  def test_minimize_extract_single_proc(self):
    ir_file = self.create_tempfile(content=PROC_2)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(
        test_sh_file.full_path,
        ["/usr/bin/env grep 'multi_proc__bytes_src' $1"],
    )
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            f'--test_executable={test_sh_file.full_path}',
            '--can_remove_params=false',
            '--can_remove_sends=true',
            '--can_remove_receives=true',
            '--can_extract_single_proc=true',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self._maybe_record_property('output', minimized_ir)
    self.assertEqual(proc_count(minimized_ir), 1)
    self.assertEqual(node_count(minimized_ir), 2)

  def test_minimize_inline_one_can_inline_other_invokes(self):
    ir_file = self.create_tempfile(content=INVOKE_TWO_DEEP)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(
        test_sh_file.full_path, ["/usr/bin/env grep 'invoke.*to_apply=mul,' $1"]
    )
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params=false',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self._maybe_record_property('output', minimized_ir)
    self.assertEqual(function_count(minimized_ir), 2)
    self.assertEqual(node_count(minimized_ir, 'invoke'), 1)

  def test_minimize_no_change_subroutine_type(self):
    ir_file = self.create_tempfile(content=ARRAY_IR)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(
        test_sh_file.full_path, ['/usr/bin/env grep invoke $1']
    )
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params=false',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self._maybe_record_property('output', minimized_ir)
    self.assertEqual(function_count(minimized_ir), 2)
    self.assertIn('fn bar(x: bits[8][8]', minimized_ir)
    self.assertIn('top fn foo(', minimized_ir)
    self.assertEqual(node_count(minimized_ir, 'invoke'), 1)

  def test_minimize_add_no_remove_params(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(
        test_sh_file.full_path, ['/usr/bin/env grep myadd $1']
    )
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params=false',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self._maybe_record_property('output', minimized_ir)
    self.assertEqual(function_count(minimized_ir), 1)
    self.assertEqual(node_count(minimized_ir), 1)
    self.assertIn('x: bits', minimized_ir)
    self.assertIn('y: bits', minimized_ir)
    self.assertIn('ret myadd', minimized_ir)

  def test_minimize_add_remove_params(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, ['/usr/bin/env grep add $1'])
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self._maybe_record_property('output', minimized_ir)
    self.assertNotIn('x: bits', minimized_ir)
    self.assertNotIn('y: bits', minimized_ir)

  def test_no_reduction_possible(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    # Shell script is run with -e so if any of the greps fail then the script
    # fails.
    self._write_sh_script(
        test_sh_file.full_path,
        [
            '/usr/bin/env grep mynot.*x $1',
            '/usr/bin/env grep myadd.*mynot.*y $1',
            '/usr/bin/env grep result.*myadd $1',
        ],
    )
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self.assertEqual(node_count(minimized_ir), 3)
    self.assertIn('mynot', minimized_ir)
    self.assertIn('myadd', minimized_ir)
    self.assertIn('result', minimized_ir)

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
    self._write_sh_script(
        test_sh_file.full_path, ['/usr/bin/env grep not.*x $1']
    )
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    # The array operation should have been stripped from the function.
    self.assertIn('array(', input_ir)
    self.assertNotIn('array(', minimized_ir)

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
    self._write_sh_script(
        test_sh_file.full_path, ['/usr/bin/env grep "tuple(.*x" $1']
    )
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self.assertRegex(
        minimized_ir,
        r'ret tuple\.\d+: \(bits\[32\]\) = tuple\(x, id=\d+\)',
    )

  def test_simplify_array(self):
    input_ir = """package foo

top fn foo() -> bits[32][3] {
  ret a: bits[32][3] = literal(value=[0, 0, 0], id=3)
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(
        test_sh_file.full_path, [r'/usr/bin/env grep "bits\[32\]\[[123]\]" $1']
    )
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self.assertRegex(
        minimized_ir,
        r'ret \w+.\d+: bits\[32\]\[1\] = literal\(value=\[0\], id=\d+\)',
    )

  def test_proc(self):
    input_ir = """package foo

chan input(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan output(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)

top proc foo(foo: bits[32], bar: bits[32], baz: bits[32], init={1, 2, 3}) {
  tkn: token = literal(value=token, id=1000)
  receive.1: (token, bits[32]) = receive(tkn, channel=input)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tkn, baz, channel=output)
  next (tuple_index.3, foo, bar)
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, [r'/usr/bin/env'])  # = true
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self.assertIn('proc foo', minimized_ir)

  def test_proc_remove_sends(self):
    input_ir = """package foo

chan input(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan output(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)

top proc foo(foo: bits[32], bar: bits[32], baz: bits[32], init={1, 2, 3}) {
  tkn: token = literal(value=token, id=1000)
  receive.1: (token, bits[32]) = receive(tkn, channel=input)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tkn, baz, channel=output)
  next (tuple_index.3, foo, bar)
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, [r'/usr/bin/env'])  # = true
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params',
            '--can_remove_sends',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self.assertIn('receive', minimized_ir)
    self.assertNotIn('send', minimized_ir)

  def test_remove_receives(self):
    input_ir = """package foo

chan input(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan output(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)

top proc foo(foo: bits[32], bar: bits[32], baz: bits[32], init={1, 2, 3}) {
  tkn: token = literal(value=token, id=1000)
  receive.1: (token, bits[32]) = receive(tkn, channel=input)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tkn, baz, channel=output)
  next (tuple_index.3, foo, bar)
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, [r'/usr/bin/env'])  # = true
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params',
            '--can_remove_receives',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self.assertNotIn('receive', minimized_ir)
    self.assertIn('send', minimized_ir)

  def test_proc_remove_sends_and_receives(self):
    input_ir = """package foo

chan input(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan output(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)

top proc foo(foo: bits[32], bar: bits[32], baz: bits[32], init={1, 2, 3}) {
  tkn: token = literal(value=token, id=1000)
  receive.1: (token, bits[32]) = receive(tkn, channel=input)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tkn, baz, channel=output)
  next (tuple_index.3, foo, bar)
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, [r'/usr/bin/env'])  # = true
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params',
            '--can_remove_receives',
            '--can_remove_sends',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self.assertNotIn('receive', minimized_ir)
    self.assertNotIn('send', minimized_ir)

  def test_proc_preserve_channels(self):
    input_ir = """package foo

chan input(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan output(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)

top proc foo(foo: bits[32], bar: bits[32], baz: bits[32], init={1, 2, 3}) {
  tkn: token = literal(value=token, id=1000)
  receive.1: (token, bits[32]) = receive(tkn, channel=input)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tkn, baz, channel=output)
  next (tuple_index.3, foo, bar)
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, [r'/usr/bin/env'])  # = true
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params',
            '--can_remove_receives',
            '--can_remove_sends',
            '--preserve_channels=input',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self.assertIn('chan input', minimized_ir)
    self.assertNotIn('chan output', minimized_ir)

  def test_new_style_proc(self):
    input_ir = """package foo

top proc foo<input: bits[32] in, output:bits[32] out>(foo: bits[32], bar: bits[32], baz: bits[32], init={1, 2, 3}) {
  chan_interface input(direction=receive, kind=streaming, strictness=proven_mutually_exclusive)
  chan_interface output(direction=send, kind=streaming, strictness=proven_mutually_exclusive)
  tkn: token = literal(value=token, id=1000)
  receive.1: (token, bits[32]) = receive(tkn, channel=input)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(tkn, baz, channel=output)
  next (tuple_index.3, foo, bar)
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, [r'/usr/bin/env'])  # = true
    # The minimizer should not crash.
    subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params',
            '--can_remove_receives',
            '--can_remove_sends',
            '--preserve_channels=input',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )

  def test_verify_return_code(self):
    # If the test script never successfully runs, then ir_minimizer_main should
    # return nonzero.
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, ['exit 1'])
    with self.assertRaises(subprocess.CalledProcessError):
      subprocess.check_call([
          IR_MINIMIZER_MAIN_PATH,
          '--test_executable=' + test_sh_file.full_path,
          '--can_remove_params',
          ir_file.full_path,
      ])

  def test_minimize_jit_mismatch(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_llvm_jit',
            '--use_optimization_pipeline',
            '--input=bits[32]:0x42; bits[32]:0x123',
            '--test_only_inject_jit_result=bits[32]:0x22',
            ir_file.full_path,
        ],
        encoding='utf-8',
        stderr=subprocess.PIPE,
    )
    # The minimizer should reduce the test case to just a literal.
    self.assertRegex(minimized_ir, 'ret .*literal')

  def test_minimize_jit_mismatch_but_no_mismatch(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    comp = subprocess.run(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_llvm_jit',
            '--use_optimization_pipeline',
            '--input=bits[32]:0x42; bits[32]:0x123',
            ir_file.full_path,
        ],
        encoding='utf-8',
        stderr=subprocess.PIPE,
        check=False,
    )
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn('main function provided does not fail', comp.stderr)

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
    self._write_sh_script(
        test_sh_file.full_path, ['/usr/bin/env grep test_node $1']
    )
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self.assertNotIn('gate_node', minimized_ir)

  def test_remove_literal_subelements(self):
    input_ir = """package foo

top fn foo() -> (bits[1], (bits[42]), bits[32]) {
  ret result: (bits[1], (bits[42]), bits[32]) = literal(value=(0, (0), 0))
}
"""
    ir_file = self.create_tempfile(content=input_ir)
    test_sh_file = self.create_tempfile()
    # The test script checks to see if `bits[42]` is in the IR.
    self._write_sh_script(
        test_sh_file.full_path, ['/usr/bin/env grep bits.42 $1']
    )
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    # All tuple elements but bits[42] should be removed.
    self.assertIn(': ((bits[42])) = literal', minimized_ir)

  def test_proc_works_with_multiple_simplifications_between_tests(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, ['/usr/bin/env grep add $1'])
    # Minimizing with small simplifications_between_tests should work.
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params',
            '--simplifications_between_tests=2',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self.assertNotIn('not', minimized_ir)

    # Minimizing with large simplifications_between_tests won't work for this
    # small example (it will remove everything), so check that the IR is
    # unchanged.
    minimized_ir = subprocess.check_output(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            '--can_remove_params',
            '--simplifications_between_tests=100',
            ir_file.full_path,
        ],
        encoding='utf-8',
    )
    self.assertIn('mynot', minimized_ir)
    self.assertIn('myadd', minimized_ir)
    self.assertIn('result', minimized_ir)

  def test_inline_single_invoke_is_triggerable(self):
    ir_file = self.create_tempfile(content=INVOKE_TWO)
    test_sh_file = self.create_tempfile()
    # The test script only checks to see if `invoke.2` is in the IR.
    self._write_sh_script(
        test_sh_file.full_path,
        ["/usr/bin/env grep 'invoke(x, to_apply=bar, id=6)' $1"],
    )
    output = subprocess.run(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--test_executable=' + test_sh_file.full_path,
            ir_file.full_path,
        ],
        encoding='utf-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    self.assertEqual(
        output.returncode,
        0,
        f'Non zero return: stderr {output.stderr}, stdout: {output.stdout}',
    )
    minimized_ir = output.stdout
    self.assertEqual(function_count(minimized_ir), 2)
    self.assertEqual(node_count(minimized_ir), 3)
    self.assertIn('ret literal', minimized_ir)
    self.assertNotIn('ret invoke', minimized_ir)

  def test_can_remove_invoke_args(self):
    ir_file = self.create_tempfile(content=INVOKE_TWO)
    test_sh_file = self.create_tempfile()
    # The test script only checks to see if invoke of bar is in the IR.
    self._write_sh_script(
        test_sh_file.full_path, ["/usr/bin/env grep 'to_apply=baz' $1"]
    )
    output = subprocess.run(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--can_remove_params',
            '--can_inline_everything=false',
            '--test_executable=' + test_sh_file.full_path,
            ir_file.full_path,
        ],
        encoding='utf-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    self.assertEqual(
        output.returncode,
        0,
        f'Non zero return: stderr {output.stderr}, stdout: {output.stdout}',
    )
    minimized_ir = output.stdout
    self._maybe_record_property('output', minimized_ir)
    self.assertIn('baz()', minimized_ir)

  def test_can_unwrap_map(self):
    ir_file = self.create_tempfile(content=INVOKE_MAP)
    test_sh_file = self.create_tempfile()
    # The test script only checks to see if bar is present
    self._write_sh_script(
        test_sh_file.full_path,
        ['/usr/bin/env grep bar $1'],
    )
    output = subprocess.run(
        [
            IR_MINIMIZER_MAIN_PATH,
            '--can_remove_params',
            '--can_inline_everything=false',
            '--test_executable=' + test_sh_file.full_path,
            ir_file.full_path,
        ],
        encoding='utf-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    self.assertEqual(
        output.returncode,
        0,
        f'Non zero return: stderr {output.stderr}, stdout: {output.stdout}',
    )
    minimized_ir = output.stdout
    self._maybe_record_property('output', minimized_ir)
    self.assertEqual(function_count(minimized_ir), 2)
    self.assertEqual(node_count(minimized_ir), 3)
    self.assertIn('ret literal', minimized_ir)
    self.assertNotIn('ret invoke', minimized_ir)


if __name__ == '__main__':
  absltest.main()
