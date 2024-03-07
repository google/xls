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

import subprocess
import textwrap

from absl import logging

from absl.testing import absltest
from absl.testing import parameterized
from xls.common import runfiles

EVAL_PROC_MAIN_PATH = runfiles.get_path("xls/tools/eval_proc_main")

PROC_IR = """package foo

chan in_ch(bits[64], id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata=\"\"\"\"\"\")
chan in_ch_2(bits[64], id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata=\"\"\"\"\"\")
chan out_ch(bits[64], id=3, kind=streaming, ops=send_only, flow_control=ready_valid, metadata=\"\"\"\"\"\")
chan out_ch_2(bits[64], id=4, kind=streaming, ops=send_only, flow_control=ready_valid, metadata=\"\"\"\"\"\")

proc test_proc(tkn: token, st: (bits[64]), init={(10)}) {
  receive.1: (token, bits[64]) = receive(tkn, channel=in_ch, id=1)

  literal.21: bits[64] = literal(value=10, id=21)
  tuple_index.23: bits[64] = tuple_index(st, index=0, id=23)

  literal.3: bits[1] = literal(value=1, id=3)
  tuple_index.7: token = tuple_index(receive.1, index=0, id=7)
  tuple_index.4: bits[64] = tuple_index(receive.1, index=1, id=4)
  receive.9: (token, bits[64]) = receive(tuple_index.7, channel=in_ch_2, id=9)
  tuple_index.10: bits[64] = tuple_index(receive.9, index=1, id=10)
  add.8: bits[64] = add(tuple_index.4, tuple_index.10, id=8)
  add.24: bits[64] = add(add.8, tuple_index.23, id=24)

  tuple_index.11: token = tuple_index(receive.9, index=0, id=11)
  send.2: token = send(tuple_index.11, add.24, predicate=literal.3, channel=out_ch, id=2)
  literal.14: bits[64] = literal(value=55, id=14)
  send.12: token = send(send.2, literal.14, predicate=literal.3, channel=out_ch_2, id=12)

  add.20: bits[64] = add(literal.21, tuple_index.23, id=20)

  tuple.22: (bits[64]) = tuple(add.20, id=22)

  next(send.12, tuple.22)
}
"""

PROC_IR_CONDITIONAL = """package foo

file_number 0 "fake_file.x"

chan input(bits[8], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata=\"\"\"\"\"\")
chan output(bits[8], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata=\"\"\"\"\"\")

proc test_proc(__token: token, init={}) {
  receive.4: (token, bits[8]) = receive(__token, channel=input, id=4)
  recv_val: bits[8] = tuple_index(receive.4, index=1, id=7, pos=[(0,7,19)])
  literal.8: bits[8] = literal(value=42, id=8, pos=[(0,8,39)])
  recv_tok: token = tuple_index(receive.4, index=0, id=6, pos=[(0,7,9)])
  do_send: bits[1] = ne(recv_val, literal.8, id=9, pos=[(0,8,33)])
  send_tok: token = send(recv_tok, recv_val, predicate=do_send, channel=output, id=10)
  after_all.12: token = after_all(__token, recv_tok, send_tok, id=12)
  next (after_all.12)
}
"""

BLOCK_IR = """package foo

block test_block(clk: clock, in_ch_data: bits[64], in_ch_2_data: bits[64], out_ch_data: bits[64], out_ch_2_data: bits[64], rst_n: bits[1], in_ch_vld: bits[1], in_ch_2_vld: bits[1], out_ch_vld: bits[1], out_ch_2_vld: bits[1], out_ch_rdy: bits[1], out_ch_2_rdy: bits[1], in_ch_rdy: bits[1], in_ch_2_rdy: bits[1]) {
  reg p0_add_49(bits[64])
  reg p1_add_49(bits[64])
  reg p2_add_49(bits[64])
  reg p3_add_49(bits[64])
  reg p0_valid(bits[1], reset_value=0, asynchronous=false, active_low=true)

  reg p1_valid(bits[1], reset_value=0, asynchronous=false, active_low=true)

  reg p2_valid(bits[1], reset_value=0, asynchronous=false, active_low=true)

  reg p3_valid(bits[1], reset_value=0, asynchronous=false, active_low=true)

  reg st((bits[64]), reset_value=(10), asynchronous=false, active_low=true)

  in_ch_data: bits[64] = input_port(name=in_ch_data, id=38)
  in_ch_2_data: bits[64] = input_port(name=in_ch_2_data, id=41)
  rst_n: bits[1] = input_port(name=rst_n, id=74)
  in_ch_vld: bits[1] = input_port(name=in_ch_vld, id=75)
  in_ch_2_vld: bits[1] = input_port(name=in_ch_2_vld, id=76)
  out_ch_rdy: bits[1] = input_port(name=out_ch_rdy, id=92)
  out_ch_2_rdy: bits[1] = input_port(name=out_ch_2_rdy, id=95)
  literal.50: bits[64] = literal(value=10, id=50)
  st__1: (bits[64]) = register_read(register=st, id=87)
  add.46: bits[64] = add(in_ch_data, in_ch_2_data, id=46)
  tuple_index.47: bits[64] = tuple_index(st__1, index=0, id=47)
  add.51: bits[64] = add(literal.50, tuple_index.47, id=51)
  tuple.52: (bits[64]) = tuple(add.51, id=52)
  add.49: bits[64] = add(add.46, tuple_index.47, id=49)
  not.127: bits[1] = not(rst_n, id=127)
  p0_add_49: bits[64] = register_read(register=p0_add_49, id=55)
  not.119: bits[1] = not(rst_n, id=119)
  p1_add_49: bits[64] = register_read(register=p1_add_49, id=59)
  not.111: bits[1] = not(rst_n, id=111)
  p2_add_49: bits[64] = register_read(register=p2_add_49, id=63)
  and.77: bits[1] = and(in_ch_vld, in_ch_2_vld, id=77)
  not.103: bits[1] = not(rst_n, id=103)
  p3_add_49: bits[64] = register_read(register=p3_add_49, id=67)
  register_read.79: bits[1] = register_read(register=p0_valid, id=79)
  register_read.81: bits[1] = register_read(register=p1_valid, id=81)
  register_read.83: bits[1] = register_read(register=p2_valid, id=83)
  register_read.85: bits[1] = register_read(register=p3_valid, id=85)
  literal.70: bits[1] = literal(value=1, id=70)
  literal.72: bits[64] = literal(value=55, id=72)
  and.88: bits[1] = and(literal.70, register_read.85, id=88)
  and.90: bits[1] = and(literal.70, register_read.85, id=90)
  not.93: bits[1] = not(literal.70, id=93)
  or.94: bits[1] = or(not.93, out_ch_rdy, id=94)
  not.96: bits[1] = not(literal.70, id=96)
  or.97: bits[1] = or(not.96, out_ch_2_rdy, id=97)
  and.98: bits[1] = and(or.94, or.97, id=98)
  p3_not_valid: bits[1] = not(register_read.85, id=99)
  p3_enable: bits[1] = or(and.98, p3_not_valid, id=100)
  p3_data_enable: bits[1] = and(p3_enable, register_read.83, id=102)
  p3_load_en: bits[1] = or(p3_data_enable, not.103, id=104)
  register_write.101: () = register_write(register_read.83, register=p3_valid, load_enable=p3_enable, reset=rst_n, id=101)
  register_write.105: () = register_write(p2_add_49, register=p3_add_49, load_enable=p3_load_en, id=105)
  p2_not_valid: bits[1] = not(register_read.83, id=107)
  p2_enable: bits[1] = or(p3_enable, p2_not_valid, id=108)
  p2_data_enable: bits[1] = and(p2_enable, register_read.81, id=110)
  p2_load_en: bits[1] = or(p2_data_enable, not.111, id=112)
  register_write.113: () = register_write(p1_add_49, register=p2_add_49, load_enable=p2_load_en, id=113)
  register_write.109: () = register_write(register_read.81, register=p2_valid, load_enable=p2_enable, reset=rst_n, id=109)
  p1_not_valid: bits[1] = not(register_read.81, id=115)
  p1_enable: bits[1] = or(p2_enable, p1_not_valid, id=116)
  p1_data_enable: bits[1] = and(p1_enable, register_read.79, id=118)
  p1_load_en: bits[1] = or(p1_data_enable, not.119, id=120)
  register_write.121: () = register_write(p0_add_49, register=p1_add_49, load_enable=p1_load_en, id=121)
  register_write.117: () = register_write(register_read.79, register=p1_valid, load_enable=p1_enable, reset=rst_n, id=117)
  p0_not_valid: bits[1] = not(register_read.79, id=123)
  p0_enable: bits[1] = or(p1_enable, p0_not_valid, id=124)
  next_state_enable: bits[1] = and(p0_enable, and.77, id=132)
  register_write.133: () = register_write(tuple.52, register=st, load_enable=next_state_enable, reset=rst_n, id=133)
  p0_data_enable: bits[1] = and(p0_enable, and.77, id=126)
  p0_load_en: bits[1] = or(p0_data_enable, not.127, id=128)
  register_write.129: () = register_write(add.49, register=p0_add_49, load_enable=p0_load_en, id=129)
  register_write.125: () = register_write(and.77, register=p0_valid, load_enable=p0_enable, reset=rst_n, id=125)

  out_ch_data: () = output_port(p3_add_49, name=out_ch_data, id=71)
  out_ch_2_data: () = output_port(literal.72, name=out_ch_2_data, id=73)
  out_ch_vld: () = output_port(and.88, name=out_ch_vld, id=89)
  out_ch_2_vld: () = output_port(and.90, name=out_ch_2_vld, id=91)
  in_ch_rdy: () = output_port(p0_enable, name=in_ch_rdy, id=134)
  in_ch_2_rdy: () = output_port(p0_enable, name=in_ch_2_rdy, id=135)

  in_pred: bits[1] = literal(value=1, id=576)
  after_all.563: token = after_all(id=563)
  trace.581: token = trace(after_all.563, in_pred, format="rst_n {:x}", data_operands=[rst_n], verbosity=2, id=581)

}
"""


BLOCK_IR_BROKEN = """package foo

block test_block(clk: clock, in_ch_data: bits[64], in_ch_2_data: bits[64], out_ch_data: bits[64], out_ch_2_data: bits[64], rst_n: bits[1], in_ch_vld: bits[1], in_ch_2_vld: bits[1], out_ch_vld: bits[1], out_ch_2_vld: bits[1], out_ch_rdy: bits[1], out_ch_2_rdy: bits[1], in_ch_rdy: bits[1], in_ch_2_rdy: bits[1]) {
  reg p0_add_49(bits[64])
  reg p1_add_49(bits[64])
  reg p2_add_49(bits[64])
  reg p3_add_49(bits[64])
  reg p0_valid(bits[1], reset_value=0, asynchronous=false, active_low=true)

  reg p1_valid(bits[1], reset_value=0, asynchronous=false, active_low=true)

  reg p2_valid(bits[1], reset_value=0, asynchronous=false, active_low=true)

  reg p3_valid(bits[1], reset_value=0, asynchronous=false, active_low=true)

  reg st((bits[64]), reset_value=(10), asynchronous=false, active_low=true)

  in_ch_data: bits[64] = input_port(name=in_ch_data, id=38)
  in_ch_2_data: bits[64] = input_port(name=in_ch_2_data, id=41)
  rst_n: bits[1] = input_port(name=rst_n, id=74)
  in_ch_vld: bits[1] = input_port(name=in_ch_vld, id=75)
  in_ch_2_vld: bits[1] = input_port(name=in_ch_2_vld, id=76)
  out_ch_rdy: bits[1] = input_port(name=out_ch_rdy, id=92)
  out_ch_2_rdy: bits[1] = input_port(name=out_ch_2_rdy, id=95)
  literal.50: bits[64] = literal(value=10, id=50)
  st__1: (bits[64]) = register_read(register=st, id=87)
  add.46: bits[64] = add(in_ch_data, in_ch_2_data, id=46)
  tuple_index.47: bits[64] = tuple_index(st__1, index=0, id=47)
  add.51: bits[64] = add(literal.50, tuple_index.47, id=51)
  tuple.52: (bits[64]) = tuple(add.51, id=52)
  add.49: bits[64] = add(add.46, tuple_index.47, id=49)
  not.127: bits[1] = not(rst_n, id=127)
  p0_add_49: bits[64] = register_read(register=p0_add_49, id=55)
  not.119: bits[1] = not(rst_n, id=119)
  p1_add_49: bits[64] = register_read(register=p1_add_49, id=59)
  not.111: bits[1] = not(rst_n, id=111)
  p2_add_49: bits[64] = register_read(register=p2_add_49, id=63)
  and.77: bits[1] = and(in_ch_vld, in_ch_2_vld, id=77)
  not.103: bits[1] = not(rst_n, id=103)
  p3_add_49: bits[64] = register_read(register=p3_add_49, id=67)
  register_read.79: bits[1] = register_read(register=p0_valid, id=79)
  register_read.81: bits[1] = register_read(register=p1_valid, id=81)
  register_read.83: bits[1] = register_read(register=p2_valid, id=83)
  register_read.85: bits[1] = register_read(register=p3_valid, id=85)
  literal.70: bits[1] = literal(value=1, id=70)
  literal.72: bits[64] = literal(value=55, id=72)
  and.88: bits[1] = and(literal.70, register_read.85, id=88)
  and.90: bits[1] = and(literal.70, register_read.85, id=90)
  not.93: bits[1] = not(literal.70, id=93)
  or.94: bits[1] = or(not.93, out_ch_rdy, id=94)
  not.96: bits[1] = not(literal.70, id=96)
  or.97: bits[1] = or(not.96, out_ch_2_rdy, id=97)
  and.98: bits[1] = and(or.94, or.97, id=98)
  p3_not_valid: bits[1] = not(register_read.85, id=99)
  p3_enable: bits[1] = or(and.98, p3_not_valid, id=100)
  p3_data_enable: bits[1] = and(p3_enable, register_read.83, id=102)
  p3_load_en: bits[1] = or(p3_data_enable, not.103, id=104)
  register_write.101: () = register_write(register_read.83, register=p3_valid, load_enable=p3_enable, reset=rst_n, id=101)
  register_write.105: () = register_write(p2_add_49, register=p3_add_49, load_enable=p3_load_en, id=105)
  p2_not_valid: bits[1] = not(register_read.83, id=107)
  p2_enable: bits[1] = or(p3_enable, p2_not_valid, id=108)
  p2_data_enable: bits[1] = and(p2_enable, register_read.81, id=110)
  p2_load_en: bits[1] = or(p2_data_enable, not.111, id=112)
  register_write.113: () = register_write(p1_add_49, register=p2_add_49, load_enable=p2_load_en, id=113)
  register_write.109: () = register_write(register_read.81, register=p2_valid, load_enable=p2_enable, reset=rst_n, id=109)
  p1_not_valid: bits[1] = not(register_read.81, id=115)
  p1_enable: bits[1] = or(p2_enable, p1_not_valid, id=116)
  p1_data_enable: bits[1] = and(p1_enable, register_read.79, id=118)
  p1_load_en: bits[1] = or(p1_data_enable, not.119, id=120)
  register_write.121: () = register_write(p0_add_49, register=p1_add_49, load_enable=p1_load_en, id=121)
  register_write.117: () = register_write(register_read.79, register=p1_valid, load_enable=p1_enable, reset=rst_n, id=117)
  p0_not_valid: bits[1] = not(register_read.79, id=123)
  p0_enable: bits[1] = or(p1_enable, p0_not_valid, id=124)
  next_state_enable: bits[1] = and(p0_enable, and.77, id=132)
  register_write.133: () = register_write(tuple.52, register=st, load_enable=next_state_enable, reset=rst_n, id=133)
  p0_data_enable: bits[1] = and(p0_enable, and.77, id=126)
  p0_load_en: bits[1] = or(p0_data_enable, not.127, id=128)
  register_write.129: () = register_write(add.49, register=p0_add_49, load_enable=p0_load_en, id=129)
  register_write.125: () = register_write(and.77, register=p0_valid, load_enable=p0_enable, reset=rst_n, id=125)
  out_ch_data: () = output_port(p3_add_49, name=out_ch_data, id=71)
  out_ch_2_data: () = output_port(literal.72, name=out_ch_2_data, id=73)
  out_ch_vld: () = output_port(and.88, name=out_ch_vld, id=89)
  literal.700: bits[1] = literal(value=0, id=700)
  out_ch_2_vld: () = output_port(literal.700, name=out_ch_2_vld, id=91)
  in_ch_rdy: () = output_port(p0_enable, name=in_ch_rdy, id=134)
  in_ch_2_rdy: () = output_port(p0_enable, name=in_ch_2_rdy, id=135)
}
"""

BLOCK_SIGNATURE_TEXT = """
module_name: "foo"
data_ports {
  direction: DIRECTION_INPUT
  name: "in_ch_data"
  width: 64
}
data_ports {
  direction: DIRECTION_INPUT
  name: "in_ch_2_data"
  width: 64
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "out_ch_data"
  width: 64
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "out_ch_2_data"
  width: 64
}
data_ports {
  direction: DIRECTION_INPUT
  name: "in_ch_vld"
  width: 1
}
data_ports {
  direction: DIRECTION_INPUT
  name: "in_ch_2_vld"
  width: 1
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "out_ch_vld"
  width: 1
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "out_ch_2_vld"
  width: 1
}
data_ports {
  direction: DIRECTION_INPUT
  name: "out_ch_rdy"
  width: 1
}
data_ports {
  direction: DIRECTION_INPUT
  name: "out_ch_2_rdy"
  width: 1
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "in_ch_rdy"
  width: 1
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "in_ch_2_rdy"
  width: 1
}
clock_name: "clk"
reset {
  name: "rst_n"
  asynchronous: false
  active_low: true
}
pipeline {
  latency: 4
  initiation_interval: 1
  pipeline_control {
    valid {
      input_name: "input_valid"
      output_name: "output_valid"
    }
  }
}
"""

BLOCK_MEMORY_IR_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_test_block_memory.ir"
)
BLOCK_MEMORY_SIGNATURE_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_test_block_memory.sig.textproto"
)


def parameterized_block_backends(func):
  return parameterized.named_parameters(
      ("block_jit", ["--backend", "block_jit"]),
      ("block_interpreter", ["--backend", "block_interpreter"]),
  )(func)


def run_command(args):
  """Runs the command described by args and returns the completion object."""
  # Don't use check=True because we want to print stderr/stdout on failure for a
  # better error message.
  # pylint: disable=subprocess-run-check
  comp = subprocess.run(
      args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
  )
  if comp.returncode != 0:
    logging.error("Failed to run: %s", " ".join(args))
    logging.error("stderr: %s", comp.stderr)
    logging.error("stdout: %s", comp.stdout)
  comp.check_returncode()
  return comp


class EvalProcTest(parameterized.TestCase):

  def test_basic(self):
    ir_file = self.create_tempfile(content=PROC_IR)
    input_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:42
          bits[64]:101
        """))
    input_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:10
          bits[64]:6
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:62
          bits[64]:127
        """))
    output_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:55
          bits[64]:55
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file.full_path,
        "--ticks",
        "2",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_channels",
        "in_ch={infile1},in_ch_2={infile2}".format(
            infile1=input_file.full_path, infile2=input_file_2.full_path
        ),
        "--expected_outputs_for_channels",
        "out_ch={outfile},out_ch_2={outfile2}".format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path
        ),
    ]

    output = run_command(shared_args + ["--backend", "ir_interpreter"])
    self.assertIn("Proc test_proc", output.stderr)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn("Proc test_proc", output.stderr)

  def test_basic_run_until_completed(self):
    ir_file = self.create_tempfile(content=PROC_IR)
    input_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:42
          bits[64]:101
        """))
    input_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:10
          bits[64]:6
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:62
          bits[64]:127
        """))
    output_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:55
          bits[64]:55
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file.full_path,
        "--ticks",
        "-1",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_channels",
        "in_ch={infile1},in_ch_2={infile2}".format(
            infile1=input_file.full_path, infile2=input_file_2.full_path
        ),
        "--expected_outputs_for_channels",
        "out_ch={outfile},out_ch_2={outfile2}".format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path
        ),
    ]

    output = run_command(shared_args + ["--backend", "ir_interpreter"])
    self.assertIn("Proc test_proc", output.stderr)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn("Proc test_proc", output.stderr)

  def test_reset_static(self):
    ir_file = self.create_tempfile(content=PROC_IR)
    input_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:42
          bits[64]:101
        """))
    input_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:10
          bits[64]:6
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:62
          bits[64]:117
        """))
    output_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:55
          bits[64]:55
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file.full_path,
        "--ticks",
        "1,1",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_channels",
        "in_ch={infile1},in_ch_2={infile2}".format(
            infile1=input_file.full_path, infile2=input_file_2.full_path
        ),
        "--expected_outputs_for_channels",
        "out_ch={outfile},out_ch_2={outfile2}".format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path
        ),
    ]

    output = run_command(shared_args + ["--backend", "ir_interpreter"])
    self.assertIn("Proc test_proc", output.stderr)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn("Proc test_proc", output.stderr)

  @parameterized_block_backends
  def test_block_filtered_traces(self, backends):
    ir_file = self.create_tempfile(content=BLOCK_IR)
    signature_file = self.create_tempfile(content=BLOCK_SIGNATURE_TEXT)
    input_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:42
          bits[64]:101
        """))
    input_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:10
          bits[64]:6
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:62
          bits[64]:127
        """))
    output_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:55
          bits[64]:55
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file.full_path,
        "--ticks",
        "2",
        "--show_trace",
        "--logtostderr",
        "--block_signature_proto",
        signature_file.full_path,
        "--inputs_for_channels",
        "in_ch={infile1},in_ch_2={infile2}".format(
            infile1=input_file.full_path, infile2=input_file_2.full_path
        ),
        "--expected_outputs_for_channels",
        "out_ch={outfile},out_ch_2={outfile2}".format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path
        ),
    ] + backends

    output = run_command(shared_args)
    self.assertIn("Cycle[6]: resetting? false", output.stderr)

    self.assertNotIn("trace: rst_n 0", output.stderr)
    self.assertNotIn("trace: rst_n 1", output.stderr)

  @parameterized_block_backends
  def test_block_traces_not_filtered(self, backends):
    ir_file = self.create_tempfile(content=BLOCK_IR)
    signature_file = self.create_tempfile(content=BLOCK_SIGNATURE_TEXT)
    input_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:42
          bits[64]:101
        """))
    input_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:10
          bits[64]:6
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:62
          bits[64]:127
        """))
    output_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:55
          bits[64]:55
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file.full_path,
        "--ticks",
        "2",
        "--show_trace",
        "--max_trace_verbosity=2",
        "--logtostderr",
        "--block_signature_proto",
        signature_file.full_path,
        "--inputs_for_channels",
        "in_ch={infile1},in_ch_2={infile2}".format(
            infile1=input_file.full_path, infile2=input_file_2.full_path
        ),
        "--expected_outputs_for_channels",
        "out_ch={outfile},out_ch_2={outfile2}".format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path
        ),
        "--show_trace",
    ] + backends

    output = run_command(shared_args)
    self.assertIn("Cycle[6]: resetting? false", output.stderr)

    self.assertIn("trace: rst_n 0", output.stderr)
    self.assertIn("trace: rst_n 1", output.stderr)

  @parameterized_block_backends
  def test_block_run_until_consumed(self, backends):
    ir_file = self.create_tempfile(content=BLOCK_IR)
    signature_file = self.create_tempfile(content=BLOCK_SIGNATURE_TEXT)
    stats_file = self.create_tempfile(content="")
    input_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:42
          bits[64]:101
        """))
    input_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:10
          bits[64]:6
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:62
          bits[64]:127
        """))
    output_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:55
          bits[64]:55
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file.full_path,
        "--ticks",
        "-1",
        "--show_trace",
        "--logtostderr",
        "--block_signature_proto",
        signature_file.full_path,
        "--inputs_for_channels",
        "in_ch={infile1},in_ch_2={infile2}".format(
            infile1=input_file.full_path, infile2=input_file_2.full_path
        ),
        "--expected_outputs_for_channels",
        "out_ch={outfile},out_ch_2={outfile2}".format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path
        ),
        "--output_stats_path",
        stats_file.full_path,
    ] + backends

    output = run_command(shared_args)
    self.assertIn("Cycle[6]: resetting? false", output.stderr)

    with open(stats_file.full_path, "r") as f:
      stats_content = f.read()
      self.assertIn("6", stats_content)

  @parameterized_block_backends
  def test_block_no_output(self, backend):
    ir_file = self.create_tempfile(content=BLOCK_IR_BROKEN)
    signature_file = self.create_tempfile(content=BLOCK_SIGNATURE_TEXT)
    input_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:42
          bits[64]:101
        """))
    input_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:10
          bits[64]:6
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:62
          bits[64]:127
        """))
    output_file_2 = self.create_tempfile(content=textwrap.dedent("""
          bits[64]:55
          bits[64]:55
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file.full_path,
        "--ticks",
        "2",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--block_signature_proto",
        signature_file.full_path,
        "--inputs_for_channels",
        "in_ch={infile1},in_ch_2={infile2}".format(
            infile1=input_file.full_path, infile2=input_file_2.full_path
        ),
        "--expected_outputs_for_channels",
        "out_ch={outfile},out_ch_2={outfile2}".format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path
        ),
    ] + backend

    comp = subprocess.run(
        shared_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        check=False,
    )
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn("Block didn't produce output", comp.stderr)

  def test_all_channels_in_a_single_file_proc(self):
    ir_file = self.create_tempfile(content=PROC_IR)
    input_file = self.create_tempfile(content=textwrap.dedent("""
          in_ch : {
            bits[64]:42
            bits[64]:101
          }
          in_ch_2 : {
            bits[64]:10
            bits[64]:6
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          out_ch : {
            bits[64]:62
            bits[64]:127
          }
          out_ch_2 : {
            bits[64]:55
            bits[64]:55
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file.full_path,
        "--ticks",
        "2",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_all_channels",
        input_file.full_path,
        "--expected_outputs_for_all_channels",
        output_file.full_path,
    ]

    output = run_command(shared_args + ["--backend", "ir_interpreter"])
    self.assertIn("Proc test_proc", output.stderr)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn("Proc test_proc", output.stderr)

  @parameterized_block_backends
  def test_all_channels_in_a_single_file_block(self, backend):
    ir_file = self.create_tempfile(content=BLOCK_IR)
    signature_file = self.create_tempfile(content=BLOCK_SIGNATURE_TEXT)
    input_file = self.create_tempfile(content=textwrap.dedent("""
          in_ch : {
            bits[64]:42
            bits[64]:101
          }
          in_ch_2 : {
            bits[64]:10
            bits[64]:6
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          out_ch : {
            bits[64]:62
            bits[64]:127
          }
          out_ch_2 : {
            bits[64]:55
            bits[64]:55
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file.full_path,
        "--ticks",
        "2",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--block_signature_proto",
        signature_file.full_path,
        "--inputs_for_all_channels",
        input_file.full_path,
        "--expected_outputs_for_all_channels",
        output_file.full_path,
    ] + backend

    output = run_command(shared_args)
    self.assertIn("Cycle[6]: resetting? false", output.stderr)

  def test_output_channels_stdout_display_proc(self):
    ir_file = self.create_tempfile(content=PROC_IR)
    input_file = self.create_tempfile(content=textwrap.dedent("""
          in_ch : {
            bits[64]:42
            bits[64]:101
          }
          in_ch_2 : {
            bits[64]:10
            bits[64]:6
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file.full_path,
        "--ticks",
        "2",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_all_channels",
        input_file.full_path,
    ]

    output = run_command(shared_args + ["--backend", "ir_interpreter"])
    self.assertIn("Proc test_proc", output.stderr)
    self.assertIn("out_ch : {", output.stdout)
    self.assertIn("out_ch_2 : {", output.stdout)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn("Proc test_proc", output.stderr)
    self.assertIn("out_ch : {", output.stdout)
    self.assertIn("out_ch_2 : {", output.stdout)

  def test_output_channels_with_no_values_stdout_display_proc(self):
    ir_file = self.create_tempfile(content=PROC_IR_CONDITIONAL)
    input_file = self.create_tempfile(content=textwrap.dedent("""
          input : {
            bits[8]:42
            bits[8]:42
            bits[8]:42
            bits[8]:42
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file.full_path,
        "--ticks",
        "4",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_all_channels",
        input_file.full_path,
    ]

    output = run_command(shared_args + ["--backend", "ir_interpreter"])
    self.assertIn("Proc test_proc", output.stderr)
    self.assertIn("output : {\n}", output.stdout)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn("Proc test_proc", output.stderr)
    self.assertIn("output : {\n}", output.stdout)

  @parameterized_block_backends
  def test_block_memory(self, backend):
    ir_file = BLOCK_MEMORY_IR_PATH
    signature_file = BLOCK_MEMORY_SIGNATURE_PATH
    channels_in_ir_file = self.create_tempfile(content=textwrap.dedent("""
          in : {
            bits[32]:0x3,
            bits[32]:0x4,
            bits[32]:0x5,
            bits[32]:0x6,
            bits[32]:0xFF,
            bits[32]:0xFF,
            bits[32]:0xFF,
            bits[32]:0xFF
          }
        """))
    channels_out_ir_file = self.create_tempfile(content=textwrap.dedent("""
          out : {
            bits[32]:7,
            bits[32]:9,
            bits[32]:11,
            bits[32]:13
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file,
        "--inputs_for_all_channels",
        channels_in_ir_file,
        "--expected_outputs_for_all_channels",
        channels_out_ir_file,
        "--block_signature_proto",
        signature_file,
        "--model_memories",
        "mem=4/bits[32]:0xAA",
        "--alsologtostderr",
        "--show_trace",
        "--ticks",
        "12",
    ] + backend

    output = run_command(shared_args)
    self.assertIn(
        "Channel Model: Consuming output for out: bits[32]:13", output.stderr
    )
    self.assertIn(
        "Memory Model: Initiated read mem[3] = bits[32]:6", output.stderr
    )


if __name__ == "__main__":
  absltest.main()
