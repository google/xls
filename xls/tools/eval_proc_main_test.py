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

import struct
import subprocess
import textwrap

from absl import logging

from absl.testing import absltest
from absl.testing import parameterized
from xls.common import runfiles
from xls.ir import xls_value_pb2
from xls.tools import node_coverage_stats_pb2
from xls.tools import proc_channel_values_pb2


EVAL_PROC_MAIN_PATH = runfiles.get_path("xls/tools/eval_proc_main")

PROC_PATH = runfiles.get_path("xls/tools/testdata/eval_proc_main_test.opt.ir")
PROC_CONDITIONAL_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_conditional_test.opt.ir"
)
BLOCK_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_test_with_trace.block.ir"
)
BLOCK_SIG_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_test.sig.textproto"
)
BLOCK_BROKEN_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_test_broken.block.ir"
)
PROC_ZERO_SIZE_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_zero_size_test.opt.ir"
)
BLOCK_ZERO_SIZE_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_zero_size_test.block.ir"
)
BLOCK_SIG_ZERO_SIZE_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_zero_size_test.sig.textproto"
)
BLOCK_MEMORY_IR_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_test.test_memory.block.ir"
)
BLOCK_MEMORY_SIGNATURE_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_test.test_memory.sig.textproto"
)
BLOCK_MEMORY_REWRITES_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_test.ram_rewrites.textproto"
)
PROC_ABSTRACT_MEMORY_IR_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_test.test_memory.ir"
)
PROC_REWRITTEN_MEMORY_IR_PATH = runfiles.get_path(
    "xls/tools/testdata/eval_proc_main_test.test_memory.opt.ir"
)

# Block generated from the proc with:
# --delay_model=unit --pipeline_stages=1 --reset=rst
# TODO(allight): Rewrite test to be writable using a dslx source.
OBSERVER_IR = '''
package ObserverTest

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=proven_mutually_exclusive, metadata="""block_ports { block_name: "ObserverTest" data_port_name: "in_data" ready_port_name: "in_rdy" valid_port_name: "in_vld" }""")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, strictness=proven_mutually_exclusive, metadata="""block_ports { block_name: "ObserverTest" data_port_name: "out_data" ready_port_name: "out_rdy" valid_port_name: "out_vld" }""")

top proc ObserverTest(st: bits[32] id=78, init={0}) {
  literal.2: token = literal(value=token, id=2)
  receive.3: (token, bits[32], bits[1]) = receive(literal.2, channel=in, blocking=false, id=3)
  tuple_index.6: bits[32] = tuple_index(receive.3, index=1, id=6)
  add.7: bits[32] = add(tuple_index.6, st, id=7)
  send.4: token = send(literal.2, st, channel=out, id=4)
  next_value.8: () = next_value(param=st, value=add.7, id=8)
}

block ObserverTest(clk: clock, in_data: bits[32], in_vld: bits[1], out_data: bits[32], rst: bits[1], out_rdy: bits[1], out_vld: bits[1], in_rdy: bits[1]) {
  reg __st(bits[32], reset_value=0, asynchronous=false, active_low=false)

  reg __in_data_reg(bits[32], reset_value=0, asynchronous=false, active_low=false)

  reg __in_data_valid_reg(bits[1], reset_value=0, asynchronous=false, active_low=false)

  reg __out_data_reg(bits[32], reset_value=0, asynchronous=false, active_low=false)

  reg __out_data_valid_reg(bits[1], reset_value=0, asynchronous=false, active_low=false)

  in_data: bits[32] = input_port(name=in_data, id=13)
  in_vld: bits[1] = input_port(name=in_vld, id=15)
  rst: bits[1] = input_port(name=rst, id=25)
  out_rdy: bits[1] = input_port(name=out_rdy, id=26)
  __in_data_reg: bits[32] = register_read(register=__in_data_reg, id=37)
  literal.16: bits[32] = literal(value=0, id=16)
  __st__1: bits[32] = register_read(register=__st, id=20)
  __out_vld_buf: bits[1] = literal(value=1, id=59)
  __in_data_valid_reg: bits[1] = register_read(register=__in_data_valid_reg, id=39)
  in_select: bits[32] = sel(__in_data_valid_reg, cases=[literal.16, __in_data_reg], id=17)
  add.21: bits[32] = add(in_select, __st__1, id=21)
  __out_data_reg: bits[32] = register_read(register=__out_data_reg, id=49)
  __out_data_valid_reg: bits[1] = register_read(register=__out_data_valid_reg, id=51)
  in_data_valid_inv: bits[1] = not(__in_data_valid_reg, id=41)
  out_data_valid_inv: bits[1] = not(__out_data_valid_reg, id=53)
  out_data_valid_load_en: bits[1] = or(out_rdy, out_data_valid_inv, id=54)
  register_write.35: () = register_write(add.21, register=__st, load_enable=out_data_valid_load_en, reset=rst, id=35)
  register_write_50: () = register_write(__st__1, register=__out_data_reg, load_enable=out_data_valid_load_en, reset=rst, id=56)
  register_write_52: () = register_write(__out_vld_buf, register=__out_data_valid_reg, load_enable=out_data_valid_load_en, reset=rst, id=57)
  in_data_valid_load_en: bits[1] = or(out_data_valid_load_en, in_data_valid_inv, id=42)
  register_write_40: () = register_write(in_vld, register=__in_data_valid_reg, load_enable=in_data_valid_load_en, reset=rst, id=45)
  in_data_load_en: bits[1] = and(in_vld, in_data_valid_load_en, id=43)
  register_write_38: () = register_write(in_data, register=__in_data_reg, load_enable=in_data_load_en, reset=rst, id=44)
  out_data: () = output_port(__out_data_reg, name=out_data, id=22)
  out_vld: () = output_port(__out_data_valid_reg, name=out_vld, id=33)
  in_rdy: () = output_port(in_data_load_en, name=in_rdy, id=36)
}
'''

OBSERVER_BLOCK_SIG = """
module_name: "ObserverTest"
data_ports {
  direction: DIRECTION_INPUT
  name: "in_data"
  width: 32
  type {
    type_enum: BITS
    bit_count: 32
  }
}
data_ports {
  direction: DIRECTION_INPUT
  name: "in_vld"
  width: 1
  type {
    type_enum: BITS
    bit_count: 1
  }
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "out_data"
  width: 32
  type {
    type_enum: BITS
    bit_count: 32
  }
}
data_ports {
  direction: DIRECTION_INPUT
  name: "out_rdy"
  width: 1
  type {
    type_enum: BITS
    bit_count: 1
  }
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "out_vld"
  width: 1
  type {
    type_enum: BITS
    bit_count: 1
  }
}
data_ports {
  direction: DIRECTION_OUTPUT
  name: "in_rdy"
  width: 1
  type {
    type_enum: BITS
    bit_count: 1
  }
}
clock_name: "clk"
reset {
  name: "rst"
  asynchronous: false
  active_low: false
}
combinational {
}
data_channels {
  name: "in"
  kind: CHANNEL_KIND_STREAMING
  supported_ops: CHANNEL_OPS_RECEIVE_ONLY
  flow_control: CHANNEL_FLOW_CONTROL_READY_VALID
  type {
    type_enum: BITS
    bit_count: 32
  }
  metadata {
    block_ports {
      block_name: "ObserverTest"
      data_port_name: "in_data"
      ready_port_name: "in_rdy"
      valid_port_name: "in_vld"
    }
    block_ports {
      block_name: "ObserverTest"
      data_port_name: "in_data"
      ready_port_name: "in_rdy"
      valid_port_name: "in_vld"
    }
  }
}
data_channels {
  name: "out"
  kind: CHANNEL_KIND_STREAMING
  supported_ops: CHANNEL_OPS_SEND_ONLY
  flow_control: CHANNEL_FLOW_CONTROL_READY_VALID
  type {
    type_enum: BITS
    bit_count: 32
  }
  metadata {
    block_ports {
      block_name: "ObserverTest"
      data_port_name: "out_data"
      ready_port_name: "out_rdy"
      valid_port_name: "out_vld"
    }
    block_ports {
      block_name: "ObserverTest"
      data_port_name: "out_data"
      ready_port_name: "out_rdy"
      valid_port_name: "out_vld"
    }
  }
}
"""

OBSERVER_INPUT_CHANNEL_VALUES = """
in: {
  bits[32]:1
  bits[32]:2
}
"""

OBSERVER_OUTPUT_PROC_CHANNEL_VALUES = """
out: {
  bits[32]:0
  bits[32]:1
  bits[32]:3
  bits[32]:3
}
"""

# Block has an extra output for the first cycle where its being reset.
OBSERVER_OUTPUT_BLOCK_CHANNEL_VALUES = """
out: {
  bits[32]:0
  bits[32]:0
  bits[32]:1
  bits[32]:3
  bits[32]:3
}
"""

MULTI_BLOCK_IR_FILE = runfiles.get_path(
    "xls/examples/dslx_module/manual_chan_caps_streaming_configured_multiproc.block.ir"
)
MULTI_BLOCK_SIG_FILE = runfiles.get_path(
    "xls/examples/dslx_module/manual_chan_caps_streaming_configured_multiproc.sig.textproto"
)
MULTI_BLOCK_MEMORY_IR_FILE = runfiles.get_path("xls/examples/delay.block.ir")
MULTI_BLOCK_MEMORY_SIG_FILE = runfiles.get_path(
    "xls/examples/delay.sig.textproto"
)


def _eight_chars(val: bytes) -> xls_value_pb2.ValueProto:
  assert len(val) == 8
  return xls_value_pb2.ValueProto(
      array=xls_value_pb2.ValueProto.Array(
          elements=[
              xls_value_pb2.ValueProto(
                  bits=xls_value_pb2.ValueProto.Bits(
                      bit_count=8, data=bytes([v])
                  )
              )
              for v in val
          ]
      )
  )


MULTI_BLOCK_INPUT_CHANNEL_VALUES = (
    proc_channel_values_pb2.ProcChannelValuesProto(
        channels=[
            proc_channel_values_pb2.ProcChannelValuesProto.Channel(
                name="some_caps_streaming_configured__external_input_wire",
                entry=[
                    _eight_chars(b"abcdabcd"),
                    _eight_chars(b"abcdabcd"),
                    _eight_chars(b"abcdabcd"),
                    _eight_chars(b"abcdabcd"),
                    _eight_chars(b"abcdabcd"),
                    _eight_chars(b"abcdabcd"),
                ],
            )
        ]
    )
)
MULTI_BLOCK_OUTPUT_CHANNEL_VALUES = (
    proc_channel_values_pb2.ProcChannelValuesProto(
        channels=[
            proc_channel_values_pb2.ProcChannelValuesProto.Channel(
                name="some_caps_streaming_configured__external_output_wire",
                entry=[
                    _eight_chars(b"ABCDABCD"),
                    _eight_chars(b"abcdabcd"),
                    _eight_chars(b"AbCdAbCd"),
                    _eight_chars(b"ABCDABCD"),
                    _eight_chars(b"abcdabcd"),
                    _eight_chars(b"AbCdAbCd"),
                ],
            )
        ]
    )
)

TOKEN = xls_value_pb2.ValueProto(token=xls_value_pb2.ValueProto.Token())
_ONE_BIT_TRUE = xls_value_pb2.ValueProto(
    bits=xls_value_pb2.ValueProto.Bits(bit_count=1, data=b"\1")
)
_ONE_BIT_FALSE = xls_value_pb2.ValueProto(
    bits=xls_value_pb2.ValueProto.Bits(bit_count=1, data=b"\0")
)


def _value_32_bits(v: int) -> xls_value_pb2.ValueProto:
  return xls_value_pb2.ValueProto(
      bits=xls_value_pb2.ValueProto.Bits(
          bit_count=32, data=struct.pack("<i", v)
      )
  )


def _value_tuple(vs) -> xls_value_pb2.ValueProto:
  res = []
  for v in vs:
    if isinstance(v, xls_value_pb2.ValueProto):
      res.append(v)
    elif isinstance(v, int):
      res.append(_value_32_bits(v))
    else:
      raise TypeError(f"Unexpected type of {v}: {type(v)}")
  return xls_value_pb2.ValueProto(
      tuple=xls_value_pb2.ValueProto.Tuple(elements=res)
  )


def parameterized_block_backends(func):
  return parameterized.named_parameters(
      ("block_jit", ["--backend", "block_jit"]),
      ("block_interpreter", ["--backend", "block_interpreter"]),
  )(func)


def parameterized_proc_backends(func):
  return parameterized.named_parameters(
      ("serial_jit", ["--backend", "serial_jit"]),
      ("ir_interpreter", ["--backend", "ir_interpreter"]),
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
    logging.error("Failed to run: %s", repr(args))
    logging.error("stderr: %s", comp.stderr)
    logging.error("stdout: %s", comp.stdout)
  comp.check_returncode()
  return comp


class EvalProcTest(parameterized.TestCase):

  def test_basic(self):
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
        PROC_PATH,
        "--ticks",
        "2",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_channels",
        "eval_proc_main_test__in_ch={infile1},eval_proc_main_test__in_ch_2={infile2}"
        .format(infile1=input_file.full_path, infile2=input_file_2.full_path),
        "--expected_outputs_for_channels",
        "eval_proc_main_test__out_ch={outfile},eval_proc_main_test__out_ch_2={outfile2}"
        .format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path
        ),
    ]

    output = run_command(shared_args + ["--backend", "ir_interpreter"])
    self.assertIn("Proc __eval_proc_main_test__test_proc_0_next", output.stderr)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn("Proc __eval_proc_main_test__test_proc_0_next", output.stderr)

  def test_basic_run_until_completed(self):
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
        BLOCK_PATH,
        "--ticks",
        "-1",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_channels",
        "eval_proc_main_test__in_ch={infile1},eval_proc_main_test__in_ch_2={infile2}"
        .format(infile1=input_file.full_path, infile2=input_file_2.full_path),
        "--expected_outputs_for_channels",
        "eval_proc_main_test__out_ch={outfile},eval_proc_main_test__out_ch_2={outfile2}"
        .format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path
        ),
    ]

    output = run_command(shared_args + ["--backend", "ir_interpreter"])
    self.assertIn("Proc __eval_proc_main_test__test_proc_0_next", output.stderr)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn("Proc __eval_proc_main_test__test_proc_0_next", output.stderr)

  def test_reset_static(self):
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
        PROC_PATH,
        "--ticks",
        "1,1",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_channels",
        "eval_proc_main_test__in_ch={infile1},eval_proc_main_test__in_ch_2={infile2}"
        .format(infile1=input_file.full_path, infile2=input_file_2.full_path),
        "--expected_outputs_for_channels",
        "eval_proc_main_test__out_ch={outfile},eval_proc_main_test__out_ch_2={outfile2}"
        .format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path
        ),
    ]

    output = run_command(shared_args + ["--backend", "ir_interpreter"])
    self.assertIn("Proc __eval_proc_main_test__test_proc_0_next", output.stderr)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn("Proc __eval_proc_main_test__test_proc_0_next", output.stderr)

  @parameterized_block_backends
  def test_block_filtered_traces(self, backends):
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
        BLOCK_PATH,
        "--ticks",
        "2",
        "--show_trace",
        "--logtostderr",
        "--block_signature_proto",
        BLOCK_SIG_PATH,
        "--inputs_for_channels",
        "eval_proc_main_test__in_ch={infile1},eval_proc_main_test__in_ch_2={infile2}"
        .format(infile1=input_file.full_path, infile2=input_file_2.full_path),
        "--expected_outputs_for_channels",
        "eval_proc_main_test__out_ch={outfile},eval_proc_main_test__out_ch_2={outfile2}"
        .format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path
        ),
    ] + backends

    output = run_command(shared_args)
    self.assertIn("Cycle[6]: resetting? false", output.stderr)

    self.assertNotIn("trace: rst_n 0", output.stderr)
    self.assertNotIn("trace: rst_n 1", output.stderr)

  @parameterized_block_backends
  def test_block_traces_not_filtered(self, backends):
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
        BLOCK_PATH,
        "--ticks",
        "2",
        "--show_trace",
        "--max_trace_verbosity=2",
        "--logtostderr",
        "--block_signature_proto",
        BLOCK_SIG_PATH,
        "--inputs_for_channels",
        "eval_proc_main_test__in_ch={infile1},eval_proc_main_test__in_ch_2={infile2}"
        .format(infile1=input_file.full_path, infile2=input_file_2.full_path),
        "--expected_outputs_for_channels",
        "eval_proc_main_test__out_ch={outfile},eval_proc_main_test__out_ch_2={outfile2}"
        .format(
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
        BLOCK_PATH,
        "--ticks",
        "-1",
        "--show_trace",
        "--logtostderr",
        "--block_signature_proto",
        BLOCK_SIG_PATH,
        "--inputs_for_channels",
        "eval_proc_main_test__in_ch={infile1},eval_proc_main_test__in_ch_2={infile2}"
        .format(infile1=input_file.full_path, infile2=input_file_2.full_path),
        "--expected_outputs_for_channels",
        "eval_proc_main_test__out_ch={outfile},eval_proc_main_test__out_ch_2={outfile2}"
        .format(
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
        BLOCK_BROKEN_PATH,
        "--ticks",
        "2",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--block_signature_proto",
        BLOCK_SIG_PATH,
        "--inputs_for_channels",
        "eval_proc_main_test__in_ch={infile1},eval_proc_main_test__in_ch_2={infile2}"
        .format(infile1=input_file.full_path, infile2=input_file_2.full_path),
        "--expected_outputs_for_channels",
        "eval_proc_main_test__out_ch={outfile},eval_proc_main_test__out_ch_2={outfile2}"
        .format(
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
    input_file = self.create_tempfile(content=textwrap.dedent("""
          eval_proc_main_test__in_ch : {
            bits[64]:42
            bits[64]:101
          }
          eval_proc_main_test__in_ch_2 : {
            bits[64]:10
            bits[64]:6
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          eval_proc_main_test__out_ch : {
            bits[64]:62
            bits[64]:127
          }
          eval_proc_main_test__out_ch_2 : {
            bits[64]:55
            bits[64]:55
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        PROC_PATH,
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
    self.assertIn("Proc __eval_proc_main_test__test_proc_0_next", output.stderr)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn("Proc __eval_proc_main_test__test_proc_0_next", output.stderr)

  @parameterized_block_backends
  def test_all_channels_in_a_single_file_block(self, backend):
    input_file = self.create_tempfile(content=textwrap.dedent("""
          eval_proc_main_test__in_ch : {
            bits[64]:42
            bits[64]:101
          }
          eval_proc_main_test__in_ch_2 : {
            bits[64]:10
            bits[64]:6
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          eval_proc_main_test__out_ch : {
            bits[64]:62
            bits[64]:127
          }
          eval_proc_main_test__out_ch_2 : {
            bits[64]:55
            bits[64]:55
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        BLOCK_PATH,
        "--ticks",
        "2",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--block_signature_proto",
        BLOCK_SIG_PATH,
        "--inputs_for_all_channels",
        input_file.full_path,
        "--expected_outputs_for_all_channels",
        output_file.full_path,
    ] + backend

    output = run_command(shared_args)
    self.assertIn("Cycle[6]: resetting? false", output.stderr)

  def test_output_channels_stdout_display_proc(self):
    input_file = self.create_tempfile(content=textwrap.dedent("""
          eval_proc_main_test__in_ch : {
            bits[64]:42
            bits[64]:101
          }
          eval_proc_main_test__in_ch_2 : {
            bits[64]:10
            bits[64]:6
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        PROC_PATH,
        "--ticks",
        "2",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_all_channels",
        input_file.full_path,
    ]

    output = run_command(shared_args + ["--backend", "ir_interpreter"])
    self.assertIn("Proc __eval_proc_main_test__test_proc_0_next", output.stderr)
    self.assertIn("eval_proc_main_test__out_ch : {", output.stdout)
    self.assertIn("eval_proc_main_test__out_ch_2 : {", output.stdout)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn("Proc __eval_proc_main_test__test_proc_0_next", output.stderr)
    self.assertIn("eval_proc_main_test__out_ch : {", output.stdout)
    self.assertIn("eval_proc_main_test__out_ch_2 : {", output.stdout)

  def test_output_channels_with_no_values_stdout_display_proc(self):
    input_file = self.create_tempfile(content=textwrap.dedent("""
          eval_proc_main_conditional_test__input : {
            bits[8]:42
            bits[8]:42
            bits[8]:42
            bits[8]:42
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        PROC_CONDITIONAL_PATH,
        "--ticks",
        "4",
        "-v=3",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_all_channels",
        input_file.full_path,
    ]

    output = run_command(shared_args + ["--backend", "ir_interpreter"])
    self.assertIn(
        "Proc __eval_proc_main_conditional_test__test_proc_0_next",
        output.stderr,
    )
    self.assertIn("output : {\n}", output.stdout)

    output = run_command(shared_args + ["--backend", "serial_jit"])
    self.assertIn(
        "Proc __eval_proc_main_conditional_test__test_proc_0_next",
        output.stderr,
    )
    self.assertIn("output : {\n}", output.stdout)

  @parameterized_block_backends
  def test_block_memory(self, backend):
    ir_file = BLOCK_MEMORY_IR_PATH
    ram_rewrites_file = BLOCK_MEMORY_REWRITES_PATH
    signature_file = BLOCK_MEMORY_SIGNATURE_PATH
    input_file = self.create_tempfile(content=textwrap.dedent("""
          in : {
            bits[32]:42
            bits[32]:101
            bits[32]:50
            bits[32]:11
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          out : {
            bits[32]:126
            bits[32]:303
            bits[32]:150
            bits[32]:33
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file,
        "--ticks",
        "17",  # Need double the ticks for the memory model to run
        "-v=1",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_all_channels",
        input_file.full_path,
        "--expected_outputs_for_all_channels",
        output_file.full_path,
        "--block_signature_proto",
        signature_file,
        "--ram_rewrites_textproto",
        ram_rewrites_file,
    ] + backend

    output = run_command(shared_args)
    self.assertIn("Memory Model", output.stderr)

  @parameterized_block_backends
  def test_multi_block_memory(self, backend):
    tick_count = 3 * 2048
    ir_file = MULTI_BLOCK_MEMORY_IR_FILE
    signature_file = MULTI_BLOCK_MEMORY_SIG_FILE
    input_channel = proc_channel_values_pb2.ProcChannelValuesProto.Channel(
        name="delay__data_in"
    )
    output_channel = proc_channel_values_pb2.ProcChannelValuesProto.Channel(
        name="delay__data_out"
    )
    # Make a little oracle to get the results.
    buffer = [3] * 2048
    for t in range(tick_count):
      input_channel.entry.append(_value_32_bits(t))
      buffer.append(t)
      output_channel.entry.append(_value_32_bits(buffer.pop(0)))

    # Create input and output args
    input_data = proc_channel_values_pb2.ProcChannelValuesProto(
        channels=[input_channel]
    )
    output_data = proc_channel_values_pb2.ProcChannelValuesProto(
        channels=[output_channel]
    )
    channels_in_ir_file = self.create_tempfile(
        content=input_data.SerializeToString()
    )
    channels_out_ir_file = self.create_tempfile(
        content=output_data.SerializeToString()
    )

    # Needed for memory model size
    rewrites_stub = """
rewrites {
  from_config {
    kind: RAM_ABSTRACT
    depth: 1024
  }
  to_config {
    kind: RAM_1RW
    depth: 1024
  }
  to_name_prefix: "ram"
}
"""

    rewrites_file = self.create_tempfile(content=rewrites_stub)

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file,
        "--top=delay_top",
        "--proto_inputs_for_all_channels",
        channels_in_ir_file,
        "--expected_proto_outputs_for_all_channels",
        channels_out_ir_file,
        "--block_signature_proto",
        signature_file,
        "--ram_rewrites_textproto",
        rewrites_file,
        "--alsologtostderr",
        "--show_trace",
        "--ticks",
        "-1",
        # f"{tick_count + 1}",
    ] + backend

    run_command(shared_args)

  @parameterized_proc_backends
  def test_proc_abstract_memory(self, backend):
    ir_file = PROC_ABSTRACT_MEMORY_IR_PATH
    ram_rewrites_file = BLOCK_MEMORY_REWRITES_PATH
    input_file = self.create_tempfile(content=textwrap.dedent("""
          in : {
            bits[32]:42
            bits[32]:101
            bits[32]:50
            bits[32]:11
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          out : {
            bits[32]:126
            bits[32]:303
            bits[32]:150
            bits[32]:33
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file,
        "--ticks",
        "17",  # Need double the ticks for the memory model to run
        "-v=1",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_all_channels",
        input_file.full_path,
        "--expected_outputs_for_all_channels",
        output_file.full_path,
        "--ram_rewrites_textproto",
        ram_rewrites_file,
        "--abstract_ram_model",
    ] + backend

    output = run_command(shared_args)
    self.assertIn("Proc Test_proc", output.stderr)

  @parameterized_proc_backends
  def test_proc_rewritten_memory(self, backend):
    ir_file = PROC_REWRITTEN_MEMORY_IR_PATH
    ram_rewrites_file = BLOCK_MEMORY_REWRITES_PATH
    input_file = self.create_tempfile(content=textwrap.dedent("""
          in : {
            bits[32]:42
            bits[32]:101
            bits[32]:50
            bits[32]:11
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          out : {
            bits[32]:126
            bits[32]:303
            bits[32]:150
            bits[32]:33
          }
        """))

    shared_args = [
        EVAL_PROC_MAIN_PATH,
        ir_file,
        "--ticks",
        "17",  # Need double the ticks for the memory model to run
        "-v=1",
        "--show_trace",
        "--logtostderr",
        "--inputs_for_all_channels",
        input_file.full_path,
        "--expected_outputs_for_all_channels",
        output_file.full_path,
        "--ram_rewrites_textproto",
        ram_rewrites_file,
    ] + backend

    output = run_command(shared_args)
    self.assertIn("Proc Test_proc", output.stderr)

  @parameterized_block_backends
  def test_observe_block(self, backend):
    ir_file = self.create_tempfile(content=OBSERVER_IR)
    sig_file = self.create_tempfile(content=OBSERVER_BLOCK_SIG)
    inp_file = self.create_tempfile(content=OBSERVER_INPUT_CHANNEL_VALUES)
    out_file = self.create_tempfile(
        content=OBSERVER_OUTPUT_BLOCK_CHANNEL_VALUES
    )
    observer_values_out = self.create_tempfile()
    run_command(
        [
            EVAL_PROC_MAIN_PATH,
            ir_file.full_path,
            f"--inputs_for_all_channels={inp_file.full_path}",
            f"--expected_outputs_for_all_channels={out_file.full_path}",
            f"--block_signature_proto={sig_file.full_path}",
            "--alsologtostderr",
            "--ticks=5",
            f"--output_node_coverage_stats_proto={observer_values_out.full_path}",
        ]
        + backend
    )
    node_coverage = node_coverage_stats_pb2.NodeCoverageStatsProto.FromString(
        observer_values_out.read_bytes()
    )
    node_stats = node_coverage_stats_pb2.NodeCoverageStatsProto.NodeStats
    node_coverage.nodes.sort(key=lambda n: n.node_id)
    # Reset signal should not be included.
    self.assertIn(
        node_stats(
            node_id=25,
            node_text="rst: bits[1] = input_port(name=rst, id=25)",
            set_bits=_ONE_BIT_FALSE,
            total_bit_count=1,
            unset_bit_count=1,
        ),
        node_coverage.nodes,
    )
    self.assertIn(
        node_stats(
            node_id=21,
            node_text="add.21: bits[32] = add(in_select, __st__1, id=21)",
            set_bits=_value_32_bits(3),
            total_bit_count=32,
            unset_bit_count=30,
        ),
        node_coverage.nodes,
    )
    # TODO(allight): Due to slight differences in how the jit works vs
    # interpreter literals are not always emitted by the block jit. This is
    # pretty irrelevant though since literals do not contribute to coverage in
    # any meaningful way.
    self.assertLen(
        [v for v in node_coverage.nodes if " = literal(" not in v.node_text],
        24,
    )

  @parameterized_proc_backends
  def test_observe_proc(self, backend):
    ir_file = self.create_tempfile(content=OBSERVER_IR)
    inp_file = self.create_tempfile(content=OBSERVER_INPUT_CHANNEL_VALUES)
    out_file = self.create_tempfile(content=OBSERVER_OUTPUT_PROC_CHANNEL_VALUES)
    observer_values_out = self.create_tempfile()
    run_command(
        [
            EVAL_PROC_MAIN_PATH,
            ir_file.full_path,
            f"--inputs_for_all_channels={inp_file.full_path}",
            f"--expected_outputs_for_all_channels={out_file.full_path}",
            "--alsologtostderr",
            "--ticks=4",
            f"--output_node_coverage_stats_proto={observer_values_out.full_path}",
        ]
        + backend
    )
    node_coverage = node_coverage_stats_pb2.NodeCoverageStatsProto.FromString(
        observer_values_out.read_bytes()
    )
    node_stats = node_coverage_stats_pb2.NodeCoverageStatsProto.NodeStats
    node_coverage.nodes.sort(key=lambda n: n.node_id)
    self.assertContainsSubsequence(
        node_coverage.nodes,
        [
            node_stats(
                node_id=2,
                node_text="literal.2: token = literal(value=token, id=2)",
                set_bits=TOKEN,
            ),
            node_stats(
                node_id=3,
                node_text=(
                    "receive.3: (token, bits[32], bits[1]) ="
                    " receive(literal.2, channel=in, blocking=false, id=3)"
                ),
                set_bits=_value_tuple([TOKEN, 0b11, _ONE_BIT_TRUE]),
                total_bit_count=33,
                unset_bit_count=30,
            ),
            node_stats(
                node_id=4,
                node_text=(
                    "send.4: token = send(literal.2, st, channel=out, id=4)"
                ),
                set_bits=TOKEN,
            ),
            node_stats(
                node_id=78,
                node_text="st: bits[32] = state_read(state_element=st, id=78)",
                set_bits=_value_32_bits(0b11),
                total_bit_count=32,
                unset_bit_count=30,
            ),
        ],
    )
    self.assertLen(node_coverage.nodes, 7)

  @parameterized_proc_backends
  def test_multi_proc(self, backend):
    ir_file = MULTI_BLOCK_IR_FILE
    channels_in_file = self.create_tempfile(
        content=MULTI_BLOCK_INPUT_CHANNEL_VALUES.SerializeToString()
    )
    channels_out_file = self.create_tempfile(
        content=MULTI_BLOCK_OUTPUT_CHANNEL_VALUES.SerializeToString()
    )
    run_command(
        [
            EVAL_PROC_MAIN_PATH,
            ir_file,
            f"--proto_inputs_for_all_channels={channels_in_file.full_path}",
            f"--expected_proto_outputs_for_all_channels={channels_out_file.full_path}",
            "--alsologtostderr",
            "--show_trace",
            "--ticks=6",
        ]
        + backend
    )

  @parameterized_block_backends
  def test_multi_block(self, backend):
    ir_file = MULTI_BLOCK_IR_FILE
    sig_file = MULTI_BLOCK_SIG_FILE
    channels_in_file = self.create_tempfile(
        content=MULTI_BLOCK_INPUT_CHANNEL_VALUES.SerializeToString()
    )
    channels_out_file = self.create_tempfile(
        content=MULTI_BLOCK_OUTPUT_CHANNEL_VALUES.SerializeToString()
    )
    run_command(
        [
            EVAL_PROC_MAIN_PATH,
            ir_file,
            f"--block_signature_proto={sig_file}",
            "--top=manual_chan_caps_streaming",
            f"--proto_inputs_for_all_channels={channels_in_file.full_path}",
            f"--expected_proto_outputs_for_all_channels={channels_out_file.full_path}",
            "--alsologtostderr",
            "--show_trace",
            "--ticks=6",
        ]
        + backend
    )

  @parameterized_proc_backends
  def test_zero_size_proc(self, backend):
    input_file = self.create_tempfile(content=textwrap.dedent("""
          eval_proc_main_zero_size_test__in_ch : {
            bits[64]:42
            bits[64]:101
          }
          eval_proc_main_zero_size_test__in_ch_2 : {
            ()
            ()
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          eval_proc_main_zero_size_test__out_ch : {
            bits[64]:43
            bits[64]:112
          }
          eval_proc_main_zero_size_test__out_ch_2 : {
            ()
            ()
          }
        """))

    run_command(
        [
            EVAL_PROC_MAIN_PATH,
            PROC_ZERO_SIZE_PATH,
            "--ticks",
            "2",
            "--logtostderr",
            "--inputs_for_all_channels",
            input_file.full_path,
            "--expected_outputs_for_all_channels",
            output_file.full_path,
        ]
        + backend
    )

  @parameterized_block_backends
  def test_zero_size_block(self, backend):
    input_file = self.create_tempfile(content=textwrap.dedent("""
          eval_proc_main_zero_size_test__in_ch : {
            bits[64]:42
            bits[64]:101
          }
          eval_proc_main_zero_size_test__in_ch_2 : {
            ()
            ()
          }
        """))
    output_file = self.create_tempfile(content=textwrap.dedent("""
          eval_proc_main_zero_size_test__out_ch : {
            bits[64]:43
            bits[64]:112
          }
          eval_proc_main_zero_size_test__out_ch_2 : {
            ()
            ()
          }
        """))

    run_command(
        [
            EVAL_PROC_MAIN_PATH,
            BLOCK_ZERO_SIZE_PATH,
            "--ticks",
            "2",
            "--logtostderr",
            "--block_signature_proto",
            BLOCK_SIG_ZERO_SIZE_PATH,
            "--inputs_for_all_channels",
            input_file.full_path,
            "--expected_outputs_for_all_channels",
            output_file.full_path,
        ]
        + backend
    )


if __name__ == "__main__":
  absltest.main()
