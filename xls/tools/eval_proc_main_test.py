# Lint as: python3
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

from xls.common import runfiles
from absl.testing import absltest

EVAL_PROC_MAIN_PATH = runfiles.get_path("xls/tools/eval_proc_main")

PROC_IR = """package foo

chan in_ch(bits[64], id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata=\"\"\"\"\"\")
chan in_ch_2(bits[64], id=2, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata=\"\"\"\"\"\")
chan out_ch(bits[64], id=3, kind=streaming, ops=send_only, flow_control=ready_valid, metadata=\"\"\"\"\"\")
chan out_ch_2(bits[64], id=4, kind=streaming, ops=send_only, flow_control=ready_valid, metadata=\"\"\"\"\"\")

proc test_proc(tkn: token, st: (), init=()) {
  receive.1: (token, bits[64]) = receive(tkn, channel_id=1, id=1)

  literal.3: bits[1] = literal(value=1, id=3)
  tuple_index.7: token = tuple_index(receive.1, index=0, id=7)
  tuple_index.4: bits[64] = tuple_index(receive.1, index=1, id=4)
  receive.9: (token, bits[64]) = receive(tuple_index.7, channel_id=2, id=9)
  tuple_index.10: bits[64] = tuple_index(receive.9, index=1, id=10)
  add.8: bits[64] = add(tuple_index.4, tuple_index.10, id=8)

  tuple_index.11: token = tuple_index(receive.9, index=0, id=11)
  send.2: token = send(tuple_index.11, add.8, predicate=literal.3, channel_id=3, id=2)
  literal.14: bits[64] = literal(value=55, id=14)
  send.12: token = send(send.2, literal.14, predicate=literal.3, channel_id=4, id=12)

  next(send.12, st)
}
"""


class EvalProcTest(absltest.TestCase):

  def test_give_me_a_name(self):
    ir_file = self.create_tempfile(content=PROC_IR)
    input_file = self.create_tempfile(content="""
bits[64]:42
bits[64]:101
""")
    input_file_2 = self.create_tempfile(content="""
bits[64]:10
bits[64]:6
""")
    output_file = self.create_tempfile(content="""
bits[64]:52
bits[64]:107
""")
    output_file_2 = self.create_tempfile(content="""
bits[64]:55
bits[64]:55
""")

    shared_args = [
        EVAL_PROC_MAIN_PATH, ir_file.full_path, "--ticks", "2", "-v=3",
        "--logtostderr", "--inputs_for_channels",
        "in_ch={infile1},in_ch_2={infile2}".format(
            infile1=input_file.full_path,
            infile2=input_file_2.full_path), "--expected_outputs_for_channels",
        "out_ch={outfile},out_ch_2={outfile2}".format(
            outfile=output_file.full_path, outfile2=output_file_2.full_path)
    ]

    output = subprocess.run(
        shared_args + ["--backend", "ir_interpreter"],
        capture_output=True,
        check=True)
    self.assertIn("Proc test_proc", output.stderr.decode("utf-8"))
    output = subprocess.run(
        shared_args + ["--backend", "serial_jit"],
        capture_output=True,
        check=True)
    self.assertIn("Proc test_proc", output.stderr.decode("utf-8"))


if __name__ == "__main__":
  absltest.main()
