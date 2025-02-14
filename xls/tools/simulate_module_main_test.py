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

from xls.common import runfiles
from xls.common import test_base

CODEGEN_MAIN_PATH = runfiles.get_path('xls/tools/codegen_main')
SIMULATE_MODULE_MAIN_PATH = runfiles.get_path('xls/tools/simulate_module_main')

ADD_IR_FUNCTION = """package add

top fn add(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
"""

ADD_IR_PROC = """package sample

file_number 0 "fake_file.x"

chan sample__operand_0(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan sample__operand_1(bits[32], id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan sample__result(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid)

top proc add(init={}) {
  __token: token = literal(value=token, id=1000)
  receive.4: (token, bits[32]) = receive(__token, channel=sample__operand_0, id=4)
  receive.7: (token, bits[32]) = receive(__token, channel=sample__operand_1, id=7)
  tok_operand_0_val: token = tuple_index(receive.4, index=0, id=5, pos=[(0,14,9)])
  tok_operand_1_val: token = tuple_index(receive.7, index=0, id=8, pos=[(0,15,9)])
  operand_0_val: bits[32] = tuple_index(receive.4, index=1, id=6, pos=[(0,14,28)])
  operand_1_val: bits[32] = tuple_index(receive.7, index=1, id=9, pos=[(0,15,28)])
  tok_recv: token = after_all(tok_operand_0_val, tok_operand_1_val, id=10)
  result_val: bits[32] = add(operand_0_val, operand_1_val, id=11, pos=[(0,18,35)])
  tok_send: token = send(tok_recv, result_val, channel=sample__result, id=12)
}
"""

PROC_WITH_NO_OUTPUT_CHANNEL = """package sample

file_number 0 "fake_file.x"

chan in0(bits[12], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan in1(bits[42], id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)

top proc no_output_channels(init={}) {
  __token: token = literal(value=token, id=1000)
  recv0: (token, bits[12]) = receive(__token, channel=in0, id=4)
  recv0_token: token = tuple_index(recv0, index=0, id=11, pos=[(0,15,22)])
  recv1: (token, bits[42], bits[1]) = receive(recv0_token, channel=in1, blocking=false, id=36)
}
"""


class SimulateModuleMainTest(test_base.TestCase):

  def test_single_arg_inline_function(self):
    ir_file = self.create_tempfile(content=ADD_IR_FUNCTION)
    verilog_file = self.create_tempfile()
    signature_file = self.create_tempfile()
    subprocess.check_call([
        CODEGEN_MAIN_PATH,
        '--generator=combinational',
        '--output_verilog_path=' + verilog_file.full_path,
        '--output_signature_path=' + signature_file.full_path,
        '--alsologtostderr',
        ir_file.full_path,
    ])
    result = subprocess.check_output([
        SIMULATE_MODULE_MAIN_PATH,
        '--verilog_simulator=iverilog',
        '--file_type=verilog',
        '--alsologtostderr',
        '--v=1',
        '--signature_file=' + signature_file.full_path,
        '--args=bits[32]:7; bits[32]:123',
        verilog_file.full_path,
    ])
    self.assertEqual('bits[32]:0x82', result.decode('utf-8').strip())

  def test_multi_arg_from_file_function(self):
    ir_file = self.create_tempfile(content=ADD_IR_FUNCTION)
    verilog_file = self.create_tempfile()
    signature_file = self.create_tempfile()
    subprocess.check_call([
        CODEGEN_MAIN_PATH,
        '--generator=combinational',
        '--output_verilog_path=' + verilog_file.full_path,
        '--output_signature_path=' + signature_file.full_path,
        '--alsologtostderr',
        ir_file.full_path,
    ])
    args_file = self.create_tempfile(content="""
      bits[32]:0xf00; bits[32]:1

      bits[32]:2; bits[32]:40

    """)
    result = subprocess.check_output([
        SIMULATE_MODULE_MAIN_PATH,
        '--verilog_simulator=iverilog',
        '--file_type=verilog',
        '--alsologtostderr',
        '--v=1',
        '--signature_file=' + signature_file.full_path,
        '--args_file=' + args_file.full_path,
        verilog_file.full_path,
    ])
    self.assertMultiLineEqual(
        'bits[32]:0xf01\nbits[32]:0x2a\n', result.decode('utf-8')
    )

  def test_multi_arg_from_file_proc(self):
    ir_file = self.create_tempfile(content=ADD_IR_PROC)
    verilog_file = self.create_tempfile()
    signature_file = self.create_tempfile()
    subprocess.check_call([
        CODEGEN_MAIN_PATH,
        '--generator=combinational',
        '--output_verilog_path=' + verilog_file.full_path,
        '--output_signature_path=' + signature_file.full_path,
        '--alsologtostderr',
        ir_file.full_path,
    ])
    channel_values_file = self.create_tempfile(content=textwrap.dedent("""
          sample__operand_1 : {
            bits[32]:0xf00
            bits[32]:2
          }
          sample__operand_0 : {
            bits[32]:1
            bits[32]:40
          }
    """))
    result = subprocess.check_output([
        SIMULATE_MODULE_MAIN_PATH,
        '--verilog_simulator=iverilog',
        '--file_type=verilog',
        '--alsologtostderr',
        '--v=1',
        '--signature_file=' + signature_file.full_path,
        '--channel_values_file=' + channel_values_file.full_path,
        '--output_channel_counts=sample__result=2',
        verilog_file.full_path,
    ])
    self.assertMultiLineEqual(
        'sample__result : {\n  bits[32]:0xf01\n  bits[32]:0x2a\n}\n',
        result.decode('utf-8'),
    )

  def test_proc_with_no_output_channels(self):
    ir_file = self.create_tempfile(content=PROC_WITH_NO_OUTPUT_CHANNEL)
    verilog_file = self.create_tempfile()
    signature_file = self.create_tempfile()
    subprocess.check_call([
        CODEGEN_MAIN_PATH,
        '--generator=combinational',
        '--output_verilog_path=' + verilog_file.full_path,
        '--output_signature_path=' + signature_file.full_path,
        '--alsologtostderr',
        ir_file.full_path,
    ])
    channel_values_file = self.create_tempfile(content=textwrap.dedent("""
          in0 : {
            bits[12]:41
            bits[12]:1
          }
          in1 : {
            bits[42]:1
            bits[42]:41
          }
    """))
    result = subprocess.check_output([
        SIMULATE_MODULE_MAIN_PATH,
        '--verilog_simulator=iverilog',
        '--file_type=verilog',
        '--alsologtostderr',
        '--v=1',
        '--signature_file=' + signature_file.full_path,
        '--channel_values_file=' + channel_values_file.full_path,
        verilog_file.full_path,
    ])
    # Empty result since there are no output channels.
    self.assertMultiLineEqual('\n', result.decode('utf-8'))


if __name__ == '__main__':
  test_base.main()
