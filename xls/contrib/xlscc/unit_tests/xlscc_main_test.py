# Copyright 2021 The XLS Authors
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

from absl.testing import absltest
from xls.common import runfiles
from xls.contrib.xlscc import hls_block_pb2

XLSCC_MAIN_PATH = runfiles.get_path("xls/contrib/xlscc/xlscc")

FUNC_CPP_SRC = """
#pragma hls_top
int my_function(int a, int b){
	return a+b;
}
"""

BLOCK_CPP_SRC = """
#include "/xls_builtin.h"

#pragma hls_top
void my_function(int& dir,
                 __xls_channel<int>& in,
                 __xls_channel<int>& out1,
                 __xls_channel<int>& out2) {


  int x = in.read();

  if (dir == 0) {
    out1.write(x);
  } else {
    out2.write(x);
  }

}
"""

BLOCK_CPP_SRC2 = """
#include "/xls_builtin.h"

#pragma hls_top
void my_function(int& dir,
                 __xls_channel<int>& out) {

  out.write(dir);

}
"""


class XlsccMainTest(absltest.TestCase):

  def test_gen_ir(self):
    cpp_file = self.create_tempfile(file_path="src.cc", content=FUNC_CPP_SRC)

    subprocess.check_call([XLSCC_MAIN_PATH, cpp_file.full_path])

  def test_gen_ir_block(self):
    cpp_file = self.create_tempfile(file_path="src.cc", content=BLOCK_CPP_SRC)

    subprocess.check_call([XLSCC_MAIN_PATH, cpp_file.full_path])

  def test_gen_combinational_verilog(self):
    cpp_file = self.create_tempfile(file_path="src.cc", content=BLOCK_CPP_SRC)

    block_out = hls_block_pb2.HLSBlock()
    block_out.name = "my_function"

    channel = hls_block_pb2.HLSChannel()
    channel.name = "dir"
    channel.is_input = True
    channel.type = hls_block_pb2.ChannelType.DIRECT_IN
    block_out.channels.add().CopyFrom(channel)

    channel = hls_block_pb2.HLSChannel()
    channel.name = "in"
    channel.is_input = True
    channel.type = hls_block_pb2.ChannelType.FIFO
    block_out.channels.add().CopyFrom(channel)

    channel = hls_block_pb2.HLSChannel()
    channel.name = "out1"
    channel.is_input = False
    channel.type = hls_block_pb2.ChannelType.FIFO
    block_out.channels.add().CopyFrom(channel)

    channel = hls_block_pb2.HLSChannel()
    channel.name = "out2"
    channel.is_input = False
    channel.type = hls_block_pb2.ChannelType.FIFO
    block_out.channels.add().CopyFrom(channel)

    block_pb_file = self.create_tempfile(file_path="block.pb")

    with open(block_pb_file.full_path, "wb") as f:
      f.write(block_out.SerializeToString())

    subprocess.check_call([
        XLSCC_MAIN_PATH,
        cpp_file.full_path,
        "--block_pb",
        block_pb_file.full_path,
    ])


if __name__ == "__main__":
  absltest.main()
