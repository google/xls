#
# Copyright 2024 The XLS Authors
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
"""Test for xls.jit.dump_llvm_artifacts_main."""

import pathlib
import struct
import subprocess

from absl.testing import absltest
from xls.common import runfiles

_DUMP_ARTIFACTS = runfiles.get_path('xls/dev_tools/dump_llvm_artifacts')

ADD_IR = """package foo

top fn muladd(a: bits[8], b: bits[8], c: bits[8]) -> bits[8] {
  umul.4: bits[8] = umul(a, b, id=4, pos=[(0,3,6)])
  ret add.5: bits[8] = add(umul.4, c, id=5, pos=[(0,3,10)])
}
"""


class DumpLlvmArtifactsMainTest(absltest.TestCase):

  def test_expected_files_exist(self):
    tmp = self.create_tempdir()
    tmp_path = pathlib.Path(tmp.full_path)
    ir = self.create_tempfile(content=ADD_IR)
    subprocess.run(
        [
            _DUMP_ARTIFACTS,
            '-out_dir',
            tmp.full_path,
            '-ir',
            ir.full_path,
            '--input=bits[8]:3',
            '--input=bits[8]:4',
            '--input=bits[8]:5',
            '--result=bits[8]:17',
        ],
        check=True,
    )
    self.assertTrue((tmp_path / 'linked.ll').exists())
    self.assertTrue((tmp_path / 'linked.opt.ll').exists())
    self.assertTrue((tmp_path / 'main.ll').exists())
    self.assertTrue((tmp_path / 'main.cc').exists())
    self.assertTrue((tmp_path / 'result.asm').exists())
    self.assertTrue((tmp_path / 'result.o').exists())
    self.assertTrue((tmp_path / 'result.ll').exists())
    self.assertTrue((tmp_path / 'result.opt.ll').exists())
    self.assertTrue((tmp_path / 'result.entrypoints.pb').exists())
    self.assertTrue((tmp_path / 'result.entrypoints.txtpb').exists())

  def test_can_detect_mismatch(self):
    tmp = self.create_tempdir()
    tmp_path = pathlib.Path(tmp.full_path)
    ir = self.create_tempfile(content=ADD_IR)
    subprocess.run(
        [
            _DUMP_ARTIFACTS,
            '-out_dir',
            tmp.full_path,
            '-ir',
            ir.full_path,
            '--input=bits[8]:3',
            '--input=bits[8]:4',
            '--input=bits[8]:5',
            '--result=bits[8]:7',  # Whoops not the right value
            '--write_result',
        ],
        check=True,
    )
    lli_res = subprocess.run(
        [
            runfiles.get_path('llvm/lli', repository='llvm-project'),
            str(tmp_path / 'linked.ll'),
        ],
        check=False,
        stdout=subprocess.PIPE,
    )
    self.assertEqual(17, struct.unpack('=B', lli_res.stdout)[0])
    self.assertEqual(lli_res.returncode, 1)

  def test_can_interpret(self):
    tmp = self.create_tempdir()
    tmp_path = pathlib.Path(tmp.full_path)
    ir = self.create_tempfile(content=ADD_IR)
    subprocess.run(
        [
            _DUMP_ARTIFACTS,
            '-out_dir',
            tmp.full_path,
            '-ir',
            ir.full_path,
            '--input=bits[8]:3',
            '--input=bits[8]:4',
            '--input=bits[8]:5',
            '--result=bits[8]:17',
            '--write_result',
        ],
        check=True,
    )
    lli_res = subprocess.run(
        [
            runfiles.get_path('llvm/lli', repository='llvm-project'),
            str(tmp_path / 'linked.ll'),
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    self.assertEqual(17, struct.unpack('=B', lli_res)[0])


if __name__ == '__main__':
  absltest.main()
