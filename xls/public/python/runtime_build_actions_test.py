#
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
"""Tests for xls.public.runtime_build_actions."""

import sys
import tempfile

from absl.testing import absltest
from xls.common.python import init_xls
from xls.public.python import runtime_build_actions


def setUpModule():
  # This is required so that module initializers are called including those
  # which register delay models.
  init_xls.init_xls(sys.argv)


class RuntimeBuildActionsTest(absltest.TestCase):

  dslx_stdlib_path = runtime_build_actions.get_default_dslx_stdlib_path()

  def test_convert_dslx_to_ir(self):
    dslx_text = 'pub fn foo(v:u8) -> u8{ v+u8:1 }'
    ir_text = runtime_build_actions.convert_dslx_to_ir(dslx_text,
                                                       '/path/to/foo', 'foo',
                                                       self.dslx_stdlib_path,
                                                       [])
    self.assertIn('package foo', ir_text)
    self.assertIn('bits[8]', ir_text)
    self.assertIn('add', ir_text)
    self.assertIn('literal', ir_text)

  def test_convert_dslx_path_to_ir(self):
    with tempfile.NamedTemporaryFile(prefix='foo', suffix='.x') as f:
      f.write(b'pub fn foo(v:u8) -> u8{ v+u8:1 }')
      f.flush()
      ir_text = runtime_build_actions.convert_dslx_path_to_ir(
          f.name, self.dslx_stdlib_path, [])
    self.assertIn('package foo', ir_text)
    self.assertIn('bits[8]', ir_text)
    self.assertIn('add', ir_text)
    self.assertIn('literal', ir_text)

  def test_optimize_ir(self):
    dslx_text = 'pub fn foo(v:u8) -> u8{ v+u8:2-u8:1 }'
    ir_text = runtime_build_actions.convert_dslx_to_ir(dslx_text,
                                                       '/path/to/foo', 'foo',
                                                       self.dslx_stdlib_path,
                                                       [])
    opt_ir_text = runtime_build_actions.optimize_ir(ir_text, '__foo__foo')
    self.assertNotEqual(ir_text, opt_ir_text)

  def test_mangle_dslx_name(self):
    mangled_name = runtime_build_actions.mangle_dslx_name('foo', 'bar')
    self.assertEqual('__foo__bar', mangled_name)

  def test_proto_to_dslx(self):
    binding_name = 'MY_TEST_MESSAGE'
    proto_def = """syntax = "proto2";

package xls_public_test;

message TestMessage {
  optional int32 test_field = 1;
}

message TestRepeatedMessage {
  repeated TestMessage messages = 1;
}"""
    text_proto = """messages: {
  test_field: 42
}
messages: {
  test_field: 64
}"""
    message_name = 'xls_public_test.TestRepeatedMessage'
    dslx_text = runtime_build_actions.proto_to_dslx(proto_def, message_name,
                                                    text_proto, binding_name)
    self.assertIn('TestMessage', dslx_text)
    self.assertIn('message', dslx_text)
    self.assertIn('test_field', dslx_text)
    self.assertIn(binding_name, dslx_text)


if __name__ == '__main__':
  absltest.main()
