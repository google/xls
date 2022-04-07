# Copyright 2022 The XLS Authors
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

"""Tests for public XLS builder APIs in Python."""

# pylint: disable=bad-continuation

import textwrap

from absl.testing import absltest
from xls.public.python import builder


class FunctionBuilderTest(absltest.TestCase):

  def test_identity_function_builder(self):
    p = builder.Package('sample_package')
    u32 = p.get_bits_type(32)
    fb = builder.FunctionBuilder('f', p)
    x = fb.add_param('x', u32)
    fb.add_identity(x)
    f = fb.build()
    self.assertEqual(f.dump_ir(), textwrap.dedent("""\
    fn f(x: bits[32]) -> bits[32] {
      ret identity.2: bits[32] = identity(x, id=2)
    }
    """))


if __name__ == '__main__':
  absltest.main()
