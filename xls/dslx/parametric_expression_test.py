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

"""Tests for xls.dslx.parametric_expression."""

from absl.testing import absltest
from xls.dslx.python import interp_value
from xls.dslx.python.cpp_parametric_expression import ParametricAdd
from xls.dslx.python.cpp_parametric_expression import ParametricConstant
from xls.dslx.python.cpp_parametric_expression import ParametricMul
from xls.dslx.python.cpp_parametric_expression import ParametricSymbol
from xls.dslx.python.cpp_pos import Pos
from xls.dslx.python.cpp_pos import Span


class ParametricExpressionTest(absltest.TestCase):

  fake_pos = Pos('<fake>', 0, 0)
  fake_span = Span(fake_pos, fake_pos)

  def test_sample_evaluation(self):
    param_0 = interp_value.Value.make_ubits(32, 0)
    param_1 = interp_value.Value.make_ubits(32, 1)
    param_2 = interp_value.Value.make_ubits(32, 2)
    param_3 = interp_value.Value.make_ubits(32, 3)
    param_6 = interp_value.Value.make_ubits(32, 6)
    param_12 = interp_value.Value.make_ubits(32, 12)
    e = ParametricMul(
        ParametricConstant(param_3),
        ParametricAdd(
            ParametricSymbol('M', self.fake_span),
            ParametricSymbol('N', self.fake_span)))
    self.assertEqual(e, e)
    self.assertEqual('(u32:3*(M+N))', str(e))
    self.assertEqual(param_6, e.evaluate(dict(N=param_2, M=param_0)))
    self.assertEqual(param_12, e.evaluate(dict(N=param_1, M=param_3)))
    self.assertEqual(set(['N', 'M']), e.get_freevars())

  def test_non_identity_equality(self):
    s0 = ParametricSymbol('s', self.fake_span)
    s1 = ParametricSymbol('s', self.fake_span)
    self.assertEqual(s0, s1)
    self.assertEqual(repr(s0), 'ParametricSymbol("s")')
    add = ParametricAdd(s0, s1)
    self.assertEqual(
        repr(add),
        'ParametricAdd(ParametricSymbol("s"), ParametricSymbol("s"))')


if __name__ == '__main__':
  absltest.main()
