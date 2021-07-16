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
"""Tests for DSLX modules with various forms of errors."""

import subprocess as subp

from xls.common import runfiles
from xls.common import test_base

_INTERP_PATH = runfiles.get_path('xls/dslx/interpreter_main')


class ImportModuleWithTypeErrorTest(test_base.TestCase):

  def _run(self, path: str) -> str:
    full_path = runfiles.get_path(path)
    p = subp.run([_INTERP_PATH, full_path],
                 stderr=subp.PIPE,
                 check=False,
                 encoding='utf-8')
    self.assertNotEqual(p.returncode, 0)
    return p.stderr

  def test_failing_test_output(self):
    stderr = self._run('xls/dslx/tests/errors/two_failing_tests.x')
    print(stderr)
    lines = [line for line in stderr.splitlines() if line.startswith('[')]
    self.assertLen(lines, 9)
    self.assertEqual(lines[0], '[ RUN UNITTEST  ] first_failing')
    self.assertEqual(lines[1], '[        FAILED ] first_failing')
    self.assertEqual(lines[2], '[ RUN UNITTEST  ] second_failing')
    self.assertEqual(lines[3], '[        FAILED ] second_failing')
    self.assertEqual(lines[4],
                     '[===============] 2 test(s) ran; 2 failed; 0 skipped.')
    self.assertRegexpMatches(lines[5], r'\[ SEED [\d ]{16} \]')
    self.assertEqual(lines[6],
                     '[ RUN QUICKCHECK        ] always_false count: 1000')
    self.assertEqual(lines[7], '[                FAILED ] always_false')
    self.assertEqual(lines[8], '[=======================] 1 quickcheck(s) ran.')

  def test_imports_module_with_type_error(self):
    stderr = self._run('xls/dslx/tests/errors/imports_has_type_error.x')
    self.assertIn('xls/dslx/tests/errors/has_type_error.x:16:3-16:4', stderr)
    self.assertIn('did not match the annotated return type', stderr)

  def test_imports_and_causes_ref_error(self):
    stderr = self._run('xls/dslx/tests/errors/imports_and_causes_ref_error.x')
    self.assertIn('TypeInferenceError', stderr)
    self.assertIn(
        'xls/dslx/tests/errors/imports_and_causes_ref_error.x:17:33-17:43',
        stderr)

  def test_imports_private_enum(self):
    stderr = self._run('xls/dslx/tests/errors/imports_private_enum.x')
    self.assertIn('xls/dslx/tests/errors/imports_private_enum.x:17:14-17:40',
                  stderr)

  def test_imports_dne(self):
    stderr = self._run('xls/dslx/tests/errors/imports_and_typedefs_dne_type.x')
    self.assertIn(
        'xls/dslx/tests/errors/imports_and_typedefs_dne_type.x:17:12-17:48',
        stderr)
    self.assertIn(
        "xls.dslx.tests.errors.mod_private_enum member 'ReallyDoesNotExist' which does not exist",
        stderr)

  def test_colon_ref_builtin(self):
    stderr = self._run('xls/dslx/tests/errors/colon_ref_builtin.x')
    self.assertIn('xls/dslx/tests/errors/colon_ref_builtin.x:16:9-16:25',
                  stderr)
    self.assertIn("Builtin 'update' has no attributes", stderr)

  def test_constant_without_type_annotation(self):
    stderr = self._run('xls/dslx/tests/errors/constant_without_type_annot.x')
    self.assertIn(
        'xls/dslx/tests/errors/constant_without_type_annot.x:15:13-15:15',
        stderr)
    self.assertIn('please annotate a type.', stderr)

  def test_enum_with_type_on_value(self):
    stderr = self._run('xls/dslx/tests/errors/enum_with_type_on_value.x')
    self.assertIn('xls/dslx/tests/errors/enum_with_type_on_value.x:16:9-16:11',
                  stderr)
    self.assertIn('A type is annotated on this enum value', stderr)

  def test_bad_annotation(self):
    stderr = self._run('xls/dslx/tests/errors/bad_annotation.x')
    self.assertIn('xls/dslx/tests/errors/bad_annotation.x:15:11-15:12', stderr)
    self.assertIn("identifier 'x' doesn't resolve to a type", stderr)

  def test_invalid_colon_ref_as_literal_type(self):
    test_path = 'xls/dslx/tests/errors/invalid_colon_ref_as_literal_type.x'
    stderr = self._run(test_path)
    self.assertIn('{}:23:25-23:26'.format(test_path), stderr)
    self.assertIn('Non-bits type used to define a numeric literal.', stderr)

  def test_invalid_parameter_cast(self):
    stderr = self._run('xls/dslx/tests/errors/invalid_parameter_cast.x')
    self.assertIn('xls/dslx/tests/errors/invalid_parameter_cast.x:16:7-16:10',
                  stderr)
    self.assertIn(
        'Old-style cast only permitted for constant arrays/tuples and literal numbers',
        stderr)

  def test_multiple_mod_level_const_bindings(self):
    stderr = self._run(
        'xls/dslx/tests/errors/multiple_mod_level_const_bindings.x')
    self.assertIn(
        'xls/dslx/tests/errors/multiple_mod_level_const_bindings.x:16:7-16:10',
        stderr)
    self.assertIn('Constant definition is shadowing an existing definition',
                  stderr)

  def test_double_define_top_level_function(self):
    stderr = self._run(
        'xls/dslx/tests/errors/double_define_top_level_function.x')
    self.assertIn(
        'xls/dslx/tests/errors/double_define_top_level_function.x:18:4-18:7',
        stderr)
    self.assertIn('defined in this module multiple times', stderr)

  def test_double_define_test_function(self):
    stderr = self._run(
        'xls/dslx/tests/errors/double_define_test_function.x')
    self.assertIn(
        'xls/dslx/tests/errors/double_define_test_function.x:24:4-24:8',
        stderr)
    self.assertIn('has same name as module member', stderr)

  def test_bad_dim(self):
    stderr = self._run('xls/dslx/tests/errors/bad_dim.x')
    self.assertIn('xls/dslx/tests/errors/bad_dim.x:15:16-15:17', stderr)
    self.assertIn('Expected start of an expression; got: +', stderr)

  def test_match_multi_pattern_with_bindings(self):
    stderr = self._run(
        'xls/dslx/tests/errors/match_multi_pattern_with_bindings.x')
    self.assertIn(
        'xls/dslx/tests/errors/match_multi_pattern_with_bindings.x:17:5-17:6',
        stderr)
    self.assertIn('Cannot have multiple patterns that bind names', stderr)

  def test_co_recursion(self):
    stderr = self._run('xls/dslx/tests/errors/co_recursion.x')
    self.assertIn('xls/dslx/tests/errors/co_recursion.x:17:3-17:6', stderr)
    self.assertIn("Cannot find a definition for name: 'bar'", stderr)

  def test_self_recursion(self):
    stderr = self._run('xls/dslx/tests/errors/self_recursion.x')
    self.assertIn('xls/dslx/tests/errors/self_recursion.x:15:4-15:21', stderr)
    self.assertIn(
        "Recursion detected while typechecking; name: 'regular_recursion'",
        stderr)

  def test_tail_call(self):
    stderr = self._run('xls/dslx/tests/errors/tail_call.x')
    self.assertIn('xls/dslx/tests/errors/tail_call.x:15:4-15:5', stderr)
    self.assertIn("Recursion detected while typechecking; name: 'f'", stderr)

  def test_let_destructure_same_name(self):
    stderr = self._run('xls/dslx/tests/errors/let_destructure_same_name.x')
    self.assertIn(
        'xls/dslx/tests/errors/let_destructure_same_name.x:17:11-17:12', stderr)
    self.assertIn("Name 'i' is defined twice in this pattern", stderr)

  def test_empty_array_error(self):
    stderr = self._run('xls/dslx/tests/errors/empty_array.x')
    self.assertIn('Cannot deduce the type of an empty array.', stderr)

  def test_invalid_array_expression_type(self):
    stderr = self._run('xls/dslx/tests/errors/invalid_array_expression_type.x')
    self.assertIn(
        'xls/dslx/tests/errors/invalid_array_expression_type.x:16:12-16:18',
        stderr)
    self.assertIn('uN[32][2] vs uN[8][2]', stderr)

  def test_invalid_array_expression_size(self):
    stderr = self._run('xls/dslx/tests/errors/invalid_array_expression_size.x')
    self.assertIn(
        'xls/dslx/tests/errors/invalid_array_expression_size.x:16:20-16:36',
        stderr)
    self.assertIn('Annotated array size 2 does not match inferred array size 1',
                  stderr)

  def test_brace_scope(self):
    stderr = self._run('xls/dslx/tests/errors/brace_scope.x')
    self.assertIn('xls/dslx/tests/errors/brace_scope.x:16:3-16:4', stderr)
    self.assertIn('Expected start of an expression; got: {', stderr)

  def test_double_define_parameter(self):
    stderr = self._run('xls/dslx/tests/errors/double_define_parameter.x')
    # TODO(leary): 2020-01-26 This should not be flagged at the IR level, we
    # should catch it in the frontend.
    self.assertIn('Could not build IR: Parameter named "x" already exists',
                  stderr)

  def test_non_constexpr_slice(self):
    stderr = self._run('xls/dslx/tests/errors/non_constexpr_slice.x')
    self.assertIn('Unable to resolve slice limit to a compile-time constant.',
                  stderr)

  def test_scan_error_pretty_printed(self):
    stderr = self._run('xls/dslx/tests/errors/no_radix.x')
    self.assertIn(
        '^^ ScanError: Invalid radix for number, expect 0b or 0x because of leading 0.',
        stderr)

  def test_negative_shift_amount_shl(self):
    stderr = self._run('xls/dslx/tests/errors/negative_shift_amount_shl.x')
    self.assertIn('Negative literal values cannot be used as shift amounts',
                  stderr)

  def test_negative_shift_amount_shr(self):
    stderr = self._run('xls/dslx/tests/errors/negative_shift_amount_shr.x')
    self.assertIn('Negative literal values cannot be used as shift amounts',
                  stderr)

  def test_over_shift(self):
    stderr = self._run('xls/dslx/tests/errors/over_shift_amount.x')
    self.assertIn('Shift amount is larger than shift value bit width of',
                  stderr)

  def test_colon_ref_of_type_alias(self):
    stderr = self._run('xls/dslx/tests/errors/colon_ref_of_type_alias.x')
    self.assertIn("Cannot use '::' on 'F32', it is not an enum or module",
                  stderr)

  def test_cast_int_to_struct(self):
    stderr = self._run('xls/dslx/tests/errors/cast_int_to_struct.x')
    self.assertIn("Cannot cast from expression type uN[32] to struct 'S'",
                  stderr)

  def test_cast_struct_to_int(self):
    stderr = self._run('xls/dslx/tests/errors/cast_struct_to_int.x')
    self.assertIn(
        "Cannot cast from expression type struct 'S' structure: S { member: uN[32] } to uN[32]",
        stderr)

  def test_destructure_fallible(self):
    stderr = self._run('xls/dslx/tests/errors/destructure_fallible.x')
    self.assertIn(
        'FailureError: The program being interpreted failed! (u8:0, u8:0)',
        stderr)

  def test_match_not_exhaustive(self):
    stderr = self._run('xls/dslx/tests/errors/match_not_exhaustive.x')
    self.assertIn('match_not_exhaustive.x:16:3-19:4', stderr)
    self.assertIn('Only matches with trailing irrefutable patterns', stderr)

  def test_bad_coverpoint_name(self):
    stderr = self._run('xls/dslx/tests/errors/coverpoint_bad_name.x')
    self.assertIn('coverpoint_bad_name.x:16:9-16:40', stderr)
    self.assertIn('A coverpoint identifer must start with', stderr)

  def test_arg0_type_mismatch(self):
    stderr = self._run('xls/dslx/tests/errors/arg0_type_mismatch.x')
    # TODO(https://github.com/google/xls/issues/438): 2021-05-24 Numbers are
    # currently reported with inaccurate spans, this will need to change once
    # that's addressed.
    self.assertIn('arg0_type_mismatch.x:18:9-18:14', stderr)
    self.assertIn(
        'uN[2] vs uN[32]: Mismatch between parameter and argument types',
        stderr)

  def test_arg1_type_mismatch(self):
    stderr = self._run('xls/dslx/tests/errors/arg1_type_mismatch.x')
    # TODO(https://github.com/google/xls/issues/438): 2021-05-24 Numbers are
    # currently reported with inaccurate spans, this will need to change once
    # that's addressed.
    self.assertIn('arg1_type_mismatch.x:19:9-19:14', stderr)
    self.assertIn(
        'uN[3] vs uN[32]: Mismatch between parameter and argument types',
        stderr)

  def test_index_struct_value(self):
    stderr = self._run('xls/dslx/tests/errors/index_struct_value.x')
    self.assertIn('index_struct_value.x:23:4-23:7', stderr)
    self.assertIn('Value to index is not an array', stderr)

  def test_non_const_array_type_dimension(self):
    stderr = self._run('xls/dslx/tests/errors/non_const_array_type_dimension.x')
    # TODO(leary): 2021-06-21 This error should become something like "can only
    # refer to constant or parametric values in dimensions".
    self.assertIn('non_const_array_type_dimension.x:16:10-16:18', stderr)
    self.assertIn('uN[32][x] vs ()', stderr)

  def test_array_type_dimension_with_width_annotated(self):
    stderr = self._run(
        'xls/dslx/tests/errors/array_type_dimension_with_width_annotated.x')
    self.assertIn('array_type_dimension_with_width_annotated.x:15:24-15:26',
                  stderr)
    self.assertIn('Please do not annotate a type on dimensions', stderr)

  def test_parametric_plus_global_as_local_const(self):
    stderr = self._run(
        'xls/dslx/tests/errors/parametric_plus_global_as_dimension.x')
    self.assertIn('parametric_plus_global_as_dimension.x:20:24-20:25', stderr)
    self.assertIn('Could not evaluate dimension expression to a constant value',
                  stderr)

  def test_signed_array_size(self):
    stderr = self._run('xls/dslx/tests/errors/signed_array_size.x')
    self.assertIn('signed_array_size.x:17:18-17:22', stderr)
    self.assertIn('Dimension SIZE must be a `u32`', stderr)

  # TODO(leary): 2021-06-30 We currently don't flag when an array dimension
  # value resolves to signed *after* parametric instantiation.
  @test_base.skip('Currently not flagging this as an error')
  def test_signed_parametric_in_array_size(self):
    self._run('xls/dslx/tests/errors/signed_parametric_in_array_size.x')

  def test_duplicate_match_arm(self):
    stderr = self._run('xls/dslx/tests/errors/duplicate_match_arm.x')
    self.assertIn('duplicate_match_arm.x:22:5-22:8', stderr)
    self.assertIn(
        'Exact-duplicate pattern match detected `FOO` -- only the first could possibly match',
        stderr)


if __name__ == '__main__':
  test_base.main()
