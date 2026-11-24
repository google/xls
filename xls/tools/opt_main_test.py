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

"""Tests for xls.tools.codegen_main."""

import concurrent
import concurrent.futures
import subprocess
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
from xls.common import runfiles
from xls.passes import optimization_pass_pipeline_pb2
from xls.passes import pass_pipeline_pb2

OPT_MAIN_PATH = runfiles.get_path('xls/tools/opt_main')
PIPELINE_METRICS_MAIN_PATH = runfiles.get_path(
    'xls/dev_tools/pass_metrics_main'
)

ADD_ZERO_IR = """package add_zero

top fn add_zero(x: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0, id=1)
  ret add.2: bits[32] = add(x, literal.1, id=2)
}
"""

DEAD_FUNCTION_IR = """package add_zero

fn dead_function(x: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0)
  ret add.2: bits[32] = add(x, literal.1)
}

top fn main(x: bits[32]) -> bits[32] {
  literal.3: bits[32] = literal(value=0)
  ret add.4: bits[32] = add(x, literal.3)
}
"""

ADD_LITERAL_IR = """package add_literal

top fn add_zero(x: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0xfff_0000)
  ret add.2: bits[32] = add(x, literal.1)
}
"""


class OptMainTest(parameterized.TestCase):

  def test_no_options(self):
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)

    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, ir_file.full_path]
    ).decode('utf-8')

    # The add with zero should be eliminated.
    self.assertIn('ret x', optimized_ir)

  def test_list_passes(self):
    output = subprocess.check_output([OPT_MAIN_PATH, '--list_passes']).decode(
        'utf-8'
    )
    self.assertIn('dce', output)

  def test_with_vlog(self):
    # Checks that enabling vlog doesn't crash.
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)

    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, '-v=5', ir_file.full_path]
    ).decode('utf-8')

    # The add with zero should be eliminated.
    self.assertIn('ret x', optimized_ir)

  def test_run_only_basic_simp_and_dce_passes(self):
    # NB This used to test a deprecated --run_only_passes flag that is
    # superseeded by the --passes flag.
    ir_file = self.create_tempfile(content=DEAD_FUNCTION_IR)

    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, '--passes=basic_simp dce', ir_file.full_path]
    ).decode('utf-8')

    # The add with zero should be eliminated.
    self.assertIn('add(', ADD_ZERO_IR)
    self.assertNotIn('add(', optimized_ir)

    # Without running DFE, the dead function should remain in the IR.
    self.assertIn('dead_function', optimized_ir)

  def test_skip_dfe(self):
    ir_file = self.create_tempfile(content=DEAD_FUNCTION_IR)

    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, ir_file.full_path]
    ).decode('utf-8')

    # Without skipping DFE, the dead function should be removed.
    self.assertNotIn('dead_function', optimized_ir)

    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, '--skip_passes=dfe', ir_file.full_path]
    ).decode('utf-8')

    # Skipping DFE should leave the dead function in the IR.
    self.assertIn('dead_function', optimized_ir)

  def test_pipeline_proto_level(self):
    ir_file = self.create_tempfile(content=DEAD_FUNCTION_IR)
    registry = optimization_pass_pipeline_pb2.OptimizationPipelineProto(
        compound_passes=[
            optimization_pass_pipeline_pb2.OptimizationPipelineProto.CompoundPass(
                long_name='compound',
                short_name='compound',
                passes=['const_fold', 'dce'],
            )
        ],
        default_pipeline=['compound'],
    )
    reg_file = self.create_tempfile(content=registry.SerializeToString())
    optimized_ir = subprocess.check_output([
        OPT_MAIN_PATH,
        ir_file.full_path,
        f'--pipeline_proto={reg_file.full_path}',
    ]).decode('utf-8')
    with self.subTest('dead function removed'):
      self.assertIn('dead_function', optimized_ir)
    manual_pipeline = subprocess.check_output([
        OPT_MAIN_PATH,
        '--passes=const_fold dce',
        ir_file.full_path,
    ]).decode('utf-8')
    with self.subTest('Equivalent to manual pass runs'):
      self.assertEqual(optimized_ir, manual_pipeline)

  def test_skip_passes_ignores_opt_levels(self):
    # --skip_passes is not a very useful flag and has a lot of weird behaviors.
    # While we still have it we should make sure that it will actually always
    # skip a pass even if its in a 'min_opt_level' set.
    ir_file = self.create_tempfile(content=DEAD_FUNCTION_IR)
    compound_pass = (
        optimization_pass_pipeline_pb2.OptimizationPipelineProto.CompoundPass
    )
    options = (
        optimization_pass_pipeline_pb2.OptimizationPipelineProto.GlobalOptions
    )
    pipeline = optimization_pass_pipeline_pb2.OptimizationPipelineProto(
        compound_passes=[
            compound_pass(
                long_name='test pass group',
                short_name='test_group',
                passes=['dfe'],
                fixedpoint=False,
                options=options(min_opt_level=1),
            )
        ],
        default_pipeline=['test_group'],
    )
    pipeline_file = self.create_tempfile(content=pipeline.SerializeToString())
    optimized_ir = subprocess.check_output([
        OPT_MAIN_PATH,
        '--skip_passes=dfe',
        f'--pipeline_proto={pipeline_file.full_path}',
        ir_file.full_path,
    ]).decode('utf-8')
    self.assertIn('dead_function', optimized_ir)

  def test_opt_level(self):
    ir_file = self.create_tempfile(content=ADD_LITERAL_IR)

    # Without specifying opt-level, the optimizer should run at maximum level
    # and convert the add with literal 0xffff_0000 into something like:
    #
    #   concat(add(0xffff, x[16:32]), x[0:16])
    #
    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, ir_file.full_path]
    ).decode('utf-8')
    self.assertIn('bits[16] = add', optimized_ir)
    self.assertIn('concat', optimized_ir)

    # Opt_level 3 should produce the same results as without opt_level
    # specified.
    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, '--opt_level=3', ir_file.full_path]
    ).decode('utf-8')
    self.assertIn('bits[16] = add', optimized_ir)
    self.assertIn('concat', optimized_ir)

    # At opt_level 1 the full width add should remain.
    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, '--opt_level=1', ir_file.full_path]
    ).decode('utf-8')
    self.assertIn('bits[32] = add', optimized_ir)
    self.assertNotIn('concat', optimized_ir)

  def test_explicit_pipeline(self):
    """Check that basic_simp gets the add-zero gone."""
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, ir_file.full_path, '--passes', 'basic_simp dce']
    ).decode('utf-8')
    self.assertNotIn('bits[32] = add', optimized_ir)

  def test_explicit_single_pass_pipeline(self):
    """Check that dce is not run."""
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, ir_file.full_path, '--passes', 'basic_simp']
    ).decode('utf-8')
    # add is not removed since the DCE was not run.
    self.assertIn('bits[32] = add', optimized_ir)

  # TODO: https://github.com/google/xls/issues/1245 - Get an example that can
  # see the fixed-point pipeline doing something.

  def test_bisect_limit(self):
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    optimized_ir = subprocess.check_output([
        OPT_MAIN_PATH,
        ir_file.full_path,
        '--passes',
        'dce dce dce dce basic_simp dce',
        '--passes_bisect_limit',
        '3',
    ]).decode('utf-8')
    # Add remains in graph since basic_simp + dce is not run
    self.assertIn('bits[32] = add', optimized_ir)

  def test_bisect_limit_allows_changes(self):
    """Check that dce is not run after basic simp."""
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    optimized_ir = subprocess.check_output([
        OPT_MAIN_PATH,
        ir_file.full_path,
        '--passes',
        'dce basic_simp dce',
        '--passes_bisect_limit',
        '2',
    ]).decode('utf-8')
    # add is not removed since the DCE was not run.
    self.assertIn('bits[32] = add', optimized_ir)

  @parameterized.parameters(range(0, 5))
  def test_proto_pipeline(self, opt_level):
    test_file = runfiles.get_path('xls/modules/aes/aes_ctr.ir')
    base_call = [OPT_MAIN_PATH, test_file, f'--opt_level={opt_level}']
    # These take ~60 seconds each so just do them in parallel.
    with concurrent.futures.ThreadPoolExecutor() as tpe:
      oracle_output = tpe.submit(
          lambda: subprocess.check_output(base_call).decode('utf-8')
      )
      test_output = tpe.submit(
          lambda: subprocess.check_output(
              base_call
              + [
                  f'--passes_proto={runfiles.get_path("xls/passes/default_opt_pass_pipeline.binarypb")}'
              ]
          ).decode('utf-8')
      )
    self.assertEqual(oracle_output.result(), test_output.result())

  def filter_non_zero_opt_level(
      self, e: pass_pipeline_pb2.PassPipelineProto.Element
  ) -> Optional[pass_pipeline_pb2.PassPipelineProto.Element]:
    """Filter out any pass marked as requiring an opt level of 1 or higher."""
    if e.options.HasField('min_opt_level') and e.options.min_opt_level >= 1:
      return None
    if e.HasField('pass_name'):
      return e
    out = []
    for i in (
        e.pipeline.elements if e.HasField('pipeline') else e.fixedpoint.elements
    ):
      filt = self.filter_non_zero_opt_level(i)
      if filt is not None:
        out.append(filt)
    if not out:
      return None
    if e.HasField('pipeline'):
      res = pass_pipeline_pb2.PassPipelineProto.Element(
          pipeline=pass_pipeline_pb2.PassPipelineProto.Pipeline(elements=out),
          options=e.options,
      )
    else:
      res = pass_pipeline_pb2.PassPipelineProto.Element(
          fixedpoint=pass_pipeline_pb2.PassPipelineProto.Pipeline(elements=out),
          options=e.options,
      )
    return res

  def test_proto_changed_pipeline(self):
    test_file = runfiles.get_path('xls/modules/aes/aes_ctr.ir')
    pipeline = pass_pipeline_pb2.PassPipelineProto().FromString(
        runfiles.get_contents_as_bytes(
            'xls/passes/default_opt_pass_pipeline.binarypb'
        )
    )
    small_pipeline = pass_pipeline_pb2.PassPipelineProto(
        top=self.filter_non_zero_opt_level(pipeline.top)
    )
    test_pipeline = self.create_tempfile(
        content=small_pipeline.SerializeToString()
    )
    with concurrent.futures.ThreadPoolExecutor() as tpe:
      oracle_output = tpe.submit(
          lambda: subprocess.check_output(
              [OPT_MAIN_PATH, test_file, '--opt_level=0']
          ).decode('utf-8')
      )
      test_output = tpe.submit(
          lambda: subprocess.check_output([
              OPT_MAIN_PATH,
              test_file,
              '--opt_level=3',
              f'--passes_proto={test_pipeline.full_path}',
          ]).decode('utf-8')
      )
    self.assertEqual(len(oracle_output.result()), len(test_output.result()))
    self.assertEqual(oracle_output.result(), test_output.result())

  def test_proto_changed_pipeline_not_equal(self):
    test_file = runfiles.get_path('xls/modules/aes/aes_ctr.ir')
    pipeline = pass_pipeline_pb2.PassPipelineProto().FromString(
        runfiles.get_contents_as_bytes(
            'xls/passes/default_opt_pass_pipeline.binarypb'
        )
    )
    small_pipeline = pass_pipeline_pb2.PassPipelineProto(
        top=self.filter_non_zero_opt_level(pipeline.top)
    )
    test_pipeline = self.create_tempfile(
        content=small_pipeline.SerializeToString()
    )
    with concurrent.futures.ThreadPoolExecutor() as tpe:
      oracle_output = tpe.submit(
          lambda: subprocess.check_output(
              [OPT_MAIN_PATH, test_file, '--opt_level=3']
          ).decode('utf-8')
      )
      test_output = tpe.submit(
          lambda: subprocess.check_output([
              OPT_MAIN_PATH,
              test_file,
              '--opt_level=3',
              f'--passes_proto={test_pipeline.full_path}',
          ]).decode('utf-8')
      )
    self.assertNotEqual(oracle_output.result(), test_output.result())

  def test_dump_metrics(self):
    ir_file = runfiles.get_path('xls/modules/aes/aes_ctr.ir')

    # Dump metrics to a file.
    metrics_file = self.create_tempfile()
    subprocess.check_call([
        OPT_MAIN_PATH,
        f'--pass_metrics_path={metrics_file.full_path}',
        ir_file,
    ])

    # Generate a table of the metrics.
    metrics_output = subprocess.check_output(
        [PIPELINE_METRICS_MAIN_PATH, metrics_file.full_path]
    ).decode('utf-8')

    # The table should include a line for cse pass (common subexpression
    # elimination).
    self.assertIn('cse', metrics_output)

  def test_list_passes_with_pipeline_proto(self):
    registry = optimization_pass_pipeline_pb2.OptimizationPipelineProto(
        compound_passes=[
            optimization_pass_pipeline_pb2.OptimizationPipelineProto.CompoundPass(
                long_name='custom list pass',
                short_name='my_custom_list_pass',
                passes=['dce'],
            )
        ],
        default_pipeline=['my_custom_list_pass'],
    )
    reg_file = self.create_tempfile(content=registry.SerializeToString())
    output = subprocess.check_output([
        OPT_MAIN_PATH,
        '--list_passes',
        f'--pipeline_proto={reg_file.full_path}',
    ]).decode('utf-8')
    self.assertIn('my_custom_list_pass', output)

  def test_pipeline_proto_and_textproto_conflict(self):
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    comp = subprocess.run(
        [
            OPT_MAIN_PATH,
            '--pipeline_proto=a.binarypb',
            '--pipeline_textproto=b.txtpb',
            ir_file.full_path,
        ],
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn('--pipeline_proto and --pipeline_textproto', comp.stderr)

  def test_ram_rewrites_conflicts_with_opt_options_proto(self):
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    empty_proto = self.create_tempfile(content='')
    comp = subprocess.run(
        [
            OPT_MAIN_PATH,
            f'--ram_rewrites_pb={empty_proto.full_path}',
            f'--opt_options_proto={empty_proto.full_path}',
            ir_file.full_path,
        ],
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn("Cannot combine 'opt_options_proto'", comp.stderr)

  def test_passes_conflicts_with_opt_options_proto(self):
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    empty_proto = self.create_tempfile(content='')
    comp = subprocess.run(
        [
            OPT_MAIN_PATH,
            '--passes=dce',
            f'--opt_options_proto={empty_proto.full_path}',
            ir_file.full_path,
        ],
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn("Cannot combine 'opt_options_proto'", comp.stderr)

  def test_empty_passes_is_noop(self):
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    optimized_ir = subprocess.check_output(
        [OPT_MAIN_PATH, '--passes=', ir_file.full_path]
    ).decode('utf-8')
    self.assertIn('bits[32] = add', optimized_ir)

  def test_passes_bisect_limit_is_error(self):
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    comp = subprocess.run(
        [
            OPT_MAIN_PATH,
            '--passes=dce basic_simp dce',
            '--passes_bisect_limit=1',
            '--passes_bisect_limit_is_error',
            ir_file.full_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn('passes bisect limit was reached', comp.stderr)
    # Optimized IR up to the bisect limit is still printed to stdout.
    self.assertIn('bits[32] = add', comp.stdout)

  def test_opt_level_help(self):
    result = subprocess.run(
        [OPT_MAIN_PATH, '--helpmatch=common_flags'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    self.assertIn('--opt_level', output)
    self.assertIn('Ranges from 0 to 3.', output)

  def test_decouple_state_access_flag(self):
    proc_ir = """package test

top proc main<cmd_ch: bits[1] in, out_ch: bits[32] out>(accum: bits[32], noop_se: bits[32], init={0, 7}) {
  chan_interface cmd_ch(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  chan_interface out_ch(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  tkn0: token = literal(value=token)
  accum: bits[32] = state_read(state_element=accum)
  noop_se: bits[32] = state_read(state_element=noop_se)
  rcv: (token, bits[1]) = receive(tkn0, channel=cmd_ch)
  tkn1: token = tuple_index(rcv, index=0)
  cmd: bits[1] = tuple_index(rcv, index=1)
  not_cmd: bits[1] = not(cmd)
  zero: bits[32] = literal(value=0)
  one: bits[32] = literal(value=1)
  accum_inc: bits[32] = add(accum, one)
  noop_inc: bits[32] = add(noop_se, one)
  out_sum: bits[32] = add(accum, noop_se)
  send_out: token = send(tkn1, out_sum, predicate=cmd, channel=out_ch)
  next_accum_blind: () = next_value(state_element=accum, value=zero, predicate=not_cmd)
  next_accum_inc: () = next_value(state_element=accum, value=accum_inc, predicate=cmd)
  next_noop_self: () = next_value(state_element=noop_se, value=noop_se, predicate=not_cmd)
  next_noop_inc: () = next_value(state_element=noop_se, value=noop_inc, predicate=cmd)
}
"""
    ir_file = self.create_tempfile(content=proc_ir)

    coupled_ir = subprocess.check_output([
        OPT_MAIN_PATH,
        '--optimize_for_best_case_throughput=true',
        '--decouple_state_access=false',
        ir_file.full_path,
    ]).decode('utf-8')
    # Pre-legalization no-op next_value is still dropped, while accum's
    # state_read stays unpredicated for legacy coupled lowering.
    self.assertNotIn('value=noop_se', coupled_ir)
    self.assertIn('state_read(state_element=accum, id=', coupled_ir)

    decoupled_ir = subprocess.check_output([
        OPT_MAIN_PATH,
        '--optimize_for_best_case_throughput=true',
        '--decouple_state_access=true',
        ir_file.full_path,
    ]).decode('utf-8')
    self.assertNotIn('value=noop_se', decoupled_ir)
    self.assertIn('state_read(state_element=accum, predicate=cmd', decoupled_ir)

  def test_empty_delay_and_area_model_flags(self):
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    optimized_ir = subprocess.check_output([
        OPT_MAIN_PATH,
        '--delay_model=',
        '--area_model=',
        ir_file.full_path,
    ]).decode('utf-8')
    self.assertIn('ret x', optimized_ir)

  @parameterized.named_parameters(
      ('delay_model', '--delay_model=unit'),
      ('area_model', '--area_model=unit'),
  )
  def test_opt_options_proto_rejects_model_flags(self, flag):
    ir_file = self.create_tempfile(content=ADD_ZERO_IR)
    proto_file = self.create_tempfile(content='')
    result = subprocess.run(
        [
            OPT_MAIN_PATH,
            f'--opt_options_proto={proto_file.full_path}',
            flag,
            ir_file.full_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(
        "Cannot combine 'opt_options_proto' and opt arguments",
        result.stdout + result.stderr,
    )


if __name__ == '__main__':
  absltest.main()
