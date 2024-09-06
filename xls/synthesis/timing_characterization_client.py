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
"""Sample points from a pb to characterize using a synthesis server.

These datapoints can be used in a delay model (where they will be interpolated)
-- the results emitted on stdout are in xls.estimator_model.DataPoints prototext
format.
"""

import multiprocessing as mp
import multiprocessing.pool as mp_pool
import os
import sys
import textwrap
from typing import Any, Dict

from absl import flags
from absl import logging

from google.protobuf import text_format
from xls.common import gfile
from xls.estimators import estimator_model_pb2
from xls.estimators.delay_model import delay_model_utils
from xls.estimators.delay_model import op_module_generator
from xls.synthesis import synthesis_pb2
from xls.synthesis import synthesis_service_pb2_grpc

_MAX_PS = flags.DEFINE_integer(
    'max_ps', 15000, 'Maximum picoseconds delay to test.'
)
_MIN_PS = flags.DEFINE_integer(
    'min_ps', 20, 'Minimum picoseconds delay to test.'
)
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    '',
    'File path at which to load and save checkpoints. Checkpoints will not be'
    ' kept if unspecified.',
)
_OUT_PATH = flags.DEFINE_string(
    'out_path', '', 'File path for the final output data points.'
)
_SAMPLES_PATH = flags.DEFINE_string(
    'samples_path', '', 'Path at which to load samples textproto.'
)
_OP_INCLUDE_LIST = flags.DEFINE_list(
    'op_include_list',
    [],
    'Names of ops from samples textproto to generate data points for. If empty,'
    ' all of them are included. Note that kIdentity is always included.',
)
_MAX_THREADS = flags.DEFINE_integer(
    'max_threads',
    max(os.cpu_count() // 2, 1),
    'Max number of threads for parallelizing the generation of data points.',
)


def check_delay_offset(results: estimator_model_pb2.DataPoints):
  # find the minimum nonzero delay, presumably from a reg-->reg connection
  minimum_delay = min([x.delay for x in results.data_points if x.delay])
  logging.vlog(0, f'USING DELAY_OFFSET {minimum_delay}')
  for dp in results.data_points:
    dp.delay_offset = minimum_delay


def save_checkpoint(
    result: estimator_model_pb2.DataPoint, checkpoint_path: str, write_lock: Any
) -> None:
  if checkpoint_path:
    write_lock.acquire()
    try:
      with gfile.open(checkpoint_path, 'a') as f:
        f.write('data_points {\n')
        f.write(textwrap.indent(text_format.MessageToString(result), '  '))
        f.write('}\n')
    finally:
      write_lock.release()


def _search_for_fmax_and_synth(
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    verilog_text: str,
    top_module_name: str,
) -> synthesis_pb2.CompileResponse:
  """Bisects the space of frequencies and sends requests to the server."""
  best_result = synthesis_pb2.CompileResponse()
  low_ps = _MIN_PS.value
  high_ps = _MAX_PS.value
  epsilon_ps = 1.0

  while high_ps - low_ps > epsilon_ps:
    current_ps = (high_ps + low_ps) / 2
    request = synthesis_pb2.CompileRequest()
    request.target_frequency_hz = int(1e12 / current_ps)
    request.module_text = verilog_text
    request.top_module_name = top_module_name
    logging.vlog(3, '--- Debug')
    logging.vlog(3, high_ps)
    logging.vlog(3, low_ps)
    logging.vlog(3, epsilon_ps)
    logging.vlog(3, current_ps)
    logging.vlog(3, '--- Request')
    logging.vlog(3, request)
    response = stub.Compile(request)
    logging.vlog(3, '--- response')
    logging.vlog(3, response.slack_ps)
    logging.vlog(3, response.max_frequency_hz)
    logging.vlog(3, response.netlist)

    if response.max_frequency_hz > 0:
      response_ps = 1e12 / response.max_frequency_hz
    else:
      response_ps = 0

    # If synthesis is insensitive to target frequency, we don't need to do
    # the binary search.  Just use the max_frequency_hz of the first response
    # (whether it passes or fails).
    if response.insensitive_to_target_freq and response.max_frequency_hz > 0:
      logging.debug('USING (@min %2.1fps).', response_ps)
      best_result = response
      break

    if response.slack_ps >= 0:
      if response.max_frequency_hz > 0:
        logging.debug(
            'PASS at %.1fps (slack %dps @min %2.1fps)',
            current_ps,
            response.slack_ps,
            response_ps,
        )
      else:
        logging.error('PASS but no maximum frequency determined.')
        logging.error(
            'ERROR: this occurs when an operator is optimized to a constant.'
        )
        logging.error('Source Verilog:\n%s', request.module_text)
        sys.exit()
      high_ps = current_ps
      if response.max_frequency_hz >= best_result.max_frequency_hz:
        best_result = response
    else:
      if response.max_frequency_hz:
        logging.debug(
            'FAIL at %.1fps (slack %dps @min %2.1fps).',
            current_ps,
            response.slack_ps,
            response_ps,
        )
      else:
        # This shouldn't happen
        logging.error('FAIL but no maximum frequency provided')
        sys.exit()
      # Speed things up if we're way off
      if current_ps < (response_ps / 2.0):
        high_ps = response_ps * 1.1
        low_ps = response_ps * 0.9
      else:
        low_ps = current_ps

  if best_result.max_frequency_hz:
    logging.debug(
        'Done at @min %2.1fps.',
        1e12 / best_result.max_frequency_hz,
    )
  else:
    logging.error('INTERNAL ERROR: no passing run.')
    sys.exit()
  return best_result


def _synthesize_ir(
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    ir_text: str,
    spec: delay_model_utils.SampleSpec,
    operand_element_counts: Dict[int, int],
) -> estimator_model_pb2.DataPoint:
  """Synthesizes the given IR text and checkpoint resulting data points."""
  logging.info(
      'Running %s with %d / %s',
      spec.op_samples.op,
      spec.point.result_width,
      ', '.join([str(x) for x in list(spec.point.operand_widths)]),
  )
  module_name = 'main'
  mod_generator_result = op_module_generator.generate_verilog_module(
      module_name, ir_text
  )

  op_comment = '// op: ' + spec.op_samples.op + ' \n'
  verilog_text = op_comment + mod_generator_result.verilog_text

  result = _search_for_fmax_and_synth(stub, verilog_text, module_name)
  logging.vlog(3, 'Result: %s', result)

  if result.max_frequency_hz > 0:
    ps = 1e12 / result.max_frequency_hz
  else:
    ps = 0

  # Add a new record to the results proto
  result_dp = estimator_model_pb2.DataPoint()
  result_dp.operation.op = spec.op_samples.op
  result_dp.operation.bit_count = spec.point.result_width
  if spec.op_samples.specialization:
    result_dp.operation.specialization = spec.op_samples.specialization
  for bit_count in list(spec.point.operand_widths):
    result_dp.operation.operands.add(bit_count=bit_count)
  for opnd_num, element_count in operand_element_counts.items():
    result_dp.operation.operands[opnd_num].element_count = element_count
  result_dp.delay = int(ps)
  # TODO(tcal) currently no support for array result type here

  return result_dp


def op_cpp_enum_to_name(cpp_enum_name: str) -> str:
  """Converts an op C++ enum (e.g., kZeroExt) to the op name (zero_ext)."""
  if not cpp_enum_name.startswith('k'):
    raise ValueError(f'Invalid op enum name {cpp_enum_name}')
  snake_case = ''
  for c in cpp_enum_name[1:]:
    if c.isupper():
      if snake_case:
        snake_case += '_'
      snake_case += c.lower()
    else:
      snake_case += c
  return snake_case


def _run_point(
    spec: delay_model_utils.SampleSpec,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    checkpoint_write_lock: Any,
) -> estimator_model_pb2.DataPoint:
  """Generate IR and Verilog, run synthesis for one op parameterization."""

  op = spec.op_samples.op
  specialization = spec.op_samples.specialization
  attributes = spec.op_samples.attributes

  op_name = op_cpp_enum_to_name(op)

  # Result type - bitwidth and optionally element count(s)
  res_bit_count = spec.point.result_width
  res_type = f'bits[{res_bit_count}]'
  for res_elem in spec.point.result_element_counts:
    res_type += f'[{res_elem}]'

  # Operand types - bitwidth and optionally element count(s)
  opnd_element_counts: Dict[int, int] = {}
  opnd_types = []
  for bw in spec.point.operand_widths:
    opnd_types.append(f'bits[{bw}]')
  for opnd_elements in spec.point.operand_element_counts:
    tot_elems = 1
    for count in opnd_elements.element_counts:
      opnd_types[opnd_elements.operand_number] += f'[{count}]'
      tot_elems *= count
    opnd_element_counts[opnd_elements.operand_number] = tot_elems

  # Handle attribute string (at most one key/value pair)
  if attributes:
    k, v = attributes.split('=')
    v = v.replace('%r', str(res_bit_count))
    attr = ((k, v),)
  else:
    attr = ()

  # TODO(tcal): complete handling for specialization == HAS_LITERAL_OPERAND
  logging.debug('types: %s : %s', res_type, ' '.join(opnd_types))
  literal_operand = None
  repeated_operand = (
      1 if (specialization == estimator_model_pb2.OPERANDS_IDENTICAL) else None
  )
  ir_text = op_module_generator.generate_ir_package(
      op_name, res_type, (opnd_types), attr, literal_operand, repeated_operand
  )
  logging.debug('ir_text:\n%s\n', ir_text)
  result_dp = _synthesize_ir(
      stub,
      ir_text,
      spec,
      opnd_element_counts,
  )

  # Checkpoint after every run.
  save_checkpoint(result_dp, _CHECKPOINT_PATH.value, checkpoint_write_lock)
  return result_dp


def load_checkpoints(checkpoint_path: str) -> estimator_model_pb2.DataPoints:
  """Loads data from a checkpoint, if available."""
  results = estimator_model_pb2.DataPoints()
  if checkpoint_path:
    with gfile.open(checkpoint_path, 'r') as f:
      contents = f.read()
      results = text_format.Parse(contents, results)
      logging.info(
          'Loaded %d prior checkpointed results from %s of size %d bytes.',
          len(results.data_points),
          checkpoint_path,
          len(contents),
      )
  return results


def run_characterization(
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Run characterization with the given synthesis service."""
  checkpointed_results = load_checkpoints(_CHECKPOINT_PATH.value)
  checkpoint_dict = delay_model_utils.map_data_points_by_key(
      checkpointed_results.data_points
  )
  samples_file = _SAMPLES_PATH.value
  op_samples_list = estimator_model_pb2.OpSamplesList()
  op_include_list = set()
  if _OP_INCLUDE_LIST.value:
    op_include_list.update(['kIdentity'] + _OP_INCLUDE_LIST.value)
  with gfile.open(samples_file, 'r') as f:
    op_samples_list = text_format.Parse(f.read(), op_samples_list)
  sample_specs_without_prior_checkpoints = []
  all_sample_spec_keys_in_order = []
  for op_samples in op_samples_list.op_samples:
    if not op_include_list or op_samples.op in op_include_list:
      for point in op_samples.samples:
        spec = delay_model_utils.SampleSpec(op_samples, point)
        spec_key = delay_model_utils.get_sample_spec_key(spec)
        all_sample_spec_keys_in_order.append(spec_key)
        if spec_key not in checkpoint_dict:
          sample_specs_without_prior_checkpoints.append(spec)
  logging.debug('Using thread pool of size %d', _MAX_THREADS.value)
  pool = mp_pool.ThreadPool(_MAX_THREADS.value)
  checkpoint_write_lock = mp.Lock()
  results_dict = delay_model_utils.map_data_points_by_key(
      pool.starmap(
          _run_point,
          (
              (request, stub, checkpoint_write_lock)
              for request in sample_specs_without_prior_checkpoints
          ),
      )
  )
  data_points_proto = estimator_model_pb2.DataPoints()
  for request_key in all_sample_spec_keys_in_order:
    if request_key in checkpoint_dict:
      data_points_proto.data_points.append(checkpoint_dict[request_key])
    else:
      data_points_proto.data_points.append(results_dict[request_key])
  logging.info(
      'Collected %d total data points.', len(data_points_proto.data_points)
  )
  check_delay_offset(data_points_proto)

  if _OUT_PATH.value:
    with gfile.open(_OUT_PATH.value, 'w') as f:
      f.write(text_format.MessageToString(data_points_proto))

  print('# proto-file: xls/estimators/estimator_model.proto')
  print('# proto-message: xls.estimator_model.DataPoints')
  print(data_points_proto)
