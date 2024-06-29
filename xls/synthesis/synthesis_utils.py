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
"""Utility functions used in synthesis."""

from typing import Callable, Optional, Sequence, TypeVar

from absl import logging
import grpc

from xls.synthesis import synthesis_pb2
from xls.synthesis import synthesis_service_pb2_grpc

T = TypeVar('T')
U = TypeVar('U')


def run_bisect(
    start_index: int,
    limit_index: int,
    considered: Sequence[T],
    frun: Callable[[T], U],
    fresult: Callable[[T, U], bool],
) -> Optional[int]:
  """Performs bisection on 'considered', using 'frun'/'fresult'.

  Args:
    start_index: Starting index to use in 'considered'.
    limit_index: Limit index to use in 'considered'
    considered: Ordered space of inputs to consider running.
    frun: Runs a value from 'considered' to produce a result.
    fresult: Processes a result to say whether it is accepted.

  Returns:
    The highest index of the 'considered' which is accepted. Returns None
    if no elements are accepted.
  """
  max_ok_index = None
  while start_index != limit_index:
    index = start_index + (limit_index - start_index) // 2
    t = considered[index]
    u = frun(t)
    ok = fresult(t, u)
    if ok:
      start_index = index + 1
      max_ok_index = index if max_ok_index is None else max(max_ok_index, index)
    else:
      limit_index = index
  return max_ok_index


def bisect_frequency(
    verilog_text: str,
    top_module_name: str,
    start_hz: int,
    limit_hz: int,
    step_hz: int,
    grpc_channel: grpc.Channel,
) -> synthesis_pb2.SynthesisSweepResult:
  """Binary searches to determine maximum frequency of a given Verilog module.

  Args:
    verilog_text: Text of the Verilog.
    top_module_name: The name of the top module to synthesize.
    start_hz: Lowest frequency (inclusive) to search.
    limit_hz: Highest frequency (inclusive) to search.
    step_hz: The minimum frequency step to use when searching.
    grpc_channel: A channel to the SynthesisService GRPC service to use for
      synthesizing the verilog.

  Returns:
    A SynthesisSweepResult containing the result of the sweep (max frequency and
    CompileResponses).
  """
  sweep_result = synthesis_pb2.SynthesisSweepResult()

  def run_sample(target_hz: int) -> synthesis_pb2.CompileResponse:
    logging.info(
        'Running with target frequency %0.3fGHz. Range: [%0.3fGHz, %0.3fGHz]',
        target_hz / 1e9,
        start_hz / 1e9,
        limit_hz / 1e9,
    )
    grpc.channel_ready_future(grpc_channel).result()
    stub = synthesis_service_pb2_grpc.SynthesisServiceStub(grpc_channel)
    request = synthesis_pb2.CompileRequest()
    request.module_text = verilog_text
    request.top_module_name = top_module_name
    request.target_frequency_hz = target_hz

    response = stub.Compile(request)

    synthesis_result = sweep_result.results.add()
    synthesis_result.target_frequency_hz = target_hz
    synthesis_result.response.CopyFrom(response)

    if response.slack_ps >= 0:
      logging.info('  PASSED TIMING')
      sweep_result.max_frequency_hz = max(
          sweep_result.max_frequency_hz, target_hz
      )
    else:
      logging.info('  FAILED TIMING (slack %dps)', response.slack_ps)
    return response

  def sample_meets_timing(
      target_hz: int, response: synthesis_pb2.CompileResponse
  ) -> bool:
    del target_hz
    return response.slack_ps >= 0

  frequencies = list(range(start_hz, limit_hz, step_hz))
  run_bisect(
      0, len(frequencies) - 1, frequencies, run_sample, sample_meets_timing
  )
  return sweep_result
