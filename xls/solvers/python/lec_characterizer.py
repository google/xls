# Lint as: python3
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
"""Characterizes time-to-LEC for a given computation."""

import time
from typing import Callable, Iterable, Optional, Tuple

from absl import logging
import grpc

from xls.delay_model import op_module_generator
from xls.ir import op_pb2
from xls.ir import xls_type_pb2
from xls.solvers.python import lec_characterizer_pb2
from xls.solvers.python import z3_lec
from xls.synthesis import client_credentials
from xls.synthesis import synthesis_pb2
from xls.synthesis import synthesis_service_pb2_grpc


class LecCharacterizer(object):
  """Driver class for measuring time-to-LEC over computations and args.

  Given a set of computation x argument types, this class manages creating
  inputs for each test, proving their equivalence (i.e., runs LEC), records
  that execution time, and saves the results. This enables us (XLS) to
  better understand the performance of our logical equivalence methods and
  improvements thereto.
  """

  # The name to use for the generated module. Hardcoded to 'main' in
  # op_module_generator.
  _FUNCTION_NAME = 'main'

  # The name to use for the generated module.
  _MODULE_NAME = 'the_module'

  def __init__(self, synthesis_server_address: str):
    """Simple init: stores the address of the synthesis server.

    Args:
      synthesis_server_address: Address on which the netlist synthesizer server
        is listening.
    """
    self._synthesis_server_address = synthesis_server_address

  def _proto_to_ir_type(self, proto: xls_type_pb2.TypeProto) -> str:
    """Converts a XLS type proto into its in-IR textual representation.

    Args:
      proto: The XLS TypeProto to convert.

    Returns:
      The string describing the type as in XLS IR.

    Raises:
      ValueError: if an invalid type is passed in (not bits, array, or tuple).
    """
    if proto.type_enum == xls_type_pb2.TypeProto.BITS:
      return 'bits[{}]'.format(proto.bit_count)
    elif proto.type_enum == xls_type_pb2.TypeProto.ARRAY:
      return '{}[{}]'.format(
          self._proto_to_ir_type(proto.array_element), proto.array_size)
    elif proto.type_enum == xls_type_pb2.TypeProto.TUPLE:
      elements = [self._proto_to_ir_type(x) for x in proto.tuple_elements]
      return '({})'.format(', '.join(elements))

    raise ValueError('Unhandled type_proto: {}'.format(
        xls_type_pb2.TypeProto.TypeEnum.Name(proto.type_enum)))

  def _generate_sources(self, op: op_pb2.OpProto,
                        operand_types: Iterable[xls_type_pb2.TypeProto],
                        output_type: xls_type_pb2.TypeProto) -> Tuple[str, str]:
    """Generates XLS IR and netlist sources for a single LEC execution.

    This function creates IR and a netlist for the given op and argument/output
    types, suitable as inputs to a LEC operation.

    Currently, this only supports a single op (per the internal operation of
    op_module_generator). In the future, it will likely be useful to see how
    execution time scales with operation composition.

    Args:
      op: The XLS IR opcode for which to generate sources.
      operand_types: The types of the arguments to use for this op execution.
      output_type: The type of the operation output.

    Returns:
      A tuple of IR and netlist sources for executing the given operation as
      text.
    """
    op_name = op_pb2.OpProto.Name(op)[3:].lower()
    operand_type_strs = [self._proto_to_ir_type(ot) for ot in operand_types]
    ir_text = op_module_generator.generate_ir_package(
        op_name, self._proto_to_ir_type(output_type), operand_type_strs, [],
        None)
    verilog_text = op_module_generator.generate_verilog_module(
        self._MODULE_NAME, ir_text).verilog_text

    creds = client_credentials.get_credentials()
    netlist_text = None
    with grpc.secure_channel(self._synthesis_server_address, creds) as channel:
      grpc.channel_ready_future(channel).result()
      stub = synthesis_service_pb2_grpc.SynthesisServiceStub(channel)

      request = synthesis_pb2.CompileRequest()
      logging.vlog(logging.INFO, 'Module text:\n %s', verilog_text)
      request.module_text = verilog_text
      request.top_module_name = self._MODULE_NAME
      # We're always going to be in a single cycle.
      request.target_frequency_hz = 1

      response = stub.Compile(request)
      netlist_text = response.netlist

    return (ir_text, netlist_text)

  def run(
      self,
      results: lec_characterizer_pb2.LecTiming,
      op: op_pb2.OpProto,
      function_type: xls_type_pb2.FunctionTypeProto,
      num_iters: int,
      cell_library_textproto: str,
      lec_fn: Callable[[str, str, str, str], bool] = z3_lec.run,
      results_fn: Optional[Callable[[lec_characterizer_pb2.LecTiming],
                                    None]] = None) -> bool:
    """Executes LEC for a single IR/netlist pair.

    Args:
      results: The LecTiming proto for this entire run.
      op: The operation to characterize.
      function_type: The signature of the op function, i.e., the types with
        which to execute.
      num_iters: The number of iterations to run for each sample.
      cell_library_textproto: Text-format proto containing the netlist's cell
        library.
      lec_fn: The function to execute for timing information. Takes in the IR
        text, the netlist text, the name of the netlist module to compare, and
        the cell library textproto. Returns True if the IR and netlist are
        proved to be equivalent.
      results_fn: A function to call for every execution completion, perhaps
        to store results to disk.

    Returns:
      True if the generated IR and netlist are proved equivalent, and False
      otherwise.
    """
    # Get or create the test case [proto] of interest.
    test_case = None
    for result_case in results.test_cases:
      if result_case.function_type == function_type:
        test_case = result_case
        break

    if test_case is None:
      test_case = results.test_cases.add()
      test_case.function_type.CopyFrom(function_type)

    # Only regenerate the sources if needed - it's slooooooooow.
    if not test_case.ir_text or not test_case.netlist_text:
      test_case.ir_text, test_case.netlist_text = self._generate_sources(
          op, function_type.parameters, function_type.return_type)

    for _ in range(num_iters):
      start_time = time.monotonic()
      are_equal = lec_fn(test_case.ir_text, test_case.netlist_text,
                         self._MODULE_NAME, cell_library_textproto)
      if not are_equal:
        logging.error('Bad comparison: ir: %s, netlist: %s', test_case.ir_text,
                      test_case.netlist_text)
        return False

      duration = time.monotonic() - start_time
      test_case.exec_times_us.append(int(duration * 1000000))

      total_time = 0
      for exec_time in test_case.exec_times_us:
        total_time += exec_time
      test_case.average_us = int(total_time / len(test_case.exec_times_us))

      if results_fn:
        results_fn(results)
