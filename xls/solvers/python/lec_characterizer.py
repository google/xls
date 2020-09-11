# Lint as: python3
# Copyright 2020 Google LLC
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

import subprocess
import time
from typing import Callable, List, Tuple

from absl import logging
import grpc

from google.protobuf import text_format
from xls.common import gfile
from xls.delay_model import op_module_generator
from xls.ir import op_pb2
from xls.ir.python import ir_parser
from xls.ir.python import type as type_mod
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

  def __init__(self, synthesis_server_args: List[str],
               synthesis_server_port: int):
    """Starts the provided netlist synthesis server.

    Args:
      synthesis_server_args: List of arguments to start the netlist synthesis
        server, including its argv[0].
      synthesis_server_port: Port on which the server will be listening.
    """
    self.synthesis_server_ = subprocess.Popen(synthesis_server_args)
    self.synthesis_server_port_ = synthesis_server_port

  def __del__(self):
    """Terminates the netlist synthesis server, if started."""
    if self.synthesis_server_:
      self.synthesis_server_.kill()
      self.synthesis_server_.wait()

  def _generate_sources(self, op: op_pb2.OpProto,
                        operand_types: List[type_mod.Type],
                        output_type: type_mod.Type) -> Tuple[str, str]:
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
    operand_type_strs = [str(ot) for ot in operand_types]
    ir_text = op_module_generator.generate_ir_package(op_name, str(output_type),
                                                      operand_type_strs, [],
                                                      None)
    verilog_text = op_module_generator.generate_verilog_module(
        self._MODULE_NAME, ir_text).verilog_text

    creds = client_credentials.get_credentials()
    netlist_text = None
    with grpc.secure_channel('localhost:{}'.format(self.synthesis_server_port_),
                             creds) as channel:
      grpc.channel_ready_future(channel).result()
      stub = synthesis_service_pb2_grpc.SynthesisServiceStub(channel)

      request = synthesis_pb2.CompileRequest()
      request.module_text = verilog_text
      request.top_module_name = self._MODULE_NAME
      # We're always going to be in a single cycle.
      request.target_frequency_hz = 1

      response = stub.Compile(request)
      netlist_text = response.netlist

    return (ir_text, netlist_text)

  def _run_sample(
      self,
      ir_text: str,
      netlist_text: str,
      num_iters: int,
      cell_library_textproto: str,
      results: lec_characterizer_pb2.LecTiming,
      results_path: str,
      lec_fn: Callable[[str, str, str, str], bool] = z3_lec.run) -> bool:
    """Executes LEC for a single IR/netlist pair.

    Args:
      ir_text: The input IR to lec_fn.
      netlist_text: The input netlist to lec_fn.
      num_iters: The number of iterations to run for each sample.
      cell_library_textproto: Text-format proto containing the netlist's cell
        library.
      results: The LecTiming proto for this entire run.
      results_path: Path to which to write output the results proto.
      lec_fn: The function to execute for timing information. Takes in the IR
        text, the netlist text, the name of the netlist module to compare, and
        the cell library textproto. Returns True if the IR and netlist are
        proved to be equivalent.

    Returns:
      True if the generated IR and netlist are proved equivalent, and False
      otherwise.
    """
    # Get or create the test case [proto] of interest.
    package = ir_parser.Parser.parse_package(ir_text)
    function_type = package.get_function(self._FUNCTION_NAME).get_type()

    test_case = None

    function_type_textproto = function_type.to_textproto()
    for result_case in results.test_cases:
      # As a reminder: we can't pass proper protos over the pybind11 boundary,
      # so it's simpler to compare textprotos.
      result_function_type_textproto = text_format.MessageToString(
          result_case.function_type)
      if result_function_type_textproto == function_type_textproto:
        test_case = result_case
        break

    if test_case is None:
      test_case = results.test_cases.add()
      text_format.Parse(function_type.to_textproto(), test_case.function_type)

    for _ in range(num_iters):
      start_time = time.monotonic()
      are_equal = lec_fn(ir_text, netlist_text, self._MODULE_NAME,
                         cell_library_textproto)
      if not are_equal:
        logging.error('Bad comparison: ir: %s, netlist: %s', ir_text,
                      netlist_text)
        return False

      duration = time.monotonic() - start_time
      test_case.exec_times_us.append(int(duration * 1000000))

      total_time = 0
      for exec_time in test_case.exec_times_us:
        total_time += exec_time
      test_case.average_us = int(total_time / len(test_case.exec_times_us))

      # Some tests could be long-running, so write after every iter for safety.
      with gfile.open(results_path, 'w') as f:
        f.write(text_format.MessageToString(results))

  def run(self,
          op: op_pb2.OpProto,
          samples: List[Tuple[List[type_mod.Type], type_mod.Type]],
          num_iters: int,
          cell_library_textproto: str,
          results_path: str,
          lec_fn: Callable[[str, str, str, str], bool] = z3_lec.run) -> bool:
    """Characterizes LEC timing across a set of data types.

    This function iterates over the input samples (collections of arg types),
    creates IR and a netlist for each, and sends them to _run_sample()
    to execute.

    Args:
      op: The IR operator to characterize.
      samples: A list of ([Arg type], Return type) tuples, each of which
        represents the input and output types for a sample to run.
      num_iters: The number of iterations to run for each sample.
      cell_library_textproto: Text-format proto containing the netlist's cell
        library.
      results_path: Path to output the results proto. If this file already
        exists, then we append the results of this execution to its contents.
        execution.
      lec_fn: The function to execute for timing information. Takes in the IR
        text, the netlist text, the name of the netlist module to compare, and
        the cell library textproto. Returns True if the IR and netlist are
        proved to be equivalent.

    Returns:
      True if the generated IR and netlist are proved equivalent, and False
      otherwise.
    """
    results = lec_characterizer_pb2.LecTiming()
    if gfile.exists(results_path):
      with gfile.open(results_path, 'r') as f:
        text_format.Parse(f.read(), results)
    else:
      results.ir_function = 'single_op_' + op_pb2.OpProto.Name(op)

    for (operand_types, output_type) in samples:
      ir_text, netlist_text = self._generate_sources(op, operand_types,
                                                     output_type)

      if not self._run_sample(ir_text, netlist_text, num_iters,
                              cell_library_textproto, results, results_path,
                              lec_fn):
        return False

    return True
