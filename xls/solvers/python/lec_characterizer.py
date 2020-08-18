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
from typing import List, Text, Tuple

import grpc

from xls.delay_model import op_module_generator
from xls.ir import op_pb2
from xls.ir.python import type as type_mod
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

  def __init__(self, synthesis_server_args: List[Text],
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
      self.synthesis_server_.terminate()

  def _generate_sources(self, op: op_pb2.OpProto,
                        operand_types: List[type_mod.Type],
                        output_type: type_mod.Type) -> Tuple[Text, Text]:
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
    module_top = 'the_module'
    op_name = op_pb2.OpProto.Name(op)[3:].lower()
    operand_type_strs = [str(ot) for ot in operand_types]
    ir_text = op_module_generator.generate_ir_package(op_name, str(output_type),
                                                      operand_type_strs, [],
                                                      None)
    verilog_text = op_module_generator.generate_verilog_module(
        'the_module', ir_text).verilog_text

    creds = client_credentials.get_credentials()
    netlist_text = None
    with grpc.secure_channel('localhost:{}'.format(self.synthesis_server_port_),
                             creds) as channel:
      grpc.channel_ready_future(channel).result()
      stub = synthesis_service_pb2_grpc.SynthesisServiceStub(channel)

      request = synthesis_pb2.CompileRequest()
      request.module_text = verilog_text
      request.top_module_name = module_top
      # We're always going to be in a single cycle.
      request.target_frequency_hz = 1

      response = stub.Compile(request)
      netlist_text = response.netlist

    return (ir_text, netlist_text)
