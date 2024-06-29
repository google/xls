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
"""Sends a Verilog file to a synthesis server."""

from absl import app
from absl import flags
from absl import logging
import grpc

from xls.synthesis import client_credentials
from xls.synthesis import synthesis_pb2
from xls.synthesis import synthesis_service_pb2_grpc

FLAGS = flags.FLAGS
flags.DEFINE_integer('port', 10000, 'Port to connect to synthesis server on.')
flags.DEFINE_string('top', 'main', 'Top level module name.')
flags.DEFINE_float(
    'ghz', None, 'Runs a single frequency, instead of bisecting.'
)

flags.mark_flag_as_required('ghz')


def main(argv):
  if len(argv) != 2:
    raise app.UsageError('Must specify verilog file.')

  with open(argv[1], 'r') as f:
    verilog_text = f.read()

  channel_creds = client_credentials.get_credentials()
  with grpc.secure_channel(f'localhost:{FLAGS.port}', channel_creds) as channel:
    grpc.channel_ready_future(channel).result()
    stub = synthesis_service_pb2_grpc.SynthesisServiceStub(channel)

    request = synthesis_pb2.CompileRequest()
    request.module_text = verilog_text
    request.top_module_name = FLAGS.top
    request.target_frequency_hz = int(FLAGS.ghz * 1e9)
    logging.info('--- Request')
    logging.info(request)

    response = stub.Compile(request)
    print(response)


if __name__ == '__main__':
  app.run(main)
