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

"""Runs the timing_characterization with OpenROAD flow-scripts tooling and libraries."""
# Assume that https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts are build
# ALSO assume that OpenROAD has pre-processed libraries
import subprocess
import portpicker
import datetime
import random
import os

from absl import app
from absl import logging
from absl import flags
from time import sleep


class WorkerConfig:
  """Configuration for workers process."""
  bazel_bin_path: str
  openroad_path: str
  debug: bool
  target: str
  yosys_bin: str
  client_bin: str
  server_bin: str
  synthesis_libraries = []
  sta_bin: str
  sta_libraries = []
  port: int
  server_extra_args = []
  client_args = []
  client_extra_arg = []


FLAGS = flags.FLAGS
flags.DEFINE_string('bazel_bin_path', 'None', 'Root directory of bazel-bin')
flags.DEFINE_string('openroad_path', 'None',
                    'Root directory of OpenROAD-flow-scripts')
flags.DEFINE_bool(
    'debug', False, 'Enable verbose debugging info for client and server')


def _do_config_task(config: WorkerConfig):
  """Extract configs from args and env """
  # Expect OpenROAD-flow-scripts to hold tools
  config.yosys_bin = f'{config.openroad_path}/tools/install/yosys/bin/yosys'
  config.sta_bin = f'{config.openroad_path}/tools/install/OpenROAD/bin/sta'
  config.server_bin = f'{config.bazel_bin_path}/xls/synthesis/yosys/yosys_sta_server_main'
  config.client_bin = f'{config.bazel_bin_path}/xls/synthesis/timing_characterization_client_main'

  if not os.path.isfile(config.yosys_bin):
    raise app.UsageError(f'Yosys tools not found with {config.yosys_bin}')

  if not os.path.isfile(config.sta_bin):
    raise app.UsageError(f'STA tool not found with {config.sta_bin}')

  if not os.path.isfile(config.server_bin):
    raise app.UsageError(f'Server tool not found with {config.server_bin}')

  if not os.path.isfile(config.client_bin):
    raise app.UsageError(f'Client tool not found with {config.client_bin}')

  config.rpc_port = portpicker.pick_unused_port()
  config.client_checkpoint_file = f'{config.target}_checkpoint.textproto'

  if config.debug:
    config.server_extra_args = ['--save_temps', '--v 1', '--alsologtostderr']
    config.client_extra_args = ['--v 3']
  else:
    config.server_extra_args = ['']
    config.client_extra_args = ['']


def _do_config_kAsap7(config: WorkerConfig):
  asap7_path = f'{config.openroad_path}/flow/platforms/asap7/lib'
  config.synthesis_libraries.append(
      f'{config.openroad_path}/flow/objects/asap7/ibex/base/lib/merged.lib')
  config.sta_libraries.append(
      f'{asap7_path}/asap7sc7p5t_AO_RVT_FF_nldm_211120.lib.gz')
  config.sta_libraries.append(
      f'{asap7_path}/asap7sc7p5t_INVBUF_RVT_FF_nldm_220122.lib.gz')
  config.sta_libraries.append(
      f'{asap7_path}/asap7sc7p5t_OA_RVT_FF_nldm_211120.lib.gz')
  config.sta_libraries.append(
      f'{asap7_path}/asap7sc7p5t_SIMPLE_RVT_FF_nldm_211120.lib.gz')
  config.sta_libraries.append(
      f'{asap7_path}/asap7sc7p5t_SEQ_RVT_FF_nldm_220123.lib')
  config.client_args.append('--max_width=8')
  config.client_args.append('--min_freq_mhz=11000')
  config.client_args.append('--max_freq_mhz=25000')


def _do_config_kNangate45(config: WorkerConfig):
  nangate45_path = f'{config.openroad_path}/flow/platforms/nangate45/lib'
  config.synthesis_libraries.append(
      f'{nangate45_path}/NangateOpenCellLibrary_typical.lib')
  config.sta_libraries.append(
      f'{nangate45_path}/NangateOpenCellLibrary_typical.lib')
  config.client_args.append('--max_width=8')
  config.client_args.append('--min_freq_mhz=1000')
  config.client_args.append('--max_freq_mhz=11000')


def _do_config_kSky130(config: WorkerConfig):
  sky130_path = f'{config.openroad_path}/flow/platforms/sky130hd/lib'
  config.synthesis_libraries.append(
      f'{sky130_path}/sky130_fd_sc_hd__tt_025C_1v80.lib')
  config.sta_libraries.append(
      f'{sky130_path}/sky130_fd_sc_hd__tt_025C_1v80.lib')
  config.client_args.append('--max_width=8')
  config.client_args.append('--min_freq_mhz=1000')
  config.client_args.append('--max_freq_mhz=11000')


def _do_worker_task(config: WorkerConfig):
  logging.info(f'Running Target   : {config.target}')

  if config.debug:
    logging.info(f'  OpenROAD dir : {config.openroad_path}')
    logging.info(f'  Server       : {config.server_bin}')
    logging.info(f'  Client       : {config.client_bin}')
    logging.info(f'  Using Yosys  : {config.yosys_bin}')
    logging.info(f'  Using STA    : {config.sta_bin}')

  server = [config.server_bin]
  server.append(f'--yosys_path={config.yosys_bin}')
  server.append(
      f"--synthesis_libraries={' '.join(config.synthesis_libraries)}")

  server.append(f'--sta_path={config.sta_bin}')
  server.append(f"--sta_libraries=\"{' '.join(config.sta_libraries)}\"")

  server.append(f'--port={config.rpc_port}')
  server.append(' '.join(config.server_extra_args))

  server_cmd = repr(' '.join(server))
  server_cmd = server_cmd.replace("'", "")

  client = [config.client_bin]
  client.append(f'--checkpoint_path {config.client_checkpoint_file}')
  client.append(' '.join(config.client_args))
  client.append(' '.join(config.client_extra_args))
  client.append(f'--port={config.rpc_port}')

  client_cmd = repr(' '.join(client))
  client_cmd = client_cmd.replace("'", "")

  # create a checkpoint file if not allready there
  with open(config.client_checkpoint_file, 'w') as f:
    f.write('')

  start = datetime.datetime.now()

  # start non-blocking process
  server_process = subprocess.Popen(
      server_cmd, stdout=subprocess.PIPE, shell=True)

  sleep(5)

  client_process = subprocess.Popen(
      client_cmd, stdout=subprocess.PIPE, shell=True, text=True)

  # block process
  client_process.communicate()
  elapsed = (datetime.datetime.now() - start)

  logging.info(
      f' Total elapsed time for worker ({config.target}) : {elapsed}')


def main(argv):
  """ Real main """
  config = WorkerConfig

  config.openroad_path = os.path.realpath(FLAGS.openroad_path)
  config.debug = FLAGS.debug
  config.bazel_bin_path = os.path.realpath(FLAGS.bazel_bin_path)

  target_task = [('asap7',     _do_config_kAsap7),
                 ('nangate45', _do_config_kNangate45),
                 ('sky130',    _do_config_kSky130),
                 ]

  for target, task in target_task:
    config.target = target
    task(config)
    _do_config_task(config)
    _do_worker_task(config)


if __name__ == '__main__':
  app.run(main)
