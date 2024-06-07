#
# Copyright 2023 The XLS Authors
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

"""Runs timing characterization to generate XLS delay models using Yosys and OpenSTA.

There are two modes:

If --openroad_path is supplied, then scripts, tooling,
and libraries are found in the OpenROAD installation.
The set of PDKs used is hardcoded (sky130, asap7, and nangate45).
Timing characterization is run for all these PDKs.

If --openroad_path is NOT supplied, then paths to
Yosys, openSTA, synthesis library, and (if different from
synthesis librarty) STA libraries must be provided.
"""

# Assume that https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts are
# build ALSO assume that OpenROAD has pre-processed libraries.

import datetime
import os
import subprocess
import time
from typing import List

from absl import app
from absl import flags
from absl import logging
import portpicker

_SAMPLES_PATH = flags.DEFINE_string(
    'samples_path', None, 'Path to proto providing sample points.'
)
_OP_INCLUDE_LIST = flags.DEFINE_list(
    'op_include_list',
    [],
    'Names of ops from samples textproto to generate data points for. If empty,'
    ' all of them are included. Note that kIdentity is always included',
)
_BAZEL_BIN_PATH = flags.DEFINE_string(
    'bazel_bin_path', None, 'Root directory of bazel-bin'
)
_OPENROAD_PATH = flags.DEFINE_string(
    'openroad_path', None, 'Root directory of OpenROAD-flow-scripts'
)
_DEBUG = flags.DEFINE_bool(
    'debug', False, 'Enable verbose debugging info for client and server'
)
_QUICK_RUN = flags.DEFINE_bool(
    'quick_run', False, 'Do a small subset for testing.'
)

# The options below are used when openroad_path is NOT specified
_YOSYS_PATH = flags.DEFINE_string(
    'yosys_path', None, 'Path to Yosys executable'
)
_STA_PATH = flags.DEFINE_string(
    'sta_path', None, 'Path to sta/opensta executable'
)
_SYNTH_LIBS = flags.DEFINE_string(
    'synth_libs', None, 'Path to synthesis library or libraries'
)
_STA_LIBS = flags.DEFINE_string(
    'sta_libs',
    None,
    'Path to static timing library or libraries; '
    'only needed if different from synth_libs',
)
_DEFAULT_DRIVER_CELL = flags.DEFINE_string(
    'default_driver_cell',
    '',
    'The default driver cell to use during synthesis',
)
_DEFAULT_LOAD = flags.DEFINE_string(
    'default_load',
    '',
    'The default load cell to use during synthesis',
)
_OUT_PATH = flags.DEFINE_string('out_path', None, 'Path for output text proto')

# The options below are used when bazel_bin_path is NOT specified
_CLIENT = flags.DEFINE_string(
    'client', None, 'Path for timing characterization client executable'
)
_SERVER = flags.DEFINE_string(
    'server', None, 'Path for timing characterization server executable'
)


class WorkerConfig:
  """Configuration for workers process."""

  bazel_bin_path: str
  openroad_path: str
  yosys_path: str
  sta_path: str
  debug: bool
  target: str
  yosys_bin: str
  synthesis_libraries: List[str] = []
  sta_bin: str
  sta_libraries: List[str] = []
  samples_path: str

  server_bin: str
  server_extra_args = []
  rpc_port: int

  client_bin: str
  client_args = []
  client_extra_args = []
  client_checkpoint_file: str


def _do_config_task(config: WorkerConfig):
  """Extract configs from args and environment."""
  if config.openroad_path:
    # Expect OpenROAD-flow-scripts to hold tools
    config.yosys_bin = f'{config.openroad_path}/tools/install/yosys/bin/yosys'
    config.sta_bin = f'{config.openroad_path}/tools/install/OpenROAD/bin/sta'
    config.client_checkpoint_file = (
        f'../../{config.target}_checkpoint.textproto'
    )
  else:
    if not _YOSYS_PATH.value:
      raise app.UsageError(
          'Must provide either --openroad_path or --yosys_path.'
      )
    config.yosys_bin = os.path.realpath(_YOSYS_PATH.value)

    if not _STA_PATH.value:
      raise app.UsageError('Must provide either --openroad_path or --sta_path.')
    config.sta_bin = os.path.realpath(_STA_PATH.value)

    if not _SYNTH_LIBS.value:
      raise app.UsageError(
          'Must provide either --openroad_path or --synth_libs.'
      )
    synth_libs = _SYNTH_LIBS.value
    assert synth_libs is not None
    config.synthesis_libraries = synth_libs.split()

    if _STA_LIBS.value:
      sta_libs = _STA_LIBS.value
      assert sta_libs is not None
      config.sta_libraries = sta_libs.split()
    else:
      config.sta_libraries = config.synthesis_libraries

    if _OUT_PATH.value:
      out_path = _OUT_PATH.value
      assert out_path is not None
      config.client_checkpoint_file = out_path
    else:
      raise app.UsageError(
          'If not using --openroad_path, then must provide --out_path.'
      )

    if not _SAMPLES_PATH.value:
      if _QUICK_RUN.value:
        config.client_args.append('--max_width=2')
      else:
        config.client_args.append('--max_width=64')
    config.client_args.append('--max_ps=15000')
    config.client_args.append('--min_ps=10')

  if config.bazel_bin_path:
    config.server_bin = (
        f'{config.bazel_bin_path}/xls/synthesis/yosys/yosys_server_main'
    )
    config.client_bin = f'{config.bazel_bin_path}/xls/synthesis/timing_characterization_client_main'
  else:
    if not _SERVER.value:
      raise app.UsageError('Must provide either --bazel_bin_path or --server.')
    config.server_bin = os.path.realpath(_SERVER.value)
    if not _CLIENT.value:
      raise app.UsageError('Must provide either --bazel_bin_path or --client.')
    config.client_bin = os.path.realpath(_CLIENT.value)

  print('server bin path:', config.server_bin)
  print('client bin path:', config.client_bin)
  print(
      'output checkpoint path:', os.path.realpath(config.client_checkpoint_file)
  )

  if not os.path.isfile(config.yosys_bin):
    raise app.UsageError(f'Yosys tools not found with {config.yosys_bin}')

  if not os.path.isfile(config.sta_bin):
    raise app.UsageError(f'STA tool not found with {config.sta_bin}')

  if not os.path.isfile(config.server_bin):
    raise app.UsageError(f'Server tool not found with {config.server_bin}')

  if not os.path.isfile(config.client_bin):
    raise app.UsageError(f'Client tool not found with {config.client_bin}')

  config.rpc_port = portpicker.pick_unused_port()

  if config.debug:
    config.server_extra_args = ['--save_temps', '--v 1', '--alsologtostderr']
    config.client_extra_args = ['--v 3']
  else:
    config.server_extra_args = ['--save_temps']
    config.client_extra_args = ['--alsologtostderr']

  if _QUICK_RUN.value:
    config.client_extra_args.append('--quick_run')

  if _SAMPLES_PATH.value:
    config.samples_path = os.path.realpath(_SAMPLES_PATH.value)
    config.client_extra_args.append(f'--samples_path={config.samples_path}')
    config.client_extra_args.append(
        '--op_include_list=' + ','.join(_OP_INCLUDE_LIST.value)
    )


def _do_config_asap7(config: WorkerConfig):
  """Configure ASAP7."""
  asap7_path = f'{config.openroad_path}/flow/platforms/asap7/lib'
  config.synthesis_libraries.append(
      f'{config.openroad_path}/flow/objects/asap7/ibex/base/lib/merged.lib'
  )
  config.sta_libraries.append(
      f'{asap7_path}/asap7sc7p5t_AO_RVT_FF_nldm_211120.lib.gz'
  )
  config.sta_libraries.append(
      f'{asap7_path}/asap7sc7p5t_INVBUF_RVT_FF_nldm_220122.lib.gz'
  )
  config.sta_libraries.append(
      f'{asap7_path}/asap7sc7p5t_OA_RVT_FF_nldm_211120.lib.gz'
  )
  config.sta_libraries.append(
      f'{asap7_path}/asap7sc7p5t_SIMPLE_RVT_FF_nldm_211120.lib.gz'
  )
  config.sta_libraries.append(
      f'{asap7_path}/asap7sc7p5t_SEQ_RVT_FF_nldm_220123.lib'
  )
  config.client_args.append('--max_width=8')
  config.client_args.append('--min_ps=100')
  config.client_args.append('--max_ps=10000')


def _do_config_nangate45(config: WorkerConfig):
  nangate45_path = f'{config.openroad_path}/flow/platforms/nangate45/lib'
  config.synthesis_libraries.append(
      f'{nangate45_path}/NangateOpenCellLibrary_typical.lib'
  )
  config.sta_libraries.append(
      f'{nangate45_path}/NangateOpenCellLibrary_typical.lib'
  )
  config.client_args.append('--max_width=8')
  config.client_args.append('--min_ps=100')
  config.client_args.append('--max_ps=10000')


def _do_config_sky130(config: WorkerConfig):
  sky130_path = f'{config.openroad_path}/flow/platforms/sky130hd/lib'
  config.synthesis_libraries.append(
      f'{sky130_path}/sky130_fd_sc_hd__tt_025C_1v80.lib'
  )
  config.sta_libraries.append(
      f'{sky130_path}/sky130_fd_sc_hd__tt_025C_1v80.lib'
  )
  config.client_args.append('--max_width=8')
  config.client_args.append('--min_ps=100')
  config.client_args.append('--max_ps=10000')


def _do_worker_task(config: WorkerConfig):
  """Run the worker task."""
  logging.info('Running Target   : {config.target}')

  if config.debug:
    logging.info('  OpenROAD dir : %s', config.openroad_path)
    logging.info('  Server       : %s', config.server_bin)
    logging.info('  Client       : %s', config.client_bin)
    logging.info('  Using Yosys  : %s', config.yosys_bin)
    logging.info('  Using STA    : %s', config.sta_bin)

  server = [repr(config.server_bin)]
  server.append(f'--yosys_path={config.yosys_bin!r}')
  server.append(
      f"--synthesis_libraries={' '.join(config.synthesis_libraries)!r}"
  )
  server.append('--return_netlist=false')

  server.append(f'--sta_path={config.sta_bin!r}')
  server.append(f"--sta_libraries={' '.join(config.sta_libraries)!r}")
  if _DEFAULT_DRIVER_CELL.value:
    server.append(f'--default_driver_cell={_DEFAULT_DRIVER_CELL.value!r}')
  if _DEFAULT_LOAD.value:
    server.append(f'--default_load={_DEFAULT_LOAD.value!r}')

  server.append(f'--port={config.rpc_port}')
  server.extend(repr(arg) for arg in config.server_extra_args)

  server_cmd = ' '.join(server)

  client = [repr(config.client_bin)]
  client.append(f'--checkpoint_path {config.client_checkpoint_file!r}')
  client.extend(repr(arg) for arg in config.client_args)
  client.extend(repr(arg) for arg in config.client_extra_args)
  client.append(f'--port={config.rpc_port}')

  client_cmd = ' '.join(client)

  # create a checkpoint file if not allready there
  with open(config.client_checkpoint_file, 'w') as f:
    f.write('')

  start = datetime.datetime.now()

  # start non-blocking process
  server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, shell=True)

  time.sleep(5)

  client_process = subprocess.Popen(
      client_cmd, stdout=subprocess.PIPE, shell=True, text=True
  )

  # block process
  client_process.communicate()
  elapsed = datetime.datetime.now() - start

  logging.info(
      ' Total elapsed time for worker (%s) : %s', config.target, elapsed
  )

  # clean up
  server_proc.kill()
  server_proc.communicate()


def main(_):
  """Real main."""
  config = WorkerConfig()

  config.debug = _DEBUG.value

  if _BAZEL_BIN_PATH.value:
    config.bazel_bin_path = os.path.realpath(_BAZEL_BIN_PATH.value)
  else:
    config.bazel_bin_path = None

  if _OPENROAD_PATH.value:
    config.openroad_path = os.path.realpath(_OPENROAD_PATH.value)
  else:
    config.openroad_path = None

  if config.openroad_path:
    target_task = [
        ('asap7', _do_config_asap7),
        ('nangate45', _do_config_nangate45),
        ('sky130', _do_config_sky130),
    ]
    for target, task in target_task:
      # Be careful re-using same config object
      config.synthesis_libraries = []
      config.sta_libraries = []

      print('Start ', target)
      config.target = target
      task(config)
      _do_config_task(config)
      _do_worker_task(config)
      print('Finish ', target)

  else:
    print('Start')
    config.target = 'user'
    _do_config_task(config)
    _do_worker_task(config)
    print('Finish')


if __name__ == '__main__':
  app.run(main)
