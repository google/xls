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
# ALSO assume that OpenROAD has pre-process libraries
import subprocess
import datetime
import random
import os

from absl import app
from absl import logging
from absl import flags
from time import sleep


class WorkerConfig:
    """Hosts all configs for workers."""
    bazel_path: str
    openroad_path: str
    debug: bool
    target: str
    yosys_bin: str
    synthesis_libraries: str
    sta_bin: str
    sta_libraries: str
    port: int
    server_extra_args: str
    client_extra_args: str


FLAGS = flags.FLAGS
flags.DEFINE_string('bazel_path', 'none', 'Root directory of bazel-bin')
flags.DEFINE_string('openroad_path', 'none', 'path to OpenROAD-flow')
flags.DEFINE_integer('debug', 0, 'debug messages 1 or 0')


def _do_config_task(config: WorkerConfig):
    """Extract configs from args and env """
    # Expect OPENROAD to point to OpenROAD-flow-scripts
    config.yosys_bin = f'{config.openroad_path}/tools/install/yosys/bin/yosys'
    config.sta_bin = f'{config.openroad_path}/tools/install/OpenROAD/bin/sta'

    if not os.path.isfile(config.yosys_bin):
        raise app.UsageError(f'Yosys tools not found with {config.yosys_bin}')

    if not os.path.isfile(config.sta_bin):
        raise app.UsageError(f'STA tool not found with {config.sta_bin}')

    config.rpc_port = random.randint(10000, 20000)
    config.client_checkpoint_file = f'{config.target}_checkpoint.textproto'

    config.client_args = '-max_width=8 --min_freq_mhz=1000 --max_freq_mhz=25000'

    if config.debug:
        config.server_extra_args = '--save_temps --v 1  --alsologtostderr'
        config.client_extra_args = '--v 3'
    else:
        config.server_extra_args = ''
        config.client_extra_args = ''


def _do_config_kAsap7(config: WorkerConfig):
    # Clearly there are better data structures to use...
    asap7_path = f'{config.openroad_path}/flow/platforms/asap7/lib'
    # OpenROAD merge cells into one lib for yosys - lets use that
    config.synthesis_libraries = f'{config.openroad_path}/flow/objects/asap7/ibex/base/lib/merged.lib'
    config.sta_libraries = f'\
    {asap7_path}/asap7sc7p5t_AO_RVT_FF_nldm_211120.lib.gz \
    {asap7_path}/asap7sc7p5t_INVBUF_RVT_FF_nldm_220122.lib.gz \
    {asap7_path}/asap7sc7p5t_OA_RVT_FF_nldm_211120.lib.gz \
    {asap7_path}/asap7sc7p5t_SIMPLE_RVT_FF_nldm_211120.lib.gz \
    {asap7_path}/asap7sc7p5t_SEQ_RVT_FF_nldm_220123.lib'
    config.client_args = '--max_width=8 --min_freq_mhz=1000 --max_freq_mhz=25000'


def _do_config_kNangate45(config: WorkerConfig):
    nangate45_path = f'{config.openroad_path}/flow/platforms/nangate45/lib'
    config.synthesis_libraries = f'{nangate45_path}/NangateOpenCellLibrary_typical.lib'
    config.sta_libraries = f'{nangate45_path}/NangateOpenCellLibrary_typical.lib'
    config.client_args = '--max_width=8 --min_freq_mhz=1000 --max_freq_mhz=11000'


def _do_config_kSky130(config: WorkerConfig):
    sky130_path = f'{config.openroad_path}/flow/platforms/sky130hd/lib'
    config.synthesis_libraries = f'{sky130_path}/sky130_fd_sc_hd__tt_025C_1v80.lib'
    config.sta_libraries = f'{sky130_path}/sky130_fd_sc_hd__tt_025C_1v80.lib'
    config.client_args = '--max_width=8 --min_freq_mhz=1000 --max_freq_mhz=11000'


def _do_worker_task(config: WorkerConfig):
    logging.info('Running')
    logging.info(f'  Using OpenROAD directory: {config.openroad_path}')
    logging.info(f'  Using Yosys : {config.yosys_bin}')
    logging.info(f'  Using STA   : {config.sta_bin}')
    logging.info(f'  With  target: {config.target}')

    server = f'{config.bazel_path}/xls/synthesis/yosys/yosys_sta_server_main \
    --yosys_path={config.yosys_bin} \
    --synthesis_libraries="{config.synthesis_libraries}" \
    --sta_path={config.sta_bin} \
    --sta_libraries="{config.sta_libraries}" \
    --port={config.rpc_port} \
    {config.server_extra_args}'

    client = f'{config.bazel_path}/xls/synthesis/timing_characterization_client_main \
    --checkpoint_path {config.client_checkpoint_file} \
    {config.client_args} {config.client_extra_args} \
    --port={config.rpc_port}'\

    # create a checkpoint file if not allready there
    with open(config.client_checkpoint_file, 'w') as f:
        f.write('')

    start = datetime.datetime.now()

    # start non-blocking process
    # server_process=subprocess.Popen(server, stdout=subprocess.PIPE,shell=True,text=True)
    server_process = subprocess.Popen(
        server, stdout=subprocess.PIPE, shell=True)

    sleep(5)
    client_process = subprocess.Popen(
        client, stdout=subprocess.PIPE, shell=True, text=True)

#    if (config.debug):
#        while server_process.poll() is None:
#           logging.info('server still working...')
#           sleep(58)

#       while client_process.poll() is None:
#           logging.info('client still working...')
#           sleep(63)

    # block process
    client_process.communicate()
    #out_value = client_process.communicate()
    elapsed = (datetime.datetime.now() - start)

    logging.info(
        f' Total elapsed time for worker ({config.target}) : {elapsed}')


def main(argv):
    """ Real main """
    config = WorkerConfig

    config.openroad_path = os.path.realpath(FLAGS.openroad_path)
    config.debug = FLAGS.debug
    config.bazel_path = os.path.realpath(FLAGS.bazel_path)

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
