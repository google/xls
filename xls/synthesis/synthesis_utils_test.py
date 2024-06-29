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
"""Tests for xls.synthesis.synthesis_utils."""

import subprocess
import time
from typing import Any, List, Optional, Sequence, Tuple

import grpc
import portpicker

from absl.testing import absltest
from xls.common import runfiles
from xls.synthesis import client_credentials
from xls.synthesis import synthesis_utils

SERVER_PATH = runfiles.get_path('xls/synthesis/fake_synthesis_server_main')


class SynthesisUtilsTest(absltest.TestCase):

  def _start_server(self, args):
    port = portpicker.pick_unused_port()
    proc = subprocess.Popen(
        [runfiles.get_path(SERVER_PATH), f'--port={port}'] + args
    )

    # allow some time for the server to open the port before continuing
    time.sleep(1)

    return port, proc

  def _run_bisect(
      self, seq: Sequence[Any], frun=None
  ) -> Tuple[Optional[int], List[Tuple[Any, Any]]]:
    frun = frun or (lambda x: -1.0 if x >= 29 else 1.0)
    results = []

    def fresult(i: int, x: float) -> bool:
      results.append((i, x))
      return x >= 0.0

    index = synthesis_utils.run_bisect(0, len(seq), seq, frun, fresult)
    return index, results

  def test_bisect_1_to_3_ghz_by_100mhz(self):
    seq = [10 + i for i in range(21)]
    max_index, results = self._run_bisect(seq)
    self.assertEqual(seq[max_index], 28)
    self.assertEqual(results, [(20, 1.0), (26, 1.0), (29, -1.0), (28, 1.0)])

  def test_bisect_2_to_3_ghz_by_100mhz(self):
    seq = [20 + i for i in range(11)]
    max_index, results = self._run_bisect(seq)
    self.assertEqual(seq[max_index], 28)
    self.assertEqual(results, [(25, 1.0), (28, 1.0), (30, -1.0), (29, -1.0)])

  def test_bisect_3_to_4_ghz_by_100mhz_does_not_meet_timing(self):
    seq = [30 + i for i in range(11)]
    max_index, results = self._run_bisect(seq)
    self.assertIsNone(max_index)
    self.assertEqual(results, [(35, -1.0), (32, -1.0), (31, -1.0), (30, -1.0)])

  def test_bisect_floats(self):
    seq = list(
        x / 100.0
        for x in range(int(1.0 * 100), int(3.0 * 100) + 1, int(0.1 * 100))
    )
    frun = lambda x: -1.0 if x >= 2.9 else 0
    max_index, results = self._run_bisect(seq, frun=frun)
    self.assertAlmostEqual(seq[max_index], 2.8)
    self.assertEqual(results, [(2.0, 0), (2.6, 0), (2.9, -1.0), (2.8, 0)])

  def test_bisect_frequencies(self):
    port, proc = self._start_server(['--max_frequency_ghz=2.0'])

    channel_creds = client_credentials.get_credentials()
    with grpc.secure_channel(f'localhost:{port}', channel_creds) as channel:
      result = synthesis_utils.bisect_frequency(
          'verilog', 'main', int(1.5e9), int(3e9), int(0.1e9), channel
      )
    self.assertEqual(result.max_frequency_hz, int(2e9))
    self.assertLen(result.results, 4)
    proc.terminate()
    proc.wait()

  def test_bisect_frequencies_infeasible(self):
    port, proc = self._start_server(['--max_frequency_ghz=2.0'])

    channel_creds = client_credentials.get_credentials()
    with grpc.secure_channel(f'localhost:{port}', channel_creds) as channel:
      result = synthesis_utils.bisect_frequency(
          'verilog', 'main', int(3e9), int(4e9), int(0.1e9), channel
      )
    self.assertEqual(result.max_frequency_hz, 0)
    proc.terminate()
    proc.wait()

  def test_bisect_frequencies_with_error(self):
    port, proc = self._start_server(
        ['--max_frequency_ghz=2.0', '--serve_errors']
    )

    channel_creds = client_credentials.get_credentials()
    with grpc.secure_channel(f'localhost:{port}', channel_creds) as channel:
      with self.assertRaises(grpc.RpcError):
        _ = synthesis_utils.bisect_frequency(
            'verilog', 'main', int(1.5e9), int(3e9), int(0.1e9), channel
        )
    proc.terminate()
    proc.wait()


if __name__ == '__main__':
  absltest.main()
