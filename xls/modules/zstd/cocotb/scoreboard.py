# Copyright 2024 The XLS Authors
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

from dataclasses import dataclass
from queue import Queue

from cocotb.clock import Clock
from cocotb.log import SimLog
from cocotb.utils import get_sim_time

from xls.modules.zstd.cocotb.channel import XLSChannelMonitor
from xls.modules.zstd.cocotb.xlsstruct import XLSStruct


@dataclass
class LatencyQueueItem:
  transaction: XLSStruct
  timestamp: int


class LatencyScoreboard:
  def __init__(self, dut, clock: Clock, req_monitor: XLSChannelMonitor, resp_monitor: XLSChannelMonitor):
    self.dut = dut
    self.log = SimLog(f"zstd.cocotb.scoreboard.{self.dut._name}")
    self.clock = clock
    self.req_monitor = req_monitor
    self.resp_monitor = resp_monitor
    self.pending_req = Queue()
    self.results = []

    self.req_monitor.add_callback(self._req_callback)
    self.resp_monitor.add_callback(self._resp_callback)

  def _current_cycle(self):
    return get_sim_time(units='step') / self.clock.period

  def _req_callback(self, transaction: XLSStruct):
    self.pending_req.put(LatencyQueueItem(transaction, self._current_cycle()))

  def _resp_callback(self, transaction: XLSStruct):
    latency_item = self.pending_req.get()
    self.results.append(self._current_cycle() - latency_item.timestamp)

  def average_latency(self):
    return sum(self.results)/len(self.results)

  def report_result(self):
    if not self.pending_req.empty():
      self.log.warning(f"There are unfulfilled requests from channel {self.req_monitor.name}")
      while not self.pending_req.empty():
        self.log.warning(f"Unfulfilled request: {self.pending_req.get()}")
    if len(self.results) > 0:
      self.log.info(f"Latency report - 1st latency: {self.results[0]}")
    if len(self.results) > 1:
      self.log.info(f"Latency report - 2nd latency: {self.results[1]}")
    if len(self.results) > 2:
      avg = sum(self.results[2:])/len(self.results[2:])
      self.log.info(f"Latency report - rest of the latencies (average): {avg}")
