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

from typing import Any, Sequence, Union

import cocotb
from cocotb.binary import BinaryValue
from cocotb.handle import SimHandleBase
from cocotb.triggers import RisingEdge
from cocotb_bus.bus import Bus
from cocotb_bus.drivers import BusDriver
from cocotb_bus.monitors import BusMonitor

Transaction = Union[BinaryValue, Sequence[BinaryValue]]

XLS_CHANNEL_SIGNALS = ["data", "rdy", "vld"]
XLS_CHANNEL_OPTIONAL_SIGNALS = []


class XLSChannel(Bus):
  _signals = XLS_CHANNEL_SIGNALS
  _optional_signals = XLS_CHANNEL_OPTIONAL_SIGNALS

  def __init__(self, entity, name, **kwargs: Any):
    super().__init__(entity, name, self._signals, self._optional_signals, **kwargs)


class XLSChannelDriver(BusDriver):
  _signals = XLS_CHANNEL_SIGNALS
  _optional_signals = XLS_CHANNEL_OPTIONAL_SIGNALS

  def __init__(self, entity: SimHandleBase, name: str, clock: SimHandleBase, **kwargs: Any):
    BusDriver.__init__(self, entity, name, clock, **kwargs)

    self.bus.data.setimmediatevalue(0)
    self.bus.vld.setimmediatevalue(0)

  async def _driver_send(self, transaction: Transaction, sync: bool = True, **kwargs: Any) -> None:
    if sync:
      await RisingEdge(self.clock)

    data_to_send = (transaction if isinstance(transaction, Sequence) else [transaction])

    for word in data_to_send:
      self.bus.vld.value = 1
      self.bus.data.value = word

      while True:
        await RisingEdge(self.clock)
        if self.bus.rdy.value:
          break

      self.bus.vld.value = 0


class XLSChannelMonitor(BusMonitor):
  _signals = XLS_CHANNEL_SIGNALS
  _optional_signals = XLS_CHANNEL_OPTIONAL_SIGNALS

  def __init__(self, entity: SimHandleBase, name: str, clock: SimHandleBase, **kwargs: Any):
    BusMonitor.__init__(self, entity, name, clock, **kwargs)

  @cocotb.coroutine
  async def _monitor_recv(self) -> None:
    while True:
      await RisingEdge(self.clock)
      if self.bus.rdy.value:
        vec = self.bus.data.value
        self._recv(vec)
