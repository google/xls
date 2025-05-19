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

"""
Extensions of the cocotb AXI RAM models to memory contents from a binary file.
"""

import os

from cocotbext.axi.axi_ram import AxiRam
from cocotbext.axi.axi_ram import AxiRamRead
from cocotbext.axi.axi_ram import AxiRamWrite
from cocotbext.axi.sparse_memory import SparseMemory


def init_axi_mem(path: os.PathLike[str], kwargs):
  with open(path, "rb") as f:
    sparse_mem = SparseMemory(size=kwargs["size"])
    sparse_mem.write(0x0, f.read())
    kwargs["mem"] = sparse_mem


class AxiRamReadFromFile(AxiRamRead):
  def __init__(self, *args, path: os.PathLike[str], **kwargs):
    init_axi_mem(path, kwargs)
    super().__init__(*args, **kwargs)


class AxiRamFromFile(AxiRam):
  def __init__(self, *args, path: os.PathLike[str], **kwargs):
    init_axi_mem(path, kwargs)
    super().__init__(*args, **kwargs)


class AxiRamWriteFromFile(AxiRamWrite):
  def __init__(self, *args, path: os.PathLike[str], **kwargs):
    init_axi_mem(path, kwargs)
    super().__init__(*args, **kwargs)
