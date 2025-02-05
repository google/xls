# Copyright 2025 The XLS Authors
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

"""Test utility that converts XLS IR to a networkx graph."""

import pathlib

import networkx as nx

from absl.testing import absltest
from xls.common import runfiles
from xls.eco import xls_ir_to_networkx

_RISCV_SIMPLE_OPT_IR_PATH = runfiles.get_path(
    "xls/examples/riscv_simple.opt.ir"
)


class XlsIrToNetworkxTest(absltest.TestCase):

  def test_riscv_simple(self):
    graph = xls_ir_to_networkx.read_xls_ir_to_networkx(
        pathlib.Path(_RISCV_SIMPLE_OPT_IR_PATH)
    )
    self.assertIsInstance(graph, nx.DiGraph)
    self.assertNotEmpty(graph.nodes())
    self.assertNotEmpty(graph.edges())


if __name__ == "__main__":
  absltest.main()
