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
"""Analyze Yosys and OpenSTA logs and gather metrics info.

Usage: gather_design_stats [--out <name.textproto>] [list of log files]

Overall and per-pipeline-stage statistics will be gathered from
provided Yosys and OpenSTA logfiles, and written to a DesignStats
textproto at a location specified by the '--out' option.
"""

import gzip
import re

from absl import app
from absl import flags

from google.protobuf import text_format
from xls.common import gfile
from xls.tools import design_stats_pb2

_OUT_PROTO = flags.DEFINE_string(
    "out", default="metrics.textproto", help="Path to output protobuf."
)
_DEBUG = flags.DEFINE_bool("debug", default=False, help="Enable debugging.")

_CHIP_AREA_TOP_REGEX = re.compile(r"Chip area for (top )?module.*: ([\d\.]+)")
_FLOP_REGEX = re.compile(r"Flop count p(\d+)mod: (\d+) objects.")
_FLOP_TOTAL_REGEX = re.compile(r"Flop count: (\d+) objects.")
_CELL_COUNT_REGEX = re.compile(r"Number of cells:\s+(\d+)")
_LONGEST_PATH_REGEX = re.compile(
    r"Longest topological path in p(\d+)mod \(length=(\d+)\)"
)
_HEADER_REGEX = re.compile(r"===\s(.*)\s===")
_PIPELINE_STAGE_REGEX = re.compile(r"p(\d+)mod")
_CHIP_AREA_MODULE_REGEX = re.compile(
    r"Chip area for module.*p(\d+)mod.*: ([\d\.]+)"
)

_TIMING_REGEX = re.compile(r"Timing p(\d+)mod")
_DATA_ARRIVAL_REGEX = re.compile(r"([\d\.]+)\s+data arrival time")
_CRITICAL_PATH_END_REGEX = re.compile(r"p(\d+)mod Endpoint.*: (.+)$")
_CRITICAL_PATH_START_REGEX = re.compile(r"p(\d+)mod Startpoint: (.+)$")
_WNS_REGEX = re.compile(r"wns (-?\d+(\.\d+)?)$")
_TNS_REGEX = re.compile(r"tns (-?\d+(\.\d+)?)$")


def save_protobuf(model: design_stats_pb2.DesignStats, path: str):
  with gfile.open(path, "w") as f:
    f.write(text_format.MessageToString(model))


def ensure_stage(model: design_stats_pb2.DesignStats, stage: int):
  while len(model.per_stage) < stage + 1:
    model.per_stage.add()


def scrape_yosys(model: design_stats_pb2.DesignStats, f):
  """Read every line of f, look for known patterns, put info into 'model'."""

  context = ""
  while line := f.readline():
    if _DEBUG.value:
      if re.search(r"p(\d+)mod", line):
        print(line)

    if line.startswith("Warning:"):
      continue

    # === find_index ===
    # === p2mod ===
    if m := re.search(_HEADER_REGEX, line):
      context = m.group(1)
      if not model.design:  # first context is the top module.
        model.design = context

    # Chip area for module '\p2mod': 41.256600
    #  (unit: square microns)
    if m := re.search(_CHIP_AREA_MODULE_REGEX, line):
      stage = int(m.group(1))
      area_um2 = float(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].area_um2 = area_um2

    # Chip area for top module '\xls_fp_four_input_adder': 259.892640
    if m := re.search(_CHIP_AREA_TOP_REGEX, line):
      area_um2 = float(m.group(2))
      model.overall.area_um2 = area_um2

    # Flop count p0mod: 96 objects.
    if m := re.search(_FLOP_REGEX, line):
      stage = int(m.group(1))
      flops = int(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].flops = flops

    # Flop count: 216 objects.
    if m := re.search(_FLOP_TOTAL_REGEX, line):
      flops = int(m.group(1))
      model.overall.flops = flops

    # Number of cells:        189
    if m := re.search(_CELL_COUNT_REGEX, line):
      cells = int(m.group(1))
      # === p2mod ===
      if mm := re.search(_PIPELINE_STAGE_REGEX, context):
        stage = int(mm.group(1))
        ensure_stage(model, stage)
        model.per_stage[stage].cells = cells
      else:
        model.overall.cells = cells

    # Longest topological path in p0mod (length=1):
    if m := re.search(_LONGEST_PATH_REGEX, line):
      stage = int(m.group(1))
      ltp = int(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].levels = ltp


def scrape_opensta(model: design_stats_pb2.DesignStats, f):
  """Read every line of f, look for known patterns, put info into 'model'."""
  current_stage = None
  while line := f.readline():
    if _DEBUG.value:
      if re.search(r"p(\d+)mod", line):
        print(line)
    # Record current module (only needed for crit_path_delay_ps)
    if m := re.search(_TIMING_REGEX, line):
      current_stage = int(m.group(1))
      ensure_stage(model, current_stage)

    #   1.50   data arrival time
    if m := re.search(_DATA_ARRIVAL_REGEX, line):
      delay = float(m.group(1))
      if current_stage is not None:
        model.per_stage[current_stage].crit_path_delay_ps = delay
      else:
        model.overall.crit_path_delay_ps = delay
      current_stage = None

    # p2mod Endpoint: $auto$ff.cc:266:slice$5367/D
    # p2mod Endpoint net name: $abc$18269$p2_shifted_fraction__5_comb[14]
    if m := re.search(_CRITICAL_PATH_END_REGEX, line):
      stage = int(m.group(1))
      name = str(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].crit_path_end = name

    # p3mod Startpoint: p2_shifted_fraction__5[3]
    if m := re.search(_CRITICAL_PATH_START_REGEX, line):
      stage = int(m.group(1))
      name = str(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].crit_path_start = name

    # wns -0.02
    # NOTE: We expect that the WNS will be returned with a negative sign if it
    # exists; we express it as a positive number in our output.
    if m := re.search(_WNS_REGEX, line):
      wns = float(m.group(1))
      model.overall.wns = -wns if wns < 0 else 0.0

    # tns -39.67
    # NOTE: We expect that the TNS will be returned with a negative sign if it
    # exists; we express it as a positive number in our output.
    if m := re.search(_TNS_REGEX, line):
      tns = float(m.group(1))
      model.overall.tns = -tns if tns < 0 else 0.0


def scrape_file(model: design_stats_pb2.DesignStats, path: str):
  if ".gz" in path:
    file_handle = gzip.open(path, "rt", encoding="utf-8")
    scrape_yosys(model, file_handle)
  else:
    file_handle = gfile.open(path, "rt")
    scrape_opensta(model, file_handle)


def main(argv):
  if _DEBUG.value:
    print(argv)
  buf = design_stats_pb2.DesignStats()
  for f in argv[1:]:
    scrape_file(buf, f)
  save_protobuf(buf, _OUT_PROTO.value)


if __name__ == "__main__":
  app.run(main)
