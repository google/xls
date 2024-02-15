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


def save_protobuf(model: design_stats_pb2.DesignStats, path: str):
  with gfile.open(path, "w") as f:
    f.write(text_format.MessageToString(model))


def ensure_stage(model: design_stats_pb2.DesignStats, stage: int):
  while len(model.per_stage) < stage + 1:
    model.per_stage.add()


def scrape_yosys(model: design_stats_pb2.DesignStats, f):
  """Read every line of f, look for known patterns, put info into 'model'."""
  chip_area_module_regex = re.compile(
      r"Chip area for module.*p(\d+)mod.*: ([\d\.]+)"
  )
  chip_area_top_regex = re.compile(r"Chip area for (top )?module.*: ([\d\.]+)")
  flop_regex = re.compile(r"Flop count p(\d+)mod: (\d+) objects.")
  flop_total_regex = re.compile(r"Flop count: (\d+) objects.")
  cell_count_regex = re.compile(r"Cell count in p(\d+)mod: (\d+)")
  total_cell_count_regex = re.compile(r"Number of cells:\s+(\d+)")
  longest_path_regex = re.compile(
      r"Longest topological path in p(\d+)mod \(length=(\d+)\)"
  )

  while line := f.readline():
    if _DEBUG.value:
      if re.search(r"p(\d+)mod", line):
        print(line)

    if line.startswith("Warning:"):
      continue

    # Chip area for module '\p2mod': 41.256600
    #  (unit: square microns)
    if m := re.search(chip_area_module_regex, line):
      stage = int(m.group(1))
      area_um2 = float(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].area_um2 = area_um2

    # Chip area for top module '\xls_fp_four_input_adder': 259.892640
    if m := re.search(chip_area_top_regex, line):
      area_um2 = float(m.group(2))
      model.overall.area_um2 = area_um2

    # Flop count p0mod: 96 objects.
    if m := re.search(flop_regex, line):
      stage = int(m.group(1))
      flops = int(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].flops = flops

    # Flop count: 216 objects.
    if m := re.search(flop_total_regex, line):
      flops = int(m.group(1))
      model.overall.flops = flops

    # Cell count in p1mod: 3406
    if m := re.search(cell_count_regex, line):
      stage = int(m.group(1))
      cells = int(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].cells = cells

    # Number of cells:        189
    if not model.overall.cells:
      if m := re.search(total_cell_count_regex, line):
        cells = int(m.group(1))
        model.overall.cells = cells

    # Longest topological path in p0mod (length=1):
    if m := re.search(longest_path_regex, line):
      stage = int(m.group(1))
      ltp = int(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].levels = ltp


def scrape_opensta(model: design_stats_pb2.DesignStats, f):
  """Read every line of f, look for known patterns, put info into 'model'."""
  current_stage = None
  timing_regex = re.compile(r"Timing p(\d+)mod")
  data_arrival_regex = re.compile(r"([\d\.]+)\s+data arrival time")
  critcal_path_end_regex = re.compile(r"p(\d+)mod Endpoint.*: (.+)$")
  critcal_path_start_regex = re.compile(r"p(\d+)mod Startpoint: (.+)$")
  wns_regex = re.compile(r"wns (-?\d+(\.\d+)?)$")
  tns_regex = re.compile(r"tns (-?\d+(\.\d+)?)$")
  while line := f.readline():
    if _DEBUG.value:
      if re.search(r"p(\d+)mod", line):
        print(line)
    # Record current module (only needed for crit_path_delay_ps)
    if m := re.search(timing_regex, line):
      current_stage = int(m.group(1))
      ensure_stage(model, current_stage)

    #   1.50   data arrival time
    if m := re.search(data_arrival_regex, line):
      delay = float(m.group(1))
      if current_stage is not None:
        model.per_stage[current_stage].crit_path_delay_ps = delay
      else:
        model.overall.crit_path_delay_ps = delay
      current_stage = None

    # p2mod Endpoint: $auto$ff.cc:266:slice$5367/D
    # p2mod Endpoint net name: $abc$18269$p2_shifted_fraction__5_comb[14]
    if m := re.search(critcal_path_end_regex, line):
      stage = int(m.group(1))
      name = str(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].crit_path_end = name

    # p3mod Startpoint: p2_shifted_fraction__5[3]
    if m := re.search(critcal_path_start_regex, line):
      stage = int(m.group(1))
      name = str(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].crit_path_start = name

    # wns -0.02
    # NOTE: We expect that the WNS will be returned with a negative sign if it
    # exists; we express it as a positive number in our output.
    if m := re.search(wns_regex, line):
      wns = float(m.group(1))
      model.overall.wns = -wns if wns < 0 else 0.0

    # tns -39.67
    # NOTE: We expect that the TNS will be returned with a negative sign if it
    # exists; we express it as a positive number in our output.
    if m := re.search(tns_regex, line):
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
