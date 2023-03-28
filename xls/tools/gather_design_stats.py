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


def scrape(model: design_stats_pb2.DesignStats, f):
  """Read every line of f, look for known patterns, put info into 'model'."""
  current_stage = -1
  while line := f.readline():
    if isinstance(line, bytes):
      line = line.decode("utf-8")
    line = line.rstrip()
    if _DEBUG.value:
      if m := re.search(r"p(\d+)mod", line):
        print(line)

    # Record current module (only needed for crit_path_delay_ps)
    if m := re.search(r"Timing p(\d+)mod", line):
      current_stage = int(m.group(1))
      ensure_stage(model, current_stage)

    #   1.50   data arrival time
    if m := re.search(r"([\d\.]+)\s+data arrival time", line):
      delay = float(m.group(1))
      if current_stage >= 0:
        model.per_stage[current_stage].crit_path_delay_ps = delay
      else:
        model.overall.crit_path_delay_ps = delay
      current_stage = -1

    # Chip area for module '\p2mod': 41.256600
    #  (unit: square microns)
    if m := re.search(r"Chip area for module.*p(\d+)mod.*: ([\d\.]+)", line):
      stage = int(m.group(1))
      area_um2 = float(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].area_um2 = area_um2

    # Chip area for top module '\xls_fp_four_input_adder': 259.892640
    if m := re.search(r"Chip area for top module.*: ([\d\.]+)", line):
      area_um2 = float(m.group(1))
      model.overall.area_um2 = area_um2

    # Flop count p0mod: 96 objects.
    if m := re.search(r"Flop count p(\d+)mod: (\d+) objects", line):
      stage = int(m.group(1))
      flops = int(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].flops = flops

    # Flop count: 216 objects.
    if m := re.search(r"Flop count: (\d+) objects", line):
      flops = int(m.group(1))
      model.overall.flops = flops

    # Longest topological path in p0mod (length=1):
    if m := re.search(
        r"Longest topological path in p(\d+)mod \(length=(\d+)\)", line
    ):
      stage = int(m.group(1))
      ltp = int(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].levels = ltp

    # Cell count in p1mod: 3406
    if m := re.search(r"Cell count in p(\d+)mod: (\d+)", line):
      stage = int(m.group(1))
      cells = int(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].cells = cells

    # p2mod Endpoint: $auto$ff.cc:266:slice$5367/D
    # p2mod Endpoint net name: $abc$18269$p2_shifted_fraction__5_comb[14]
    if m := re.search(r"p(\d+)mod Endpoint.*: (.+)$", line):
      stage = int(m.group(1))
      name = str(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].crit_path_end = name

    # p3mod Startpoint: p2_shifted_fraction__5[3]
    if m := re.search(r"p(\d+)mod Startpoint: (.+)$", line):
      stage = int(m.group(1))
      name = str(m.group(2))
      ensure_stage(model, stage)
      model.per_stage[stage].crit_path_start = name


def scrape_file(model: design_stats_pb2.DesignStats, path: str):
  if ".gz" in path:
    gz_file_handle = gfile.open(path, "rb")
    file_handle = gzip.GzipFile(fileobj=gz_file_handle, mode="r")
  else:
    file_handle = gfile.open(path, "rt")
  scrape(model, file_handle)


def main(argv):
  if _DEBUG.value:
    print(argv)
  buf = design_stats_pb2.DesignStats()
  for f in argv[1:]:
    scrape_file(buf, f)
  save_protobuf(buf, _OUT_PROTO.value)


if __name__ == "__main__":
  app.run(main)
