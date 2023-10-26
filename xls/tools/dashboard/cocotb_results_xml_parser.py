#!/usr/bin/env python3
# Copyright 2023 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import sys
import xml.etree.ElementTree as ET


def parse_testcase(testcase):
  name = testcase.get("name")
  classname = testcase.get("classname")
  sim_time_ns = testcase.get("sim_time_ns")
  ratio_time = testcase.get("ratio_time")
  time = testcase.get("time")

  return {
    "name": f"{classname}.{name}",
    "group": "cocotb",
    "type": "table",
    "value": [
      ["Name", "Value"],
      ["real time [s]", time],
      ["sim time [ns]", sim_time_ns],
      ["ratio_time", ratio_time],
    ],
  }


if __name__ == "__main__":
  data = sys.stdin.read()
  tree = ET.ElementTree(ET.fromstring(data))

  json_data = []
  for testcase in tree.getroot().iter("testcase"):
    json_data += [parse_testcase(testcase)]

  print(json.dumps(json_data))
