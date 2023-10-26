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
import re
import sys


def parse_testcase(data):
  name_regex = r"Test output for (.*):$"
  (name,) = re.findall(name_regex, data, re.MULTILINE)

  test_regex = r"\[ RUN UNITTEST  ]\s*(\w*)\s*.*?\[\s*(OK|FAILED)\s*\]"
  test_matches = re.findall(test_regex, data, re.DOTALL | re.MULTILINE)

  return [
    {
      "name": f"{name}",
      "group": "dslx",
      "type": "table",
      "value": [
        ["Test Name", "Status"],
      ]
      + [[test, status] for test, status in test_matches],
    }
  ]


if __name__ == "__main__":
  data = sys.stdin.read()
  json_data = parse_testcase(data)
  print(json.dumps(json_data))
