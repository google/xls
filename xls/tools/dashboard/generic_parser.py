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


def parse_data(data):
  regex = r"={3,} DASHBOARD JSON BEGIN ={3,}\s*(.*?)={3,} DASHBOARD JSON END ={3,}"
  (dashboard_json,) = re.findall(regex, data, re.DOTALL | re.MULTILINE)
  return json.loads(dashboard_json)


if __name__ == "__main__":
  data = sys.stdin.read()
  json_data = parse_data(data)
  print(json.dumps(json_data))
