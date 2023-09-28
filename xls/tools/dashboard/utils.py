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
from typing import Dict

from xls.tools.dashboard.validate_dashboard_json import validate_dashboard_json

DUMP_PREFIX = "\n==== DASHBOARD JSON BEGIN ====\n"
DUMP_SUFFIX = "\n==== DASHBOARD JSON END ====\n"


def get_dashboard_json_dump_str(data: Dict) -> str:
  """Get dashboard JSON in a standardized format used when dumping data to log"""
  validate_dashboard_json(data)
  return DUMP_PREFIX + json.dumps(data) + DUMP_SUFFIX
