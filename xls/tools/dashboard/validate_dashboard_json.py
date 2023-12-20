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
from typing import Dict, Union

from jsonschema import validate
from jsonschema.exceptions import ValidationError

schema = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Dashboard information",
  "description": "Structure with information for the dashboard generation script",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Name of the test",
      },
      "desc": {
        "type": "string",
        "description": "Description of the test (optional)",
      },
      "group": {
        "type": "string",
        "description": "Group to which the test belongs",
      },
      "type": {
        "type": "string",
        "description": "Type of the value stored in the value field",
      },
      "value": {
        "description": "Value extracted from the test",
      },
      "label": {
        "type": "string",
        "description": "Label of the test (optional)",
      },
      "log": {
        "type": "string",
        "description": "Path to the log file (optional)",
      },
    },
    "allOf": [
      {
        "if": {
          "properties": {
            "type": {
              "const": "table",
            },
          },
        },
        "then": {
          "properties": {
            "value": {
              "type": "array",
              "items": {
                "type": "array",
              },
            },
          },
        },
      },
    ],
    "required": ["name", "group", "type", "value"],
  },
}


class DashboardJSONValidationError(Exception):
  pass


def validate_dashboard_json(data: Union[str, Dict]) -> None:
  """Checks if data is in the right format for dashboard generation script.
  If it is not, the function raises DashboardJSONValidationError
  """
  if isinstance(data, str):
    data = json.loads(data)

  try:
    validate(instance=data, schema=schema)
  except ValidationError:
    raise DashboardJSONValidationError(
      "Provided JSON does not match Dashboard JSON format"
    )

  for item in data:
    if item["type"] == "table":
      first_column_length = len(item["value"][0])
      if not all(len(l) == first_column_length for l in item["value"]):
        raise DashboardJSONValidationError(
          "Table value has different column lengths"
        )
