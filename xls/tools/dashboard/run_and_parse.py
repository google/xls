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
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Union, Optional
from tempfile import NamedTemporaryFile, TemporaryDirectory

from xls.tools.dashboard.validate_dashboard_json import validate_dashboard_json

BazelLabel = str
DashboardJSONs = Sequence[Dict]
SystemPath = Union[str, Path]


@dataclass
class FileParserData:
  parser: str
  file: str


@dataclass
class OutputParserData:
  parser: str


@dataclass
class TestData:
  name: str
  output_parsers: List[OutputParserData] = field(default_factory=list)
  file_parsers: List[FileParserData] = field(default_factory=list)


def run_test_and_capture_output(
  test: BazelLabel, log_file: SystemPath, cwd: SystemPath
) -> None:
  cmd = f"bazel build {test} && bazel test {test} --test_output=all > {log_file}"
  try:
    subprocess.check_call(cmd, shell=True, cwd=cwd)
  except subprocess.CalledProcessError:
    raise Exception(
      f"Error while executing Bazel test with the follwing command:\n{cmd}"
    )


def parse_file(parser: SystemPath, file_to_parse: SystemPath, cwd: Optional[SystemPath]) -> str:
  cmd = f"cat {file_to_parse} | {parser}"
  try:
    return subprocess.check_output(cmd, shell=True, cwd=cwd).decode("utf-8")

  except subprocess.CalledProcessError:
    raise Exception(f"Error while parsing Bazel log using {parser} script")


def get_test_output_path(test_name: str) -> Path:
  path = test_name.replace("@//", "").replace(":", "/")
  return Path("bazel-testlogs", path, "test.outputs", "outputs.zip")


def unzip_test_output(test_outputs: SystemPath, target_directory: SystemPath) -> None:
  shutil.unpack_archive(test_outputs, target_directory, "zip")


def run_and_parse(
  tests: Sequence[TestData], root_dir: SystemPath, working_dir: SystemPath
) -> DashboardJSONs:
  jsons = []
  for test in tests:
    with NamedTemporaryFile() as log_file, TemporaryDirectory() as tmp_dir:
      run_test_and_capture_output(test.name, log_file.name, root_dir)

      # Parse log

      for output_parser_data in test.output_parsers:
        json_fragment = parse_file(
          output_parser_data.parser, log_file.name, working_dir
        )
        json_obj_fragment = json.loads(json_fragment)
        validate_dashboard_json(json_obj_fragment)
        jsons += json_obj_fragment

      # Parse files

      if test.file_parsers:
        output_path = root_dir / get_test_output_path(test.name)
        unzip_test_output(output_path, tmp_dir)

      for file_parser_data in test.file_parsers:
        filepath = Path(tmp_dir, file_parser_data.file)
        json_fragment = parse_file(
          file_parser_data.parser, filepath, cwd=root_dir
        )
        json_obj_fragment = json.loads(json_fragment)
        validate_dashboard_json(json_obj_fragment)
        jsons += json_obj_fragment

  return jsons

def parse_files(data: Sequence[FileParserData]) -> DashboardJSONs:
  jsons = []
  for item in data:
    json_fragment = parse_file(item.parser, Path(item.file), cwd=None)
    json_obj_fragment = json.loads(json_fragment)
    validate_dashboard_json(json_obj_fragment)
    jsons += json_obj_fragment

  return jsons
