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


import argparse
import os
from pathlib import Path
from typing import Tuple, Sequence

from xls.tools.dashboard.json_to_markdown import json_to_markdown
from xls.tools.dashboard.mkdocs_creator import DefaultDashboardCreator
from xls.tools.dashboard.run_and_parse import (FileParserData,
                                               OutputParserData, TestData,
                                               run_and_parse)


def tuple_of_two(string: str) -> Tuple[str, str]:
  return tuple(string.rsplit(",", 1))


def tuple_of_three(string: str) -> Tuple[str, str, str]:
  return tuple(string.rsplit(",", 2))


def parse_args_to_dict(args) -> Sequence[TestData]:
  args_dict = dict()

  for test, parser in args.parser:
    if test not in args_dict:
      args_dict[test] = TestData(test)
    args_dict[test].output_parsers += [OutputParserData(parser)]

  for test, parser, file in args.file:
    if test not in args_dict:
      args_dict[test] = TestData(test)
    args_dict[test].file_parsers += [FileParserData(parser, file)]

  return args_dict.values()


def check_args(args) -> None:
  """Checks correctness of the arguments passed to the script and
  rises appropriate errors when they are not valid.

  In order not to run Bazel, we are not checking labels of the Bazel tests
  """

  root_dir = Path(args.root_directory)
  if not root_dir.is_dir():
    raise FileNotFoundError(f"Root directory {str(root_dir)} does not exist")

  working_dir = Path(args.working_directory)
  if not working_dir.is_dir():
    raise FileNotFoundError(f"Working directory {str(working_dir)} does not exist")

  output_dir = Path(args.output_directory)
  if not output_dir.is_absolute():
    output_dir = working_dir / output_dir
  if output_dir.exists():
    raise FileExistsError(f"Output directory {str(output_dir)} exists")

  for _, parser in args.parser:
    parser_path = Path(parser)
    if not parser_path.exists():
      raise FileNotFoundError(f"Output parser {str(parser_path)} does not exist")

  for _, parser, _ in args.file:
    parser_path = Path(parser)
    if not parser_path.exists():
      raise FileNotFoundError(f"File parser {str(parser_path)} does not exist")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run tests and parse results")
  parser.add_argument("-r", "--root-directory", help="Root directory", required=True)
  parser.add_argument("-w", "--working-directory", help="Working directory", required=True)
  parser.add_argument("-o", "--output-directory", help="Directory for generated dashboard", required=True)
  parser.add_argument("-p", "--parser", type=tuple_of_two, action="append", help="Output parser")
  parser.add_argument("-f", "--file", type=tuple_of_three, action="append", help="Test and parser pair")
  parser.add_argument("-t", "--title", default="Dashboard", help="Test and parser pair")
  args = parser.parse_args()

  check_args(args)

  title = args.title
  root_dir = Path(args.root_directory)
  runfiles_dir = Path(os.getcwd())
  working_dir = Path(args.working_directory)
  output_dir = Path(args.output_directory)
  if not output_dir.is_absolute():
    output_dir = Path(working_dir, args.output_directory)

  tests = parse_args_to_dict(args)

  json_objs = run_and_parse(tests, root_dir, runfiles_dir)
  page = json_to_markdown(json_objs, title=title)

  mk = DefaultDashboardCreator(title, page)
  mk.build(output_dir)
