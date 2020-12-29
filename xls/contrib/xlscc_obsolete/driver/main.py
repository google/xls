# Lint as: python3
#
# Copyright 2020 The XLS Authors
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
"""Drives translator to convert C++ to XLS IR."""

import argparse

import sys
import pycparser

from google.protobuf import text_format
from xls.contrib.xlscc_obsolete.parse import ext_c_parser
from xls.contrib.xlscc_obsolete.translate import hls_types_pb2
import xls.contrib.xlscc_obsolete.translate.translator as xlscc_translator


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("cpp", help="C++ file to process.")
  parser.add_argument("output_filename", help="XLS IR file output path.")
  parser.add_argument(
      "-T", "--types_proto", help="Parse types from protobuffer.", default=None)
  parser.add_argument(
      "-D",
      "--define",
      help="Define a preprocessor symbol.",
      action="append",
      default=[])
  parser.add_argument(
      "--channel_in",
      help="Define an input channel.",
      action="append",
      default=[])
  parser.add_argument(
      "--channel_out",
      help="Define an output channel.",
      action="append",
      default=[])
  parser.add_argument(
      "-N", "--package_name", help="Package name.", default="test_package")
  parser.add_argument(
      "-W", "--wrapper", help="Generate a wrapper file at path.", default=None)
  args = parser.parse_args()

  hls_types_by_name = {}

  if args.types_proto:
    with open(args.types_proto, mode="rb") as proto_file:
      file_content = proto_file.read()
      hls_types = hls_types_pb2.HLSTypes()
      if args.types_proto.endswith("pb"):
        hls_types.ParseFromString(file_content)
      elif args.types_proto.endswith("pbtext"):
        text_format.Parse(file_content, hls_types)
      else:
        print("Unknown extension for protobuf: {args.types_proto}",
              file=sys.stderr)
        sys.exit(1)

      for named_type in hls_types.hls_types:
        hls_types_by_name[named_type.name] = named_type.hls_type

  translator = xlscc_translator.Translator(args.package_name, hls_types_by_name,
                                           args.channel_in, args.channel_out)

  my_parser = ext_c_parser.XLSccParser()
  my_parser.is_intrinsic_type = translator.is_intrinsic_type
  defines = ["-DHLS_STATIC=static", "-D__SYNTHESIS__=1"]
  defines += ["-D" + d for d in args.define]
  ast = pycparser.parse_file(
      args.cpp,
      use_cpp=True,
      cpp_args=defines,
      parser=my_parser)

  translator.parse(ast)

  with open(args.output_filename, "w") as text_file:
    text_file.write(translator.gen_ir().dump_ir())

if __name__ == "__main__":
  main()
