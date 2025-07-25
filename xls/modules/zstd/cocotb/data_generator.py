# Copyright 2024 The XLS Authors
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

"""Module for generating and decompressing test frames."""

import pathlib
import enum

from xls.common import runfiles
import subprocess
import tempfile

class BlockType(enum.Enum):
  """Enum encoding of ZSTD block types."""

  RAW = 0
  RLE = 1
  COMPRESSED = 2
  RANDOM = 3

  def __str__(self):
    return self.name

  @staticmethod
  def from_string(s):
    try:
      return BlockType[s]
    except KeyError as e:
      raise ValueError(str(e)) from e

class LiteralType(enum.Enum):
  """Enum encoding of ZSTD literal types."""

  RAW = 0
  RLE = 1
  COMPRESSED = 2
  RANDOM = 3

  def __str__(self):
    return self.name

  @staticmethod
  def from_string(s):
    try:
      return BlockType[s]
    except KeyError as e:
      raise ValueError(str(e)) from e

def CallDecodecorpus(args):
  decodecorpus = pathlib.Path(
    runfiles.get_path("decodecorpus", repository = "zstd")
  )
  cmd = args
  cmd.insert(0, str(decodecorpus))
  cmd_concat = " ".join(cmd)
  subprocess.run(cmd_concat, shell=True, check=True)

def DecompressFrame(data):
  zstd_cli = pathlib.Path(runfiles.get_path("zstd_cli", repository = "zstd"))
  with tempfile.NamedTemporaryFile(mode='wb') as input_data, \
       tempfile.NamedTemporaryFile(mode='wb') as output_data:
    input_data.write(data)
    input_data.flush()
    cmd = f"{str(zstd_cli)} -f -d {input_data.name} -o {output_data.name}"
    output_data.close()
    subprocess.run(cmd, shell=True, check=True)
    with open(output_data.name, "rb") as output_data:
      return output_data.read()

def GenerateFrame(seed, btype, output_path, ltype=LiteralType.RANDOM):
  args = []
  args.append("-s" + str(seed))
  if (btype != BlockType.RANDOM):
    args.append("--block-type=" + str(btype.value))

  if (ltype != LiteralType.RANDOM):
    args.append("--literal-type=" + str(ltype.value))

  args.append("--content-size")
  # Test payloads up to 16KB
  args.append("--max-content-size-log=14")
  args.append("-p" + output_path)
  args.append("-vvvvvvv")

  CallDecodecorpus(args)
