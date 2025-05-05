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

from pathlib import Path
from enum import Enum

from xls.common import runfiles
import subprocess
import zstandard

class BlockType(Enum):
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
      raise ValueError(str(e))

def CallDecodecorpus(args):
  decodecorpus = Path(runfiles.get_path("decodecorpus", repository = "zstd"))
  cmd = args
  cmd.insert(0, str(decodecorpus))
  cmd_concat = " ".join(cmd)
  subprocess.run(cmd_concat, shell=True, check=True)

def DecompressFrame(data):
  dctx = zstandard.ZstdDecompressor()
  return dctx.decompress(data)

def GenerateFrame(seed, btype, output_path):
  args = []
  args.append("-s" + str(seed))
  if (btype != BlockType.RANDOM):
    args.append("--block-type=" + str(btype.value))
  args.append("--content-size")
  # Test payloads up to 16KB
  args.append("--max-content-size-log=14")
  args.append("-p" + output_path)
  args.append("-vvvvvvv")

  CallDecodecorpus(args)
