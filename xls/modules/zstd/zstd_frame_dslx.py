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

"""ZSTD test frame generator for DSLX tests.

This module interacts with the data_generator module and underlying
Decodecorpus in order to generate inputs and expected outputs for the ZSTD
Decoder tests in DSLX.

It generates the ZSTD frame and then decodes it with the reference ZSTD
library. Both the encoded frame and decoded data are written to a DSLX file by
converting raw bytes of the frame and decoded data into DSLX structures.

Resulting file can be included in the ZSTD Decoder DSLX test and used as the
inputs and expected output for the testbench.
"""

import argparse
import math
import random
import tempfile
import pathlib

from xls.modules.zstd.cocotb import data_generator


def GenerateTestData(seed, btype):
  with tempfile.NamedTemporaryFile() as tmp:
    data_generator.GenerateFrame(seed, btype, tmp.name)
    tmp.seek(0)
    return tmp.read()


def Bytes2DSLX(frames, bytes_per_word, array_name):
  """Converts a list of byte frames to a DSLX constant array declaration.

  Args:
    frames (List[bytes]): List of byte sequences representing frames.
    bytes_per_word (int): Number of bytes per word in the output format.
    array_name (str): Name of the resulting DSLX constant array.

  Returns:
    str: A string containing the DSLX constant array declaration.
  """
  frames_hex = []
  maxlen = max(len(frame) for frame in frames)
  maxlen_size = math.ceil(maxlen / bytes_per_word)
  bits_per_word = bytes_per_word * 8
  for i, frame in enumerate(frames):
    frame_hex = []
    for i in range(0, len(frame), bytes_per_word):
      # reverse byte order to make them little endian
      word = bytes(reversed(frame[i : i + bytes_per_word])).hex()
      frame_hex.append(f"uN[{bits_per_word}]:0x{word}")

    array_length = len(frame_hex)
    if len(frame) < maxlen:
      frame_hex += [f"uN[{bits_per_word}]:0x0", "..."]

    frame_array = (
      f"DataArray<{bits_per_word}, {maxlen_size}>{{\n"
      f"  length: u32:{len(frame)},\n"
      f"  array_length: u32:{array_length},\n"
      f"  data: uN[{bits_per_word}][{maxlen_size}]:[{', '.join(frame_hex)}]\n"
      f"}}"
    )
    frames_hex.append(frame_array)

  frames_str = ",\n".join(frames_hex)
  frames_array = (
    f"pub const {array_name}:DataArray<\n"
    f"  u32:{bits_per_word},\n"
    f"  u32:{maxlen_size}\n"
    f">[{len(frames_hex)}] = [{frames_str}];\n"
  )
  return frames_array


def GenerateDataStruct():
  return (
    "pub struct DataArray<BITS_PER_WORD: u32, LENGTH: u32>{\n"
    "  data: uN[BITS_PER_WORD][LENGTH],\n"
    "  length: u32,\n"
    "  array_length: u32\n"
    "}\n"
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-n", help="Number of testcases to generate", type=int, default=1
  )
  parser.add_argument(
    "--seed", help="Seed for the testcases generator", type=int, default=0
  )
  parser.add_argument(
    "--btype",
    help=(
      "Block types allowed in the generated testcases. If multiple block types "
      "are supplied, generated testcases will cycle through them"
    ),
    type=data_generator.BlockType.from_string,
    choices=list(data_generator.BlockType),
    default=data_generator.BlockType.RANDOM,
    nargs="+",
  )
  parser.add_argument(
    "-o",
    "--output",
    help="Filename of the DSLX output file",
    type=pathlib.Path,
    default=pathlib.Path("frames_test_data.x"),
  )
  parser.add_argument(
    "--bytes-per-word",
    help="Width of a word in memory, in bytes",
    type=int,
    default=8,
  )
  args = parser.parse_args()

  random.seed(args.seed)
  byte_frames = [
    GenerateTestData(random.randrange(2**32), args.btype[i % len(args.btype)])
    for i in range(args.n)
  ]
  with open(args.output, "w") as dslx_output:
    dslx_output.write(GenerateDataStruct())

    dslx_frames = Bytes2DSLX(byte_frames, args.bytes_per_word, "FRAMES")
    dslx_output.write(dslx_frames)

    byte_frames_decompressed = list(
      map(data_generator.DecompressFrame, byte_frames)
    )
    dslx_frames_decompressed = Bytes2DSLX(
      byte_frames_decompressed, args.bytes_per_word, "DECOMPRESSED_FRAMES"
    )
    dslx_output.write(dslx_frames_decompressed)


if __name__ == "__main__":
  main()
