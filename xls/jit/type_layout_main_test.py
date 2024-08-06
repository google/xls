#
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
"""Test for xls.jit.type_layout_main."""

import struct
import subprocess

from absl.testing import absltest
from xls.common import runfiles
from xls.jit import type_layout_pb2

_TYPE_LAYOUT = "xls/jit/type_layout_main"

# A type-layout of a 2-length array of a tuple of a 13-bit and a 21 bit integer.
# The layout places the elements in order padded to u16 and u32 as appropriate.
_PAIR_OF_TUPLES_LAYOUT = type_layout_pb2.TypeLayoutProto(
    type="(bits[13],bits[21])[2]",
    size=12,
    elements=[
        # arr[0].0
        type_layout_pb2.ElementLayoutProto(
            offset=0, data_size=2, padded_size=2
        ),
        # arr[0].1
        type_layout_pb2.ElementLayoutProto(
            offset=2, data_size=3, padded_size=4
        ),
        # arr[1].0
        type_layout_pb2.ElementLayoutProto(
            offset=6, data_size=2, padded_size=2
        ),
        # arr[1].1
        type_layout_pb2.ElementLayoutProto(
            offset=8, data_size=3, padded_size=4
        ),
    ],
)


class TypeLayoutMainTest(absltest.TestCase):

  def test_mask(self):
    layout = self.create_tempfile(
        content=_PAIR_OF_TUPLES_LAYOUT.SerializeToString()
    )
    out = subprocess.run(
        [
            runfiles.get_path(_TYPE_LAYOUT),
            "-layout_proto",
            layout.full_path,
            "-mask",
        ],
        check=True,
        stdout=subprocess.PIPE,
        encoding=None,
    ).stdout
    # Platform endianness, no alignment: u16, u32, u16, u32
    # All bits which are within the 13 & 21 bit elements of the type should be
    # set.  Expect 13-bits to be set in the first and third element and 21 bits
    # set in the other two.
    self.assertEqual(
        out,
        struct.pack(
            "=HIHI",
            0b0001_1111_1111_1111,  # max 13-bit value
            0b0000_0000_0001_1111_1111_1111_1111_1111,  # max 21 bit value
            0b0001_1111_1111_1111,  # max 13-bit value
            0b0000_0000_0001_1111_1111_1111_1111_1111,  # max 21 bit value
        ),
    )

  def test_decode(self):
    layout = self.create_tempfile(
        content=_PAIR_OF_TUPLES_LAYOUT.SerializeToString()
    )
    with subprocess.Popen(
        [
            runfiles.get_path(_TYPE_LAYOUT),
            "-layout_proto",
            layout.full_path,
            "-decode",
        ],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
    ) as proc:
      # NB All padding bits set
      unpadded_00 = 0x1E12
      unpadded_01 = 0x001F_FC45
      unpadded_10 = 0x1112
      unpadded_11 = 0x001F_E345
      pad_bits_0 = 0b1110_0000_0000_0000
      pad_bits_1 = 0b1111_1111_1110_0000_0000_0000_0000_0000
      result = proc.communicate(
          struct.pack(
              "=HIHI",
              unpadded_00 | pad_bits_0,
              unpadded_01 | pad_bits_1,
              unpadded_10 | pad_bits_0,
              unpadded_11 | pad_bits_1,
          )
      )[0]
    # Ensure that padding bits are correctly ignored.
    self.assertEqual(
        result.decode("UTF-8"),
        f"[(bits[13]:{unpadded_00}, bits[21]:{unpadded_01}),"
        f" (bits[13]:{unpadded_10}, bits[21]:{unpadded_11})]",
    )

  def test_encode(self):
    layout = self.create_tempfile(
        content=_PAIR_OF_TUPLES_LAYOUT.SerializeToString()
    )
    out = subprocess.run(
        [
            runfiles.get_path(_TYPE_LAYOUT),
            "-layout_proto",
            layout.full_path,
            "-encode",
            (
                "[(bits[13]:0x12, bits[21]:0x345), (bits[13]:0x112,"
                " bits[21]:0x1345)]"
            ),
        ],
        check=True,
        stdout=subprocess.PIPE,
        encoding=None,
    ).stdout
    # Platform endianness, no alignment: u16, u32, u16, u32
    self.assertEqual(out, struct.pack("=HIHI", 0x12, 0x345, 0x112, 0x1345))


if __name__ == "__main__":
  absltest.main()
