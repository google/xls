# Copyright 2025 The XLS Authors
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

"""ZSTD test debugger for cocotb tests.

This module iterates over given ZSTD-encoded file and finds
the executed sequence which is decoded into the nth byte
specified with `--byte` flag. It skips decoding literals
and executing sequences, because it only cares about how
much data is decoded in each block and from which sequence
comes given decoded byte.

For ZSTD reference, see https://www.rfc-editor.org/rfc/rfc8878.html
"""

import argparse
import pathlib
import sys
from enum import Enum

class BlockType(Enum):
  RAW_BLOCK = 0
  RLE_BLOCK = 1
  COMPRESSED_BLOCK = 2

class BlockHeader:
  def __init__(self, last_block, block_type, block_content_size):
    self.last_block = last_block
    self.block_type = block_type
    self.block_content_size = block_content_size

class LiteralsBlockType(Enum):
  RAW_LITERALS_BLOCK = 0
  RLE_LITERALS_BLOCK = 1
  COMPRESSED_LITERALS_BLOCK = 2
  TREELESS_LITERALS_BLOCK = 3

class LiteralsSectionHeader:
  def __init__(self, block_type, regenerated_size, compressed_size = 0, has_four_streams = False):
    self.block_type = block_type
    self.regenerated_size = regenerated_size
    self.compressed_size = compressed_size
    self.has_four_streams = has_four_streams

def read_magic_number(encoded_file):
  magic_number = int.from_bytes(encoded_file.read(4))
  print(f"Magic number: {hex(magic_number)}")

def read_frame_header(encoded_file):
  frame_header_descriptor = int.from_bytes(encoded_file.read(1))
  print(f"Frame header descriptor: {hex(frame_header_descriptor)}")

  # 6th bit of a frame header indicates if a window descriptor is present
  SINGLE_SEGMENT_FLAG_MASK = 0b100000
  if frame_header_descriptor & SINGLE_SEGMENT_FLAG_MASK == 0:
    # Calculate a required window_size for the encoded file
    # and compare it with the ZSTD decoder history buffer size.
    # Based on window_size calculation from: RFC 8878
    # https://datatracker.ietf.org/doc/html/rfc8878#name-window-descriptor
    window_descriptor = int.from_bytes(encoded_file.read(1))

    MANTISSA_BITS = 0b111
    mantissa = window_descriptor & MANTISSA_BITS

    EXPONENT_BITS = 0b11111000
    exponent = (window_descriptor & EXPONENT_BITS) >> 3

    window_log = 10 + exponent
    window_base = 1 << window_log
    window_add = (window_base / 8) * mantissa
    window_size = int(window_base + window_add)
    print(f"Required window size: {window_size} bytes")

  DICTIONARY_ID_MASK = 0b11
  if frame_header_descriptor & DICTIONARY_ID_MASK != 0:
    dictionary_id_size = 2 ** ((frame_header_descriptor & DICTIONARY_ID_MASK) - 1)
    dictionary_id = encoded_file.read(dictionary_id_size)
    print(f"Dictionary ID: {dictionary_id.hex()}")

  frame_content_size_flag = frame_header_descriptor >> 6
  if frame_content_size_flag == 0:
    fcs_field_size = (frame_header_descriptor & SINGLE_SEGMENT_FLAG_MASK) != 0
  else:
    fcs_field_size = 2 ** frame_content_size_flag

  if fcs_field_size != 0:
    frame_content_size = int.from_bytes(encoded_file.read(fcs_field_size), byteorder="little")
    print(f"Frame content used bytes: {fcs_field_size}")
    print(f"Frame content size: {frame_content_size}")

def read_block_header(encoded_file, block_starting_byte_index):
  encoded_file.seek(block_starting_byte_index)
  block_header_bytes = encoded_file.read(3)
  block_header = int.from_bytes(block_header_bytes, byteorder="little")

  LAST_BLOCK_FLAG_MASK = 0b1
  last_block_flag = block_header & LAST_BLOCK_FLAG_MASK

  BLOCK_TYPE_FLAG_MASK = 0b110
  block_type_flag = (block_header & BLOCK_TYPE_FLAG_MASK) >> 1

  block_size = block_header >> 3
  block_content_size = block_size
  if block_type_flag == BlockType.RLE_BLOCK:
    block_content_size = 1

  return BlockHeader(last_block_flag, block_type_flag, block_content_size)

def find_block_containing_byte_index(encoded_file, byte_index):
  current_byte_index = encoded_file.tell()
  while current_byte_index < byte_index:
    last_block_starting_byte_index = current_byte_index
    block_header = read_block_header(encoded_file, current_byte_index)
    print(f"Block at {last_block_starting_byte_index}:")
    print(f"  Last block flag: {block_header.last_block}")
    print(f"  Block content type: {block_header.block_type}")
    print(f"  Block content size: {block_header.block_content_size}")
    current_byte_index = encoded_file.tell() + block_header.block_content_size
    encoded_file.seek(current_byte_index)

  print(f"Block containing index is located at position {last_block_starting_byte_index}")

  return last_block_starting_byte_index

def read_literals_section_header(encoded_file):
  literal_section_header = encoded_file.read(1)

  LITERALS_BLOCK_TYPE_MASK = 0b11
  literals_block_type = LiteralsBlockType(literal_section_header[0] & LITERALS_BLOCK_TYPE_MASK)

  size_format = (literal_section_header[0] >> 2) & 0b11
  if (literals_block_type == LiteralsBlockType.RAW_LITERALS_BLOCK or
      literals_block_type == LiteralsBlockType.RLE_LITERALS_BLOCK):

    if size_format == 0b00 or size_format == 0b10:
      regenerated_size = literal_section_header[0] >> 3

      return LiteralsSectionHeader(literals_block_type, regenerated_size)

    if size_format == 0b01:
      literal_section_header_second_byte = encoded_file.read(1)

      regenerated_size = (literal_section_header[0] >> 4) + (literal_section_header_second_byte[0] << 4)

      return LiteralsSectionHeader(literals_block_type, regenerated_size)

    if size_format == 0b11:
      literal_section_header_remaining_bytes = encoded_file.read(2)

      regenerated_size = ((literal_section_header[0] >> 4) +
        (literal_section_header_remaining_bytes[0] << 4) +
        (literal_section_header_remaining_bytes[1] << 12))

      return LiteralsSectionHeader(literals_block_type, regenerated_size)

  if (literals_block_type == LiteralsBlockType.COMPRESSED_LITERALS_BLOCK or
      literals_block_type == LiteralsBlockType.TREELESS_LITERALS_BLOCK):

    if size_format == 0b00 or size_format == 0b01:
      literal_section_header_remaining_bytes = encoded_file.read(2)

      regenerated_size = (literal_section_header[0] >> 4) + ((literal_section_header_remaining_bytes[0] & 0b111111) << 4)
      compressed_size = (literal_section_header_remaining_bytes[0] >> 6) + (literal_section_header_remaining_bytes[1] << 2)
      has_four_streams = size_format == 0b01

      return LiteralsSectionHeader(literals_block_type, regenerated_size, compressed_size, has_four_streams)

    if size_format == 0b10:
      literal_section_header_remaining_bytes = encoded_file.read(3)

      regenerated_size = ((literal_section_header[0] >> 4) +
                          (literal_section_header_remaining_bytes[0] << 4) +
                          ((literal_section_header_remaining_bytes[1] & 0b11) << 12))
      compressed_size = ((literal_section_header_remaining_bytes[1] >> 2) +
                         (literal_section_header_remaining_bytes[2] << 6))
      has_four_streams = True

      return LiteralsSectionHeader(literals_block_type, regenerated_size, compressed_size, has_four_streams)

    if size_format == 0b11:
      literal_section_header_remaining_bytes = encoded_file.read(4)

      regenerated_size = ((literal_section_header[0] >> 4) +
                          (literal_section_header_remaining_bytes[0] << 4) +
                          ((literal_section_header_remaining_bytes[1] & 0b111111) << 12))
      compressed_size = ((literal_section_header_remaining_bytes[1] >> 6) +
                         (literal_section_header_remaining_bytes[2] << 2) +
                         (literal_section_header_remaining_bytes[3] << 10))
      has_four_streams = True

      return LiteralsSectionHeader(literals_block_type, regenerated_size, compressed_size, has_four_streams)

class SymbolCompressionMode(Enum):
  PREDEFINED_MODE = 0
  RLE_MODE = 1
  FSE_COMPRESSED_MODE = 2
  REPEAT_MODE = 3

# The predefined FSE distribution tables for PREDEFINED_MODE
# https://datatracker.ietf.org/doc/html/rfc8878#name-default-distributions
LITERALS_LENGTH_DEFAULT_DIST = [
  4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,  1,  2,  2,
  2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1]
MATCH_LENGTHS_DEFAULT_DIST = [
  1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1]
OFFSET_CODES_DEFAULT_DIST = [
  1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]

class SequencesSectionHeader:
  def __init__(self, num_of_sequences = 0, symbol_compression_modes = 0):
    self.num_of_sequences = num_of_sequences

    self.literal_lengths_mode = SymbolCompressionMode(symbol_compression_modes >> 6)
    self.offsets_mode = SymbolCompressionMode((symbol_compression_modes >> 4) & 0b11)
    self.match_lengths_mode = SymbolCompressionMode((symbol_compression_modes >> 2) & 0b11)

def read_sequences_section_header(encoded_file):
  byte0 = encoded_file.read(1)[0]
  if byte0 == 0:
    return SequencesSectionHeader()

  num_of_sequences = 0
  if byte0 < 128:
    num_of_sequences = byte0
  elif byte0 < 255:
    byte1 = encoded_file.read(1)[0]
    num_of_sequences = ((byte0 - 128) << 8) + byte1
  elif byte0 == 255:
    remaining_bytes = encoded_file.read(2)
    byte1 = remaining_bytes[0]
    byte2 = remaining_bytes[1]
    num_of_sequences = byte1 + (byte2 << 8) + 0x7F00

  symbol_compression_modes = encoded_file.read(1)[0]

  return SequencesSectionHeader(num_of_sequences, symbol_compression_modes)

class BitReader:
  def __init__(self, file):
    self.file = file
    self.byte_pos = file.tell()
    self.current_byte = None
    self.bitpos = 0

  def read_bits(self, n):
    if self.current_byte == None:
      self.current_byte = self.file.read(1)[0]

    result = 0
    bits_read = 0
    while bits_read < n:
      remaining_bits_in_byte = 8 - self.bitpos
      bits_to_take = min(n - bits_read, remaining_bits_in_byte)
      mask = (1 << bits_to_take) - 1
      bits = (self.current_byte >> self.bitpos) & mask
      result |= bits << bits_read
      bits_read += bits_to_take
      self.bitpos += bits_to_take
      if self.bitpos == 8:
        self.current_byte = self.file.read(1)[0]
        self.byte_pos += 1
        self.bitpos = 0

    return result

  def rewind_bits(self, n):
    self.bitpos -= n
    while self.bitpos < 0:
      self.byte_pos -= 1
      self.bitpos += 8

    self.file.seek(self.byte_pos)
    self.current_byte = self.file.read(1)[0]

  def read_previous_bits(self, n):
    self.rewind_bits(n)
    bits = self.read_bits(n)
    self.rewind_bits(n)
    return bits

# Explanation of FSE tables can be found here
# https://datatracker.ietf.org/doc/html/rfc8878#name-fse
class FSETable():
  def __init__(self, accuracy_log, num_symbols, symbol_frequencies):
    self.accuracy_log = accuracy_log

    table_size = 1 << accuracy_log
    self.symbols = [0] * table_size
    self.num_of_bits = [0] * table_size
    self.baseline = [0] * table_size
    self.current_state = 0

    MAX_SYMBOLS = 255
    state = [0] * MAX_SYMBOLS

    # Allocate positions for 'less than 1' probability symbols starting at the end
    # https://datatracker.ietf.org/doc/html/rfc8878#section-4.1.1-12
    last_element_index = table_size
    for symbol_index in range(num_symbols):
      if symbol_frequencies[symbol_index] == -1:
        last_element_index -= 1
        self.symbols[last_element_index] = symbol_index
        state[symbol_index] = 1

    symbol_table_position = 0
    symbol_table_position_step = (table_size >> 1) + (table_size >> 3) + 3
    table_size_mask = table_size - 1
    for symbol_index in range(num_symbols):
      # Skip if already occupied by less than 1
      # https://datatracker.ietf.org/doc/html/rfc8878#section-4.1.1-15
      if (symbol_frequencies[symbol_index] <= 0):
        continue

      state[symbol_index] = symbol_frequencies[symbol_index]

      # A symbol occupies as many positions as its probability
      # https://datatracker.ietf.org/doc/html/rfc8878#section-4.1.1-18
      for index in range(symbol_frequencies[symbol_index]):
        self.symbols[symbol_table_position] = symbol_index

        # Calculating symbol table position
        # https://datatracker.ietf.org/doc/html/rfc8878#section-4.1.1-15
        symbol_table_position = (symbol_table_position + symbol_table_position_step) & table_size_mask
        while symbol_table_position >= last_element_index:
          symbol_table_position = (symbol_table_position + symbol_table_position_step) & table_size_mask

    # Assigning symbols baselines
    # https://datatracker.ietf.org/doc/html/rfc8878#section-4.1.1-19
    for index in range(table_size):
      symbol = self.symbols[index]
      state_value = state[symbol]
      state[symbol] += 1
      self.num_of_bits[index] = accuracy_log - highest_set_bit(state_value)
      self.baseline[index] = (state_value << self.num_of_bits[index]) - table_size

  def update_table_state(self, bit_reader: BitReader):
    # Updating FSE table state
    # https://datatracker.ietf.org/doc/html/rfc8878#section-4.1.1-21
    bits = self.num_of_bits[self.current_state]
    value = bit_reader.read_previous_bits(bits)
    self.current_state = self.baseline[self.current_state] + value

  def get_symbol(self):
    return self.symbols[self.current_state]

class RLETable():
  def __init__(self, symbol):
    self.symbol = symbol
    self.accuracy_log = 0

  def update_table_state(self, bit_reader: BitReader):
    self.nop = 0

  def get_symbol(self):
    return self.symbol

def highest_set_bit(num):
  if num == 0:
    return -1

  return num.bit_length() - 1

def fill_mask(num_of_bits):
  return (1 << num_of_bits) - 1

def use_less_bits_if_needed(bits,
                            decoded_value,
                            bit_reader,
                            remaining_probabilities):
  # Find if the value can be stored with lower num of bits
  # https://datatracker.ietf.org/doc/html/rfc8878#section-4.1.1-4
  lower_bits = bits - 1
  lower_bits_mask = fill_mask(lower_bits)
  cutoff = fill_mask(bits) - remaining_probabilities - 1
  low = decoded_value & lower_bits_mask
  if low < cutoff:
      bit_reader.rewind_bits(1)
      return low
  elif decoded_value > lower_bits_mask:
      return decoded_value - cutoff

  return decoded_value

# Reading FSE table
def read_fse_table(encoded_file):
  bit_reader = BitReader(encoded_file)
  accuracy_log = bit_reader.read_bits(4) + 5

  remaining_probabilities = 1 << accuracy_log
  MAX_SYMBOLS = 255
  distribution_table = [0] * MAX_SYMBOLS
  current_symbol = 0
  while remaining_probabilities > 0:
    # Read value for symbol with highest possible num of bits
    bits = highest_set_bit(remaining_probabilities + 1) + 1
    decoded_value = bit_reader.read_bits(bits)
    decoded_value = use_less_bits_if_needed(bits, decoded_value, bit_reader, remaining_probabilities)

    symbol_probability = decoded_value - 1
    distribution_table[current_symbol] = symbol_probability
    remaining_probabilities -= abs(symbol_probability)

    current_symbol += 1

    # Handle 0 value probabilities
    # https://datatracker.ietf.org/doc/html/rfc8878#section-4.1.1-7
    if symbol_probability == 0:
      repeat = bit_reader.read_bits(2)

      while repeat > 0:
        for _ in range(repeat):
          distribution_table[current_symbol] = 0
          current_symbol += 1

        if repeat == 3:
          repeat = bit_reader.read_bits(2)
        else:
          repeat = 0

  # Reset encoded file reader by skipping over remaining bits
  if bit_reader.bitpos == 0:
    encoded_file.seek(encoded_file.tell() - 1)

  return FSETable(accuracy_log, current_symbol, distribution_table)

# Sequences baselines and extra bits
# https://datatracker.ietf.org/doc/html/rfc8878#name-sequence-codes-for-lengths-
LITERALS_LENGTH_BASELINES = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28,
  32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
LITERALS_LENGTH_NUM_OF_BITS = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
  1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
MATCH_LENGTH_BASELINES = [
  3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
  31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99,
  131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539]
MATCH_LENGTH_NUM_OF_BITS = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
  2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

class DecodedSequence():
  def __init__(self, literal_length, offset_length, match_length):
    self.literal_length = literal_length
    self.offset_length = offset_length
    self.match_length = match_length

  def __str__(self):
    return f"LL:{self.literal_length}, OF:{self.offset_length}, ML:{self.match_length}"

def decode_sequence(bit_reader: BitReader,
                    literal_lengths_fse_table,
                    offsets_fse_table,
                    match_lengths_fse_table,
                    last_sequence):
  of_code = offsets_fse_table.get_symbol()
  ll_code = literal_lengths_fse_table.get_symbol()
  ml_code = match_lengths_fse_table.get_symbol()

  offset = (1 << of_code) + bit_reader.read_previous_bits(of_code)
  match_length = MATCH_LENGTH_BASELINES[ml_code] + bit_reader.read_previous_bits(MATCH_LENGTH_NUM_OF_BITS[ml_code])
  literal_length = LITERALS_LENGTH_BASELINES[ll_code] + bit_reader.read_previous_bits(LITERALS_LENGTH_NUM_OF_BITS[ll_code])

  if not last_sequence:
    literal_lengths_fse_table.update_table_state(bit_reader)
    match_lengths_fse_table.update_table_state(bit_reader)
    offsets_fse_table.update_table_state(bit_reader)

  return DecodedSequence(literal_length, offset, match_length)

def decode_sequences(encoded_file,
                     num_sequences,
                     last_block_byte_index,
                     literal_lengths_fse_table,
                     offsets_fse_table,
                     match_lengths_fse_table):
  encoded_file.seek(last_block_byte_index)
  last_block_byte = encoded_file.read(1)[0]
  encoded_file.seek(last_block_byte_index)
  bit_reader = BitReader(encoded_file)
  padding = highest_set_bit(last_block_byte)
  if padding > 0:
    bit_reader.read_bits(padding)

  literal_lengths_fse_table.current_state = bit_reader.read_previous_bits(literal_lengths_fse_table.accuracy_log)
  offsets_fse_table.current_state = bit_reader.read_previous_bits(offsets_fse_table.accuracy_log)
  match_lengths_fse_table.current_state = bit_reader.read_previous_bits(match_lengths_fse_table.accuracy_log)

  decoded_sequences = []
  for sequence in range(num_sequences):
    decoded_sequences.append(decode_sequence(bit_reader, literal_lengths_fse_table, offsets_fse_table, match_lengths_fse_table, sequence == (num_sequences - 1)))

  return decoded_sequences

# Calculating repeat offsets
# https://datatracker.ietf.org/doc/html/rfc8878#name-repeat-offsets
repeat_offsets = [1, 4, 8]
def handle_repeated_offsets(decoded_sequence):
  global repeat_offsets
  offset = -1

  if decoded_sequence.offset_length > 3:
    # Simple case where we subtract 3 from the offset
    # https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.4-4.1
    offset = decoded_sequence.offset_length - 3

    repeat_offsets[2] = repeat_offsets[1]
    repeat_offsets[1] = repeat_offsets[0]
    repeat_offsets[0] = offset
  elif decoded_sequence.literal_length != 0:
    # Rotate repeated offsets
    # https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.5-8
    if decoded_sequence.offset_length == 1:
      offset == repeat_offsets[0]
    else:
      if decoded_sequence.offset_length == 3:
        repeat_offsets[2] = repeat_offsets[1]

      repeat_offsets[1] = repeat_offsets[0]
      repeat_offsets[0] = offset
  else:
    # Special case when LL is 0, then repeat offsets are shifted by one
    # https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.5-3
    offset_index = decoded_sequence.offset_length

    if offset_index < 3:
      offset = repeat_offsets[offset_index]
    else:
      # 'an offset_value of 3 means Repeated_Offset1 - 1_byte'
      offset = repeat_offsets[0] - 1

    if offset_index > 1:
      repeat_offsets[2] = repeat_offsets[1]

    repeat_offsets[1] = repeat_offsets[0]
    repeat_offsets[0] = offset

  decoded_sequence.offset_length = offset


literal_lengths_fse_table = None
offsets_fse_table = None
match_lengths_fse_table = None
block_cnt = 0

def debug_block(encoded_file, current_decoded_data_size, searched_for_byte_index, test_name):
  global block_cnt
  global literal_lengths_fse_table
  global offsets_fse_table
  global match_lengths_fse_table

  block_starting_byte_index = encoded_file.tell()
  block_header = read_block_header(encoded_file, block_starting_byte_index)

  print(f"Debugging {block_cnt} block at {block_starting_byte_index}")
  print("Block:")
  print(f"  Block content size: {block_header.block_content_size}")
  last_block_byte_index = encoded_file.tell() + block_header.block_content_size - 1
  print(f"  Block last byte: {last_block_byte_index}")

  current_byte_index = encoded_file.tell()
  literals_section_header = read_literals_section_header(encoded_file)
  literals_section_header_size = encoded_file.tell() - current_byte_index

  print(f"  Literals header size: {literals_section_header_size}")
  print(f"  Literals block type: {literals_section_header.block_type}")
  print(f"  Compressed size: {literals_section_header.compressed_size}")
  print(f"  Has four streams: {literals_section_header.has_four_streams}")

  literals_section_content_size = 0
  if literals_section_header.block_type == LiteralsBlockType.RAW_LITERALS_BLOCK:
    literals_section_content_size = literals_section_header.regenerated_size
    literals_block = encoded_file.read(literals_section_content_size)

  if literals_section_header.block_type == LiteralsBlockType.RLE_LITERALS_BLOCK:
    print(f"  Literals block contains a single byte which should be repeated {literals_section_header.regenerated_size} times")
    literals_section_content_size = 1
    literals_block = encoded_file.read(literals_section_content_size)

  if literals_section_header.block_type == LiteralsBlockType.COMPRESSED_LITERALS_BLOCK:
    literals_section_content_size = literals_section_header.compressed_size
    literals_block = encoded_file.read(literals_section_content_size)

  if literals_section_header.block_type == LiteralsBlockType.TREELESS_LITERALS_BLOCK:
    literals_section_content_size = literals_section_header.compressed_size
    literals_block = encoded_file.read(literals_section_content_size)

  print(f"  Literals section content size {literals_section_content_size}")

  sequences_block_content_size = block_header.block_content_size - literals_section_content_size - literals_section_header_size
  print(f"  Sequences block content size: {sequences_block_content_size}")

  sequences_section_header = read_sequences_section_header(encoded_file)
  print(f"  Num of sequences: {sequences_section_header.num_of_sequences}")
  print(f"  Literal lengths mode: {sequences_section_header.literal_lengths_mode}")
  print(f"  Offsets mode: {sequences_section_header.offsets_mode}")
  print(f"  Match lengths mode: {sequences_section_header.match_lengths_mode}")

  if sequences_section_header.num_of_sequences == 0:
    return literals_section_header.regenerated_size

  # Order of values matches: literal lengths, offsets and match lengths
  # Default distribution lengths and accuracies used by the PREDEFINED_MODE
  # https://datatracker.ietf.org/doc/html/rfc8878#name-default-distributions
  #             LL OF ML
  # Lengths:    36 29 53
  # Accuracies: 6  5  6

  if sequences_section_header.literal_lengths_mode == SymbolCompressionMode.PREDEFINED_MODE:
    literal_lengths_fse_table = FSETable(6, 36, LITERALS_LENGTH_DEFAULT_DIST)
  elif sequences_section_header.literal_lengths_mode == SymbolCompressionMode.RLE_MODE:
    literal_lengths_fse_table = RLETable(encoded_file.read(1)[0])
  elif sequences_section_header.literal_lengths_mode == SymbolCompressionMode.FSE_COMPRESSED_MODE:
    literal_lengths_fse_table = read_fse_table(encoded_file)

  if sequences_section_header.offsets_mode == SymbolCompressionMode.PREDEFINED_MODE:
    offsets_fse_table = FSETable(5, 29, OFFSET_CODES_DEFAULT_DIST)
  elif sequences_section_header.offsets_mode == SymbolCompressionMode.RLE_MODE:
    offsets_fse_table = RLETable(encoded_file.read(1)[0])
  elif sequences_section_header.offsets_mode == SymbolCompressionMode.FSE_COMPRESSED_MODE:
    offsets_fse_table = read_fse_table(encoded_file)

  if sequences_section_header.match_lengths_mode == SymbolCompressionMode.PREDEFINED_MODE:
    match_lengths_fse_table = FSETable(6, 53, MATCH_LENGTHS_DEFAULT_DIST)
  elif sequences_section_header.match_lengths_mode == SymbolCompressionMode.RLE_MODE:
    match_lengths_fse_table = RLETable(encoded_file.read(1)[0])
  elif sequences_section_header.match_lengths_mode == SymbolCompressionMode.FSE_COMPRESSED_MODE:
    match_lengths_fse_table = read_fse_table(encoded_file)

  print(f"  LL accuracy log: {literal_lengths_fse_table.accuracy_log}")
  print(f"  OF accuracy log: {offsets_fse_table.accuracy_log}")
  print(f"  ML accuracy log: {match_lengths_fse_table.accuracy_log}")

  decoded_sequences = decode_sequences(encoded_file,
                                       sequences_section_header.num_of_sequences,
                                       last_block_byte_index,
                                       literal_lengths_fse_table,
                                       offsets_fse_table,
                                       match_lengths_fse_table)

  total_literal_lengths = 0
  total_match_lengths = 0
  for (index, decoded_sequence) in enumerate(decoded_sequences):

    comes_from_match = False
    comes_from_literal = False
    if current_decoded_data_size + decoded_sequence.match_length >= searched_for_byte_index:
      comes_from_match = True
    elif current_decoded_data_size + decoded_sequence.match_length + decoded_sequence.literal_length >= searched_for_byte_index:
      comes_from_literal = True

    total_literal_lengths += decoded_sequence.literal_length
    total_match_lengths += decoded_sequence.match_length
    current_decoded_data_size += decoded_sequence.literal_length + decoded_sequence.match_length

    handle_repeated_offsets(decoded_sequence)

    if current_decoded_data_size >= searched_for_byte_index:
      insert_vertical_separation()

      if comes_from_match:
        origin = "history match"
      elif comes_from_literal:
        origin = "literal copying"
      else:
        origin = "<error>"

      print(f"** Byte {searched_for_byte_index} comes from the block {block_cnt} from sequence {index} from the block at byte index {block_starting_byte_index}")
      print(f"** The copied value comes from {origin} part of the sequence: ")
      print(f"** LL: {decoded_sequence.literal_length} OF: {decoded_sequence.offset_length} ML: {decoded_sequence.match_length} (after handling repeated offsets)")
      print(f"|| {test_name:>15} | 0x{searched_for_byte_index:016X} | {block_cnt:>15} | {index:>20} | {str(decoded_sequence):>30} | {origin:>15} ||\n")

      block_cnt += 1
      return (literals_section_header.regenerated_size + total_literal_lengths + total_match_lengths, True)

  decoded_content_size = literals_section_header.regenerated_size + total_match_lengths

  print(f"  Total match lengths: {total_match_lengths}")
  print(f"  Literals size: {literals_section_header.regenerated_size}")
  print(f"  Decoded content size: {decoded_content_size}")

  encoded_file.seek(last_block_byte_index + 1)

  block_cnt += 1
  return (decoded_content_size, block_header.last_block)

def insert_vertical_separation():
  print()

def debug_file(file_path, byte_index, test_name):
  with open(file_path, "rb") as encoded_file:
    read_magic_number(encoded_file)
    read_frame_header(encoded_file)
    insert_vertical_separation()

    total_decoded_content_size = 0
    finished = False
    while not finished:
      print(f"Currently decoded {total_decoded_content_size} bytes of data")
      decoded_content_size, finished = debug_block(encoded_file, total_decoded_content_size, byte_index, test_name)
      total_decoded_content_size += decoded_content_size
      finished = finished or total_decoded_content_size >= byte_index

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--file",
    help="Filename of the tested ZSTD encoded file",
    type=pathlib.Path,
    required=True
  )
  parser.add_argument(
    "--name",
    help="Test name",
    default="name"
  )
  parser.add_argument(
    "--byte",
    help="Byte which information will be printed",
    type=int,
    required=True
  )
  args = parser.parse_args()
  debug_file(args.file, args.byte, args.name)

if __name__ == "__main__":
  main()
