import math
import cocotb
import os

from pathlib import Path
from enum import IntEnum
from pprint import pformat

from cocotb.utils import get_sim_time

from cocotbext.axi.sparse_memory import SparseMemory
from cocotbext.axi.axi_channels import AxiWMonitor

from xls.modules.zstd.cocotb import data_generator
from xls.modules.zstd.cocotb.channel import XLSChannel
from xls.modules.zstd.cocotb.xlsstruct import xls_dataclass, XLSStruct
from xls.modules.zstd.zstd_dec_cocotb_common import (
    configure_decoder, start_decoder, reset_dut, prepare_test_environment,
    reverse_expected_huffman_codes, fields_as_array, print_fse_ram_contents,
    check_status, check_output, get_clock_time, CLOCK_PERIOD_PS
)
from xls.modules.zstd.cocotb.memory import AxiRamFromFile
from xls.modules.zstd.perf_report import report_test_result

def check_if_ram_contents_are_valid(mem, expected, name="") -> bool:
  for i, value in enumerate(expected):
    if mem[i].value != value:
      print(f"[{name}] index: {i} value: {mem[i].value}, expected: {value}")
      return False
  return True

def print_ram_contents(mem, name="", size=None):
  for i in range(size):
    print(f"{name} [{i}]\t: {hex(mem[i]) if not hasattr(mem[i], 'value') else hex(mem[i].value)}")

def printc(*args, **kwargs):
  GREEN = '\033[92m'
  ENDC = '\033[0m'
  print(GREEN, *args, ENDC, **kwargs)

class BlockType(IntEnum):
  RAW = 0,
  RLE = 1,
  COMPRESSED = 2,
  RESERVED = 3,

@xls_dataclass
class BlockHeader(XLSStruct):
  last: 1
  btype: 2
  size: 21

@xls_dataclass
class BlockHeaderDecoderResp(XLSStruct):
  status: 2
  header: BlockHeader.total_width
  rle_symbol: 8

@xls_dataclass
class BlockHeaderDecoderReq(XLSStruct):
  addr: 32

@xls_dataclass
class RawBlockDecoderReq(XLSStruct):
  id: 32
  addr: 32
  length: 32
  last_block: 1

class RawBlockDecoderStatus(IntEnum):
  OKAY = 0
  ERROR = 1

@xls_dataclass
class RawBlockDecoderResp(XLSStruct):
  status: 1

@xls_dataclass
class RleBlockDecoderReq(XLSStruct):
  id: 32
  symbol: 8
  length: 21
  last_block: 1

@xls_dataclass
class RleBlockDecoderResp(XLSStruct):
  status: 1

@xls_dataclass
class CompressedBlockDecoderReq(XLSStruct):
  addr: 32
  length: 21
  id: 32
  last_block: 1

@xls_dataclass
class CompressedBlockDecoderResp(XLSStruct):
  status: 1

@xls_dataclass
class LiteralsHeaderDecoderReq(XLSStruct):
  addr: 32

class LiteralsBlockType(IntEnum):
  RAW        = 0
  RLE        = 1
  COMP       = 2
  COMP_4     = 3
  TREELESS   = 4
  TREELESS_4 = 5

@xls_dataclass
class LiteralsHeader(XLSStruct):
  literal_type: 3
  regenerated_size: 20
  compressed_size: 20

@xls_dataclass
class LiteralsHeaderDecoderResp(XLSStruct):
  header: LiteralsHeader.total_width
  symbol: 8
  length: 3
  status: 1

@xls_dataclass
class SequenceConfDecoderReq(XLSStruct):
  addr: 32

@xls_dataclass
class SequenceConf(XLSStruct):
  sequence_count: 17
  literals_mode: 2
  offset_mode: 2
  match_mode: 2

@xls_dataclass
class SequenceConfDecoderResp(XLSStruct):
  header: SequenceConf.total_width
  length: 3
  status: 1

@xls_dataclass
class RawLiteralsDecoderReq(XLSStruct):
  id: 32
  addr: 32
  length: 32
  literals_last: 1

@xls_dataclass
class RawLiteralsDecoderResp(XLSStruct):
  status: 1

@xls_dataclass
class RleLiteralsDecoderReq(XLSStruct):
  id: 32
  symbol: 8
  length: 20
  literals_last: 1

@xls_dataclass
class RleLiteralsDecoderResp(XLSStruct):
  status: 1


@xls_dataclass
class HuffmanControlAndSequenceCtrl(XLSStruct):
  base_addr: 32
  len: 32
  new_config: 1
  multi_stream: 1
  id: 32
  literals_last: 1


@xls_dataclass
class HuffmanControlAndSequenceResp(XLSStruct):
  status: 1


class CompressionMode(IntEnum):
  PREDEFINED = 0
  RLE = 1
  COMPRESSED = 2
  REPEAT = 3

@xls_dataclass
class HuffmanWeightsDecoderReq(XLSStruct):
  addr: 32

@xls_dataclass
class HuffmanWeightsDecoderResp(XLSStruct):
  status: 1
  tree_description_size: 32

MAX_WEIGHT = 11
WEIGHT_LOG = math.ceil(math.log2(MAX_WEIGHT + 1))
VALID_W = 1

class WeightType(IntEnum):
  RAW = 0
  FSE = 1

@xls_dataclass
class CodeBuilderOutput(XLSStruct):
  symbol_valid_7: VALID_W
  symbol_valid_6: VALID_W
  symbol_valid_5: VALID_W
  symbol_valid_4: VALID_W
  symbol_valid_3: VALID_W
  symbol_valid_2: VALID_W
  symbol_valid_1: VALID_W
  symbol_valid_0: VALID_W
  code_length_7: WEIGHT_LOG
  code_length_6: WEIGHT_LOG
  code_length_5: WEIGHT_LOG
  code_length_4: WEIGHT_LOG
  code_length_3: WEIGHT_LOG
  code_length_2: WEIGHT_LOG
  code_length_1: WEIGHT_LOG
  code_length_0: WEIGHT_LOG
  code_7: MAX_WEIGHT
  code_6: MAX_WEIGHT
  code_5: MAX_WEIGHT
  code_4: MAX_WEIGHT
  code_3: MAX_WEIGHT
  code_2: MAX_WEIGHT
  code_1: MAX_WEIGHT
  code_0: MAX_WEIGHT


@xls_dataclass
class HuffmanRawWeightsDecoderReq(XLSStruct):
  addr: 32
  n_symbols: 8


@xls_dataclass
class HuffmanRawWeightsDecoderResp(XLSStruct):
  status: 1


@xls_dataclass
class HuffmanFseWeightsDecoderReq(XLSStruct):
  addr: 32
  length: 8


@xls_dataclass
class HuffmanFseWeightsDecoderResp(XLSStruct):
  status: 1


class CompressionMode(IntEnum):
  PREDEFINED = 0
  RLE = 1
  COMPRESSED = 2
  REPEAT = 3


@xls_dataclass
class FseLookupCtrlReq(XLSStruct):
  ll_mode: 2
  ml_mode: 2
  of_mode: 2


@xls_dataclass
class FseLookupCtrlResp(XLSStruct):
  ll_accuracy_log: 7
  ml_accuracy_log: 7
  of_accuracy_log: 7


@xls_dataclass
class HuffmanDecoderStart(XLSStruct):
    new_config: 1
    id: 32
    literals_last: 1
    last_stream: 1


async def detailed_testing_routine(dut,
  pregenerated_path,
  expected_fse_huffman_lookups=None,
  expected_huffman_weights=None,
  expected_huffman_codes=None,
  expected_fse_lookups=None
):
  (axi_buses, cpu, clock) = prepare_test_environment(dut)

  encoded_file = Path(pregenerated_path)

  BLOCK_HEADER_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_block_header_dec__ZstdDecoderInst__ZstdDecoder_0__BlockHeaderDecoder_0__32_64_next_inst2
  BLOCK_HEADER_REQ_CHANNEL_NAME = "zstd_dec__bh_req"
  BLOCK_HEADER_RESP_CHANNEL_NAME = "zstd_dec__bh_resp"
  MAX_ENCODED_FRAME_SIZE_B = 2**32

  RAW_BLOCK_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_raw_block_dec__ZstdDecoderInst__ZstdDecoder_0__RawBlockDecoder_0__32_64_next_inst135
  RAW_BLOCK_REQ_CHANNEL_NAME = "zstd_dec__raw_req"
  RAW_BLOCK_RESP_CHANNEL_NAME = "zstd_dec__raw_resp"

  RLE_BLOCK_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_rle_block_dec__ZstdDecoderInst__ZstdDecoder_0__RleBlockDecoder_0__64_next_inst142
  RLE_BLOCK_REQ_CHANNEL_NAME = "zstd_dec__rle_req"
  RLE_BLOCK_RESP_CHANNEL_NAME = "zstd_dec__rle_resp"

  COMPRESSED_BLOCK_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_comp_block_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__32_64_8_4_4_16_256_64_6_92_1_8_16_1_8_32_1_6_32_8_9_8_1_8_16_1_13_9_1_8_16_1_15_15_32_1_9_8_1_8_16_1_next_inst4
  COMPRESSED_BLOCK_REQ_CHANNEL_NAME = "zstd_dec__comp_block_req"
  COMPRESSED_BLOCK_RESP_CHANNEL_NAME = "zstd_dec__comp_block_resp"

  LITERALS_HEADER_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_literals_block_header_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__LiteralsHeaderDecoder_0__32_64_next_inst32
  LITERALS_HEADER_REQ_CHANNEL_NAME = "zstd_dec__lit_header_req"
  LITERALS_HEADER_RESP_CHANNEL_NAME = "zstd_dec__lit_header_resp"

  RAW_LITERALS_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_raw_literals_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__RawLiteralsDecoder_0__32_64_next_inst136
  RAW_LITERALS_DECODER_REQ_CHANNEL_NAME = "zstd_dec__raw_lit_req"
  RAW_LITERALS_DECODER_RESP_CHANNEL_NAME = "zstd_dec__raw_lit_resp"

  RLE_LITERALS_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_rle_literals_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__RleLiteralsDecoder_0__64_next_inst143
  RLE_LITERALS_DECODER_REQ_CHANNEL_NAME = "zstd_dec__rle_lit_req"
  RLE_LITERALS_DECODER_RESP_CHANNEL_NAME = "zstd_dec__rle_lit_resp"

  HUFFMAN_LITERALS_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_huffman_ctrl__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__HuffmanLiteralsDecoder_0__HuffmanControlAndSequence_0__32_64_next_inst21
  HUFFMAN_LITERALS_DECODER_REQ_CHANNEL_NAME = "zstd_dec__huffman_lit_req"
  HUFFMAN_LITERALS_DECODER_RESP_CHANNEL_NAME = "zstd_dec__huffman_lit_resp"

  HUFFMAN_LITERALS_DECODER_WEIGHTS_REQ_CHANNEL_NAME = "zstd_dec__weights_dec_req"
  HUFFMAN_LITERALS_DECODER_WEIGHTS_RESP_CHANNEL_NAME =  "zstd_dec__weights_dec_resp"

  HUFFMAN_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_huffman_decoder__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__HuffmanLiteralsDecoder_0__HuffmanDecoder_0_next_inst26
  HUFFMAN_DECODER_DONE_CHANNEL_NAME = "zstd_dec__decoder_done"

  HUFFMAN_LITERALS_WEIGHT_CODE_BUILDER_INST = dut.ZstdDecoder.xls_modules_zstd_huffman_code_builder__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__HuffmanLiteralsDecoder_0__WeightCodeBuilder_0__256_8_32_7_next_inst20
  HUFFMAN_LITERALS_WEIGHT_CODES_CHANNEL_NAME = "zstd_dec__code_builder_codes"

  HUFFMAN_WEIGHTS_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_huffman_weights_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__HuffmanLiteralsDecoder_0__HuffmanWeightsDecoder_0__32_64_4_8_16_1_8_32_1_9_8_1_8_16_1_6_32_8_next_inst28
  HUFFMAN_WEIGHTS_DECODER_SEL_REQ_CHANNEL_NAME = "zstd_dec__decoded_weights_sel_req"
  HUFFMAN_WEIGHTS_DECODER_SEL_RESP_CHANNEL_NAME = "zstd_dec__decoded_weights_sel_resp"

  HUFFMAN_RAW_WEIGHTS_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_huffman_weights_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__HuffmanLiteralsDecoder_0__HuffmanWeightsDecoder_0__HuffmanRawWeightsDecoder_0__32_64_96_7_6_32_8_next_inst31
  HUFFMAN_RAW_WEIGHTS_DECODER_SEL_REQ_CHANNEL_NAME = "zstd_dec__raw_weights_req"
  HUFFMAN_RAW_WEIGHTS_DECODER_SEL_RESP_CHANNEL_NAME = "zstd_dec__raw_weights_resp"

  HUFFMAN_FSE_WEIGHTS_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_huffman_weights_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__HuffmanLiteralsDecoder_0__HuffmanWeightsDecoder_0__HuffmanFseWeightsDecoder_0__32_64_4_8_16_1_8_32_1_64_7_9_8_1_8_16_1_6_32_8_next_inst29
  HUFFMAN_FSE_WEIGHTS_DECODER_SEL_REQ_CHANNEL_NAME = "zstd_dec__fse_weights_req"
  HUFFMAN_FSE_WEIGHTS_DECODER_SEL_RESP_CHANNEL_NAME = "zstd_dec__fse_weights_resp"

  HUFFMAN_CONTROL_AND_SEQUENCE_INTERNAL_INST = dut.ZstdDecoder.xls_modules_zstd_huffman_ctrl__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__HuffmanLiteralsDecoder_0__HuffmanControlAndSequence_0__HuffmanControlAndSequenceMultiStreamHandler_0__HuffmanControlAndSequenceInternal_0__32_next_inst23
  HUFFMAN_CONTROL_AND_SEQUENCE_INTERNAL_REQ_CHANNEL_NAME = "zstd_dec__hcs_internal_req"
  HUFFMAN_CONTROL_AND_SEQUENCE_INTERNAL_RESP_CHANNEL_NAME = "zstd_dec__hcs_internal_resp"

  SEQUENCE_HEADER_DECODER_INST = dut.ZstdDecoder.xls_modules_zstd_sequence_conf_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__SequenceDecoder_0__SequenceConfDecoder_0__32_64_next_inst145
  SEQUENCE_HEADER_REQ_CHANNEL_NAME = "zstd_dec__scd_req"
  SEQUENCE_HEADER_RESP_CHANNEL_NAME = "zstd_dec__scd_resp"

  FSE_LOOKUP_CTRL_INST = dut.ZstdDecoder.xls_modules_zstd_sequence_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__SequenceDecoder_0__SequenceDecoderCtrl_0__FseLookupCtrl_0_next_inst149
  FSE_LOOKUP_CTRL_REQ_CHANNEL_NAME = "zstd_dec__flc_req"
  FSE_LOOKUP_CTRL_RESP_CHANNEL_NAME = "zstd_dec__flc_resp"

  block_header_req = XLSChannel(BLOCK_HEADER_DECODER_INST, BLOCK_HEADER_REQ_CHANNEL_NAME, dut.clk)
  block_header_resp = XLSChannel(BLOCK_HEADER_DECODER_INST, BLOCK_HEADER_RESP_CHANNEL_NAME, dut.clk)

  raw_block_req = XLSChannel(RAW_BLOCK_DECODER_INST, RAW_BLOCK_REQ_CHANNEL_NAME, dut.clk)
  raw_block_resp = XLSChannel(RAW_BLOCK_DECODER_INST, RAW_BLOCK_RESP_CHANNEL_NAME, dut.clk)

  rle_block_req = XLSChannel(RLE_BLOCK_DECODER_INST, RLE_BLOCK_REQ_CHANNEL_NAME, dut.clk)
  rle_block_resp = XLSChannel(RLE_BLOCK_DECODER_INST, RLE_BLOCK_RESP_CHANNEL_NAME, dut.clk)

  compressed_block_req = XLSChannel(COMPRESSED_BLOCK_DECODER_INST, COMPRESSED_BLOCK_REQ_CHANNEL_NAME, dut.clk)
  compressed_block_resp = XLSChannel(COMPRESSED_BLOCK_DECODER_INST, COMPRESSED_BLOCK_RESP_CHANNEL_NAME, dut.clk)

  literals_header_req = XLSChannel(LITERALS_HEADER_DECODER_INST, LITERALS_HEADER_REQ_CHANNEL_NAME, dut.clk)
  literals_header_resp = XLSChannel(LITERALS_HEADER_DECODER_INST, LITERALS_HEADER_RESP_CHANNEL_NAME, dut.clk)

  raw_literals_dec_req = XLSChannel(RAW_LITERALS_DECODER_INST, RAW_LITERALS_DECODER_REQ_CHANNEL_NAME, dut.clk)
  raw_literals_dec_resp = XLSChannel(RAW_LITERALS_DECODER_INST, RAW_LITERALS_DECODER_RESP_CHANNEL_NAME, dut.clk)

  rle_literals_dec_req = XLSChannel(RLE_LITERALS_DECODER_INST, RLE_LITERALS_DECODER_REQ_CHANNEL_NAME, dut.clk)
  rle_literals_dec_resp = XLSChannel(RLE_LITERALS_DECODER_INST, RLE_LITERALS_DECODER_RESP_CHANNEL_NAME, dut.clk)

  huffman_literals_dec_req = XLSChannel(HUFFMAN_LITERALS_DECODER_INST, HUFFMAN_LITERALS_DECODER_REQ_CHANNEL_NAME, dut.clk)
  huffman_literals_dec_resp = XLSChannel(HUFFMAN_LITERALS_DECODER_INST, HUFFMAN_LITERALS_DECODER_RESP_CHANNEL_NAME, dut.clk)

  huffman_weights_req = XLSChannel(HUFFMAN_LITERALS_DECODER_INST, HUFFMAN_LITERALS_DECODER_WEIGHTS_REQ_CHANNEL_NAME, dut.clk)
  huffman_weights_resp = XLSChannel(HUFFMAN_LITERALS_DECODER_INST, HUFFMAN_LITERALS_DECODER_WEIGHTS_RESP_CHANNEL_NAME, dut.clk)

  huffman_weights_sel_req = XLSChannel(HUFFMAN_WEIGHTS_DECODER_INST, HUFFMAN_WEIGHTS_DECODER_SEL_REQ_CHANNEL_NAME, dut.clk)
  huffman_weights_sel_resp = XLSChannel(HUFFMAN_WEIGHTS_DECODER_INST, HUFFMAN_WEIGHTS_DECODER_SEL_RESP_CHANNEL_NAME, dut.clk)

  huffman_raw_weights_req = XLSChannel(HUFFMAN_RAW_WEIGHTS_DECODER_INST, HUFFMAN_RAW_WEIGHTS_DECODER_SEL_REQ_CHANNEL_NAME, dut.clk)
  huffman_raw_weights_resp = XLSChannel(HUFFMAN_RAW_WEIGHTS_DECODER_INST, HUFFMAN_RAW_WEIGHTS_DECODER_SEL_RESP_CHANNEL_NAME, dut.clk)

  huffman_fse_weights_req = XLSChannel(HUFFMAN_FSE_WEIGHTS_DECODER_INST, HUFFMAN_FSE_WEIGHTS_DECODER_SEL_REQ_CHANNEL_NAME, dut.clk)
  huffman_fse_weights_resp = XLSChannel(HUFFMAN_FSE_WEIGHTS_DECODER_INST, HUFFMAN_FSE_WEIGHTS_DECODER_SEL_RESP_CHANNEL_NAME, dut.clk)

  huffman_codes = XLSChannel(HUFFMAN_LITERALS_WEIGHT_CODE_BUILDER_INST, HUFFMAN_LITERALS_WEIGHT_CODES_CHANNEL_NAME, dut.clk)
  huffman_done = XLSChannel(HUFFMAN_DECODER_INST, HUFFMAN_DECODER_DONE_CHANNEL_NAME, dut.clk)

  sequence_header_req = XLSChannel(SEQUENCE_HEADER_DECODER_INST, SEQUENCE_HEADER_REQ_CHANNEL_NAME, dut.clk)
  sequence_header_resp = XLSChannel(SEQUENCE_HEADER_DECODER_INST, SEQUENCE_HEADER_RESP_CHANNEL_NAME, dut.clk)

  huffman_control_and_sequence_internal_req = XLSChannel(HUFFMAN_CONTROL_AND_SEQUENCE_INTERNAL_INST, HUFFMAN_CONTROL_AND_SEQUENCE_INTERNAL_REQ_CHANNEL_NAME, dut.clk)
  huffman_control_and_sequence_internal_resp = XLSChannel(HUFFMAN_CONTROL_AND_SEQUENCE_INTERNAL_INST, HUFFMAN_CONTROL_AND_SEQUENCE_INTERNAL_RESP_CHANNEL_NAME, dut.clk)

  fse_lookup_ctrl_req = XLSChannel(FSE_LOOKUP_CTRL_INST, FSE_LOOKUP_CTRL_REQ_CHANNEL_NAME, dut.clk)
  fse_lookup_ctrl_resp = XLSChannel(FSE_LOOKUP_CTRL_INST, FSE_LOOKUP_CTRL_RESP_CHANNEL_NAME, dut.clk)

  await reset_dut(dut, 50)

  memory_bus = axi_buses["memory"]
  mem_size = MAX_ENCODED_FRAME_SIZE_B
  ibuf_addr = 0x0
  obuf_addr = mem_size // 2
  await reset_dut(dut, 50)

  AXI_DATA_W = 64
  AXI_DATA_W_BYTES = AXI_DATA_W // 8

  # Initialise testbench memory with generated ZSTD frame

  with open(encoded_file, "rb") as f:
      expected_decoded_frame = data_generator.DecompressFrame(f.read())
      reference_memory = SparseMemory(mem_size)
      reference_memory.write(obuf_addr, expected_decoded_frame)
      expected_packet_count = (len(expected_decoded_frame) + (AXI_DATA_W_BYTES - 1)) // AXI_DATA_W_BYTES

  memory = AxiRamFromFile(
    bus=memory_bus, clock=dut.clk, reset=dut.rst, path=str(encoded_file), size=mem_size
  )

  await configure_decoder(dut, cpu, ibuf_addr, obuf_addr)
  output_monitor = AxiWMonitor(memory_bus.write.w, dut.clk, dut.rst)

  check_status_thread = await cocotb.start(check_status(dut, cpu))
  check_output_thread = await cocotb.start(check_output(expected_packet_count, memory, reference_memory, output_monitor, obuf_addr, clock, encoded_file))

  await start_decoder(cpu)

  score = 0
  block_cnt = 0
  expected_fse_count = 0
  expected_fse_huffman_count = 0
  expected_huffman_weights_count = 0
  expected_huffman_codes_count = 0

  while True:
    req = await block_header_req.recv_as(BlockHeaderDecoderReq)
    score += 1
    printc(f"[block {block_cnt}] Requested block header, score: {score}")

    resp = await block_header_resp.recv_as(BlockHeaderDecoderResp)
    block_header = BlockHeader.from_int(resp.header)
    score += 1
    printc(f"[block {block_cnt}] Decoded block header: {block_header}, score {score}")
    printc(f"SIM TIME: {get_sim_time(units='step')}")

    block_type = BlockType(block_header.btype)
    match block_type:
      case BlockType.RAW:
        req = await raw_block_req.recv_as(RawBlockDecoderReq)
        score +=1
        printc(f"[block {block_cnt}] Requested decoding a RAW block, score: {score}")

        resp = await raw_block_resp.recv_as(RawBlockDecoderResp)
        score +=1
        printc(f"[block {block_cnt}] Decoded a RAW block, score: {score}")

      case BlockType.RLE:
        req = await rle_block_req.recv_as(RleBlockDecoderReq)
        score +=1
        printc(f"[block {block_cnt}] Requested decoding an RLE block, score: {score}")

        resp = await rle_block_resp.recv_as(RleBlockDecoderResp)
        score +=1
        printc(f"[block {block_cnt}] Decoded an RLE block, score: {score}")

      case BlockType.COMPRESSED:
        req = await compressed_block_req.recv_as(CompressedBlockDecoderReq)
        score +=1
        printc(f"[block {block_cnt}] Requested decoding a COMPRESSED block, score: {score}")

        req = await literals_header_req.recv_as(LiteralsHeaderDecoderReq)
        score +=1
        printc(f"[block {block_cnt}] Requested decoding a literals header, score: {score}")

        resp = await literals_header_resp.recv_as(LiteralsHeaderDecoderResp)
        literal_header = LiteralsHeader.from_int(resp.header)
        score +=1
        printc(f"[block {block_cnt}] Decoded a literals header: {literal_header}, score: {score}")

        literal_type = LiteralsBlockType(literal_header.literal_type)
        match literal_type:
          case LiteralsBlockType.RAW:
            req = await raw_literals_dec_req.recv_as(RawLiteralsDecoderReq)
            score +=1
            printc(f"[block {block_cnt}] Requested decoding RAW literals, score: {score}")

            resp = await raw_literals_dec_resp.recv_as(RawLiteralsDecoderResp)
            score +=1
            printc(f"[block {block_cnt}] Decoded RAW literals, score: {score}")

          case LiteralsBlockType.RLE:
            req = await rle_literals_dec_req.recv_as(RleLiteralsDecoderReq)
            score +=1
            printc(f"[block {block_cnt}] Requested decoding RLE literals, score: {score}")

            resp = await rle_literals_dec_resp.recv_as(RleLiteralsDecoderResp)
            score +=1
            printc(f"[block {block_cnt}] Decoded RLE literals, score: {score}")

          case LiteralsBlockType.COMP | LiteralsBlockType.TREELESS | LiteralsBlockType.COMP_4 | LiteralsBlockType.TREELESS_4:
            req = await huffman_literals_dec_req.recv_as(HuffmanControlAndSequenceCtrl)
            score +=1
            printc(f"SIM TIME: {get_sim_time(units='step')}")

            if not literal_type in [LiteralsBlockType.TREELESS, LiteralsBlockType.TREELESS_4]:
              weight_dec_req = await huffman_weights_req.recv_as(HuffmanWeightsDecoderReq)
              score +=1
              printc(f"[block {block_cnt}] Requested decoding COMPRESSED literals, score: {score}")

              sel_req = await huffman_weights_sel_req.recv()
              score +=1

              huffman_weights_type = WeightType(sel_req)
              if huffman_weights_type == WeightType.RAW:

                req = await huffman_raw_weights_req.recv_as(HuffmanRawWeightsDecoderReq)
                score +=1
                printc(f"[block {block_cnt}] Requested decoding RAW Huffman weights, score: {score}")

                resp = await huffman_raw_weights_resp.recv_as(HuffmanRawWeightsDecoderResp)
                score +=1
                printc(f"[block {block_cnt}] Decoded RAW Huffman weights, score: {score}")

              else:
                req = await huffman_fse_weights_req.recv_as(HuffmanFseWeightsDecoderReq)
                score +=1
                printc(f"[block {block_cnt}] Requested decoding FSE Huffman weights, score: {score}")

                resp = await huffman_fse_weights_resp.recv_as(HuffmanFseWeightsDecoderResp)
                score +=1
                printc(f"[block {block_cnt}] Decoded FSE Huffman weights, score: {score}")

                huffman_fse_mem = dut.huffman_literals_weights_fse_ram_ram.mem
                if expected_fse_huffman_lookups is not None:
                  valid = check_if_ram_contents_are_valid(huffman_fse_mem, [x.value for x in expected_fse_huffman_lookups[expected_fse_huffman_count]], "huffman_fse")
                  if not valid:
                    print_fse_ram_contents(huffman_fse_mem, "huffman_fse_mem", len(expected_fse_huffman_lookups[expected_fse_huffman_count]))
                    print_fse_ram_contents(expected_fse_huffman_lookups[expected_fse_huffman_count], "expected_huffman_fse_mem", len(expected_fse_huffman_lookups[expected_fse_huffman_count]))
                    assert False, f"Huffman FSE table in block {block_cnt} is invalid"
                  expected_fse_huffman_count += 1
                  printc(f"[block {block_cnt}] Verified FSE Huffman weights, score: {score}")
                  score +=1

              score +=1
              printc(f"[block {block_cnt}] Decoded Huffman weights, score: {score}")

              huffman_mem = dut.huffman_literals_weights_mem_ram_ram.mem
              if expected_huffman_weights is not None:
                valid = check_if_ram_contents_are_valid(huffman_mem, expected_huffman_weights[expected_huffman_weights_count], "huffman_weights")
                if not valid:
                  print_ram_contents(huffman_mem, "huffman_weights", len(expected_huffman_weights[expected_huffman_weights_count]))
                  printc("VS.")
                  print_ram_contents(expected_huffman_weights[expected_huffman_weights_count], "expected_huffman_weights", len(expected_huffman_weights[expected_huffman_weights_count]))
                  assert False, f"Huffman weights table in block {block_cnt} is invalid"
                expected_huffman_weights_count += 1
                score +=1
                printc(f"[block {block_cnt}] Checked huffman weights, score: {score}")

              codes = []
              symbol_cnt = 0

              j = 32
              while (j > 0):
                data = await huffman_codes.recv_as(CodeBuilderOutput)
                score +=1
                printc(f"[block {block_cnt}] Received huffman codes, score: {score}")

                symbol_valid_array = fields_as_array(data, "symbol_valid", 8)
                code_length_array = fields_as_array(data, "code_length", 8)
                code_array = fields_as_array(data, "code", 8)
                for symbol_valid, code_length, code in zip(symbol_valid_array, code_length_array, code_array):
                  if symbol_valid == 1:
                    new_codes = [{"symbol": symbol_cnt, "code": code, "length": code_length}]
                    codes += new_codes
                  symbol_cnt += 1
                j -= 1

              if expected_huffman_codes is not None:
                if codes != reverse_expected_huffman_codes(expected_huffman_codes[expected_huffman_codes_count]):
                  by_codes = lambda d: d['symbol']
                  printc(pformat(sorted(codes, key=by_codes)))
                  printc("VS.")
                  printc(pformat(sorted(reverse_expected_huffman_codes(expected_huffman_codes[expected_huffman_codes_count]), key=by_codes)))
                  assert False, f"Huffman weights table in block {block_cnt} is invalid"
                expected_huffman_codes_count += 1
                score +=1
                printc(f"[block {block_cnt}] Checked huffman codes, score: {score}")

              resp = await huffman_literals_dec_resp.recv_as(HuffmanControlAndSequenceResp)
              score +=1
              printc(f"[block {block_cnt}] Decoded Huffman literals, score: {score}")


        req = await sequence_header_req.recv_as(SequenceConfDecoderReq)
        score +=1
        printc(f"[block {block_cnt}] Requested decoding sequence header, score: {score}")

        resp = await sequence_header_resp.recv_as(SequenceConfDecoderResp)
        header = SequenceConf.from_int(resp.header)
        score +=1
        printc(f"[block {block_cnt}] Decoded sequence header, score: {score}")

        if header.sequence_count != 0:
          req = await fse_lookup_ctrl_req.recv_as(FseLookupCtrlReq)
          score +=1
          printc(f"[block {block_cnt}] Requested decoding FSE lookups, score: {score}")

          ll_mode = CompressionMode(req.ll_mode)
          ml_mode = CompressionMode(req.ml_mode)
          of_mode = CompressionMode(req.of_mode)

          req = await fse_lookup_ctrl_resp.recv_as(FseLookupCtrlResp)
          score +=1
          printc(f"[block {block_cnt}] Decoded FSE lookups, score: {score}")

          if expected_fse_lookups is not None:
            if ll_mode != CompressionMode.PREDEFINED:
              printc(f"LL mode: {ll_mode}")
              ll_mem = dut.ll_fse_ram.mem
              if expected_fse_lookups is not None:
                expected_ll_mem = expected_fse_lookups[expected_fse_count]["ll"]
                valid = check_if_ram_contents_are_valid(ll_mem, [x.value for x in expected_ll_mem])
                if not valid:
                  print_fse_ram_contents(ll_mem, "ll_fse_mem", len(expected_ll_mem))
                  printc("VS.")
                  print_fse_ram_contents(expected_ll_mem, "expected_ll_fse_mem", len(expected_ll_mem))
                  assert False, f"LL FSE lookup table in block {block_cnt} is invalid"

            if of_mode != CompressionMode.PREDEFINED:
              printc(f"OF mode: {of_mode}")
              of_mem = dut.of_fse_ram.mem
              expected_of_mem = expected_fse_lookups[expected_fse_count]["of"]
              valid = check_if_ram_contents_are_valid(of_mem, [x.value for x in expected_of_mem])
              if not valid:
                print_fse_ram_contents(of_mem, "of_fse_mem", len(expected_of_mem))
                printc("VS.")
                print_fse_ram_contents(expected_of_mem, "expected_of_fse_mem", len(expected_of_mem))
                assert False, f"OF FSE lookup table in block {block_cnt} is invalid"

            if ml_mode != CompressionMode.PREDEFINED:
              printc(f"ML mode: {ml_mode}")
              ml_mem = dut.ml_fse_ram.mem
              expected_ml_mem = expected_fse_lookups[expected_fse_count]["ml"]
              valid = check_if_ram_contents_are_valid(ml_mem, [x.value for x in expected_ml_mem])
              if not valid:
                print_fse_ram_contents(ml_mem, "ml_fse_mem", len(expected_ml_mem))
                printc("VS.")
                print_fse_ram_contents(expected_ml_mem, "expected_of_fse_mem", len(expected_ml_mem))
                assert False, f"ML FSE lookup table in block {block_cnt} is invalid"

            expected_fse_count += 1

          resp = await compressed_block_resp.recv_as(CompressedBlockDecoderResp)
          score +=1
          printc(f"[block {block_cnt}] Decoded COMPRESSED block, score: {score}")

      case BlockType.RESERVED:
        raise Exception("Decoded block header of type RESERVED.")

    block_cnt += 1

    if block_header.last:
      break

  decode_times = await check_output_thread
  (decode_start, decode_first_packet, decode_last_packet) = decode_times
  await check_status_thread

  decode_end = get_clock_time(clock)
  latency = decode_first_packet - decode_start
  duration = decode_end - decode_start
  total_decoded_bytes = expected_packet_count * AXI_DATA_W_BYTES
  bytes_per_clock = total_decoded_bytes / duration
  BYTES_IN_GIGABYTE = 1024 * 1024 * 1024
  CLOCKS_PER_SECOND = 1e12 / CLOCK_PERIOD_PS
  gigabytes_per_second = bytes_per_clock * CLOCKS_PER_SECOND / BYTES_IN_GIGABYTE
  print(f"Duration: {duration} cycles")
  print(f"Latency (clocks till first data): {latency} cycles")
  print(f"Total decoded bytes: {total_decoded_bytes} bytes")
  print(f"Decoding throughput: {gigabytes_per_second:.04f} GB/s")

  test_name = os.path.basename(pregenerated_path)
  report_test_result(test_name, duration, latency, total_decoded_bytes, gigabytes_per_second)
