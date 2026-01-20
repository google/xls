from xls.modules.zstd.cocotb import xlsstruct


SYMBOL_W = 8
NUM_OF_BITS_W = 8
BASE_W = 16

@xlsstruct.xls_dataclass
class FseTableRecord(xlsstruct.XLSStruct):
  base: BASE_W
  num_of_bits: NUM_OF_BITS_W
  symbol: SYMBOL_W
