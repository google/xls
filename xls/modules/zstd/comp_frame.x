pub struct DataArray<BITS_PER_WORD: u32, LENGTH: u32>{
  data: uN[BITS_PER_WORD][LENGTH],
  length: u32,
  array_length: u32
}
pub const FRAMES:DataArray<
  u32:64,
  u32:5
>[1] = [DataArray<64, 5>{
  length: u32:33,
  array_length: u32:5,
  data: uN[64][5]:[uN[64]:0x001a3384fd2fb528, uN[64]:0xc1d3500000850000, uN[64]:0xdcf0529b98db8a06, uN[64]:0x308fa3120a430001, uN[64]:0x50]
}];
pub const DECOMPRESSED_FRAMES:DataArray<
  u32:64,
  u32:4
>[1] = [DataArray<64, 4>{
  length: u32:26,
  array_length: u32:4,
  data: uN[64][4]:[uN[64]:0x529b98db8a06c1d3, uN[64]:0x529b98db8a06dcf0, uN[64]:0x529b98db8a06dcf0, uN[64]:0xdcf0]
}];
