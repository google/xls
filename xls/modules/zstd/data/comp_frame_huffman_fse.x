pub struct DataArray<BITS_PER_WORD: u32, LENGTH: u32>{
  data: uN[BITS_PER_WORD][LENGTH],
  length: u32,
  array_length: u32
}
pub const FRAMES:DataArray<
  u32:64,
  u32:8
>[1] = [DataArray<64, 8>{
  length: u32:64,
  array_length: u32:8,
  data: uN[64][8]:[uN[64]:0x007e4f84fd2fb528, uN[64]:0x00068e00017d0000, uN[64]:0xd5764f39f0080008, uN[64]:0x04000400045c4f40, uN[64]:0xcfefff3e7fefff00, uN[64]:0x5dff77afbdffef3f, uN[64]:0x1de190b0000301fb, uN[64]:0x807e83a8084e0c21]
}];
pub const DECOMPRESSED_FRAMES:DataArray<
  u32:64,
  u32:16
>[1] = [DataArray<64, 16>{
  length: u32:126,
  array_length: u32:16,
  data: uN[64][16]:[uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0xe6e6e6e6e680e6e6, uN[64]:0xe6e6e6b3e6e6e6e6, uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0x80e6e6e6e6e6e6e6, uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0xe6b3e6e6e6e6e6e6, uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0xe6e6e6e6b3b3e6e6, uN[64]:0xe6e6e6b3e6e6e6b3, uN[64]:0xe6e6e6e6e6e6b3e6, uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0xe6e6e6e6e6e6e6b3, uN[64]:0xb3e6e6b3b3e6b3e6, uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0xe6e6b3e6e6b3]
}];
