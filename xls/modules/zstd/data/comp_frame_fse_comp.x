pub struct DataArray<BITS_PER_WORD: u32, LENGTH: u32>{
  data: uN[BITS_PER_WORD][LENGTH],
  length: u32,
  array_length: u32
}
pub const FRAMES:DataArray<
  u32:64,
  u32:9
>[1] = [DataArray<64, 9>{
  length: u32:66,
  array_length: u32:9,
  data: uN[64][9]:[uN[64]:0x00545084fd2fb528, uN[64]:0x4236d000018d0000, uN[64]:0x1d98357537f4050f, uN[64]:0x8d92b5aed6d7791b, uN[64]:0x51538ed729019574, uN[64]:0x701101fb8611a803, uN[64]:0x8acfff857107d159, uN[64]:0x548604b38e0a63fd, uN[64]:0xc551]
}];
pub const DECOMPRESSED_FRAMES:DataArray<
  u32:64,
  u32:11
>[1] = [DataArray<64, 11>{
  length: u32:84,
  array_length: u32:11,
  data: uN[64][11]:[uN[64]:0x373737f4050f4236, uN[64]:0x3737373737373737, uN[64]:0x3737373737373737, uN[64]:0x3737373737373737, uN[64]:0x373737f4050f4237, uN[64]:0x3737373737373737, uN[64]:0x3737373737373737, uN[64]:0x3737373737373737, uN[64]:0xd6d7791b1d983575, uN[64]:0x290195748d92b5ae, uN[64]:0x51538ed7]
}];
