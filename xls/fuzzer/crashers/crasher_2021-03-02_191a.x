// Copyright 2021 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// BEGIN_CONFIG
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--use_system_verilog"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=4"
//   codegen_args: "--reset_data_path=false"
//   simulate: false
//   use_system_verilog: true
//   calls_per_sample: 1
// }
// inputs {
//   function_args {
//     args: "bits[43]:0x1000; bits[15]:0x2aaa"
//     args: "bits[43]:0x1000_0000; bits[15]:0x111"
//     args: "bits[43]:0x400; bits[15]:0x2106"
//     args: "bits[43]:0x3ff_ffff_ffff; bits[15]:0x5f6f"
//     args: "bits[43]:0x2000; bits[15]:0x794c"
//     args: "bits[43]:0x3ff_ffff_ffff; bits[15]:0x4000"
//     args: "bits[43]:0x2; bits[15]:0x5502"
//     args: "bits[43]:0x3ff_ffff_ffff; bits[15]:0x2"
//     args: "bits[43]:0x0; bits[15]:0x4256"
//     args: "bits[43]:0x20_0000; bits[15]:0x5301"
//     args: "bits[43]:0x200; bits[15]:0x1000"
//     args: "bits[43]:0x3ff_ffff_ffff; bits[15]:0x2000"
//     args: "bits[43]:0x100; bits[15]:0x100"
//     args: "bits[43]:0x282_e94e_62bf; bits[15]:0x7233"
//     args: "bits[43]:0x8; bits[15]:0x400"
//     args: "bits[43]:0x80_0000; bits[15]:0x2000"
//     args: "bits[43]:0x4000; bits[15]:0x488c"
//     args: "bits[43]:0x400_0000; bits[15]:0x5555"
//     args: "bits[43]:0x8_0000_0000; bits[15]:0x0"
//     args: "bits[43]:0x20; bits[15]:0x4e8"
//     args: "bits[43]:0x2_0000; bits[15]:0x5e0f"
//     args: "bits[43]:0x8; bits[15]:0x4209"
//     args: "bits[43]:0x400_0000_0000; bits[15]:0x20"
//     args: "bits[43]:0x0; bits[15]:0x3c02"
//     args: "bits[43]:0x4; bits[15]:0x804"
//     args: "bits[43]:0x555_5555_5555; bits[15]:0x4545"
//     args: "bits[43]:0x8_0000; bits[15]:0x4a08"
//     args: "bits[43]:0x7ee_c3dc_df0b; bits[15]:0x5f0b"
//     args: "bits[43]:0x20_0000; bits[15]:0xc00"
//     args: "bits[43]:0x2aa_aaaa_aaaa; bits[15]:0x0"
//     args: "bits[43]:0x200; bits[15]:0x200"
//     args: "bits[43]:0x200; bits[15]:0x4000"
//     args: "bits[43]:0x4000; bits[15]:0x100"
//     args: "bits[43]:0x1000; bits[15]:0x0"
//     args: "bits[43]:0x2_0000; bits[15]:0x8"
//     args: "bits[43]:0x1_0000_0000; bits[15]:0x10"
//     args: "bits[43]:0x2000_0000; bits[15]:0x0"
//     args: "bits[43]:0x100_0000; bits[15]:0x80"
//     args: "bits[43]:0x8_0000; bits[15]:0x0"
//     args: "bits[43]:0x4000; bits[15]:0x4000"
//     args: "bits[43]:0x800; bits[15]:0x4"
//     args: "bits[43]:0x3ff_ffff_ffff; bits[15]:0x2"
//     args: "bits[43]:0x1000; bits[15]:0x7fff"
//     args: "bits[43]:0x20_0000_0000; bits[15]:0x5555"
//     args: "bits[43]:0x16b_26a2_3499; bits[15]:0x3491"
//     args: "bits[43]:0x80_0000_0000; bits[15]:0x4"
//     args: "bits[43]:0x10_0000; bits[15]:0x800"
//     args: "bits[43]:0x200; bits[15]:0xa00"
//     args: "bits[43]:0x200_0000; bits[15]:0x400"
//     args: "bits[43]:0x64d_4d48_9cf5; bits[15]:0x58f9"
//     args: "bits[43]:0x80_0000_0000; bits[15]:0x0"
//     args: "bits[43]:0x1_0000; bits[15]:0x2c90"
//     args: "bits[43]:0x2aa_aaaa_aaaa; bits[15]:0x2253"
//     args: "bits[43]:0x4_0000_0000; bits[15]:0x34f3"
//     args: "bits[43]:0x8000_0000; bits[15]:0x3400"
//     args: "bits[43]:0x4; bits[15]:0x2"
//     args: "bits[43]:0x100; bits[15]:0x180"
//     args: "bits[43]:0x100; bits[15]:0x54a1"
//     args: "bits[43]:0x10_0000_0000; bits[15]:0x1"
//     args: "bits[43]:0x165_e997_6d9d; bits[15]:0x5555"
//     args: "bits[43]:0x4_0000_0000; bits[15]:0x100"
//     args: "bits[43]:0x1000; bits[15]:0x3fff"
//     args: "bits[43]:0x4; bits[15]:0x283"
//     args: "bits[43]:0x20_0000; bits[15]:0x2b"
//     args: "bits[43]:0x1000; bits[15]:0x4000"
//     args: "bits[43]:0x200_0000_0000; bits[15]:0x3fff"
//     args: "bits[43]:0x0; bits[15]:0x4100"
//     args: "bits[43]:0x0; bits[15]:0x200"
//     args: "bits[43]:0x400_0000; bits[15]:0x0"
//     args: "bits[43]:0x8_0000_0000; bits[15]:0x4000"
//     args: "bits[43]:0x4_0000_0000; bits[15]:0x5476"
//     args: "bits[43]:0x177_7c94_4ed4; bits[15]:0x4cd4"
//     args: "bits[43]:0x80; bits[15]:0xc1"
//     args: "bits[43]:0x1_0000; bits[15]:0x0"
//     args: "bits[43]:0x10_0000; bits[15]:0x1000"
//     args: "bits[43]:0x40_0000_0000; bits[15]:0x1"
//     args: "bits[43]:0x2000_0000; bits[15]:0x5801"
//     args: "bits[43]:0x40_0000; bits[15]:0x5331"
//     args: "bits[43]:0x400; bits[15]:0x4600"
//     args: "bits[43]:0xcd_d1c1_8151; bits[15]:0x2151"
//     args: "bits[43]:0x1b4_bbbd_edc9; bits[15]:0x4"
//     args: "bits[43]:0x4; bits[15]:0x40"
//     args: "bits[43]:0x2_0000; bits[15]:0x1081"
//     args: "bits[43]:0x400_0000; bits[15]:0x3fa3"
//     args: "bits[43]:0x800; bits[15]:0x4000"
//     args: "bits[43]:0x100_0000_0000; bits[15]:0x20"
//     args: "bits[43]:0x100; bits[15]:0x7c9"
//     args: "bits[43]:0x10; bits[15]:0x690"
//     args: "bits[43]:0x1_0000_0000; bits[15]:0x100"
//     args: "bits[43]:0x400_0000_0000; bits[15]:0x1"
//     args: "bits[43]:0x2; bits[15]:0x6003"
//     args: "bits[43]:0x4; bits[15]:0x800"
//     args: "bits[43]:0x2000_0000; bits[15]:0x80"
//     args: "bits[43]:0x3ff_ffff_ffff; bits[15]:0x80"
//     args: "bits[43]:0x80_0000_0000; bits[15]:0x40"
//     args: "bits[43]:0x8_0000; bits[15]:0x928"
//     args: "bits[43]:0x2_0000; bits[15]:0x4806"
//     args: "bits[43]:0x40_0000; bits[15]:0x100"
//     args: "bits[43]:0x8000_0000; bits[15]:0x4"
//     args: "bits[43]:0x800_0000; bits[15]:0x4100"
//     args: "bits[43]:0x400; bits[15]:0x400"
//     args: "bits[43]:0x1; bits[15]:0x181"
//     args: "bits[43]:0x377_16b1_b482; bits[15]:0x3402"
//     args: "bits[43]:0x200; bits[15]:0x2aaa"
//     args: "bits[43]:0x555_5555_5555; bits[15]:0x200"
//     args: "bits[43]:0x10; bits[15]:0x400"
//     args: "bits[43]:0x400_0000; bits[15]:0x2000"
//     args: "bits[43]:0x4000; bits[15]:0x75ad"
//     args: "bits[43]:0x80; bits[15]:0x4400"
//     args: "bits[43]:0x400_0000; bits[15]:0x10"
//     args: "bits[43]:0x2; bits[15]:0x195"
//     args: "bits[43]:0x4000; bits[15]:0x4800"
//     args: "bits[43]:0x80_0000; bits[15]:0x200"
//     args: "bits[43]:0x80; bits[15]:0x800"
//     args: "bits[43]:0x0; bits[15]:0x1000"
//     args: "bits[43]:0x80_0000; bits[15]:0x48"
//     args: "bits[43]:0x10; bits[15]:0x4"
//     args: "bits[43]:0x40_0000_0000; bits[15]:0x4440"
//     args: "bits[43]:0x20; bits[15]:0x20"
//     args: "bits[43]:0x200; bits[15]:0x200"
//     args: "bits[43]:0x1000; bits[15]:0x7fff"
//     args: "bits[43]:0x2_0000; bits[15]:0x910"
//     args: "bits[43]:0x40_0000; bits[15]:0x4000"
//     args: "bits[43]:0x4_0000; bits[15]:0x1000"
//     args: "bits[43]:0x2aa_aaaa_aaaa; bits[15]:0x2eee"
//     args: "bits[43]:0x40_0000_0000; bits[15]:0x2cc7"
//     args: "bits[43]:0x10_0000_0000; bits[15]:0x2aaa"
//     args: "bits[43]:0x2000; bits[15]:0x10"
//   }
// }
// END_CONFIG
const W32_V2 = u32:0x2;
type x23 = (u9, (u9,));
type x29 = u3;
fn x3(x4: u44) -> (u9, (u9,)) {
  let x5: u3 = u3:0x4;
  let x6: (u3,) = (x5,);
  let x7: u9 = ((x5) ++ (x5)) ++ (x5);
  let x8: (u44, u44) = (x4, x4);
  let x9: u9 = one_hot_sel(x5, [x7, x7, x7]);
  let x10: u3 = for (i, x): (u4, u3) in u4:0x0..u4:0x6 {
    x
  }(x5);
  let x11: u9 = (x7) * (x7);
  let x12: (u9,) = (x11,);
  let x13: u3 = (x6).0;
  let x14: u9 = for (i, x): (u4, u9) in u4:0x0..u4:0x3 {
    x
  }(x7);
  let x15: s55 = s55:0x8000000000000;
  let x16: s55 = one_hot_sel(x10, [x15, x15, x15]);
  let x17: u44 = (x8).0;
  let x18: u9 = rev(x11);
  let x19: u9 = !(x9);
  let x20: u9 = for (i, x): (u4, u9) in u4:0x0..u4:0x7 {
    x
  }(x14);
  let x21: u9 = for (i, x): (u4, u9) in u4:0x0..u4:0x0 {
    x
  }(x9);
  let x22: u9 = for (i, x): (u4, u9) in u4:0x0..u4:0x1 {
    x
  }(x11);
  (x9, x12)
}
fn main(x0: s43, x1: u15) -> (u15, u15, u15, (u30,), u1, u23, (u1,), u15, u1, u15, (u30,), u15, u1, u30, u15, u1, u15, x23[W32_V2], x23[0x4], u55, u15, u15, x23[0x4], (u30,), u15, u1, u15, u15) {
  let x2: x23[W32_V2] = map(u44[0x2]:[u44:0x8000000, u44:0x400], x3);
  let x24: u15 = (x1)[:];
  let x25: u15 = one_hot_sel(u4:0x1, [x1, x1, x1, x1]);
  let x26: u1 = (x25) != (((x24) as u15));
  let x27: u15 = ctz(x24);
  let x28: x29[0x5] = ((x24) as x29[0x5]);
  let x30: u15 = one_hot_sel(x26, [x1]);
  let x31: u15 = one_hot_sel(x26, [x30]);
  let x32: u1 = ((x1) != (u15:0x0)) && ((x24) != (u15:0x0));
  let x33: u16 = one_hot(x1, u1:0x1);
  let x34: (u1,) = (x26,);
  let x35: u23 = u23:0x400000;
  let x36: u15 = one_hot_sel(x26, [x25]);
  let x37: u30 = u30:0x2000000;
  let x38: u15 = !(x1);
  let x39: (u15,) = (x31,);
  let x40: u55 = (((x26) ++ (x1)) ++ (x35)) ++ (x33);
  let x41: u15 = one_hot_sel(x32, [x31]);
  let x42: x23[0x4] = (x2) ++ (x2);
  let x43: u30 = !(x37);
  let x44: s20 = s20:0x400;
  let x45: x23[0x4] = (x2) ++ (x2);
  let x46: u1 = for (i, x): (u4, u1) in u4:0x0..u4:0x7 {
    x
  }(x26);
  let x47: u55 = one_hot_sel(x26, [x40]);
  let x48: (u30,) = (x43,);
  let x49: u15 = x38;
  (x1, x24, x25, x48, x26, x35, x34, x30, x46, x24, x48, x49, x46, x43, x41, x32, x31, x2, x42, x47, x24, x24, x42, x48, x38, x26, x1, x36)
}
