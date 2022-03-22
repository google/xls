// Copyright 2020 The XLS Authors
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

// options: {"input_is_dslx": true, "ir_converter_args": ["--top=main"], "convert_to_ir": true, "optimize_ir": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=3"], "simulate": false, "simulator": null}
// args: bits[27]:0x6baaccb; bits[20]:0x907c2; bits[1]:0x1; bits[7]:0xa; bits[27]:0x1eaedb5
// args: bits[27]:0x0; bits[20]:0x6b5e0; bits[1]:0x1; bits[7]:0x57; bits[27]:0x5f3c152
// args: bits[27]:0x80000; bits[20]:0x4a94a; bits[1]:0x0; bits[7]:0x2d; bits[27]:0x2965974
// args: bits[27]:0x1c32523; bits[20]:0x2c2c0; bits[1]:0x1; bits[7]:0x13; bits[27]:0x6d84d27
// args: bits[27]:0x1174908; bits[20]:0x79d38; bits[1]:0x1; bits[7]:0x52; bits[27]:0x2309307
// args: bits[27]:0x36a9029; bits[20]:0xb586b; bits[1]:0x0; bits[7]:0x50; bits[27]:0x5236833
// args: bits[27]:0x56cf562; bits[20]:0x525c5; bits[1]:0x0; bits[7]:0x1b; bits[27]:0x519e728
// args: bits[27]:0xf6eb5e; bits[20]:0x8369f; bits[1]:0x1; bits[7]:0x54; bits[27]:0x4701b38
// args: bits[27]:0xdad80a; bits[20]:0xc710b; bits[1]:0x1; bits[7]:0x68; bits[27]:0x5591098
// args: bits[27]:0x63a8aca; bits[20]:0xcfe4e; bits[1]:0x1; bits[7]:0x11; bits[27]:0x209efb6
// args: bits[27]:0x7568226; bits[20]:0x2e6b3; bits[1]:0x1; bits[7]:0x71; bits[27]:0x1151291
// args: bits[27]:0x5512fef; bits[20]:0xc45d5; bits[1]:0x0; bits[7]:0x14; bits[27]:0x3455a62
// args: bits[27]:0x1973942; bits[20]:0x2ab47; bits[1]:0x0; bits[7]:0x3b; bits[27]:0x73dd738
// args: bits[27]:0x10; bits[20]:0x8db84; bits[1]:0x1; bits[7]:0x42; bits[27]:0x3464c11
// args: bits[27]:0x3437d99; bits[20]:0x2ebf9; bits[1]:0x1; bits[7]:0x60; bits[27]:0x56ef43e
// args: bits[27]:0xffccec; bits[20]:0xc1cb7; bits[1]:0x1; bits[7]:0x71; bits[27]:0x43dbe0a
// args: bits[27]:0xf78472; bits[20]:0x3df55; bits[1]:0x0; bits[7]:0xe; bits[27]:0x625cd5c
// args: bits[27]:0x20000; bits[20]:0x2e861; bits[1]:0x1; bits[7]:0x4a; bits[27]:0x3714ff2
// args: bits[27]:0x5f4433; bits[20]:0xacf83; bits[1]:0x1; bits[7]:0x6c; bits[27]:0x65ee38d
// args: bits[27]:0x1dd1a3e; bits[20]:0x5cd2e; bits[1]:0x0; bits[7]:0x2; bits[27]:0x40000
// args: bits[27]:0x508b4ba; bits[20]:0x88630; bits[1]:0x0; bits[7]:0x3b; bits[27]:0x77588f2
// args: bits[27]:0x15fb666; bits[20]:0xc9e28; bits[1]:0x0; bits[7]:0x2; bits[27]:0x436f75b
// args: bits[27]:0x7a4fb81; bits[20]:0x22585; bits[1]:0x1; bits[7]:0x9; bits[27]:0x59b3ba4
// args: bits[27]:0x10; bits[20]:0x3fe66; bits[1]:0x1; bits[7]:0x7f; bits[27]:0x16f8fa1
// args: bits[27]:0x58629a9; bits[20]:0xae1b7; bits[1]:0x1; bits[7]:0x3; bits[27]:0x5e10079
// args: bits[27]:0x22dd36b; bits[20]:0xdcff9; bits[1]:0x1; bits[7]:0x25; bits[27]:0x6db5087
// args: bits[27]:0x619a6ca; bits[20]:0xd6739; bits[1]:0x1; bits[7]:0x2f; bits[27]:0x200000
// args: bits[27]:0x2a0c550; bits[20]:0xd984f; bits[1]:0x1; bits[7]:0x3b; bits[27]:0x30a28be
// args: bits[27]:0x7881e4c; bits[20]:0xa7509; bits[1]:0x1; bits[7]:0x70; bits[27]:0x80000
// args: bits[27]:0x100000; bits[20]:0x6048; bits[1]:0x0; bits[7]:0x1; bits[27]:0x2af954a
// args: bits[27]:0x2; bits[20]:0x1; bits[1]:0x0; bits[7]:0x50; bits[27]:0x210a2b7
// args: bits[27]:0x431d020; bits[20]:0xc201; bits[1]:0x0; bits[7]:0x7f; bits[27]:0x63fde8c
// args: bits[27]:0x569da07; bits[20]:0x20000; bits[1]:0x0; bits[7]:0x51; bits[27]:0x4e481eb
// args: bits[27]:0x2f43050; bits[20]:0x30ff0; bits[1]:0x1; bits[7]:0xe; bits[27]:0x69598aa
// args: bits[27]:0x254fd4b; bits[20]:0x7ffff; bits[1]:0x0; bits[7]:0x10; bits[27]:0x2aecb78
// args: bits[27]:0x2310d6a; bits[20]:0xb3411; bits[1]:0x1; bits[7]:0x35; bits[27]:0x46d9b4e
// args: bits[27]:0x56f25b5; bits[20]:0xe226a; bits[1]:0x1; bits[7]:0x75; bits[27]:0x2ee2ee5
// args: bits[27]:0x4c2d545; bits[20]:0xecc76; bits[1]:0x0; bits[7]:0x74; bits[27]:0x7ed13fe
// args: bits[27]:0x4edc5eb; bits[20]:0x6ec36; bits[1]:0x1; bits[7]:0x71; bits[27]:0x7ffffff
// args: bits[27]:0x47f7d47; bits[20]:0x8a208; bits[1]:0x1; bits[7]:0x51; bits[27]:0xc4a23a
// args: bits[27]:0x24f2001; bits[20]:0x2e29c; bits[1]:0x0; bits[7]:0x4f; bits[27]:0xda0b9c
// args: bits[27]:0x4a1590a; bits[20]:0x6673b; bits[1]:0x1; bits[7]:0x20; bits[27]:0x4e2e0
// args: bits[27]:0x1b58397; bits[20]:0x9453e; bits[1]:0x0; bits[7]:0x17; bits[27]:0x5476182
// args: bits[27]:0x3c402c8; bits[20]:0xa11b8; bits[1]:0x0; bits[7]:0xa; bits[27]:0x50dc4bc
// args: bits[27]:0x4000; bits[20]:0xf8140; bits[1]:0x1; bits[7]:0x63; bits[27]:0x46ac0d
// args: bits[27]:0xafcd12; bits[20]:0xa3b1a; bits[1]:0x1; bits[7]:0x18; bits[27]:0x2aaaaaa
// args: bits[27]:0x1c6583b; bits[20]:0x15265; bits[1]:0x1; bits[7]:0x58; bits[27]:0x662a4bf
// args: bits[27]:0x26a4f79; bits[20]:0x46e9d; bits[1]:0x1; bits[7]:0x5; bits[27]:0x50d5e59
// args: bits[27]:0x163a94b; bits[20]:0xe2cb7; bits[1]:0x0; bits[7]:0x55; bits[27]:0x2113560
// args: bits[27]:0x3cabd71; bits[20]:0x841fe; bits[1]:0x0; bits[7]:0x77; bits[27]:0x4d9a1bf
// args: bits[27]:0x2a9196e; bits[20]:0x18c8b; bits[1]:0x1; bits[7]:0x6e; bits[27]:0x2f80027
// args: bits[27]:0x29b0198; bits[20]:0x63562; bits[1]:0x0; bits[7]:0x14; bits[27]:0x100
// args: bits[27]:0x47e2146; bits[20]:0x14ad9; bits[1]:0x1; bits[7]:0x66; bits[27]:0x2000000
// args: bits[27]:0x1b69262; bits[20]:0xf9ef1; bits[1]:0x1; bits[7]:0x71; bits[27]:0x2b4376f
// args: bits[27]:0x4420ccb; bits[20]:0x74baa; bits[1]:0x0; bits[7]:0x31; bits[27]:0x739191c
// args: bits[27]:0x6d351fa; bits[20]:0x3d4df; bits[1]:0x0; bits[7]:0x6a; bits[27]:0x5dd075e
// args: bits[27]:0x4306806; bits[20]:0x71b45; bits[1]:0x1; bits[7]:0x5b; bits[27]:0x56c01c8
// args: bits[27]:0x3dfb35a; bits[20]:0x28ae8; bits[1]:0x0; bits[7]:0x42; bits[27]:0x29e88ee
// args: bits[27]:0x6dcda8d; bits[20]:0xf391a; bits[1]:0x0; bits[7]:0x4d; bits[27]:0x3df06f9
// args: bits[27]:0x2d5932d; bits[20]:0x5e785; bits[1]:0x0; bits[7]:0x5b; bits[27]:0x4335758
// args: bits[27]:0x70bf311; bits[20]:0xe875f; bits[1]:0x0; bits[7]:0x9; bits[27]:0x1f26dd8
// args: bits[27]:0x101aafe; bits[20]:0xc9690; bits[1]:0x0; bits[7]:0x4b; bits[27]:0x5ea140
// args: bits[27]:0x3eeb8e0; bits[20]:0xfa431; bits[1]:0x1; bits[7]:0x75; bits[27]:0x58656b6
// args: bits[27]:0x26bc0da; bits[20]:0x85f6a; bits[1]:0x1; bits[7]:0x25; bits[27]:0x6a8629e
// args: bits[27]:0x589543f; bits[20]:0x1000; bits[1]:0x1; bits[7]:0x3c; bits[27]:0x1c01e9f
// args: bits[27]:0x5e2eff9; bits[20]:0xc6314; bits[1]:0x0; bits[7]:0x4a; bits[27]:0x766a2cf
// args: bits[27]:0x2000; bits[20]:0x68943; bits[1]:0x0; bits[7]:0x51; bits[27]:0x3992b68
// args: bits[27]:0x5ff696c; bits[20]:0xaac0d; bits[1]:0x1; bits[7]:0x29; bits[27]:0x228dc6c
// args: bits[27]:0x71d3229; bits[20]:0x5f0d3; bits[1]:0x0; bits[7]:0x48; bits[27]:0x39e98b6
// args: bits[27]:0x276fd05; bits[20]:0xd4bf4; bits[1]:0x0; bits[7]:0x3f; bits[27]:0x100
// args: bits[27]:0x3aa487; bits[20]:0x138fb; bits[1]:0x1; bits[7]:0x0; bits[27]:0x7fec8ec
// args: bits[27]:0x3e1496b; bits[20]:0x67b39; bits[1]:0x0; bits[7]:0x4c; bits[27]:0x1cbf3cb
// args: bits[27]:0x2000; bits[20]:0x3457e; bits[1]:0x0; bits[7]:0x4c; bits[27]:0x3b0dc29
// args: bits[27]:0x6a82787; bits[20]:0x9ae07; bits[1]:0x1; bits[7]:0x4d; bits[27]:0x54fed08
// args: bits[27]:0x405828d; bits[20]:0xd4fde; bits[1]:0x1; bits[7]:0x61; bits[27]:0x2aaaaaa
// args: bits[27]:0x69529a; bits[20]:0xdbb47; bits[1]:0x1; bits[7]:0x5e; bits[27]:0x1000000
// args: bits[27]:0x6fb5866; bits[20]:0x64020; bits[1]:0x0; bits[7]:0x9; bits[27]:0x7d4dc93
// args: bits[27]:0x6f1d246; bits[20]:0x6693; bits[1]:0x1; bits[7]:0x51; bits[27]:0x5df8fb4
// args: bits[27]:0x1a1822; bits[20]:0xd0aeb; bits[1]:0x0; bits[7]:0x2; bits[27]:0x53df94e
// args: bits[27]:0x21395bb; bits[20]:0x8360e; bits[1]:0x0; bits[7]:0x2; bits[27]:0x426f236
// args: bits[27]:0x7091f42; bits[20]:0x328cc; bits[1]:0x0; bits[7]:0x4c; bits[27]:0x10
// args: bits[27]:0x2a45366; bits[20]:0xbb060; bits[1]:0x0; bits[7]:0x75; bits[27]:0x7555313
// args: bits[27]:0x0; bits[20]:0x81e49; bits[1]:0x0; bits[7]:0x30; bits[27]:0x535d5ae
// args: bits[27]:0x5b3f4c; bits[20]:0x84979; bits[1]:0x0; bits[7]:0x78; bits[27]:0x4f0aa83
// args: bits[27]:0x52e98c2; bits[20]:0x612db; bits[1]:0x1; bits[7]:0x20; bits[27]:0x3deaa85
// args: bits[27]:0x68cb9f; bits[20]:0x44c77; bits[1]:0x1; bits[7]:0x5e; bits[27]:0x747bf35
// args: bits[27]:0x4b3098c; bits[20]:0x4ff7e; bits[1]:0x1; bits[7]:0x19; bits[27]:0x7cfb12c
// args: bits[27]:0x4; bits[20]:0x20000; bits[1]:0x1; bits[7]:0x23; bits[27]:0x20000
// args: bits[27]:0x2cd31cf; bits[20]:0x4db54; bits[1]:0x1; bits[7]:0x58; bits[27]:0x3686110
// args: bits[27]:0x6356787; bits[20]:0xfca4d; bits[1]:0x0; bits[7]:0x79; bits[27]:0x3b38e43
// args: bits[27]:0x1b861f6; bits[20]:0x250c2; bits[1]:0x0; bits[7]:0x43; bits[27]:0x55c4a38
// args: bits[27]:0x3a62592; bits[20]:0x4b091; bits[1]:0x0; bits[7]:0x9; bits[27]:0x4cbb824
// args: bits[27]:0x38ff5fa; bits[20]:0x3924e; bits[1]:0x1; bits[7]:0x5c; bits[27]:0x1373696
// args: bits[27]:0x16004ec; bits[20]:0x7bc67; bits[1]:0x1; bits[7]:0x58; bits[27]:0xa9eef
// args: bits[27]:0x340f784; bits[20]:0x4f7e0; bits[1]:0x0; bits[7]:0x3; bits[27]:0x143c911
// args: bits[27]:0x3ffffff; bits[20]:0x23112; bits[1]:0x0; bits[7]:0x1; bits[27]:0x481ce92
// args: bits[27]:0x80; bits[20]:0x80321; bits[1]:0x1; bits[7]:0x6c; bits[27]:0x400
// args: bits[27]:0x15749b3; bits[20]:0xd229c; bits[1]:0x0; bits[7]:0x71; bits[27]:0x723308d
// args: bits[27]:0x5494a58; bits[20]:0x69063; bits[1]:0x1; bits[7]:0x0; bits[27]:0x3822e37
// args: bits[27]:0x20000; bits[20]:0x10; bits[1]:0x0; bits[7]:0x10; bits[27]:0x4000
// args: bits[27]:0x61f6aa4; bits[20]:0x65cf4; bits[1]:0x1; bits[7]:0x37; bits[27]:0x179d298
// args: bits[27]:0x4be7389; bits[20]:0x95c28; bits[1]:0x1; bits[7]:0x1b; bits[27]:0x78c88b4
// args: bits[27]:0x2afeb7f; bits[20]:0x7974f; bits[1]:0x1; bits[7]:0x66; bits[27]:0x5d6be67
// args: bits[27]:0x2621511; bits[20]:0x31e55; bits[1]:0x1; bits[7]:0x27; bits[27]:0x49a3f8e
// args: bits[27]:0x1; bits[20]:0xaa15a; bits[1]:0x0; bits[7]:0x5d; bits[27]:0x35881d1
// args: bits[27]:0x12eb28b; bits[20]:0x5fb15; bits[1]:0x0; bits[7]:0x0; bits[27]:0x7dbb276
// args: bits[27]:0x1f2553a; bits[20]:0x51f2c; bits[1]:0x0; bits[7]:0x2b; bits[27]:0x304939a
// args: bits[27]:0x3ce2eb; bits[20]:0xc2e13; bits[1]:0x0; bits[7]:0x31; bits[27]:0x2fb5358
// args: bits[27]:0x361f314; bits[20]:0x0; bits[1]:0x1; bits[7]:0x1e; bits[27]:0x1148e3d
// args: bits[27]:0x3e8a350; bits[20]:0xeb0ac; bits[1]:0x1; bits[7]:0x4c; bits[27]:0x2561e1a
// args: bits[27]:0xf96d92; bits[20]:0x70408; bits[1]:0x1; bits[7]:0x27; bits[27]:0x60ae0f8
// args: bits[27]:0x4d9ee66; bits[20]:0xbfa93; bits[1]:0x1; bits[7]:0x0; bits[27]:0x31b6a7
// args: bits[27]:0x200; bits[20]:0xfc6e6; bits[1]:0x0; bits[7]:0x61; bits[27]:0x7ea413f
// args: bits[27]:0x44a9359; bits[20]:0x8000; bits[1]:0x1; bits[7]:0x39; bits[27]:0x3801037
// args: bits[27]:0x3dfc4a1; bits[20]:0xee92a; bits[1]:0x0; bits[7]:0x1f; bits[27]:0x63cb7db
// args: bits[27]:0x5824ba9; bits[20]:0xccec8; bits[1]:0x1; bits[7]:0x17; bits[27]:0x185bd2f
// args: bits[27]:0x534816c; bits[20]:0xb4c0e; bits[1]:0x1; bits[7]:0x5a; bits[27]:0x65104b
// args: bits[27]:0x53f21e1; bits[20]:0xf01e0; bits[1]:0x0; bits[7]:0x33; bits[27]:0x5532c13
// args: bits[27]:0x5bc2dd4; bits[20]:0x6abe2; bits[1]:0x0; bits[7]:0x3b; bits[27]:0x7746d30
// args: bits[27]:0x1c81b6; bits[20]:0x725f0; bits[1]:0x1; bits[7]:0xd; bits[27]:0x4
// args: bits[27]:0x3d66458; bits[20]:0x523d3; bits[1]:0x0; bits[7]:0x20; bits[27]:0x21c09a7
// args: bits[27]:0x4c0f8c; bits[20]:0x1417; bits[1]:0x1; bits[7]:0x59; bits[27]:0x665e7c1
// args: bits[27]:0x26d664; bits[20]:0x48936; bits[1]:0x0; bits[7]:0x4a; bits[27]:0x87a3bf
// args: bits[27]:0x1f36c86; bits[20]:0x4626b; bits[1]:0x0; bits[7]:0x60; bits[27]:0x113f22
// args: bits[27]:0x24df4e2; bits[20]:0x299c5; bits[1]:0x1; bits[7]:0x2; bits[27]:0x5196af5
// args: bits[27]:0x41444d4; bits[20]:0x2026; bits[1]:0x0; bits[7]:0xa; bits[27]:0x6dd9d
// args: bits[27]:0x4aef726; bits[20]:0xbe670; bits[1]:0x0; bits[7]:0x2; bits[27]:0x5f93f48
// args: bits[27]:0x6576466; bits[20]:0x4a973; bits[1]:0x1; bits[7]:0x15; bits[27]:0x3cf2eb6
fn main(x15127: u27, x15128: u20, x15129: u1, x15130: u7, x15131: u27) -> (u11, u10, u2, u27, u10, u2, u11) {
    let x15132: u2 = (u2:0x2);
    let x15133: u2 = (x15132) + (x15132);
    let x15134: u7 = -(x15130);
    let x15135: u10 = (u10:0x3ff);
    let x15136: u11 = (u11:0x1);
    (x15136, x15135, x15132, x15131, x15135, x15133, x15136)
}


