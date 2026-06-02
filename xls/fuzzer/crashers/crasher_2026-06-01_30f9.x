// Copyright 2026 The XLS Authors
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
// # proto-message: xls.fuzzer.CrasherConfigurationProto
// exception: "/xls/tools/codegen_main returned a non-zero exit status (1): /xls/tools/codegen_main --output_signature_path=module_sig.textproto --delay_model=unit --nouse_system_verilog --output_block_ir_path=sample.block.ir --generator=pipeline --pipeline_stages=3 --worst_case_throughput=1 --reset=rst --reset_active_low=false --reset_asynchronous=true --reset_data_path=true sample.opt.ir --logtostderr\n\nINVALID_ARGUMENT: Impossible to schedule proc __sample__main_0_next as specified with clock period 5__sample__main_0_next: cannot achieve full throughput."
// issue: "none"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=3"
//   codegen_args: "--worst_case_throughput=1"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=true"
//   codegen_args: "--reset_data_path=true"
//   simulate: true
//   simulator: "iverilog"
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve full throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: true
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x6"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x9_387e"
//       values: "bits[21]:0x17_f23c"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0x2000"
//       values: "bits[21]:0x4"
//       values: "bits[21]:0x6_7600"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x4_0000"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0xa_aaaa"
//       values: "bits[21]:0xb_5193"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x3_5162"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x1e_7ac2"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0xa_aaaa"
//       values: "bits[21]:0xa_aaaa"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0xa_aaaa"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x4000"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0xa_aaaa"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x4_0000"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x1e_3b7f"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0x1"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x40"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0xa_aaaa"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x100"
//       values: "bits[21]:0x16_6eb5"
//       values: "bits[21]:0x4_9c1e"
//       values: "bits[21]:0x1b_5d09"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x100"
//       values: "bits[21]:0x1e_90e8"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0xf_5b05"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x1_0000"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x10_3879"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x1_0000"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0xa_aaaa"
//       values: "bits[21]:0xf_9ef7"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x10_8179"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0xa_015b"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0x4_12cd"
//       values: "bits[21]:0x1b_d940"
//       values: "bits[21]:0x2000"
//       values: "bits[21]:0xa_aaaa"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0xa_aaaa"
//       values: "bits[21]:0xf_399a"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x2000"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0xd_7572"
//       values: "bits[21]:0xa_aaaa"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0x1f_ffff"
//       values: "bits[21]:0xa_aaaa"
//       values: "bits[21]:0x7_5a63"
//       values: "bits[21]:0x15_5555"
//       values: "bits[21]:0x0"
//       values: "bits[21]:0xf_ffff"
//       values: "bits[21]:0x1f_ffff"
//     }
//     inputs {
//       channel_name: "sample__x20"
//       values: "(bits[16]:0x7fff)"
//       values: "(bits[16]:0x7fff)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0x38ee)"
//       values: "(bits[16]:0x2633)"
//       values: "(bits[16]:0x5450)"
//       values: "(bits[16]:0x2201)"
//       values: "(bits[16]:0x800)"
//       values: "(bits[16]:0x7600)"
//       values: "(bits[16]:0x80)"
//       values: "(bits[16]:0x1313)"
//       values: "(bits[16]:0x97c0)"
//       values: "(bits[16]:0x7fff)"
//       values: "(bits[16]:0x8000)"
//       values: "(bits[16]:0xffff)"
//       values: "(bits[16]:0x59c5)"
//       values: "(bits[16]:0x5162)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0x7cfb)"
//       values: "(bits[16]:0xffbe)"
//       values: "(bits[16]:0xffff)"
//       values: "(bits[16]:0x80)"
//       values: "(bits[16]:0x6595)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0x7e71)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0xbf9e)"
//       values: "(bits[16]:0x0)"
//       values: "(bits[16]:0x28a8)"
//       values: "(bits[16]:0x26e2)"
//       values: "(bits[16]:0x5f9f)"
//       values: "(bits[16]:0x100)"
//       values: "(bits[16]:0x0)"
//       values: "(bits[16]:0x8595)"
//       values: "(bits[16]:0xaeaa)"
//       values: "(bits[16]:0x95d5)"
//       values: "(bits[16]:0x4818)"
//       values: "(bits[16]:0x6480)"
//       values: "(bits[16]:0x3)"
//       values: "(bits[16]:0xfffa)"
//       values: "(bits[16]:0xeef6)"
//       values: "(bits[16]:0x68be)"
//       values: "(bits[16]:0x0)"
//       values: "(bits[16]:0x0)"
//       values: "(bits[16]:0xbfff)"
//       values: "(bits[16]:0xaaaa)"
//       values: "(bits[16]:0xdefb)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0x0)"
//       values: "(bits[16]:0xfeff)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0xcdba)"
//       values: "(bits[16]:0x3898)"
//       values: "(bits[16]:0x80)"
//       values: "(bits[16]:0xfeff)"
//       values: "(bits[16]:0xaaaa)"
//       values: "(bits[16]:0x5d7d)"
//       values: "(bits[16]:0x41)"
//       values: "(bits[16]:0x5d10)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0x41f6)"
//       values: "(bits[16]:0xffff)"
//       values: "(bits[16]:0xffff)"
//       values: "(bits[16]:0xffff)"
//       values: "(bits[16]:0xc3d5)"
//       values: "(bits[16]:0x4000)"
//       values: "(bits[16]:0x5c07)"
//       values: "(bits[16]:0xffff)"
//       values: "(bits[16]:0xffff)"
//       values: "(bits[16]:0x110)"
//       values: "(bits[16]:0xce2f)"
//       values: "(bits[16]:0x981e)"
//       values: "(bits[16]:0x5d31)"
//       values: "(bits[16]:0x2200)"
//       values: "(bits[16]:0x8)"
//       values: "(bits[16]:0x0)"
//       values: "(bits[16]:0xfefd)"
//       values: "(bits[16]:0x5c54)"
//       values: "(bits[16]:0xafcf)"
//       values: "(bits[16]:0x1000)"
//       values: "(bits[16]:0x2ff3)"
//       values: "(bits[16]:0x9bd2)"
//       values: "(bits[16]:0x2f8c)"
//       values: "(bits[16]:0x1000)"
//       values: "(bits[16]:0xf7ff)"
//       values: "(bits[16]:0xffff)"
//       values: "(bits[16]:0x3879)"
//       values: "(bits[16]:0x0)"
//       values: "(bits[16]:0x10)"
//       values: "(bits[16]:0x1000)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0xaaaa)"
//       values: "(bits[16]:0x9b33)"
//       values: "(bits[16]:0xaaaa)"
//       values: "(bits[16]:0x400)"
//       values: "(bits[16]:0xb1e1)"
//       values: "(bits[16]:0x80)"
//       values: "(bits[16]:0x5634)"
//       values: "(bits[16]:0x7fff)"
//       values: "(bits[16]:0x2a10)"
//       values: "(bits[16]:0x7fff)"
//       values: "(bits[16]:0x31f)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0xaaaa)"
//       values: "(bits[16]:0x2054)"
//       values: "(bits[16]:0x0)"
//       values: "(bits[16]:0x0)"
//       values: "(bits[16]:0xb16b)"
//       values: "(bits[16]:0x0)"
//       values: "(bits[16]:0x87de)"
//       values: "(bits[16]:0x8)"
//       values: "(bits[16]:0x1)"
//       values: "(bits[16]:0x5555)"
//       values: "(bits[16]:0xd7fe)"
//       values: "(bits[16]:0x9800)"
//       values: "(bits[16]:0x4745)"
//       values: "(bits[16]:0xaaaa)"
//       values: "(bits[16]:0xffff)"
//       values: "(bits[16]:0xffff)"
//       values: "(bits[16]:0x7fdb)"
//       values: "(bits[16]:0xcaf0)"
//       values: "(bits[16]:0x7a61)"
//       values: "(bits[16]:0x800)"
//       values: "(bits[16]:0x805)"
//       values: "(bits[16]:0xffff)"
//       values: "(bits[16]:0xc0e3)"
//     }
//   }
// }
// 
// END_CONFIG
proc main {
    x1: chan<u18> out;
    x6: chan<u21> in;
    x20: chan<(u16,)> in;
    config(x1: chan<u18> out, x6: chan<u21> in, x20: chan<(u16,)> in) {
        (x1, x6, x20)
    }
    init {
        u18:262143
    }
    next(x0: u18) {
        {
            let x2: token = send(join(), x1, x0);
            let x3: u18 = x0 << x0;
            let x4: u35 = u35:0x2_aaaa_aaaa;
            let x5: u35 = x4[x4+:u35];
            let x7: (token, u21) = recv(x2, x6);
            let x8: token = x7.0;
            let x9: u21 = x7.1;
            let x10: (u6, u14) = match x3 {
                u18:32768 => (u6:0x3f, u14:0x1555),
                _ => (u6:0x3f, u14:0x400),
            };
            let x11: token = x7.0;
            let x12: bool = or_reduce(x9);
            let x13: bool = xor_reduce(x12);
            let x15: u35 = rev(x5);
            let x17: u35 = {
                let x16: (xN[bool:0x0][35], xN[bool:0x0][35]) = umulp(x13 as u35, x4);
                x16.0 + x16.1
            };
            let x18: u21 = x3 as u21 & x9;
            let x19: u18 = u18:0x0;
            let x21: (token, (u16,)) = recv(x2, x20);
            let x22: token = x21.0;
            let x23: (u16,) = x21.1;
            let x24: u35 = x4 ^ x18 as u35;
            let x25: xN[bool:0x0][42] = x9 ++ x18;
            let x26: bool = x18 == x4 as u21;
            let x27: (u42, u12, u64) = match x0 {
                u18:0x3_ffff..=u18:0x0 | u18:0b10_1010_1010_1010_1010 => (u42:0x0, u12:0x0, u64:0x0),
                u18:0x2_0000 | u18:0x1_ffff..u18:173894 => (u42:0x3ff_ffff_ffff, u12:0xaa3, u64:0x7fff_ffff_ffff_ffff),
                u18:0x1_ffff..u18:261172 | u18:0x0 => (u42:0x2aa_aaaa_aaaa, u12:0xaec, u64:0x13b6_08db_4efd_733c),
                u18:0x3_2c11 | u18:0b1_0101_0101_0101_0101 => (u42:0xfb_107c_55a7, u12:0x7ff, u64:0xaaaa_aaaa_aaaa_aaaa),
                _ => (u42:0x155_5555_5555, xN[bool:0x0][12]:0xac8, u64:1073741824),
            };
            let x28: u35 = x5[x0+:u35];
            let x29: bool = x13 as bool | x26;
            let x30: bool = one_hot_sel(x12, [x29]);
            let x31: u60 = u60:0xdcf_d37c_3d22_1c8d;
            let x32: u35 = signex(x28, x24);
            let x33: bool = x10 == x10;
            let x34: u24 = u24:0x55_5555;
            let x35: u18 = bit_slice_update(x0, x18, x30);
            let x36: u11 = x5[24+:u11];
            let x37: bool = x30[x3+:bool];
            let x38: bool = or_reduce(x5);
            let x39: bool = or_reduce(x9);
            let x40: u35 = -x28;
            let x41: u35 = x12 as u35 & x4;
            x35
        }
    }
}
