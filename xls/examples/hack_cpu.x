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

// An incomplete implementation of a the Hack CPU from the https://www.nand2tetris.org/ course.
//
// Implemented:
// - ALU
// - A instructions: `@value`: load value in register A.
//   0bo_aaaaaaaaaaaaaaa
//     ^----------------: opcode signifying A-instruction
//       ^^^^^^^^^^^^^^^: 15bit value to load in register A
//
// - C instructions: `DEST=COMP;JUMP`: perform COMP, assign to DEST
//                    and JUMP conditionally.
//   0boxx_acccccc_ddd_jjj
//     ^------------------: opcode signifying C-instruction
//      ^^----------------: unused
//         ^^^^^^^--------: comp bits
//                 ^^^----: dest bits
//                     ^^^: jump bits
//
// - RAM addressing
// - ROM PC addressing
//
// Limitations:
// - missing screen and keyboard memory based IO
// - smaller RAM and ROM size

fn decode_a_instruction(ins: u16) -> (u15) {
  let value: u15 = ins[0+:u15];
  (value,)
}

#[test]
fn decode_a_test() {
  // @21
  let (value) = decode_a_instruction(u16:0b0_000000000010101);
  assert_eq(value, u15:21);
}

const COMP_0 = u7:0b0101010;
const COMP_1 = u7:0b0111111;
const COMP_MINUS_1 = u7:0b0111010;
const COMP_D = u7:0b0001100;
const COMP_A = u7:0b0110000;
const COMP_M = u7:0b1110000;
const COMP_NOT_D = u7:0b0001101;
const COMP_NOT_A = u7:0b0110001;
const COMP_NOT_M = u7:0b1110001;
const COMP_MINUS_D = u7:0b0001111;
const COMP_MINUS_A = u7:0b0110011;
const COMP_MINUS_M = u7:0b1110011;
const COMP_D_PLUS_1 = u7:0b0011111;
const COMP_A_PLUS_1 = u7:0b0110111;
const COMP_M_PLUS_1 = u7:0b1110111;
const COMP_D_MINUS_1 = u7:0b0001110;
const COMP_A_MINUS_1 = u7:0b0110010;
const COMP_M_MINUS_1 = u7:0b1110010;
const COMP_D_PLUS_A = u7:0b0000010;
const COMP_D_PLUS_M = u7:0b1000010;
const COMP_D_MINUS_A = u7:0b0010011;
const COMP_D_MINUS_M = u7:0b1010011;
const COMP_A_MINUS_D = u7:0b0000111;
const COMP_M_MINUS_D = u7:0b1000111;
const COMP_D_AND_A = u7:0b0000000;
const COMP_D_AND_M = u7:0b1000000;
const COMP_D_OR_A = u7:0b0010101;
const COMP_D_OR_M = u7:0b1010101;

const DEST_NULL = u3:0b000;
const DEST_M = u3:0b001;
const DEST_D = u3:0b010;
const DEST_A = u3:0b100;

const JUMP_NULL = u3:0b000;
const JUMP_JGT = u3:0b001;
const JUMP_JEQ = u3:0b010;
const JUMP_JGE = u3:0b011;
const JUMP_JLT = u3:0b100;
const JUMP_JNE = u3:0b101;
const JUMP_JLE = u3:0b110;
const JUMP_JMP = u3:0b111;

fn decode_c_instruction(ins: u16) -> (u7, u3, u3) {
  let comp: u7 = ins[6+:u7];
  let dest: u3 = ins[3+:u3];
  let jump: u3 = ins[0+:u3];
  (comp, dest, jump)
}

#[test]
fn decode_c_test() {
  // MD=D+1
  let (comp, dest, jump) = decode_c_instruction(u16:0b1110011111011000);
  assert_eq(comp, COMP_D_PLUS_1);
  assert_eq(dest, DEST_M|DEST_D);
  assert_eq(jump, JUMP_NULL);
  // M=1
  let (comp, dest, jump) = decode_c_instruction(u16:0b111_0111111_001_000);
  assert_eq(comp, COMP_1);
  assert_eq(dest, DEST_M);
  assert_eq(jump, JUMP_NULL);
  // D+1;JLE
  let (comp, dest, jump) = decode_c_instruction(u16:0b111_0011111_000_110);
  assert_eq(comp, COMP_D_PLUS_1);
  assert_eq(dest, DEST_NULL);
  assert_eq(jump, JUMP_JLE);
}

fn encode_a_instruction(a: u15) -> u16 {
  u1:0 ++ a
}

#[test]
fn encode_a_test() {
  let value = encode_a_instruction(u15:21);
  assert_eq(value, u16:21);
}

fn encode_c_instruction(comp: u7, dest: u3, jump: u3) -> u16 {
  u3:0b111 ++ comp ++ dest ++ jump
}

#[test]
fn encode_c_test() {
  let value = encode_c_instruction(COMP_D_PLUS_1, DEST_M|DEST_D, JUMP_NULL);
  assert_eq(value, u16:0b1110011111011000);
  let value = encode_c_instruction(COMP_1, DEST_M, JUMP_NULL);
  assert_eq(value, u16:0b111_0111111_001_000);
  let value = encode_c_instruction(COMP_D_PLUS_1, DEST_NULL, JUMP_JLE);
  assert_eq(value, u16:0b111_0011111_000_110);
}

fn run_a_instruction(pc: u16, ins: u16, rd: u16, ra: u16, rm: u16) -> (u16, u16, u16, u16, u1) {
  let (value) = decode_a_instruction(ins);
  let ra' = value as u16;
  let wm = u1:0;
  (pc + u16:1, rd, ra', rm, wm)
}

#[test]
fn run_a_test() {
  // @21
  let (pc, rd, ra, rm, wm) = run_a_instruction(u16:0, encode_a_instruction(u15:21), u16:1, u16:2, u16:3);
  assert_eq(pc, u16:1);
  assert_eq(rd, u16:1);
  assert_eq(ra, u16:21);
  assert_eq(rm, u16:3);
  assert_eq(wm, u1:0);
}

fn z_bit(z: u1, v: u16) -> u16 {
  if z { u16:0 } else { v }
}

#[test]
fn z_bit_test() {
  let z = z_bit(u1:1, u16:42);
  assert_eq(z, u16:0);
  let v = z_bit(u1:0, u16:42);
  assert_eq(v, u16:42);
}

fn n_bit(n: u1, v: u16) -> u16 {
  if n { !v } else { v }
}

#[test]
fn n_bit_test() {
  let n = n_bit(u1:1, u16:42);
  assert_eq(n, !u16:42);
  let v = n_bit(u1:0, u16:42);
  assert_eq(v, u16:42);
}

fn alu(x: u16, y:u16, c:u6) -> (u16, u1, u1) {
  let (zx, nx, zy, ny, f, no) = (c[5+:u1], c[4+:u1], c[3+:u1], c[2+:u1], c[1+:u1], c[0+:u1]);
  let x' = z_bit(zx, x);
  let x'' = n_bit(nx, x');
  let y' = z_bit(zy, y);
  let y'' = n_bit(ny, y');
  let sum: u16 = (x'' + y'');
  let and: u16 = x'' & y'';
  let output: u16 = if f { sum } else { and };
  let output': u16 = if no { !output } else { output };
  let zr = (output' == u16:0);
  let ng = (output' as s16 < s16:0);
  (output', zr, ng)
}

#[test]
fn alu_test() {
  let (output, zr, ng) = alu(u16:2, u16:8, COMP_D_PLUS_1[0:6]);
  assert_eq(output, u16:3);
  assert_eq(zr, u1:0);
  assert_eq(ng, u1:0);
  let (output, zr, ng) = alu(u16:2, u16:8, COMP_D_AND_A[0:6]);
  assert_eq(output, u16:0);
  assert_eq(zr, u1:1);
  assert_eq(ng, u1:0);
  let (output, zr, ng) = alu(u16:8, u16:2, COMP_A_MINUS_D[0:6]);
  assert_eq(output, s16:-6 as u16);
  assert_eq(zr, u1:0);
  assert_eq(ng, u1:1);
}

fn run_c_instruction(pc: u16, ins: u16, rd: u16, ra: u16, rm: u16) -> (u16, u16, u16, u16, u1) {
  let (comp, dest, jump) = decode_c_instruction(ins);
  let x = rd;
  let a = comp[6+:u1];
  let y = if a { rm } else { ra };
  let (output, zr, ng) = alu(x, y, comp[0+:u6]);
  let rd' = if (dest & DEST_D) == DEST_D { output } else { rd };
  let ra' = if (dest & DEST_A) == DEST_A { output } else { ra };
  let (rm', wm) = if (dest & DEST_M) == DEST_M { (output, u1:1) } else { (rm, u1:0) };
  let flags: u3 = ng ++ zr ++ !ng;
  let pc' = if (jump & flags) != u3:0 { ra' } else { pc + u16:1 };
  (pc', rd', ra', rm', wm)
}

#[test]
fn run_c_test() {
  // MD=D+1
  let (pc, rd, ra, rm, _wm) = run_c_instruction(u16:0, encode_c_instruction(u7:0b0011111, u3:0b011, u3:0b000), u16:1, u16:2, u16:3);
  assert_eq(pc, u16:1);
  assert_eq(rd, u16:2);
  assert_eq(ra, u16:2);
  assert_eq(rm, u16:2);
  // D+1;JLE
  let (pc, rd, ra, rm, _wm) = run_c_instruction(u16:0, encode_c_instruction(u7:0b0011111, u3:0b000, u3:0b110), s16:-1 as u16, u16:2, u16:3);
  assert_eq(pc, u16:2);
  assert_eq(rd, s16:-1 as u16);
  assert_eq(ra, u16:2);
  assert_eq(rm, u16:3);
  // 0;JMP
  let (pc, rd, ra, rm, _wm) = run_c_instruction(u16:0, encode_c_instruction(u7:0b0101010, u3:0b000, u3:0b111), u16:1, u16:2, u16:3);
  assert_eq(pc, u16:2);
  assert_eq(rd, u16:1);
  assert_eq(ra, u16:2);
  assert_eq(rm, u16:3);
}


fn cpu(pc: u16, rd: u16, ra: u16, ram: u16[32], rom: u16[32]) -> (u16, u16, u16, u16[32])  {
  let ins = rom[pc];
  let rm = ram[ra];
  let (pc', rd', ra', rm', wm) = match ins[15+:u1] {
    u1:0 => run_a_instruction(pc, ins, rd, ra, rm),
    u1:1 => run_c_instruction(pc, ins, rd, ra, rm),
  };
  (pc', rd', ra', if wm { update(ram, ra', rm') } else { ram })
}

#[test]
fn run_cpu() {
  let rom = u16[32]:[
    encode_a_instruction(u15:16),                            // 00 @16
    encode_c_instruction(COMP_1, DEST_M, JUMP_NULL),         // 01 M=1
    encode_a_instruction(u15:17),                            // 02 @17
    encode_c_instruction(COMP_0, DEST_M, JUMP_NULL),         // 03 M=0
    encode_a_instruction(u15:16),                            // 04 @16
    encode_c_instruction(COMP_M, DEST_D, JUMP_NULL),         // 05 D=M
    encode_a_instruction(u15:0),                             // 06 @0
    encode_c_instruction(COMP_M_MINUS_D, DEST_D, JUMP_NULL), // 07 D=M-D
    encode_a_instruction(u15:18),                            // 08 @18
    encode_c_instruction(COMP_D, DEST_NULL, JUMP_JLT),       // 09 D;JLT
    encode_a_instruction(u15:16),                            // 10 @16
    encode_c_instruction(COMP_M, DEST_D, JUMP_NULL),         // 11 D=M
    encode_a_instruction(u15:17),                            // 12 @17
    encode_c_instruction(COMP_D_PLUS_M, DEST_M, JUMP_NULL),  // 13 M=D+M
    encode_a_instruction(u15:16),                            // 14 @16
    encode_c_instruction(COMP_M_PLUS_1, DEST_M, JUMP_NULL),  // 15 M=M+1
    encode_a_instruction(u15:4),                             // 16 @4
    encode_c_instruction(COMP_0, DEST_NULL, u3:0b111),       // 17 0;JMP
    encode_a_instruction(u15:17),                            // 18 @17
    encode_c_instruction(COMP_M, DEST_D, JUMP_NULL),         // 19 D=M
    encode_a_instruction(u15:1),                             // 20 @1
    encode_c_instruction(COMP_D, DEST_M, JUMP_NULL),         // 21 M=D
    encode_a_instruction(u15:22),                            // 22 @22
    encode_c_instruction(COMP_0, DEST_NULL, u3:0b111),       // 23 0;JMP
    u16:0, ...
  ];
  let pc = u16:0;
  let ra = u16:0;
  let rd = u16:0;
  let ram = u16[32]:[u16:4, 0, ...];
  let (pc'', _rd, _ra, ram'') = for (_, (pc', rd', ra', ram')): (u32, (u16, u16, u16, u16[32])) in range(u32:0, u32:100) {
    cpu(pc', rd', ra', ram', rom)
  }((pc, rd, ra, ram));
  assert_eq(ram''[u16:1], u16:10);
  assert_eq(pc'', u16:22);
}
