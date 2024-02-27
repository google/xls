// Copyright 2022 The XLS Authors
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

import std;

// TTU registers.
const T0 = u2:0;
const T1 = u2:1;
const T2 = u2:2;
const T3 = u2:3;

// Instructions.
const LOAD = u8:0;
const STORE = u8:1;
const MUL = u8:2;
const ADD = u8:3;
const END = u8:255;

// Regular ALU registers.
const R0 = u2:0;
const R1 = u2:1;
const R2 = u2:2;
const R3 = u2:3;

pub struct LoopBundle {
    start: u32,
    end: u32,
    stride: u16,
}

pub struct TtuBundle<NumBundles: u32> {
    loops: LoopBundle[NumBundles],
    ttu_id: u2,
    register: u2,
}

pub struct TtuState<NumRegisters: u32> {
    regs: u2[NumRegisters],
    pc: u16,
}

pub struct Register<Bits: u32> {
    is_ttu_reg: u1,
    reg: bits[Bits],
}

pub struct Instruction {
    opcode: u8,
    src_reg0: Register<2>,
    src_reg1: Register<2>,
    dst_reg: Register<2>,
}

#[test]
fn ttu_test() {
    let loop = LoopBundle {start:u32: 0, end: u32: 10, stride: u16:1};
    assert_eq(loop.start, u32:0)
}
