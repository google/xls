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

// DSLX 4-path nested ALU example from the XLS Symbolic Execution Proposal.

pub enum Opcode : u2 {
    ADD = 0,
    AND = 1,
    INVALID = 2,
}

pub enum Status : u2 {
    OK = 0,
    OVERFLOW = 1,
    INVALID_OP = 2,
}

pub fn execute_alu(op: Opcode, a: u8, b: u8) -> (Status, u8) {
    if op == Opcode::ADD {
        let sum = (a as u9) + (b as u9);
        if sum > u9:255 {
            (Status::OVERFLOW, u8:0)
        } else {
            (Status::OK, sum as u8)
        }
    } else if op == Opcode::AND {
        (Status::OK, a & b)
    } else {
        (Status::INVALID_OP, u8:0)
    }
}
