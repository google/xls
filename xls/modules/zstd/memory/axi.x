// Copyright 2024 The XLS Authors
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

pub enum AxiAxSize : u3 {
    MAX_1B_TRANSFER = 0,
    MAX_2B_TRANSFER = 1,
    MAX_4B_TRANSFER = 2,
    MAX_8B_TRANSFER = 3,
    MAX_16B_TRANSFER = 4,
    MAX_32B_TRANSFER = 5,
    MAX_64B_TRANSFER = 6,
    MAX_128B_TRANSFER = 7,
}

pub enum AxiWriteResp : u3 {
    OKAY = 0,
    EXOKAY = 1,
    SLVERR = 2,
    DECERR = 3,
    DEFER = 4,
    TRANSFAULT = 5,
    RESERVED = 6,
    UNSUPPORTED = 7,
}

pub enum AxiReadResp : u3 {
    OKAY = 0,
    EXOKAY = 1,
    SLVERR = 2,
    DECERR = 3,
    PREFETCHED = 4,
    TRANSFAULT = 5,
    OKAYDIRTY = 6,
    RESERVED = 7,
}

pub enum AxiAxBurst : u2 {
    FIXED = 0,
    INCR = 1,
    WRAP = 2,
    RESERVED = 3,
}

pub enum AxiAwCache : u4 {
    DEV_NO_BUF = 0b0000,
    DEV_BUF = 0b0001,
    NON_C_NON_BUF = 0b0010,
    NON_C_BUF = 0b0011,
    WT_NO_ALLOC = 0b0110,
    WT_RD_ALLOC = 0b0110,
    WT_WR_ALLOC = 0b1110,
    WT_ALLOC = 0b1110,
    WB_NO_ALLOC = 0b0111,
    WB_RD_ALLOC = 0b0111,
    WB_WR_ALLOC = 0b1111,
    WB_ALLOC = 0b1111,
}

pub enum AxiArCache : u4 {
    DEV_NO_BUF = 0b0000,
    DEV_BUF = 0b0001,
    NON_C_NON_BUF = 0b0010,
    NON_C_BUF = 0b0011,
    WT_NO_ALLOC = 0b1010,
    WT_RD_ALLOC = 0b1110,
    WT_WR_ALLOC = 0b1010,
    WT_ALLOC = 0b1110,
    WB_NO_ALLOC = 0b1011,
    WB_RD_ALLOC = 0b1111,
    WB_WR_ALLOC = 0b1011,
    WB_ALLOC = 0b1111,
}

pub struct AxiAw<ADDR_W: u32, ID_W: u32> {
    id: uN[ID_W],
    addr: uN[ADDR_W],
    size: AxiAxSize,
    len: u8,
    burst: AxiAxBurst,
}

pub struct AxiW<DATA_W: u32, STRB_W: u32> {
    data: uN[DATA_W],
    strb: uN[STRB_W],
    last: u1
}

pub struct AxiB<ID_W: u32> {
    resp: AxiWriteResp,
    id: uN[ID_W]
}

pub struct AxiAr<ADDR_W: u32, ID_W: u32> {
    id: uN[ID_W],
    addr: uN[ADDR_W],
    region: u4,
    len: u8,
    size: AxiAxSize,
    burst: AxiAxBurst,
    cache: AxiArCache,
    prot: u3,
    qos: u4,
}

pub struct AxiR<DATA_W: u32, ID_W: u32> {
    id: uN[ID_W],
    data: uN[DATA_W],
    resp: AxiReadResp,
    last: u1,
}
