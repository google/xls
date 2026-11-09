// Copyright 2023-2024 The XLS Authors
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

// AXI Package

import std;

pub enum AXI_AXSIZE_ENCODING : u3 {
    MAX_1B_TRANSFER = 0,
    MAX_2B_TRANSFER = 1,
    MAX_4B_TRANSFER = 2,
    MAX_8B_TRANSFER = 3,
    MAX_16B_TRANSFER = 4,
    MAX_32B_TRANSFER = 5,
    MAX_64B_TRANSFER = 6,
    MAX_128B_TRANSFER = 7,
}

pub enum AXI_WRITE_RESPONSE_CODES : u3 {
    OKAY = 0,
    EXOKAY = 1,
    SLVERR = 2,
    DECERR = 3,
    DEFER = 4,
    TRANSFAULT = 5,
    RESERVED = 6,
    UNSUPPORTED = 7,
}

pub enum AXI_READ_RESPONSE_CODES : u3 {
    OKAY = 0,
    EXOKAY = 1,
    SLVERR = 2,
    DECERR = 3,
    PREFETCHED = 4,
    TRANSFAULT = 5,
    OKAYDIRTY = 6,
    RESERVED = 7,
}

pub enum AXI_AXBURST_ENCODING : u2 {
    FIXED = 0,
    INCR = 1,
    WRAP = 2,
    RESERVED = 3,
}

pub enum AXI_AWCACHE_ENCODING : u4 {
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

pub enum AXI_ARCACHE_ENCODING : u4 {
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

pub struct AxiAwBundle<ADDR_W: u32, ID_W: u32> {
    awid: uN[ID_W],
    awaddr: uN[ADDR_W],
    awsize: AXI_AXSIZE_ENCODING,
    awlen: uN[8],
    awburst: AXI_AXBURST_ENCODING,
}

pub struct AxiWBundle<DATA_W: u32, STRB_W: u32> { wdata: uN[DATA_W], wstrb: uN[STRB_W], wlast: u1 }

pub struct AxiBBundle<ID_W: u32> { bresp: AXI_WRITE_RESPONSE_CODES, bid: uN[ID_W] }

pub fn simpleAxiAwBundle<ADDR_W: u32, ID_W: u32>(addr: uN[ADDR_W], id: uN[ID_W]) -> AxiAwBundle {
    AxiAwBundle {
        awid: id,
        awaddr: addr,
        awsize: AXI_AXSIZE_ENCODING::MAX_8B_TRANSFER,
        awlen: uN[8]:0,
        awburst: AXI_AXBURST_ENCODING::FIXED
    }
}

// TODO: #984, zero! does not work with parametric types
pub fn zeroAxiAwBundle<ADDR_W: u32, ID_W: u32>() -> AxiAwBundle {
    AxiAwBundle {
        awid: uN[ID_W]:0,
        awaddr: uN[ADDR_W]:0,
        awsize: AXI_AXSIZE_ENCODING::MAX_8B_TRANSFER,
        awlen: u8:0,
        awburst: AXI_AXBURST_ENCODING::FIXED
    }
}

pub fn simpleAxiWBundle<DATA_W: u32, STRB_W: u32>(data: uN[DATA_W]) -> AxiWBundle {
    let strb = std::unsigned_max_value<STRB_W>();
    AxiWBundle { wdata: data, wstrb: strb, wlast: u1:0 }
}

// TODO: #984, zero! does not work with parametric types
pub fn zeroAxiWBundle<DATA_W: u32, STRB_W: u32>() -> AxiWBundle {
    AxiWBundle { wdata: uN[DATA_W]:0, wstrb: uN[STRB_W]:0, wlast: u1:0 }
}

pub fn simpleAxiBBundle<ID_W: u32>() -> AxiBBundle {
    AxiBBundle { bresp: AXI_WRITE_RESPONSE_CODES::OKAY, bid: uN[ID_W]:0 }
}

// TODO: #984, zero! does not work with parametric types
pub fn zeroAxiBBundle<ID_W: u32>() -> AxiBBundle {
    AxiBBundle { bresp: AXI_WRITE_RESPONSE_CODES::OKAY, bid: uN[ID_W]:0 }
}

pub struct AxiArBundle<ADDR_W: u32, ID_W: u32> {
    arid: uN[ID_W],
    araddr: uN[ADDR_W],
    arregion: uN[4],
    arlen: uN[8],
    arsize: AXI_AXSIZE_ENCODING,
    arburst: AXI_AXBURST_ENCODING,
    arcache: AXI_ARCACHE_ENCODING,
    arprot: uN[3],
    arqos: uN[4],
}

pub struct AxiRBundle<DATA_W: u32, ID_W: u32> {
    rid: uN[ID_W],
    rdata: uN[DATA_W],
    rresp: AXI_READ_RESPONSE_CODES,
    rlast: uN[1],
}

pub fn simpleAxiArBundle<ADDR_W: u32, ID_W: u32>
    (addr: uN[ADDR_W], id: uN[ID_W], arlen: u8) -> AxiArBundle {
    AxiArBundle {
        arid: id,
        araddr: addr,
        arregion: uN[4]:0,
        arlen,
        arsize: AXI_AXSIZE_ENCODING::MAX_1B_TRANSFER,
        arburst: AXI_AXBURST_ENCODING::FIXED,
        arcache: AXI_ARCACHE_ENCODING::DEV_NO_BUF,
        arprot: uN[3]:0,
        arqos: uN[4]:0
    }
}

// TODO: #984, zero! does not work with parametric types
pub fn zeroAxiArBundle<ADDR_W: u32, ID_W: u32>() -> AxiArBundle {
    AxiArBundle {
        arid: uN[ID_W]:0,
        araddr: uN[ADDR_W]:0,
        arregion: uN[4]:0,
        arlen: uN[8]:0,
        arsize: AXI_AXSIZE_ENCODING::MAX_1B_TRANSFER,
        arburst: AXI_AXBURST_ENCODING::FIXED,
        arcache: AXI_ARCACHE_ENCODING::DEV_NO_BUF,
        arprot: uN[3]:0,
        arqos: uN[4]:0
    }
}

pub fn simpleAxiRBundle<DATA_W: u32, ID_W: u32>(data: uN[DATA_W], id: uN[ID_W]) -> AxiRBundle {
    AxiRBundle { rid: id, rdata: data, rresp: AXI_READ_RESPONSE_CODES::OKAY, rlast: uN[1]:0 }
}

pub fn zeroAxiRBundle<DATA_W: u32, ID_W: u32>() -> AxiRBundle {
    AxiRBundle {
        rid: uN[ID_W]:0, rdata: uN[DATA_W]:0, rresp: AXI_READ_RESPONSE_CODES::OKAY, rlast: uN[1]:0
    }
}
