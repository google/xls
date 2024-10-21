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

// This file includes various helpers used in the implementation
// of AxiReader and AxiWriter.

import std;

import xls.modules.zstd.memory.axi;

// Returns a value rounded down to a multiple of ALIGN
pub fn align<ALIGN: u32, N:u32, LOG_ALIGN:u32 = {std::clog2(ALIGN)}>(x: uN[N]) -> uN[N] {
    x & !(all_ones!<uN[LOG_ALIGN]>() as uN[N])
}

#[test]
fn test_align() {
    assert_eq(align<u32:1>(u32:0x1000), u32:0x1000);
    assert_eq(align<u32:1>(u32:0x1001), u32:0x1001);

    assert_eq(align<u32:2>(u32:0x1000), u32:0x1000);
    assert_eq(align<u32:2>(u32:0x1001), u32:0x1000);
    assert_eq(align<u32:2>(u32:0x1002), u32:0x1002);

    assert_eq(align<u32:4>(u32:0x1000), u32:0x1000);
    assert_eq(align<u32:4>(u32:0x1001), u32:0x1000);
    assert_eq(align<u32:4>(u32:0x1002), u32:0x1000);
    assert_eq(align<u32:4>(u32:0x1003), u32:0x1000);
    assert_eq(align<u32:4>(u32:0x1004), u32:0x1004);

    assert_eq(align<u32:8>(u32:0x1000), u32:0x1000);
    assert_eq(align<u32:8>(u32:0x1001), u32:0x1000);
    assert_eq(align<u32:8>(u32:0x1002), u32:0x1000);
    assert_eq(align<u32:8>(u32:0x1003), u32:0x1000);
    assert_eq(align<u32:8>(u32:0x1004), u32:0x1000);
    assert_eq(align<u32:8>(u32:0x1005), u32:0x1000);
    assert_eq(align<u32:8>(u32:0x1006), u32:0x1000);
    assert_eq(align<u32:8>(u32:0x1007), u32:0x1000);
    assert_eq(align<u32:8>(u32:0x1008), u32:0x1008);
}

// "Returns the remainder left after aligning the value to ALIGN
pub fn offset<ALIGN: u32, N:u32, LOG_ALIGN:u32 = {std::clog2(ALIGN)}>(x: uN[N]) -> uN[LOG_ALIGN] {
    type Offset = uN[LOG_ALIGN];
    checked_cast<Offset>(x & (all_ones!<uN[LOG_ALIGN]>() as uN[N]))
}

#[test]
fn test_offset() {
    const OFFSET_W = std::clog2(u32:1);
    type Offset = uN[OFFSET_W];

    assert_eq(offset<u32:1>(u32:0x1000), Offset:0x0);
    assert_eq(offset<u32:1>(u32:0x1001), Offset:0x0);

    const OFFSET_W = std::clog2(u32:2);
    type Offset = uN[OFFSET_W];

    assert_eq(offset<u32:2>(u32:0x1000), Offset:0x0);
    assert_eq(offset<u32:2>(u32:0x1001), Offset:0x1);
    assert_eq(offset<u32:2>(u32:0x1002), Offset:0x0);

    const OFFSET_W = std::clog2(u32:4);
    type Offset = uN[OFFSET_W];

    assert_eq(offset<u32:4>(u32:0x1000), Offset:0x0);
    assert_eq(offset<u32:4>(u32:0x1001), Offset:0x1);
    assert_eq(offset<u32:4>(u32:0x1002), Offset:0x2);
    assert_eq(offset<u32:4>(u32:0x1003), Offset:0x3);
    assert_eq(offset<u32:4>(u32:0x1004), Offset:0x0);

    const OFFSET_W = std::clog2(u32:8);
    type Offset = uN[OFFSET_W];

    assert_eq(offset<u32:8>(u32:0x1000), Offset:0x0);
    assert_eq(offset<u32:8>(u32:0x1001), Offset:0x1);
    assert_eq(offset<u32:8>(u32:0x1002), Offset:0x2);
    assert_eq(offset<u32:8>(u32:0x1003), Offset:0x3);
    assert_eq(offset<u32:8>(u32:0x1004), Offset:0x4);
    assert_eq(offset<u32:8>(u32:0x1005), Offset:0x5);
    assert_eq(offset<u32:8>(u32:0x1006), Offset:0x6);
    assert_eq(offset<u32:8>(u32:0x1007), Offset:0x7);
    assert_eq(offset<u32:8>(u32:0x1008), Offset:0x0);
}

// Returns a tuple representing the byte lanes used in a transaction.
// The first value indicates the starting byte lane of the data bus that holds
// valid data in the initial transaction, while the second value identifies
// the last byte lane containing valid data in the entire transfer.
pub fn get_lanes<
    DATA_W_DIV8: u32,
    ADDR_W: u32,
    LANE_W: u32 = {std::clog2(DATA_W_DIV8)}
>(addr: uN[ADDR_W], len: uN[ADDR_W]) -> (uN[LANE_W], uN[LANE_W]) {
    type Lane = uN[LANE_W];

    let low_lane = checked_cast<Lane>(offset<DATA_W_DIV8>(addr));
    let len_mod = checked_cast<Lane>(std::mod_pow2(len, DATA_W_DIV8 as uN[ADDR_W]));
    const MAX_LANE = std::unsigned_max_value<LANE_W>();

    let high_lane = low_lane + len_mod + MAX_LANE;
    (low_lane, high_lane)
}

#[test]
fn test_get_lanes() {
    const DATA_W_DIV8 = u32:32 / u32:8;
    const ADDR_W = u32:16;
    const LANE_W = std::clog2(DATA_W_DIV8);

    type Addr = uN[ADDR_W];
    type Length = uN[ADDR_W];
    type Lane = uN[LANE_W];

    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x0, Length:0x1), (Lane:0, Lane:0));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x0, Length:0x2), (Lane:0, Lane:1));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x0, Length:0x3), (Lane:0, Lane:2));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x0, Length:0x4), (Lane:0, Lane:3));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x0, Length:0x5), (Lane:0, Lane:0));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x0, Length:0x6), (Lane:0, Lane:1));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x0, Length:0x7), (Lane:0, Lane:2));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x0, Length:0x8), (Lane:0, Lane:3));

    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x1, Length:0x1), (Lane:1, Lane:1));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x1, Length:0x2), (Lane:1, Lane:2));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x1, Length:0x3), (Lane:1, Lane:3));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x1, Length:0x4), (Lane:1, Lane:0));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x1, Length:0x5), (Lane:1, Lane:1));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x1, Length:0x6), (Lane:1, Lane:2));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x1, Length:0x7), (Lane:1, Lane:3));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0x1, Length:0x8), (Lane:1, Lane:0));

    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0xFFE, Length:0x1), (Lane:2, Lane:2));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0xFFE, Length:0x2), (Lane:2, Lane:3));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0xFFE, Length:0x3), (Lane:2, Lane:0));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0xFFE, Length:0x4), (Lane:2, Lane:1));

    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0xFFE, Length:0xFFE), (Lane:2, Lane:3));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0xFFE, Length:0xFFF), (Lane:2, Lane:0));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0xFFE, Length:0x1000), (Lane:2, Lane:1));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0xFFE, Length:0x1001), (Lane:2, Lane:2));
    assert_eq(get_lanes<DATA_W_DIV8>(Addr:0xFFE, Length:0x1002), (Lane:2, Lane:3));
}

// Returns a mask for the data transfer, used in the tkeep and tstrb signals
// in the AXI Stream. The mask sets ones for byte lanes greater than or equal
// to the low byte lane and less than or equal to the high byte lane.
pub fn lane_mask<
    DATA_W_DIV8: u32,
    LANE_W: u32 = {std::clog2(DATA_W_DIV8)},
    ITER_W: u32 = {std::clog2(DATA_W_DIV8) + u32:1}
>(low_lane: uN[LANE_W], high_lane: uN[LANE_W]) -> uN[DATA_W_DIV8] {

    type Iter = uN[ITER_W];
    const ITER_MAX = Iter:1 << LANE_W;

    type Mask = uN[DATA_W_DIV8];

    let low_mask = for (i, mask) in Iter:0..ITER_MAX {
        if i >= low_lane as Iter {
            mask | Mask:0x1 << i
        } else { mask }
    }(uN[DATA_W_DIV8]:0);

    let high_mask = for (i, mask) in Iter:0..ITER_MAX {
        if i <= high_lane as Iter {
            mask | Mask:0x1 << i
        } else { mask }
    }(uN[DATA_W_DIV8]:0);

    low_mask & high_mask
}

#[test]
fn test_lane_mask() {
    const DATA_W_DIV8 = u32:32/u32:8;
    const LANE_W = std::clog2(DATA_W_DIV8);

    type Mask = uN[DATA_W_DIV8];
    type Lane = uN[LANE_W];

    assert_eq(lane_mask<DATA_W_DIV8>(Lane:0, Lane:0), Mask:0b0001);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:0, Lane:1), Mask:0b0011);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:0, Lane:2), Mask:0b0111);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:0, Lane:3), Mask:0b1111);

    assert_eq(lane_mask<DATA_W_DIV8>(Lane:1, Lane:0), Mask:0b0000);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:1, Lane:1), Mask:0b0010);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:1, Lane:2), Mask:0b0110);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:1, Lane:3), Mask:0b1110);

    assert_eq(lane_mask<DATA_W_DIV8>(Lane:2, Lane:0), Mask:0b0000);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:2, Lane:1), Mask:0b0000);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:2, Lane:2), Mask:0b0100);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:2, Lane:3), Mask:0b1100);

    assert_eq(lane_mask<DATA_W_DIV8>(Lane:3, Lane:0), Mask:0b0000);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:3, Lane:1), Mask:0b0000);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:3, Lane:2), Mask:0b0000);
    assert_eq(lane_mask<DATA_W_DIV8>(Lane:3, Lane:3), Mask:0b1000);
}

// Returns the number of bits remaining to 4kB
pub fn bytes_to_4k_boundary<ADDR_W: u32>(addr: uN[ADDR_W]) -> uN[ADDR_W] {
    const AXI_4K_BOUNDARY = uN[ADDR_W]:0x1000;
    AXI_4K_BOUNDARY - std::mod_pow2(addr, AXI_4K_BOUNDARY)
}

#[test]
fn test_bytes_to_4k_boundary() {
    assert_eq(bytes_to_4k_boundary(u32:0x0), u32:0x1000);
    assert_eq(bytes_to_4k_boundary(u32:0x1), u32:0xFFF);
    assert_eq(bytes_to_4k_boundary(u32:0xFFF), u32:0x1);
    assert_eq(bytes_to_4k_boundary(u32:0x1000), u32:0x1000);
    assert_eq(bytes_to_4k_boundary(u32:0x1001), u32:0xFFF);
    assert_eq(bytes_to_4k_boundary(u32:0x1FFF), u32:0x1);
}

// Returns the AxSIZE value for the given bus width.
// Assumes that whole bus width will be used to siplify AxiReader proc
pub fn axsize<DATA_W_DIV8: u32>() -> axi::AxiAxSize {
    match (DATA_W_DIV8) {
        u32:1  => axi::AxiAxSize::MAX_1B_TRANSFER,
        u32:2  => axi::AxiAxSize::MAX_2B_TRANSFER,
        u32:4  => axi::AxiAxSize::MAX_4B_TRANSFER,
        u32:8  => axi::AxiAxSize::MAX_8B_TRANSFER,
        u32:16 => axi::AxiAxSize::MAX_16B_TRANSFER,
        u32:32 => axi::AxiAxSize::MAX_32B_TRANSFER,
        u32:64 => axi::AxiAxSize::MAX_64B_TRANSFER,
        _      => axi::AxiAxSize::MAX_128B_TRANSFER,
    }
}
