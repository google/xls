// Copyright 2023 The XLS Authors
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

// This file contains implementation of a Buffer structure that acts as
// a simple FIFO. Additionally, the file provides various functions that
// can simplify access to the stored.
//
// The utility functions containing the `_checked` suffix serve two purposes:
// they perform the actual operation and return information on whether
// the operation was successful. If you are sure that the precondition is
// always true, you can use the function with the same name but without
// the `_checked` suffix.

import std;

// Structure to hold the buffered data
pub struct Buffer<CAPACITY: u32> {
    content: bits[CAPACITY],
    length: u32
}

// Status values reported by the functions operating on a Buffer
pub enum BufferStatus : u2 {
    OK = 0,
    NO_ENOUGH_SPACE = 1,
    NO_ENOUGH_DATA = 2,
}

// Structure for returning Buffer and BufferStatus together
pub struct BufferResult<CAPACITY: u32> {
    buffer: Buffer<CAPACITY>,
    status: BufferStatus
}

// Checks whether a `buffer` can fit `data`
pub fn buffer_can_fit<DSIZE: u32, CAPACITY: u32>(buffer: Buffer<CAPACITY>, data: bits[DSIZE]) -> bool {
    buffer.length + DSIZE <= CAPACITY
}

#[test]
fn test_buffer_can_fit() {
    let buffer = Buffer<u32:32> { content: u32:0, length: u32:0 };
    assert_eq(buffer_can_fit(buffer, bits[0]:0), true);
    assert_eq(buffer_can_fit(buffer, u16:0), true);
    assert_eq(buffer_can_fit(buffer, u32:0), true);
    assert_eq(buffer_can_fit(buffer, u33:0), false);

    let buffer = Buffer<u32:32> { content: u32:0, length: u32:16 };
    assert_eq(buffer_can_fit(buffer, bits[0]:0), true);
    assert_eq(buffer_can_fit(buffer, u16:0), true);
    assert_eq(buffer_can_fit(buffer, u17:0), false);
    assert_eq(buffer_can_fit(buffer, u32:0), false);

    let buffer = Buffer<u32:32> { content: u32:0, length: u32:32 };
    assert_eq(buffer_can_fit(buffer, bits[0]:0), true);
    assert_eq(buffer_can_fit(buffer, u1:0), false);
    assert_eq(buffer_can_fit(buffer, u16:0), false);
    assert_eq(buffer_can_fit(buffer, u32:0), false);
}

// Checks whether a `buffer` has at least `length` amount of data
pub fn buffer_has_at_least<CAPACITY: u32>(buffer: Buffer<CAPACITY>, length: u32) -> bool {
    length <= buffer.length
}

#[test]
fn test_buffer_has_at_least() {
    let buffer = Buffer { content: u32:0, length: u32:0 };
    assert_eq(buffer_has_at_least(buffer, u32:0), true);
    assert_eq(buffer_has_at_least(buffer, u32:16), false);
    assert_eq(buffer_has_at_least(buffer, u32:32), false);
    assert_eq(buffer_has_at_least(buffer, u32:33), false);

    let buffer = Buffer { content: u32:0, length: u32:16 };
    assert_eq(buffer_has_at_least(buffer, u32:0), true);
    assert_eq(buffer_has_at_least(buffer, u32:16), true);
    assert_eq(buffer_has_at_least(buffer, u32:32), false);
    assert_eq(buffer_has_at_least(buffer, u32:33), false);

    let buffer = Buffer { content: u32:0, length: u32:32 };
    assert_eq(buffer_has_at_least(buffer, u32:0), true);
    assert_eq(buffer_has_at_least(buffer, u32:16), true);
    assert_eq(buffer_has_at_least(buffer, u32:32), true);
    assert_eq(buffer_has_at_least(buffer, u32:33), false);
}

// Returns a new buffer with `data` appended to the original `buffer`.
// It will fail if the buffer cannot fit the data. For calls that need better
// error handling, check `buffer_append_checked`
pub fn buffer_append<CAPACITY: u32, DSIZE: u32> (buffer: Buffer<CAPACITY>, data: bits[DSIZE]) -> Buffer<CAPACITY> {
    if buffer_can_fit(buffer, data) == false {
        trace_fmt!("Not enough space in the buffer! {} + {} <= {}", buffer.length, DSIZE, CAPACITY);
        fail!("not_enough_space", buffer)
    } else {
        Buffer {
            content: (data as bits[CAPACITY] << buffer.length) | buffer.content,
            length: DSIZE + buffer.length
        }
    }
}

#[test]
fn test_buffer_append() {
    let buffer = Buffer { content: u32:0, length: u32:0 };
    let buffer = buffer_append(buffer, u16:0xBEEF);
    assert_eq(buffer, Buffer { content: u32:0xBEEF, length: u32:16 });
    let buffer = buffer_append(buffer, u16:0xDEAD);
    assert_eq(buffer, Buffer { content: u32:0xDEADBEEF, length: u32:32 });
}

// Returns a new buffer with the `data` appended to the original `buffer` if
// the buffer has enough space. Otherwise, it returns an unmodified buffer
// along with an error. The results are stored in the BufferResult structure.
pub fn buffer_append_checked<CAPACITY: u32, DSIZE: u32> (buffer: Buffer<CAPACITY>, data: bits[DSIZE]) -> BufferResult<CAPACITY> {
    if buffer_can_fit(buffer, data) == false {
        BufferResult { status: BufferStatus::NO_ENOUGH_SPACE, buffer }
    } else {
        BufferResult {
            status: BufferStatus::OK,
            buffer: buffer_append(buffer, data)
        }
    }
}

#[test]
fn test_buffer_append_checked() {
    let buffer = Buffer { content: u32:0, length: u32:0 };

    let result1 = buffer_append_checked(buffer, u16:0xBEEF);
    assert_eq(result1, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0xBEEF, length: u32:16 }
    });

    let result2 = buffer_append_checked(result1.buffer, u16:0xDEAD);
    assert_eq(result2, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0xDEADBEEF, length: u32:32 }
    });

    let result3 = buffer_append_checked(result2.buffer, u16:0xCAFE);
    assert_eq(result3, BufferResult {
        status: BufferStatus::NO_ENOUGH_SPACE,
        buffer: result2.buffer
    });
}

// Returns `length` amount of data from a `buffer` and a new buffer with
// the data removed. Since the Buffer structure acts as a simple FIFO,
// it pops the data in the same order as they were added to the buffer.
// If the buffer does not have enough data to meet the specified length,
// the function will fail. For calls that need better error handling,
// check `buffer_pop_checked`.
pub fn buffer_pop<CAPACITY: u32>(buffer: Buffer<CAPACITY>, length: u32) -> (Buffer<CAPACITY>, bits[CAPACITY]) {
    if buffer_has_at_least(buffer, length) == false {
        trace_fmt!("Not enough data in the buffer!");
        fail!("not_enough_data", (buffer, bits[CAPACITY]:0))
    } else {
        let mask = (bits[CAPACITY]:1 << length) - bits[CAPACITY]:1;
        (
            Buffer {
                content: buffer.content >> length,
                length: buffer.length - length
            },
            buffer.content & mask
        )
    }
}

#[test]
fn test_buffer_pop() {
    let buffer = Buffer { content: u32:0xDEADBEEF, length: u32:32 };
    let (buffer, data) = buffer_pop(buffer, u32:16);
    assert_eq(data, u32:0xBEEF);
    assert_eq(buffer, Buffer { content: u32:0xDEAD, length: u32:16 });
    let (buffer, data) = buffer_pop(buffer, u32:16);
    assert_eq(data, u32:0xDEAD);
    assert_eq(buffer, Buffer { content: u32:0, length: u32:0 });
}

// Returns `length` amount of data from a `buffer`, a new buffer with
// the data removed and a positive status, if the buffer contains enough data.
// Otherwise, it returns unmodified buffer, zeroed data field and error.
// Since the Buffer structure acts as a simple FIFO, it pops the data in
// the same order as they were added to the buffer.
// The results are stored in the BufferResult structure.
pub fn buffer_pop_checked<CAPACITY: u32> (buffer: Buffer<CAPACITY>, length: u32) -> (BufferResult<CAPACITY>, bits[CAPACITY]) {
    if buffer_has_at_least(buffer, length) == false {
        (
            BufferResult { status: BufferStatus::NO_ENOUGH_DATA, buffer },
            bits[CAPACITY]:0
        )
    } else {
        let (buffer_leftover, content) = buffer_pop(buffer, length);
        (
            BufferResult {
                status: BufferStatus::OK,
                buffer: buffer_leftover
            },
            content
        )
    }
}

#[test]
fn test_buffer_pop_checked() {
    let buffer = Buffer { content: u32:0xDEADBEEF, length: u32:32 };

    let (result1, data1) = buffer_pop_checked(buffer, u32:16);
    assert_eq(result1, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0xDEAD, length: u32:16 }
    });
    assert_eq(data1, u32:0xBEEF);

    let (result2, data2) = buffer_pop_checked(result1.buffer, u32:16);
    assert_eq(result2, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0, length: u32:0 }
    });
    assert_eq(data2, u32:0xDEAD);

    let (result3, data3) = buffer_pop_checked(result2.buffer, u32:16);
    assert_eq(result3, BufferResult {
        status: BufferStatus::NO_ENOUGH_DATA,
        buffer: result2.buffer
    });
    assert_eq(data3, u32:0);
}

// Behaves like `buffer_pop` except that the length of the popped data can be
// set using a DSIZE function parameter. For calls that need better error
// handling, check `buffer_fixed_pop_checked`.
pub fn buffer_fixed_pop<DSIZE: u32, CAPACITY: u32> (buffer: Buffer<CAPACITY>) -> (Buffer<CAPACITY>, bits[DSIZE]) {
    let (buffer, value) = buffer_pop(buffer, DSIZE);
    (buffer, value as bits[DSIZE])
}

#[test]
fn test_buffer_fixed_pop() {
    let buffer = Buffer { content: u32:0xDEADBEEF, length: u32:32 };
    let (buffer, data) = buffer_fixed_pop<u32:16>(buffer);
    assert_eq(data, u16:0xBEEF);
    assert_eq(buffer, Buffer { content: u32:0xDEAD, length: u32:16 });
    let (buffer, data) = buffer_fixed_pop<u32:16>(buffer);
    assert_eq(data, u16:0xDEAD);
    assert_eq(buffer, Buffer { content: u32:0, length: u32:0 });
}

// Behaves like `buffer_pop_checked` except that the length of the popped data
// can be set using a DSIZE function parameter.
pub fn buffer_fixed_pop_checked<DSIZE: u32, CAPACITY: u32> (buffer: Buffer<CAPACITY>) -> (BufferResult<CAPACITY>, bits[DSIZE]) {
    let (result, value) = buffer_pop_checked(buffer, DSIZE);
    (result, value as bits[DSIZE])
}

#[test]
fn test_buffer_fixed_pop_checked() {
    let buffer = Buffer { content: u32:0xDEADBEEF, length: u32:32 };
    let (result1, data1) = buffer_fixed_pop_checked<u32:16>(buffer);
    assert_eq(result1, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0xDEAD, length: u32:16 }
    });
    assert_eq(data1, u16:0xBEEF);

    let (result2, data2) = buffer_fixed_pop_checked<u32:16>(result1.buffer);
    assert_eq(result2, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer { content: u32:0, length: u32:0 }
    });
    assert_eq(data2, u16:0xDEAD);

    let (result3, data3) = buffer_fixed_pop_checked<u32:16>(result2.buffer);
    assert_eq(result3, BufferResult {
        status: BufferStatus::NO_ENOUGH_DATA,
        buffer: result2.buffer
    });
    assert_eq(data3, u16:0);
}

// Returns `length` amount of data from a `buffer`.
// It will fail if the buffer has no sufficient amount of data.
// For calls that need better error handling, check `buffer_peek_checked`.
pub fn buffer_peek<CAPACITY: u32>(buffer: Buffer<CAPACITY>, length: u32) -> bits[CAPACITY] {
    if buffer_has_at_least(buffer, length) == false {
        trace_fmt!("Not enough data in the buffer!");
        fail!("not_enough_data", bits[CAPACITY]:0)
    } else {
        let mask = (bits[CAPACITY]:1 << length) - bits[CAPACITY]:1;
        buffer.content & mask
    }
}

#[test]
fn test_buffer_peek() {
    let buffer = Buffer { content: u32:0xDEADBEEF, length: u32:32 };
    assert_eq(buffer_peek(buffer, u32:0), u32:0);
    assert_eq(buffer_peek(buffer, u32:16), u32:0xBEEF);
    assert_eq(buffer_peek(buffer, u32:32), u32:0xDEADBEEF);
}

// Returns a new buffer with the `data` and a positive status if
// the buffer has enough data. Otherwise, it returns a zeroed-data and error.
// The results are stored in the BufferResult structure.
pub fn buffer_peek_checked<CAPACITY: u32> (buffer: Buffer<CAPACITY>, length: u32) -> (BufferStatus, bits[CAPACITY]) {
    if buffer_has_at_least(buffer, length) == false {
        (BufferStatus::NO_ENOUGH_DATA, bits[CAPACITY]:0)
    } else {
        let mask = (bits[CAPACITY]:1 << length) - bits[CAPACITY]:1;
        (BufferStatus::OK, buffer.content & mask)
    }
}

#[test]
fn test_buffer_peek_checked() {
    let buffer = Buffer { content: u32:0xDEADBEEF, length: u32:32 };

    let (status1, data1) = buffer_peek_checked(buffer, u32:0);
    assert_eq(status1, BufferStatus::OK);
    assert_eq(data1, u32:0);

    let (status2, data2) = buffer_peek_checked(buffer, u32:16);
    assert_eq(status2, BufferStatus::OK);
    assert_eq(data2, u32:0xBEEF);

    let (status3, data3) = buffer_peek_checked(buffer, u32:32);
    assert_eq(status3, BufferStatus::OK);
    assert_eq(data3, u32:0xDEADBEEF);

    let (status4, data4) = buffer_peek_checked(buffer, u32:64);
    assert_eq(status4, BufferStatus::NO_ENOUGH_DATA);
    assert_eq(data4, u32:0);
}

// Creates a new buffer
pub fn buffer_new<CAPACITY: u32>() -> Buffer<CAPACITY> {
    Buffer { content: bits[CAPACITY]:0, length: u32:0 }
}

#[test]
fn test_buffer_new() {
    assert_eq(buffer_new<u32:32>(), Buffer { content: u32:0, length: u32:0 });
}
