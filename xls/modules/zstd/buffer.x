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

// This file contains implementation of a simple buffer that can store
// data from multiple transfers over the XLS channel and glue them together.
// It contains various operations that can simplify access to the stored data.
//
// Utility functions containing the `_checked` suffix, in addition
// to performing the actual operation, return information whether the operation
// was successful. If the precondition is guaranteed to be true, one can use
// the function with the same name but without the suffix.

import std

// Structure to hold the buffered data
pub struct Buffer<BSIZE: u32> {
    contents: bits[BSIZE],
    length: u32,
}

// Status values reported by the functions operating on a Buffer
pub enum BufferStatus: u2 {
    OK = 0,
    NO_ENOUGH_SPACE = 1,
    NO_ENOUGH_DATA = 2,
}

// Structure for returning Buffer and BufferStatus together
pub struct BufferResult<BSIZE: u32> {
    buffer: Buffer<BSIZE>,
    status: BufferStatus,
}

// Checks whether a `buffer` can fit `data`
pub fn buffer_can_fit<BSIZE: u32, DSIZE: u32>(buffer: Buffer<BSIZE>, data: bits[DSIZE]) -> bool {
    buffer.length + DSIZE <= BSIZE
}

#[test]
fn test_buffer_fits() {
  let buffer = Buffer<u32:32> { contents: u32:0, length: u32:0 };
  assert_eq(buffer_can_fit(buffer, bits[0]:0), true);
  assert_eq(buffer_can_fit(buffer, u16:0), true);
  assert_eq(buffer_can_fit(buffer, u32:0), true);
  assert_eq(buffer_can_fit(buffer, u33:0), false);

  let buffer = Buffer<u32:32> { contents: u32:0, length: u32:16 };
  assert_eq(buffer_can_fit(buffer, bits[0]:0), true);
  assert_eq(buffer_can_fit(buffer, u16:0), true);
  assert_eq(buffer_can_fit(buffer, u17:0), false);
  assert_eq(buffer_can_fit(buffer, u32:0), false);

  let buffer = Buffer<u32:32> { contents: u32:0, length: u32:32 };
  assert_eq(buffer_can_fit(buffer, bits[0]:0), true);
  assert_eq(buffer_can_fit(buffer, u1:0), false);
  assert_eq(buffer_can_fit(buffer, u16:0), false);
  assert_eq(buffer_can_fit(buffer, u32:0), false);
}

// Checks whether a `buffer` has at least `length` amount of data
pub fn buffer_has_at_least<BSIZE: u32>(buffer: Buffer<BSIZE>, length: u32) -> bool {
    length <= buffer.length
}

#[test]
fn test_buffer_has_at_least() {
  let buffer = Buffer { contents: u32:0, length: u32:0 };
  assert_eq(buffer_has_at_least(buffer, u32:0), true);
  assert_eq(buffer_has_at_least(buffer, u32:16), false);
  assert_eq(buffer_has_at_least(buffer, u32:32), false);
  assert_eq(buffer_has_at_least(buffer, u32:33), false);

  let buffer = Buffer { contents: u32:0, length: u32:16 };
  assert_eq(buffer_has_at_least(buffer, u32:0), true);
  assert_eq(buffer_has_at_least(buffer, u32:16), true);
  assert_eq(buffer_has_at_least(buffer, u32:32), false);
  assert_eq(buffer_has_at_least(buffer, u32:33), false);

  let buffer = Buffer { contents: u32:0, length: u32:32 };
  assert_eq(buffer_has_at_least(buffer, u32:0), true);
  assert_eq(buffer_has_at_least(buffer, u32:16), true);
  assert_eq(buffer_has_at_least(buffer, u32:32), true);
  assert_eq(buffer_has_at_least(buffer, u32:33), false);
}

// Appends `data` to a `buffer`. Returns the updated buffer
// It will fail if the buffer cannot fit the data. For calls that need
// better error handling, check `buffer_append_checked`
pub fn buffer_append<BSIZE: u32, DSIZE: u32>(buffer: Buffer<BSIZE>, data: bits[DSIZE]) -> Buffer<BSIZE> {
    if buffer_can_fit(buffer, data) == false {
        trace_fmt!("Not enough space in the buffer! {} + {} <= {}", buffer.length, DSIZE, BSIZE);
        fail!("not_enough_space", ());
    } else {()};

    Buffer {
        contents: (data as bits[BSIZE] << buffer.length) | buffer.contents,
        length: DSIZE + buffer.length,
    }
}

#[test]
fn test_buffer_append() {
    let buffer = Buffer { contents: u32:0, length: u32:0 };
    let buffer = buffer_append(buffer, u16:0xBEEF);
    assert_eq(buffer, Buffer {contents: u32:0xBEEF, length: u32:16});
    let buffer = buffer_append(buffer, u16:0xDEAD);
    assert_eq(buffer, Buffer {contents: u32:0xDEADBEEF, length: u32:32});
}

// Appends `data` to a `buffer` if possible. Returns the status of the operation
// and the updated buffer in a BufferResult structure
pub fn buffer_append_checked<BSIZE: u32, DSIZE: u32>(buffer: Buffer<BSIZE>, data: bits[DSIZE]) -> BufferResult<BSIZE> {
    if buffer_can_fit(buffer, data) == false {
        BufferResult {
            status: BufferStatus::NO_ENOUGH_SPACE,
            buffer: buffer,
        }
    } else {
        BufferResult {
            status: BufferStatus::OK,
            buffer: Buffer {
                contents: (data as bits[BSIZE] << buffer.length) | buffer.contents,
                length: DSIZE + buffer.length,
            }
        }
    }
}

#[test]
fn test_buffer_append_checked() {
    let buffer = Buffer { contents: u32:0, length: u32:0 };

    let result1 = buffer_append_checked(buffer, u16:0xBEEF);
    assert_eq(result1, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer {contents: u32:0xBEEF, length: u32:16},
    });

    let result2 = buffer_append_checked(result1.buffer, u16:0xDEAD);
    assert_eq(result2, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer {contents: u32:0xDEADBEEF, length: u32:32},
    });

    let result3 = buffer_append_checked(result2.buffer, u16:0xCAFE);
    assert_eq(result3, BufferResult {
        status: BufferStatus::NO_ENOUGH_SPACE,
        buffer: result2.buffer,
    });
}

// Pops `length` amount of data from a `buffer`. Returns the modified buffer
// and the popped data. The length of the data is the same as the buffer size.
// It will fail if the buffer has no sufficient amount of data.
// For calls that need better error handling, check `buffer_pop_checked`.
pub fn buffer_pop<BSIZE: u32>(buffer: Buffer<BSIZE>, length:u32) -> (Buffer<BSIZE>, bits[BSIZE]) {
    if buffer_has_at_least(buffer, length) == false {
        trace_fmt!("Not enough data in the buffer!");
        fail!("not_enough_data", ());
    } else {()};

    let mask = (bits[BSIZE]:1 << length) - bits[BSIZE]:1;
    (
        Buffer {
            contents: buffer.contents >> length,
            length: buffer.length - length
        },
        buffer.contents & mask
    )
}

#[test]
fn test_buffer_pop() {
    let buffer = Buffer {contents: u32:0xDEADBEEF, length: u32:32};
    let (buffer, data) = buffer_pop(buffer, u32:16);
    assert_eq(data, u32:0xBEEF);
    assert_eq(buffer, Buffer {contents: u32:0xDEAD, length: u32:16});
    let (buffer, data) = buffer_pop(buffer, u32:16);
    assert_eq(data, u32:0xDEAD);
    assert_eq(buffer, Buffer {contents: u32:0, length: u32:0});
}

// Pops `length` amount of data from a `buffer` if possible. The length of the
// popped data is the same as the buffer size. Returns the status of the operation
// and the updated buffer in a BufferResult structure, and the popped data.
pub fn buffer_pop_checked<BSIZE: u32>(buffer: Buffer<BSIZE>, length:u32) -> (BufferResult<BSIZE>, bits[BSIZE]) {
    if buffer_has_at_least(buffer, length) == false {
        (
            BufferResult {
                status: BufferStatus::NO_ENOUGH_DATA,
                buffer: buffer,
            },
            bits[BSIZE]:0
        )
    } else {
        let mask = (bits[BSIZE]:1 << length) - bits[BSIZE]:1;
        (
            BufferResult {
                status: BufferStatus::OK,
                buffer: Buffer {
                    contents: buffer.contents >> length,
                    length: buffer.length - length
                }
            },
            buffer.contents & mask
        )
    }
}

#[test]
fn test_buffer_pop_checked() {
    let buffer = Buffer {contents: u32:0xDEADBEEF, length: u32:32};

    let (result1, data1) = buffer_pop_checked(buffer, u32:16);
    assert_eq(result1, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer {contents: u32:0xDEAD, length: u32:16},
    });
    assert_eq(data1, u32:0xBEEF);

    let (result2, data2) = buffer_pop_checked(result1.buffer, u32:16);
    assert_eq(result2, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer {contents: u32:0, length: u32:0},
    });
    assert_eq(data2, u32:0xDEAD);

    let (result3, data3) = buffer_pop_checked(result2.buffer, u32:16);
    assert_eq(result3, BufferResult {
        status: BufferStatus::NO_ENOUGH_DATA,
        buffer: result2.buffer,
    });
    assert_eq(data3, u32:0);
}

// Pops `length` amount of data from a `buffer`. Returns the modified buffer
// and the popped data. The length of the popped data is a function parameter.
// It will fail if the buffer has no sufficient amount of data.
// For calls that need better error handling, check `buffer_sized_pop_checked`.
pub fn buffer_sized_pop<BSIZE: u32, WSIZE: u32>(buffer: Buffer<BSIZE>) -> (Buffer<BSIZE>, bits[WSIZE]) {
    let (buffer, value) = buffer_pop(buffer, WSIZE);
    (buffer, value as bits[WSIZE])
}

#[test]
fn test_buffer_sized_pop() {
    let buffer = Buffer {contents: u32:0xDEADBEEF, length: u32:32};
    let (buffer, data) = buffer_sized_pop<u32:32, u32:16>(buffer);
    assert_eq(data, u16:0xBEEF);
    assert_eq(buffer, Buffer {contents: u32:0xDEAD, length: u32:16});
    let (buffer, data) = buffer_sized_pop<u32:32, u32:16>(buffer);
    assert_eq(data, u16:0xDEAD);
    assert_eq(buffer, Buffer {contents: u32:0, length: u32:0});
}

// Pops `length` amount of data from a `buffer` if possible. The length of
// the popped data is a function parameter. Returns the status of the operation
// and the updated buffer in a BufferResult structure, and the popped data.
pub fn buffer_sized_pop_checked<BSIZE: u32, WSIZE: u32>(buffer: Buffer<BSIZE>) -> (BufferResult<BSIZE>, bits[WSIZE]) {
    let (result, value) = buffer_pop_checked(buffer, WSIZE);
    (result, value as bits[WSIZE])
}

#[test]
fn test_buffer_sized_pop_checked() {
    let buffer = Buffer {contents: u32:0xDEADBEEF, length: u32:32};
    let (result1, data1) = buffer_sized_pop_checked<u32:32, u32:16>(buffer);
    assert_eq(result1, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer {contents: u32:0xDEAD, length: u32:16},
    });
    assert_eq(data1, u16:0xBEEF);

    let (result2, data2) = buffer_sized_pop_checked<u32:32, u32:16>(result1.buffer);
    assert_eq(result2, BufferResult {
        status: BufferStatus::OK,
        buffer: Buffer {contents: u32:0, length: u32:0},
    });
    assert_eq(data2, u16:0xDEAD);

    let (result3, data3) = buffer_sized_pop_checked<u32:32, u32:16>(result2.buffer);
    assert_eq(result3, BufferResult {
        status: BufferStatus::NO_ENOUGH_DATA,
        buffer: result2.buffer,
    });
    assert_eq(data3, u16:0);
}

// Returns `length` amount of data from a `buffer`.
// It will fail if the buffer has no sufficient amount of data.
// For calls that need better error handling, check `buffer_peek_checked`.
pub fn buffer_peek<BSIZE: u32>(buffer: Buffer<BSIZE>, length:u32) -> bits[BSIZE] {
    if buffer_has_at_least(buffer, length) == false {
        trace_fmt!("Not enough data in the buffer!");
        fail!("not_enough_data", ());
    } else {()};

    let mask = (bits[BSIZE]:1 << length) - bits[BSIZE]:1;
    buffer.contents & mask
}

#[test]
fn test_buffer_peek() {
    let buffer = Buffer {contents: u32:0xDEADBEEF, length: u32:32};
    assert_eq(buffer_peek(buffer, u32:0), u32:0);
    assert_eq(buffer_peek(buffer, u32:16), u32:0xBEEF);
    assert_eq(buffer_peek(buffer, u32:32), u32:0xDEADBEEF);
}

// Returns `length` amount of data from a `buffer` if possible and the status of the operation.
pub fn buffer_peek_checked<BSIZE: u32>(buffer: Buffer<BSIZE>, length:u32) -> (BufferStatus, bits[BSIZE]) {
    if buffer_has_at_least(buffer, length) == false {
        (BufferStatus::NO_ENOUGH_DATA, bits[BSIZE]:0)
    } else {
        let mask = (bits[BSIZE]:1 << length) - bits[BSIZE]:1;
        (BufferStatus::OK, buffer.contents & mask)
    }
}

#[test]
fn test_buffer_peek_checked() {
    let buffer = Buffer {contents: u32:0xDEADBEEF, length: u32:32};

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
pub fn buffer_new<BSIZE: u32>() -> Buffer<BSIZE> {
    Buffer {
        contents: bits[BSIZE]:0,
        length: u32:0,
    }
}

#[test]
fn test_buffer_new() {
    assert_eq(buffer_new<u32:32>(), Buffer {contents: u32:0, length: u32:0});
}
