#![feature(type_inference_v2)]
#![feature(channel_attributes)]

// Copyright 2025 The XLS Authors
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

// tagged reorder queue implementation
//
// This proc implements a reorder queue that accepts tagged data on `ch_in` and outputs the data in
// tag-order on `ch_out`. It is useful for re-sequencing responses that may arrive out-of-order from
// an upstream source (e.g., memory controllers, parallel processors).
//
// The reorder queue expects a sequence of input tags 0, 1, ..., BUFFER_SIZE-1, 0, 1, ...
// Data may arrive on `ch_in` in any order, e.g., tag 3 may arrive before tag 0.
// The reorder queue buffers incoming data and sends it on `ch_out` only when the
// data for the next expected tag (`head`) becomes available.
//
// For example, if the buffer size is 4, the queue could receive data in the following sequence:
//   (2, A2), (0, A0), (0, B0), (1, A1), (3, A3), ...
// and would output:
//   (_), A0, (_), A1, A2, A3, B0, ...
// where `(_)` represents no output for that cycle.
//
// Generic Parameters:
//   T_WIDTH: The bit-width of the data elements.
//   BUFFER_SIZE: The number of tags in use, from 0 to BUFFER_SIZE-1. This
//     corresponds to the number of parallel transactions that can be in-flight.
//
// Channels:
//   ch_in: Accepts (Tag, T) tuples, where Tag is between 0 and BUFFER_SIZE-1.
//   ch_out: Emits data T in tag order.
//
// Usage Constraints:
//
// 1. Tag Reuse: The sender MUST guarantee that only one transaction for any given tag is in-flight
//    at a time. If data for tag `T` is sent on `ch_in`, new data for tag `T` MUST NOT be sent until
//    the data for the first instance of `T` has been emitted by `ch_out`.
//
//    NOTE: If new data for tag `T` arrives on `ch_in` in the exact same cycle that the previous
//          data for tag `T` is being sent on `ch_out`, this will be handled safely; the previous
//          data will be emitted on `ch_out`, and the new data will be buffered in the queue.
//
//          However, sending data for tag `T` while its previous instance is still buffered and
//          waiting in the queue will trigger an assertion during simulation; when synthesized, it
//          will corrupt the queue's internal state and may cause it to deadlock.
//
//    TODO: epastor - Add an option to backpressure on tag reuse, rather than asserting.
//    This requires either a way to predicate a receive on the data being received, or an internal
//    skid buffer to store input that had a tag collision.
//
// 2. Backpressure: If `ch_out` cannot accept data (i.e., applies backpressure) when the queue has
//    data ready to send, the queue will stall. This stall prevents it from accepting any new input
//    on `ch_in` until `ch_out` becomes ready. In other words, backpressure on `ch_out` propagates
//    directly to `ch_in`.
//
//    Designers should ensure that downstream modules can accept data from `ch_out` at a sufficient
//    rate, or use a FIFO to decouple `ch_in` or `ch_out` from modules that may stall.
//
//    TODO: https://github.com/google/xls/issues/1453 - Use a non-blocking send to decouple input &
//    output backpressure.

import std;

struct HeadPosition<INDEX_WIDTH: u32> { tag: uN[INDEX_WIDTH], generation_parity: u1 }

struct BufferEntry<T_WIDTH: u32> { data: uN[T_WIDTH], generation_parity: u1 }

struct ReorderQueueState<T_WIDTH: u32, BUFFER_SIZE: u32, INDEX_WIDTH: u32> {
    head: HeadPosition<INDEX_WIDTH>,
    buffer: BufferEntry<T_WIDTH>[BUFFER_SIZE],
}

pub proc reorder_queue<T_WIDTH: u32, BUFFER_SIZE: u32, TAG_WIDTH:
u32 = {
    u32:1 + std::flog2(BUFFER_SIZE - u32:1)}>
{
    type Tag = uN[TAG_WIDTH];
    type T = uN[T_WIDTH];
    type Entry = BufferEntry<T_WIDTH>;
    ch_in: chan<(Tag, T)> in;
    ch_out: chan<T> out;

    config(ch_in: chan<(Tag, T)> in, ch_out: chan<T> out) { (ch_in, ch_out) }

    init {
        ReorderQueueState {
            head: HeadPosition { tag: Tag:0, generation_parity: u1:0 },
            buffer: Entry[BUFFER_SIZE]:[Entry { data: zero!<T>(), generation_parity: u1:1 }, ...],
        }
    }

    next(state: ReorderQueueState<T_WIDTH, BUFFER_SIZE, TAG_WIDTH>) {
        let (tok, in_val, in_valid) = recv_non_blocking(join(), ch_in, zero!<(Tag, T)>());
        let (in_tag, in_data) = in_val;

        // Determine output availability and select output data
        // Data is available if it's already in the buffer for the correct generation...
        let output_available =
            state.buffer[state.head.tag].generation_parity == state.head.generation_parity;
        // ...or if it's arriving right now for tag `head` and we're waiting for it (bypass).
        let can_bypass = in_valid && (in_tag == state.head.tag) && !output_available;

        // Assert against tag reuse: if data is received for a tag that still has data buffered in
        // the queue, and that data is not already being sent out in this cycle, it's a violation.
        if in_valid {
            let ready_parity = if in_tag >= state.head.tag {
                state.head.generation_parity
            } else {
                !state.head.generation_parity
            };
            let in_tag_is_buffered = state.buffer[in_tag].generation_parity == ready_parity;
            assert!(state.head.tag == in_tag || !in_tag_is_buffered, "tag_collision");
        };

        let should_send = output_available || can_bypass;
        let out_data = if can_bypass { in_data } else { state.buffer[state.head.tag].data };

        // Write to buffer if input received
        let next_buffer = if in_valid {
            update(
                state.buffer, in_tag,
                Entry { data: in_data, generation_parity: !state.buffer[in_tag].generation_parity })
        } else {
            state.buffer
        };

        // TODO: https://github.com/google/xls/issues/1453 - Make this a non-blocking send so we can
        // handle backpressure by not advancing the head, rather than by stalling the entire proc.
        send_if(tok, ch_out, should_send, out_data);

        // If outputted, advance head and potentially flip generation on wrap-around
        let advance = should_send;
        let next_head = if advance {
            if state.head.tag == ((BUFFER_SIZE - 1) as Tag) {
                HeadPosition { tag: Tag:0, generation_parity: !state.head.generation_parity }
            } else {
                HeadPosition {
                    tag: state.head.tag + Tag:1,
                    generation_parity: state.head.generation_parity,
                }
            }
        } else {
            state.head
        };

        ReorderQueueState { head: next_head, buffer: next_buffer }
    }
}

// Example instantiation:
pub proc reorder_queue_32_16 {
    type Tag = uN[4];
    type T = uN[32];

    config(ch_in: chan<(Tag, T)> in, ch_out: chan<T> out) {
        spawn reorder_queue<32, 16>(ch_in, ch_out);
    }

    init {  }

    next(state: ()) {  }
}

// Testing:
const TEST_T_WIDTH = u32:8;
const TEST_BUFFER_SIZE = u32:4;
const TEST_TAG_WIDTH = u32:2;

type TestTag = uN[TEST_TAG_WIDTH];
type TestT = uN[TEST_T_WIDTH];

// Test sequence:
// Tick  | Send        | Queue State After | Output
// -------------------------------------------------------------------------
// Init  |             |  h=0,hg=F,sg=TTTT |
// 0     | send(1,11)  |  h=0,hg=F,sg=TFTT | none
// 1     | send(3,13)  |  h=0,hg=F,sg=TFTF | none
// 2     | send(0,10)  |  h=1,hg=F,sg=FFTF | 10 (bypass)
// 3     | send(2,12)  |  h=2,hg=F,sg=FFFF | 11 (buffered)
// 4     | -           |  h=3,hg=F,sg=FFFF | 12 (buffered)
// 5     | send(3,23)  |  h=0,hg=T,sg=FFFT | 13 (conflict resolved: buffered)
// 6     | send(0,20)  |  h=1,hg=T,sg=TFFT | 20 (bypass)
// 7     | send(1,21)  |  h=2,hg=T,sg=TTFT | 21 (bypass)
// 8     | send(2,22)  |  h=3,hg=T,sg=TTTT | 22 (bypass)
// 9     | -           |  h=0,hg=F,sg=TTTT | 23 (buffered)
// 10    | -           |  h=0,hg=F,sg=TTTT | none
// 11    | -           |  h=0,hg=F,sg=TTTT | none

struct TestTick { send: bool, tag: TestTag, data: TestT, recv: bool, output: TestT }

const TEST_LENGTH = u32:12;
const TEST_TICKS = TestTick[12]:[
    TestTick { send: true, tag: 1, data: 11, recv: false, output: 0 },
    TestTick { send: true, tag: 3, data: 13, recv: false, output: 0 },
    TestTick { send: true, tag: 0, data: 10, recv: true, output: 10 },
    TestTick { send: true, tag: 2, data: 12, recv: true, output: 11 },
    TestTick { send: false, tag: 0, data: 0, recv: true, output: 12 },
    TestTick { send: true, tag: 3, data: 23, recv: true, output: 13 },
    TestTick { send: true, tag: 0, data: 20, recv: true, output: 20 },
    TestTick { send: true, tag: 1, data: 21, recv: true, output: 21 },
    TestTick { send: true, tag: 2, data: 22, recv: true, output: 22 },
    TestTick { send: false, tag: 0, data: 0, recv: true, output: 23 },
    TestTick { send: false, tag: 0, data: 0, recv: false, output: 0 },
    TestTick { send: false, tag: 0, data: 0, recv: false, output: 0 },
];

#[test_proc]
proc reorder_queue_test {
    to_queue: chan<(TestTag, TestT)> out;
    from_queue: chan<TestT> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (ch_in_s, ch_in_r) = #[channel(depth=0)]
                                 chan<(TestTag, TestT)>("queue_in_ch");
        let (ch_out_s, ch_out_r) = #[channel(depth=0)]
                                   chan<TestT>("queue_out_ch");
        spawn reorder_queue<TEST_T_WIDTH, TEST_BUFFER_SIZE, TEST_TAG_WIDTH>(ch_in_r, ch_out_s);
        (ch_in_s, ch_out_r, terminator)
    }

    init { () }

    next(state: ()) {
        let tok = for (i, tok) in u32:0..TEST_LENGTH {
            trace_fmt!("tick: {}", i);
            let send_tok = send_if(
                tok, to_queue, TEST_TICKS[i].send, (TEST_TICKS[i].tag, TEST_TICKS[i].data));
            if TEST_TICKS[i].send {
                trace_fmt!("sent: (tag: {}, data: {})", TEST_TICKS[i].tag, TEST_TICKS[i].data);
            };
            if TEST_TICKS[i].recv {
                // TODO(epastor): This should really be non-blocking, but we'd need better
                // timing-sensitive test specifications for this to work; it currently fails when
                // simulated without the schedule taken into account, because the simulator doesn't
                // guarantee that the `reorder_queue`'s send will execute before this resolves.
                let (recv_tok, out_data) = recv(send_tok, from_queue);
                trace_fmt!("received: {}, expected: {}", out_data, TEST_TICKS[i].output);
                assert!(out_data == TEST_TICKS[i].output, "output_data_mismatch");
                recv_tok
            } else {
                let (recv_tok, out_data, received) =
                    recv_non_blocking(send_tok, from_queue, zero!<TestT>());
                if received { trace_fmt!("unexpectedly received: {}", out_data); };
                assert!(!received, "output_validity_mismatch");
                recv_tok
            }
        }(join());

        send(tok, terminator, true);
    }
}
