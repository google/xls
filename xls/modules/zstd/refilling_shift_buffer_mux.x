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

// This file contains a RamMux implementation that can be used to connect
// a single proc with two RAM instances, by using a single RAM interface and
// switching between the RAMs, when requested. The switching occurs only after
// each request has received the corresponding response.
// Additionally, a "naive" implementation is provided that does not ensure
// any synchronization when switching RAMs.

import xls.modules.zstd.refilling_shift_buffer;

struct RefillingShiftBufferMuxState {
    sel: u1,
}

pub proc RefillingShiftBufferMux<
    AXI_DATA_W: u32, SB_LENGTH_W: u32,
    INIT_SEL: u1 = {u1:0},
>{
    type State = RefillingShiftBufferMuxState;

    type SBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<AXI_DATA_W, SB_LENGTH_W>;
    type SBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<SB_LENGTH_W>;

    init {
        State { sel: INIT_SEL }
    }

    sel_req_r: chan<u1> in;
    sel_resp_s: chan<()> out;

    ctrl0_r: chan<SBCtrl> in;
    data0_s: chan<SBOutput> out;

    ctrl1_r: chan<SBCtrl> in;
    data1_s: chan<SBOutput> out;

    ctrl_s: chan<SBCtrl> out;
    data_r: chan<SBOutput> in;


    config (
        sel_req_r: chan<u1> in,
        sel_resp_s: chan<()> out,

        ctrl0_r: chan<SBCtrl> in,
        data0_s: chan<SBOutput> out,

        ctrl1_r: chan<SBCtrl> in,
        data1_s: chan<SBOutput> out,

        ctrl_s: chan<SBCtrl> out,
        data_r: chan<SBOutput> in,
    ) {
        (
            sel_req_r, sel_resp_s,
            ctrl0_r, data0_s,
            ctrl1_r, data1_s,
            ctrl_s, data_r,
        )
    }

    next (state: State) {
        let tok0 = join();

        let (tok1, sel, sel_valid) = recv_non_blocking(tok0, sel_req_r, state.sel);
        let tok2_0 = send(tok0, sel_resp_s, ());

        let (tok2_0, ctrl0, ctrl0_valid) = recv_if_non_blocking(tok1, ctrl0_r, sel == u1:0, zero!<SBCtrl>());
        let (tok2_1, ctrl1, ctrl1_valid) = recv_if_non_blocking(tok1, ctrl1_r, sel == u1:1, zero!<SBCtrl>());
        let tok2 = join(tok2_0, tok2_1);

        let (ctrl, ctrl_valid) = if ctrl0_valid {
            (ctrl0, true)
        } else if ctrl1_valid {
            (ctrl1, true)
        } else {
            (zero!<SBCtrl>(), false)
        };

        let tok3 = send_if(tok2, ctrl_s, ctrl_valid, ctrl);
        let (tok4, data) = recv_if(tok3, data_r, ctrl_valid, zero!<SBOutput>());

        let do_recv_data0 = (sel == u1:0) && ctrl_valid;
        send_if(tok4, data0_s, do_recv_data0, data);

        let do_recv_data1 = (sel == u1:1) && ctrl_valid;
        send_if(tok4, data1_s, do_recv_data1, data);

        State { sel }
    }
}

const TEST_AXI_DATA_W = u32:64;
const TEST_SB_LENGTH_W = u32:32;

proc RefillingShiftBufferStub<
    AXI_DATA_W: u32, SB_LENGTH_W: u32
> {
    type SBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<AXI_DATA_W, SB_LENGTH_W>;
    type SBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<SB_LENGTH_W>;

    type Length = uN[SB_LENGTH_W];
    type Data = uN[AXI_DATA_W];

    ctrl_r: chan<SBCtrl> in;
    data_s: chan<SBOutput> out;

    init { u32:0 }

    config (
        ctrl_r: chan<SBCtrl> in,
        data_s: chan<SBOutput> out,
    ) {
        (ctrl_r, data_s)
    }

    next(cnt: u32) {
        let tok = join();
        let (tok, ctrl) = recv(tok, ctrl_r);
        let tok = send(tok, data_s, SBOutput { data: cnt as Data, length: ctrl.length, error: false });
        cnt + u32:1
    }
}

#[test_proc]
proc RefillingShitBufferMuxTest
{
    type SBOutput = refilling_shift_buffer::RefillingShiftBufferOutput<TEST_AXI_DATA_W, TEST_SB_LENGTH_W>;
    type SBCtrl = refilling_shift_buffer::RefillingShiftBufferCtrl<TEST_SB_LENGTH_W>;

    type Length = uN[TEST_SB_LENGTH_W];
    type Data = uN[TEST_AXI_DATA_W];

    terminator: chan<bool> out;

    sel_req_s: chan<u1> out;
    sel_resp_r: chan<()> in;

    ctrl0_s: chan<SBCtrl> out;
    data0_r: chan<SBOutput> in;

    ctrl1_s: chan<SBCtrl> out;
    data1_r: chan<SBOutput> in;

    init {}

    config(terminator: chan<bool> out) {
        let (sel_req_s, sel_req_r) = chan<u1>("sel_req");
        let (sel_resp_s, sel_resp_r) = chan<()>("sel_resp");

        let (ctrl_s, ctrl_r) = chan<SBCtrl>("ctrl");
        let (data_s, data_r) = chan<SBOutput>("data");

        let (ctrl0_s, ctrl0_r) = chan<SBCtrl>("ctrl0");
        let (data0_s, data0_r) = chan<SBOutput>("data0");

        let (ctrl1_s, ctrl1_r) = chan<SBCtrl>("ctrl1");
        let (data1_s, data1_r) = chan<SBOutput>("data1");

        spawn RefillingShiftBufferMux<TEST_AXI_DATA_W, TEST_SB_LENGTH_W>(
            sel_req_r, sel_resp_s,
            ctrl0_r, data0_s,
            ctrl1_r, data1_s,
            ctrl_s, data_r,
        );

        spawn RefillingShiftBufferStub<TEST_AXI_DATA_W, TEST_SB_LENGTH_W> (
            ctrl_r, data_s,
        );

        (
            terminator,
            sel_req_s, sel_resp_r,
            ctrl0_s, data0_r,
            ctrl1_s, data1_r,
        )
    }

    next(state: ()) {
        let tok = join();

        let tok = send(tok, ctrl0_s, SBCtrl { length: Length:0xA1 });
        let tok = send(tok, ctrl0_s, SBCtrl { length: Length:0xA2 });
        let tok = send(tok, ctrl0_s, SBCtrl { length: Length:0xA3 });
        let tok = send(tok, ctrl0_s, SBCtrl { length: Length:0xA4 });
        let tok = send(tok, ctrl0_s, SBCtrl { length: Length:0xA5 });
        let tok = send(tok, ctrl0_s, SBCtrl { length: Length:0xA6 });
        let tok = send(tok, ctrl0_s, SBCtrl { length: Length:0xA7 });
        let tok = send(tok, ctrl0_s, SBCtrl { length: Length:0xA8 });

        let (tok, data) = recv(tok, data0_r);
        assert_eq(data, SBOutput { data: Data:0, length: Length:0xA1, error: false });
        let (tok, data) = recv(tok, data0_r);
        assert_eq(data, SBOutput { data: Data:1, length: Length:0xA2, error: false });
        let (tok, data) = recv(tok, data0_r);
        assert_eq(data, SBOutput { data: Data:2, length: Length:0xA3, error: false });
        let (tok, data) = recv(tok, data0_r);
        assert_eq(data, SBOutput { data: Data:3, length: Length:0xA4, error: false });
        let (tok, data) = recv(tok, data0_r);
        assert_eq(data, SBOutput { data: Data:4, length: Length:0xA5, error: false });
        let (tok, data) = recv(tok, data0_r);
        assert_eq(data, SBOutput { data: Data:5, length: Length:0xA6, error: false });
        let (tok, data) = recv(tok, data0_r);
        assert_eq(data, SBOutput { data: Data:6, length: Length:0xA7, error: false });
        let (tok, data) = recv(tok, data0_r);
        assert_eq(data, SBOutput { data: Data:7, length: Length:0xA8, error: false });

        let tok = send(tok, sel_req_s, u1:1);
        let (tok, _) = recv(tok, sel_resp_r);

        let tok = send(tok, ctrl1_s, SBCtrl { length: Length:0xB1 });
        let tok = send(tok, ctrl1_s, SBCtrl { length: Length:0xB2 });
        let tok = send(tok, ctrl1_s, SBCtrl { length: Length:0xB3 });
        let tok = send(tok, ctrl1_s, SBCtrl { length: Length:0xB4 });
        let tok = send(tok, ctrl1_s, SBCtrl { length: Length:0xB5 });
        let tok = send(tok, ctrl1_s, SBCtrl { length: Length:0xB6 });
        let tok = send(tok, ctrl1_s, SBCtrl { length: Length:0xB7 });
        let tok = send(tok, ctrl1_s, SBCtrl { length: Length:0xB8 });

        let (tok, data) = recv(tok, data1_r);
        assert_eq(data, SBOutput { data: Data:8, length: Length:0xB1, error: false});
        let (tok, data) = recv(tok, data1_r);
        assert_eq(data, SBOutput { data: Data:9, length: Length:0xB2, error: false});
        let (tok, data) = recv(tok, data1_r);
        assert_eq(data, SBOutput { data: Data:10, length: Length:0xB3, error: false});
        let (tok, data) = recv(tok, data1_r);
        assert_eq(data, SBOutput { data: Data:11, length: Length:0xB4, error: false});
        let (tok, data) = recv(tok, data1_r);
        assert_eq(data, SBOutput { data: Data:12, length: Length:0xB5, error: false});
        let (tok, data) = recv(tok, data1_r);
        assert_eq(data, SBOutput { data: Data:13, length: Length:0xB6, error: false});
        let (tok, data) = recv(tok, data1_r);
        assert_eq(data, SBOutput { data: Data:14, length: Length:0xB7, error: false});
        let (tok, data) = recv(tok, data1_r);
        assert_eq(data, SBOutput { data: Data:15, length: Length:0xB8, error: false});

        send(tok, terminator, true);
    }
}
