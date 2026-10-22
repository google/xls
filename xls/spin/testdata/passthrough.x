#![feature(type_inference_v2)]

proc Passthrough {
    req_r: chan<u32> in;
    resp_s: chan<u32> out;

    config(req_r: chan<u32> in, resp_s: chan<u32> out) { (req_r, resp_s) }
    init { () }
    next(state: ()) {
        let (tok, data) = recv(join(), req_r);
        let tok = send(tok, resp_s, data);
    }
}

#[test_proc]
proc PassthroughTest {
    terminator: chan<bool> out;
    req_s: chan<u32> out;
    resp_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<u32>("req");
        let (resp_s, resp_r) = chan<u32>("resp");
        spawn Passthrough(req_r, resp_s);
        (terminator, req_s, resp_r)
    }
    // count=10..1: forward one value per tick; fire terminator at count=1.
    // count=0: no-op so SPIN extra iterations produce no channel events.
    // Passthrough's blocking recv then stalls naturally until __terminated.
    init { u32:10 }
    next(count: u32) {
        let tok = join();
        let tok = send_if(tok, req_s, count > u32:0, count);
        let (tok, received_data) = recv_if(tok, resp_r, count > u32:0, u32:0);
        assert_eq(count, received_data);
        send_if(tok, terminator, count == u32:1, true);
        if count > u32:0 { count - u32:1 } else { u32:0 }
    }
}
