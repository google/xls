#![feature(type_inference_v2)]

// recv_if with true predicate -- always receives when called
proc RecvIfTrue {
    input_r: chan<u32> in;
    config(input_r: chan<u32> in) { (input_r,) }
    init { () }
    next(state: ()) {
        let (_, _) = recv_if(join(), input_r, true, u32:0);
    }
}

// recv_if with false predicate -- never receives
proc RecvIfFalse {
    input_r: chan<u32> in;
    config(input_r: chan<u32> in) { (input_r,) }
    init { () }
    next(state: ()) {
        let (_, _) = recv_if(join(), input_r, false, u32:0);
    }
}

// send_if with false predicate -- never sends
proc SendIfFalse {
    output_s: chan<u32> out;
    config(output_s: chan<u32> out) { (output_s,) }
    init { () }
    next(state: ()) {
        send_if(join(), output_s, false, u32:0);
    }
}

// recv_if_non_blocking with true predicate -- non-blocking poll, enabled
proc RecvIfNonBlockingTrue {
    input_r: chan<u32> in;
    config(input_r: chan<u32> in) { (input_r,) }
    init { () }
    next(state: ()) {
        let (_, _, _) = recv_if_non_blocking(join(), input_r, true, u32:0);
    }
}

// recv_if_non_blocking with false predicate -- non-blocking poll, disabled
proc RecvIfNonBlockingFalse {
    input_r: chan<u32> in;
    config(input_r: chan<u32> in) { (input_r,) }
    init { () }
    next(state: ()) {
        let (_, _, _) = recv_if_non_blocking(join(), input_r, false, u32:0);
    }
}

#[test_proc]
proc ConditionalIoTest {
    terminator: chan<bool> out;
    recv_if_true_s: chan<u32> out;

    config(terminator: chan<bool> out) {
        let (recv_if_true_s, recv_if_true_r) = chan<u32>("recv_if_true");
        let (_recv_if_false_s, recv_if_false_r) = chan<u32>("recv_if_false");
        let (send_if_false_s, _send_if_false_r) = chan<u32>("send_if_false");
        let (_recv_if_nb_true_s, recv_if_nb_true_r) = chan<u32>("recv_if_nb_true");
        let (_recv_if_nb_false_s, recv_if_nb_false_r) = chan<u32>("recv_if_nb_false");
        spawn RecvIfTrue(recv_if_true_r);
        spawn RecvIfFalse(recv_if_false_r);
        spawn SendIfFalse(send_if_false_s);
        spawn RecvIfNonBlockingTrue(recv_if_nb_true_r);
        spawn RecvIfNonBlockingFalse(recv_if_nb_false_r);
        (terminator, recv_if_true_s)
    }
    // state=0: send to recv_if_true; state=1: send to terminator.
    // Two-cycle design ensures the DSLX trace records RecvIfTrue's RECV
    // (cycle 1) before the terminator fires. SPIN interleaves the same events
    // in the same relative order, so both traces match.
    init { u32:0 }
    next(state: u32) {
        let tok = send_if(join(), recv_if_true_s, state == u32:0, u32:42);
        send_if(tok, terminator, state == u32:1, true);
        state + u32:1
    }
}
