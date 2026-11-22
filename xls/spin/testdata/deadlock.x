proc A {
    req0_r: chan<u32> in;
    resp0_s: chan<u32> out;
    resp1_s: chan<u32> out;

    init { u32:0 }

    config(
        req0_r: chan<u32> in,
        resp0_s: chan<u32> out,
        resp1_s: chan<u32> out
    )  {
        (req0_r, resp0_s, resp1_s)
    }

    next(state: u32) {
        // This does not work in the DSLX interpreter
        let (tok, _) = recv(join(), req0_r);
        let tok =  send(join(), resp0_s, state);

        // This does, but has exactly the same meaning as the code above
        // let tok =  send(join(), resp0_s, state);
        // let (tok, _) = recv(join(), req0_r);

        let tok = send(join(), resp1_s, state);
        state + u32:1
    }
}

proc B {
    req0_r: chan<u32> in;
    resp0_s: chan<u32> out;

    init {}
    config(req0_r: chan<u32> in, resp0_s: chan<u32> out) {
        (req0_r, resp0_s)
    }

    next(state: ()) {
        let (tok, data) = recv(join(), req0_r);
        // Data dependency enforces correct ordering of IO operations
        let tok = send(join(), resp0_s, data);
    }
}

#[test_proc]
proc Test {
    terminator: chan<bool> out;
    resp1_r: chan<u32> in;

    init {}
    config(terminator: chan<bool> out)  {
        let (req0_s, req0_r) = chan<u32>("req0");
        let (resp0_s, resp0_r) = chan<u32>("resp0");
        let (resp1_s, resp1_r) = chan<u32>("resp1");

        spawn A(req0_r, resp0_s, resp1_s);
        spawn B(resp0_r, req0_s);

        (terminator, resp1_r)
    }

    next(state: ()) {
        let tok = const for (i, tok) in u32:0..u32:10 {
            let (tok, data) = recv(tok, resp1_r);
            trace_fmt!("Data: {}", data);
            tok
        }(join());

        send(tok, terminator, true);
    }
}
