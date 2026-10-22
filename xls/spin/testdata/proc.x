#![feature(type_inference_v2)]

proc MyProc {
    req_r: chan<u8> in;
    resp_s: chan<u32> out;

    config(
        req_r: chan<u8> in,
        resp_s: chan<u32> out
    ) { (req_r, resp_s) }

    init { u16:0 }

    next(state: u16) {
        let (tok, data) = recv(join(), req_r);
        let tok = send(tok, resp_s, (data as u16 + state) as u32);
        state + u16:1
    }
}
