// Copyright 2026 The XLS Authors
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

#![feature(explicit_state_access)]
#![feature(generics)]

fn const_match_literal_bool(a: u32, b: u32) -> u32 {
    const match false {
        true => a,
        false => b,
    }
}

#[test]
fn test_const_match_literal_bool() { assert_eq(const_match_literal_bool(u32:1, u32:2), u32:2); }

fn const_match_const_bool(a: u32, b: u32) -> u32 {
    const A = true;
    const match A {
        true => a,
        false => b,
    }
}

#[test]
fn test_const_match_const_bool() { assert_eq(const_match_const_bool(u32:1, u32:2), u32:1); }

fn const_match_param_bool_case1<A: bool>(a: u32, b: u32) -> u32 {
    const match A {
        true => a,
        false => b,
    }
}

fn const_match_param_bool_case2<A: u32>(a: u32, b: u32) -> u32 {
    trace_fmt!("Param used: {}", A);
    const COND = true;
    const match COND {
        true => a,
        false => b,
    }
}

#[test]
fn test_const_match_param_bool() {
    assert_eq(const_match_param_bool_case1<false>(u32:1, u32:2), u32:2);
    assert_eq(const_match_param_bool_case2<u32:5>(u32:1, u32:2), u32:1);
}

type ComplexType = (u32, (u32, u32, u32));

fn const_match_tuple<A: ComplexType>() -> u32 {
    match A {
        (u32:1, (u32:3, ..)) => u32:0,
        (u32:2, (u32:2, _, _)) => u32:1,
        (u32:1, ..) => u32:2,
        (u32:3, (x, u32:1, u32:1)) => x,
        _ => u32:4,
    }
}

#[test]
fn test_const_match_tuple() {
    const TUPLE1 = (u32:1, (u32:3, u32:2, u32:1));
    const TUPLE2 = (u32:2, (u32:2, u32:5, u32:4));
    const TUPLE3 = (u32:1, (u32:2, u32:3, u32:4));
    const TUPLE4 = (u32:3, (u32:3, u32:1, u32:1));
    const TUPLE5 = (u32:5, (u32:2, u32:3, u32:4));
    assert_eq(const_match_tuple<TUPLE1>(), u32:0);
    assert_eq(const_match_tuple<TUPLE2>(), u32:1);
    assert_eq(const_match_tuple<TUPLE3>(), u32:2);
    assert_eq(const_match_tuple<TUPLE4>(), u32:3);
    assert_eq(const_match_tuple<TUPLE5>(), u32:4);
}

fn const_match_typecheck<A: bool>(a: u32, b: u16) -> u32 {
    let result = const match A {
        true => a,
        _ => b,
    };
    result as u32
}

#[test]
fn test_const_match_typecheck() {
    assert_eq(const_match_typecheck<true>(u32:1, u16:2), u32:1);
    assert_eq(const_match_typecheck<false>(u32:1, u16:2), u32:2);
}

fn const_match_types<A: u32>() -> u32 {
    const B = u32:9;
    const match A {
        u32:0..u32:3 => u32:0,
        u32:4 => u32:1,
        u32:5 | u32:6 | u32:7 => u32:2,
        u32:8 => u32::MAX,
        B => u32:800,
        _ => u32:1000,
    }
}

#[test]
fn test_const_match_types() { assert_eq(const_match_types<u32:8>(), u32::MAX); }

proc Falsy {
    req_r: chan<()> in,
    resp_s: chan<bool> out,
}

impl Falsy {
    fn new(req_r: chan<()> in, resp_s: chan<bool> out) -> Self {
        Self { req_r, resp_s }
    }

    fn next(self) {
        let (tok, _d) = recv(join(), self.req_r);
        let tok = send(tok, self.resp_s, false);
    }
}

proc Truthy {
    req_r: chan<()> in,
    resp_s: chan<bool> out,
}

impl Truthy {
    fn new(req_r: chan<()> in, resp_s: chan<bool> out) -> Self {
        Self { req_r, resp_s }
    }

    fn next(self) {
        let (tok, _d) = recv(join(), self.req_r);
        let tok = send(tok, self.resp_s, true);
    }
}

proc Foo<CONFIG: bool> {}

impl Foo<CONFIG> {
    fn new(req_r: chan<()> in, resp_s: chan<bool> out) -> Self {
        const match CONFIG {
            true => Truthy::new(req_r, resp_s).spawn(),
            false => Falsy::new(req_r, resp_s).spawn(),
        };
        Self {}
    }
}

#[test]
proc test_const_match_in_config {
    req_s: chan<()>[2] out,
    resp_r: chan<bool>[2] in,
    terminator: chan<bool> out,
}

impl test_const_match_in_config {
    fn new(terminator: chan<bool> out) -> Self {
        let (req_s, req_r) = chan<()>[2]("req");
        let (resp_s, resp_r) = chan<bool>[2]("resp");
        Foo<true>::new(req_r[0], resp_s[0]).spawn();
        Foo<false>::new(req_r[1], resp_s[1]).spawn();
        Self { req_s, resp_r, terminator }
    }

    fn next(self) {
        let tok = send(join(), self.req_s[0], ());
        let (tok, resp) = recv(tok, self.resp_r[0]);
        assert_eq(resp, true);
        let tok = send(join(), self.req_s[1], ());
        let (tok, resp) = recv(tok, self.resp_r[1]);
        assert_eq(resp, false);
        let tok = send(tok, self.terminator, true);
    }
}

fn main() -> (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32) {
    const TUPLE1 = (u32:1, (u32:3, u32:2, u32:1));
    const TUPLE2 = (u32:2, (u32:2, u32:5, u32:4));
    const TUPLE3 = (u32:1, (u32:2, u32:3, u32:4));
    const TUPLE4 = (u32:3, (u32:3, u32:1, u32:1));
    const TUPLE5 = (u32:5, (u32:2, u32:3, u32:4));
    (
        const_match_literal_bool(u32:1, u32:2), const_match_const_bool(u32:1, u32:2),
        const_match_param_bool_case1<false>(u32:1, u32:2),
        const_match_param_bool_case2<u32:6>(u32:1, u32:2), const_match_tuple<TUPLE1>(),
        const_match_tuple<TUPLE2>(), const_match_tuple<TUPLE3>(), const_match_tuple<TUPLE4>(),
        const_match_tuple<TUPLE5>(), const_match_typecheck<true>(u32:1, u16:2),
        const_match_typecheck<false>(u32:1, u16:2),
    )
}

proc ConstMatchInst {}

impl ConstMatchInst {
    fn new(req_r: chan<()> in, resp_s: chan<bool> out) -> Self {
        const CONFIG = true;
        const match CONFIG {
            true => Truthy::new(req_r, resp_s).spawn(),
            false => Falsy::new(req_r, resp_s).spawn(),
        };
        Self {}
    }
}
