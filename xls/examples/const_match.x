#![feature(type_inference_v2)]

fn match_bool(a: u32, b: u32) -> u32 {
    const A = true;
    let result = const match A {
        true => a,
        false => b
    };
    result
}

fn match_bool_param<A: bool>(a: u32, b: u32) -> u32 {
    const match A {
        true => a,
        false => b
    }
}

fn matcher_types<A: u32>() -> u32 {
    const B = u32:9;
    const match A {
        u32:0..u32:3 => u32:0,
        u32:4 => u32:1,
        u32:5 | u32:6 | u32:7 => u32:2,
        // TODO use colon reference
        u32:8 => u32:3,
        B => u32:800,
        _ => u32:1000,
    }
}

type ARG = (u32, (u32, u32, u32));
fn matcher_tuple<A: ARG>() -> u32 {
    const match A {
        (u32:1, (u32:3, ..)) => u32:0,
        (u32:2, (u32:2, _, _)) => u32:1,
        (u32:1, ..) => u32:2,
        (u32:3, (x, u32:1, u32:1)) => x,
        _ => u32:4,
    }
}

#[test]
fn test_all_uniform_types() {
    assert_eq(match_bool(u32:1, u32:2), u32:1);
    assert_eq(match_bool_param<true>(u32:1, u32:2), u32:1);
    assert_eq(match_bool_param<false>(u32:1, u32:2), u32:2);
    assert_eq(matcher_types<u32:1>(), u32:0);
    assert_eq(matcher_types<u32:4>(), u32:1);
    assert_eq(matcher_types<u32:7>(), u32:2);
    assert_eq(matcher_types<u32:8>(), u32:3);
    assert_eq(matcher_types<u32:9>(), u32:800);
    assert_eq(matcher_types<u32:100>(), u32:1000);
    let tuple1 = (u32:1, (u32:3, u32:0, u32:0));
    let tuple2 = (u32:2, (u32:2, u32:0, u32:0));
    let tuple3 = (u32:1, (u32:4, u32:0, u32:0));
    let tuple4 = (u32:3, (u32:5, u32:1, u32:1));
    let tuple5 = (u32:7, (u32:5, u32:1, u32:1));
    assert_eq(matcher_tuple<tuple1>(), u32:0);
    assert_eq(matcher_tuple<tuple2>(), u32:1);
    assert_eq(matcher_tuple<tuple3>(), u32:2);
    assert_eq(matcher_tuple<tuple4>(), u32:5);
    assert_eq(matcher_tuple<tuple5>(), u32:4);
}

fn main() -> (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32) {
    let tuple1 = (u32:1, (u32:3, u32:0, u32:0));
    let tuple2 = (u32:2, (u32:2, u32:0, u32:0));
    let tuple3 = (u32:1, (u32:4, u32:0, u32:0));
    let tuple4 = (u32:3, (u32:5, u32:1, u32:1));
    let tuple5 = (u32:7, (u32:5, u32:1, u32:1));
    (
        match_bool(u32:1, u32:2),
        match_bool_param<true>(u32:1, u32:2),
        match_bool_param<false>(u32:1, u32:2),
        matcher_types<u32:1>(),
        matcher_types<u32:4>(),
        matcher_types<u32:7>(),
        matcher_types<u32:8>(),
        matcher_types<u32:9>(),
        matcher_types<u32:100>(),
        matcher_tuple<tuple1>(),
        matcher_tuple<tuple2>(),
        matcher_tuple<tuple3>(),
        matcher_tuple<tuple4>(),
        matcher_tuple<tuple5>(),
    )
}

// step 2: do not typecheck the unused branch
// fn foo<A: bool>() -> u32 {
//     const B1 = u32:1;
//     const B2 = u16:2;
//     let result = const match A {
//         true => B1,
//         _ => B2
//     };
//     result as u32
// }
