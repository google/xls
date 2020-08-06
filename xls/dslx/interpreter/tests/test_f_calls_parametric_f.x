fn [N: u8] p(_: bits[N]) -> u8 {
  N
}

fn f() -> u8 {
  match false {
    // TODO(cdleary): 2020-08-05 Turn this match arm into a wildcard match when
    // https://github.com/google/xls/issues/75 is resolved.
    false => p(u8:0);
    _ => u8:0
  }
}

test t {
  assert_eq(u8:8, f())
}
