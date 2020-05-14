import std

fn umul_2(x: u4) -> u4 {
  std::umul(x, u4:2) as u4
}

fn umul_2_widening(x: u4) -> u6 {
  std::umul(x, u2:2)
}

fn [N: u4] umul_2_parametric(x: uN[N]) -> uN[N] {
  std::umul(x, uN[N]:2) as uN[N]
}

fn main() -> u4[8] {
  let x0 = u4[8]:[0, 1, 2, 3, 4, 5, 6, 7] in
  let result_0 = map(x0, std::bounded_minus_1) in
  let result_1 = map(x0, umul_2) in
  let result_2 = map(x0, umul_2_parametric) in
  let result_3 = map(x0, clz) in
  map(map(map(x0, std::bounded_minus_1), umul_2), clz)
}

test maps {
  let x0 = u4[8]:[0, 1, 2, 3, 4, 5, 6, 7] in
  let expected = u4[8]:[0, 2, 4, 6, 8, 10, 12, 14] in
  let expected_u6 = u6[8]:[0, 2, 4, 6, 8, 10, 12, 14] in
  let _: () = assert_eq(expected, map(x0, umul_2)) in
  let _: () = assert_eq(expected_u6, map(x0, umul_2_widening)) in
  let _: () = assert_eq(expected, map(x0, umul_2_parametric)) in
  ()
}
