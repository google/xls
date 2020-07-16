fn double(n: u32) -> u32 {
  n * u32:2
}

fn [A: u32, B: u32 = double(A)] self_append(x: uN[A]) -> uN[B] {
  x++x
}

fn main() -> (u10, u20) {
  let x1 = self_append(u5:5) in
  let x2 = self_append(u10:10) in
  (x1, x2)
}

test derived_parametric_functions {
  let arr = map([u2:1, u2:2], self_append) in
  let _ = assert_eq(u4:5, arr[u32:0]) in
  let _ = assert_eq(u4:10, arr[u32:1]) in

  let _ = assert_eq(u4:5, self_append(u2:1)) in
  let _ = assert_eq(u6:18, self_append(u3:2)) in
  ()
}
