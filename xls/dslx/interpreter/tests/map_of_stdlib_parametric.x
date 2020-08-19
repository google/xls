import std

fn main(x: s3[3]) -> s3[3] {
  let y: s3[3] = map(x, std::abs);
  y
}

test main {
  let got: s3[3] = main(s3[3]:[-1, 1, 0]);
  assert_eq(s3[3]:[1, 1, 0], got)
}
