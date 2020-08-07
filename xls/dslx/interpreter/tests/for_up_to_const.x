// Contains a for loop that iterates up to a CONST value.

const FOO = u32:3;

fn f() -> u32 {
  for (i, accum): (u32, u32) in range(u32:0, FOO) {
    accum+i
  }(u32:0)
}

test f {
  assert_eq(f(), u32:3)
}
