
fn main() -> u32 {
  let x0 = clz(u32:0x0005a000) in
  let x1 = clz(x0) in
  clz(x1)
}

test ctz {
  let _ = assert_eq(u3:2, ctz(u3:0b100)) in
  let _ = assert_eq(u3:1, ctz(u3:0b010)) in
  let _ = assert_eq(u3:0, ctz(u3:0b001)) in
  let _ = assert_eq(u3:0, ctz(u3:0b111)) in
  let _ = assert_eq(u3:3, ctz(u3:0b000)) in
  ()
}

