
fn main() -> u32 {
  let x0 = clz(u32:0x0005a000) in
  let x1 = clz(x0) in
  clz(x1)
}

test clz {
  let _ = assert_eq(u3:0, clz(u3:0b111)) in
  let _ = assert_eq(u3:1, clz(u3:0b011)) in
  let _ = assert_eq(u3:2, clz(u3:0b001)) in
  let _ = assert_eq(u3:3, clz(u3:0b000)) in
  ()
}

