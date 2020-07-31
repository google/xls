// (Dummy) wrapper around reverse.
fn [N: u32] wrapper(x: bits[N]) -> bits[N] {
  rev(x)
}

// Target for IR conversion that works on u3s.
fn main(x: u3) -> u3 {
  wrapper(x)
}

// Reverse examples.
#![unittest]
reverse {
  let _ = assert_eq(u3:0b100, main(u3:0b001)) in
  let _ = assert_eq(u3:0b001, main(u3:0b100)) in
  let _ = assert_eq(bits[0]:0, rev(bits[0]:0)) in
  let _ = assert_eq(u1:1, rev(u1:1)) in
  let _ = assert_eq(u2:0b10, rev(u2:0b01)) in
  let _ = assert_eq(u2:0b00, rev(u2:0b00)) in
  ()
}

// Reversing a value twice gets you the original value.
#![quickcheck]
fn prop_double_reverse(x: u32) -> bool {
  x == rev(rev(x))
}

// Reversing a value means that the lsb becomes the msb.
#![quickcheck]
fn prop_lsb_becomes_msb(x: u32) -> bool {
  let reversed_x = rev(x) in
  x[0:1] == reversed_x[-1:]
}
