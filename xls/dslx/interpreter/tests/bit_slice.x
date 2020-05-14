test bit_slice {
  let _ = assert_eq(u2:0b11, bit_slice(u6:0b100111, u6:0, u2:0)) in
  let _ = assert_eq(u2:0b11, bit_slice(u6:0b100111, u6:1, u2:0)) in
  let _ = assert_eq(u2:0b01, bit_slice(u6:0b100111, u6:2, u2:0)) in
  let _ = assert_eq(u2:0b00, bit_slice(u6:0b100111, u6:3, u2:0)) in

  let _ = assert_eq(u3:0b111, bit_slice(u6:0b100111, u6:0, u3:0)) in
  let _ = assert_eq(u3:0b011, bit_slice(u6:0b100111, u6:1, u3:0)) in
  let _ = assert_eq(u3:0b001, bit_slice(u6:0b100111, u6:2, u3:0)) in
  let _ = assert_eq(u3:0b100, bit_slice(u6:0b100111, u6:3, u3:0)) in
  ()
}
