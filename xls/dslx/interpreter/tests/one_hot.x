test one_hot {
  // LSb has priority.
  let _ = assert_eq(u4:0b1000, one_hot(u3:0b000, true)) in
  let _ = assert_eq(u4:0b0001, one_hot(u3:0b001, true)) in
  let _ = assert_eq(u4:0b0010, one_hot(u3:0b010, true)) in
  let _ = assert_eq(u4:0b0001, one_hot(u3:0b011, true)) in
  let _ = assert_eq(u4:0b0100, one_hot(u3:0b100, true)) in
  let _ = assert_eq(u4:0b0001, one_hot(u3:0b101, true)) in
  let _ = assert_eq(u4:0b0010, one_hot(u3:0b110, true)) in
  let _ = assert_eq(u4:0b0001, one_hot(u3:0b111, true)) in
  // MSb has priority.
  let _ = assert_eq(u4:0b1000, one_hot(u3:0b000, false)) in
  let _ = assert_eq(u4:0b0001, one_hot(u3:0b001, false)) in
  let _ = assert_eq(u4:0b0010, one_hot(u3:0b010, false)) in
  let _ = assert_eq(u4:0b0010, one_hot(u3:0b011, false)) in
  let _ = assert_eq(u4:0b0100, one_hot(u3:0b100, false)) in
  let _ = assert_eq(u4:0b0100, one_hot(u3:0b101, false)) in
  let _ = assert_eq(u4:0b0100, one_hot(u3:0b110, false)) in
  let _ = assert_eq(u4:0b0100, one_hot(u3:0b111, false)) in
  ()
}
