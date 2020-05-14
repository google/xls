fn cast_to_array(x: u6) -> u2[3] {
  x as u2[3]
}

fn cast_from_array(a: u2[3]) -> u6 {
  a as u6
}

fn concat_arrays(a: u2[3], b: u2[3]) -> u2[6] {
  a ++ b
}

test cast_to_array {
  let a_value: u6 = u6:0b011011 in
  let a: u2[3] = cast_to_array(a_value) in
  let a_array = u2[3]:[1, 2, 3] in
  let _ = assert_eq(a, a_array) in
  // Note: converting back from array to bits gives the original value.
  let _ = assert_eq(a_value, cast_from_array(a)) in

  let b_value: u6 = u6:0b111001 in
  let b_array: u2[3] = u2[3]:[3, 2, 1] in
  let b: u2[3] = cast_to_array(b_value) in
  let _ = assert_eq(b, b_array) in
  let _ = assert_eq(b_value, cast_from_array(b)) in

  // Concatenation of bits is analogous to concatenation of their converted
  // arrays. That is:
  //
  //  convert(concat(a, b)) == concat(convert(a), convert(b))
  let concat_value: u12 = a_value ++ b_value in
  let concat_array: u2[6] = concat_value as u2[6] in
  let _ = assert_eq(concat_array, concat_arrays(a_array, b_array)) in

  // Show a few classic "endianness" example using 8-bit array values.
  let x = u32:0xdeadbeef in
  let _ = assert_eq(x as u8[4], u8[4]:[0xde, 0xad, 0xbe, 0xef]) in
  let y = u16:0xbeef in
  let _ = assert_eq(y as u8[2], u8[2]:[0xbe, 0xef]) in

  ()
}
