test numerical_conversions {
  let s8_m2 = s8:-2 in
  let u8_m2 = u8:-2 in
  // Sign extension (source type is signed).
  let _ = assert_eq(s32:-2, s8_m2 as s32) in
  let _ = assert_eq(u32:-2, s8_m2 as u32) in
  let _ = assert_eq(s16:-2, s8_m2 as s16) in
  let _ = assert_eq(u16:-2, s8_m2 as u16) in
  // Zero extension (source type is unsigned).
  let _ = assert_eq(u32:0xfe, u8_m2 as u32) in
  let _ = assert_eq(s32:0xfe, u8_m2 as s32) in
  // Nop (bitwidth is unchanged).
  let _ = assert_eq(s8:-2, s8_m2 as s8) in
  let _ = assert_eq(s8:-2, u8_m2 as s8) in
  let _ = assert_eq(u8:-2, u8_m2 as u8) in
  let _ = assert_eq(s8:-2, u8_m2 as s8) in
  ()
}
