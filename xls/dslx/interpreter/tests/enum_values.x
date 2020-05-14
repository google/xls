enum MyEnum : u2 {
  A = 0,
  B = 1,
  C = 2,
  D = 3,
}

test enum_values {
  let a: MyEnum = MyEnum::A in
  // Cast some to unsigned.
  let b_u2: u2 = MyEnum::B as u2 in
  let c_u2: u2 = MyEnum::C as u2 in
  // Cast one to signed and sign extend it.
  let d_s2: s2 = MyEnum::D as s2 in
  let d_signext: s3 = d_s2 as s3 in
  let _ = assert_eq(d_signext, s3:0b111) in

  // Extend values to u3 and sum them up.
  let sum = (a as u2 as u3) + (b_u2 as u3) + (c_u2 as u3) + (d_s2 as u2 as u3) in
  let _ = assert_eq(sum, u3:6) in

  // A bunch of equality/comparison checks.
  let _ = assert_eq(a, MyEnum::A) in
  let _ = assert_eq(true, a == MyEnum::A) in
  let _ = assert_eq(false, a != MyEnum::A) in
  let _ = assert_eq(a, a as u2 as MyEnum) in
  let _ = assert_eq(a, a as u2 as MyEnum) in
  ()
}

test enum_values_widen_from_unsigned {
  let d_s4: s4 = MyEnum::D as s4 in
  assert_eq(s4:0b0011, d_s4)
}

test enum_values_narrow_from_unsigned {
  let d_s1: s1 = MyEnum::D as s1 in
  assert_eq(s1:0b1, d_s1)
}

enum MyEnumSigned : s2 {
  A = 0,
  B = 1,
  C = 2,
  D = 3,
}

test enum_values_widen_from_signed {
  let d_s4: s4 = MyEnumSigned::D as s4 in
  assert_eq(s4:0b1111, d_s4)
}

test enum_values_narrow_from_signed {
  let d_s1: s1 = MyEnumSigned::D as s1 in
  assert_eq(s1:0b1, d_s1)
}
