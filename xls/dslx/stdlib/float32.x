// 32-bit floating point routines.

pub struct F32 {
  sign: u1,  // sign bit
  bexp: u8,  // biased exponent
  sfd: u23,  // significand (no hidden bit)
}

pub enum FloatTag : u3 {
  NAN       = 0,
  INFINITY  = 1,
  SUBNORMAL = 2,
  ZERO      = 3,
  NORMAL    = 4,
}

pub type TaggedF32 = (FloatTag, F32);

pub fn qnan() -> F32 {
  F32 { sign: false, bexp: u8:0xff, sfd: u23:0x400000 }
}

pub fn zero(sign: u1) -> F32 { F32 { sign: sign, bexp: u8:0, sfd: u23:0 } }
pub fn one(sign: u1) -> F32 { F32 { sign: sign, bexp: u8:127, sfd: u23:0 } }

pub fn inf(sign: u1) -> F32 {
  F32 { sign: sign, bexp: u8:0xff, sfd: u23:0 }
}

// Accessor helpers for the F32 typedef.
pub fn unbiased_exponent(f: F32) -> u9 { (f.bexp as u9) - u9:127 }
pub fn bias(unbiased_exponent_in: u9) -> u8 { (unbiased_exponent_in + u9:127) as u8 }

pub fn flatten(x: F32) -> u32 { x.sign ++ x.bexp ++ x.sfd }

pub fn unflatten(x: u32) -> F32 {
  F32 { sign: (x >> u32:31) as u1, bexp: (x >> u32:23) as u8, sfd: x as u23 }
}

pub fn tag(input_float: F32) -> FloatTag {
  match (input_float.bexp, input_float.sfd) {
    (  u8:0, u23:0) => FloatTag::ZERO;
    (  u8:0,     _) => FloatTag::SUBNORMAL;
    (u8:255, u23:0) => FloatTag::INFINITY;
    (u8:255,     _) => FloatTag::NAN;
    (     _,     _) => FloatTag::NORMAL;
  }
}

test tag {
  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, sfd: u23:0 }), FloatTag::ZERO) in
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:0, sfd: u23:0 }), FloatTag::ZERO) in
  let _ = assert_eq(tag(zero(u1:0)), FloatTag::ZERO) in
  let _ = assert_eq(tag(zero(u1:1)), FloatTag::ZERO) in

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, sfd: u23:1 }), FloatTag::SUBNORMAL) in
  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, sfd: u23:0x7f_ffff }), FloatTag::SUBNORMAL) in

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:12, sfd: u23:0 }), FloatTag::NORMAL) in
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:254, sfd: u23:0x7f_ffff }), FloatTag::NORMAL) in
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:1, sfd: u23:1 }), FloatTag::NORMAL) in

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:255, sfd: u23:0 }), FloatTag::INFINITY) in
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:255, sfd: u23:0 }), FloatTag::INFINITY) in
  let _ = assert_eq(tag(inf(u1:0)), FloatTag::INFINITY) in
  let _ = assert_eq(tag(inf(u1:1)), FloatTag::INFINITY) in

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:255, sfd: u23:1 }), FloatTag::NAN) in
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:255, sfd: u23:0x7f_ffff }), FloatTag::NAN) in
  let _ = assert_eq(tag(qnan()), FloatTag::NAN) in
  ()
}

pub fn subnormals_to_zero(x: F32) -> F32 {
  zero(x.sign) if x.bexp == u8:0 else x
}

pub fn fixed_fraction(input_float: F32) -> u23 {
  let input_significand_magnitude: u25 = u2:0b01 ++ input_float.sfd in
  let unbiased_input_float_exponent: u9 = unbiased_exponent(input_float) in

  let input_fixed_magnitude: u25 = match sgt(unbiased_input_float_exponent, u9:0) {
    true =>
      let significand_left_shift = unbiased_input_float_exponent as u3 in
      input_significand_magnitude << (significand_left_shift as u25);
    _ =>
      let significand_right_shift = (-unbiased_input_float_exponent) as u5 in
      input_significand_magnitude >> (significand_right_shift as u25)
  } in

  let input_fraction_part_magnitude: u24 = input_fixed_magnitude as u23 as u24 in
  let fixed_fraction: u24 = (u24:1<<u24:23) - input_fraction_part_magnitude
    if input_float.sign && input_fraction_part_magnitude != u24:0
    else input_fraction_part_magnitude
  in
  fixed_fraction as u23
}

// Returns a normalized F32 with the given components. 'sfd_with_hidden' is the
// significand including the hidden bit. This function only normalizes in the
// direction of decreasing the exponent. Input must be a normal number or
// zero. Dernormals are flushed to zero in the result.
pub fn normalize(sign: u1, exp: u8, sfd_with_hidden: u24) -> F32 {
  let leading_zeros = clz(sfd_with_hidden) as u8 in
  match (exp <= leading_zeros, leading_zeros) {
    // Significand is zero.
    (_, u8:24) => zero(sign);
    // Flush denormals to zero.
    (true, _) => zero(sign);
    // Normalize.
    _ => F32 { sign: sign,
               bexp: exp - (leading_zeros as u8),
               sfd: (sfd_with_hidden << (leading_zeros as u24))[:23] };
  }
}

// Returns whether or not the given F32 represents an infinite quantity.
pub fn is_inf(x: F32) -> u1 {
  (x.bexp == u8:255) & (x.sfd == u23:0)
}

// Returns whether or not the given F32 represents NaN.
pub fn is_nan(x: F32) -> u1 {
  (x.bexp == u8:255) & (x.sfd != u23:0)
}

test normalize {
  let _ = assert_eq(normalize(u1:0, u8:0x12, u24:0xfe_dcba),
                    F32 { sign: u1:0, bexp: u8:0x12, sfd: u23:0x7e_dcba }) in

  let _ = assert_eq(normalize(u1:0, u8:0x01, u24:0),
                    F32 { sign: u1:0, bexp: u8:0, sfd: u23:0 }) in
  let _ = assert_eq(normalize(u1:0, u8:0xfe, u24:0),
                    F32 { sign: u1:0, bexp: u8:0, sfd: u23:0 }) in

  let _ = assert_eq(normalize(u1:1, u8:100, u24:1),
                    F32 { sign: u1:1, bexp: u8:77, sfd: u23:0 }) in
  let _ = assert_eq(normalize(u1:1,
                              u8:10,
                              u24:0b0000_0000_1000_1111_0000_0101),
                    F32 { sign: u1:1, bexp: u8:2, sfd: u23:0b000_1111_0000_0101_0000_0000 }) in
  let _ = assert_eq(normalize(u1:1,
                              u8:10,
                              u24:0b1000_0000_1000_1111_0000_0101),
                    F32 { sign: u1:1, bexp: u8:10, sfd: u23:0b000_0000_1000_1111_0000_0101 }) in

  // Denormals should be flushed to zero.
  let _ = assert_eq(normalize(u1:1,
                              u8:5,
                              u24:0b0000_0000_1000_1111_0000_0101),
                    zero(u1:1)) in
  let _ = assert_eq(normalize(u1:0,
                              u8:2,
                              u24:0b0010_0000_1000_1111_0000_0101),
                    zero(u1:0)) in
  ()
}

