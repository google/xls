#![cfg(let_terminator_is_semi = true)]

// DSLX standard library routines.

// Returns unsigned mul of x (N bits) and y (M bits) as an N+M bit value.
pub fn [N: u32, M: u32] umul(x: uN[N], y: uN[M]) -> uN[N+M] {
  (x as uN[N+M]) * (y as uN[N+M])
}

// Returns signed mul of x (N bits) and y (M bits) as an N+M bit value.
pub fn [N: u32, M: u32] smul(x: sN[N], y: sN[M]) -> sN[N+M] {
  (x as sN[N+M]) * (y as sN[N+M])
}

test smul {
  let _ = assert_eq(s6:4, smul(s3:2, s3:2));
  let _ = assert_eq(s4:0b1111, smul(s2:0b11, s2:0b01));
  ()
}

test umul {
  let _ = assert_eq(u6:4, umul(u3:2, u3:2));
  let _ = assert_eq(u4:0b0011, umul(u2:0b11, u2:0b01));
  let _ = assert_eq(u4:0b1001, umul(u2:0b11, u2:0b11));
  ()
}

// Returns the value of x-1 with saturation at 0.
pub fn [N: u32] bounded_minus_1(x: uN[N]) -> uN[N] {
  x if x == uN[N]:0 else x-uN[N]:1
}

// Extracts the LSb (least significant bit) from the value `x` and returns it.
pub fn [N: u32] lsb(x: uN[N]) -> u1 {
  x as u1
}

test lsb {
  let _ = assert_eq(u1:0, lsb(u2:0b00));
  let _ = assert_eq(u1:1, lsb(u2:0b01));
  let _ = assert_eq(u1:1, lsb(u2:0b11));
  let _ = assert_eq(u1:0, lsb(u2:0b10));
  ()
}

// Returns the absolute value of x as a signed number.
pub fn [BITS: u32] abs(x: sN[BITS]) -> sN[BITS] {
  -x if x < sN[BITS]:0 else x
}

// Converts an array of N bools to a bits[N] value.
pub fn [N: u32] convert_to_bits(x: bool[N]) -> uN[N] {
  for (i, accum): (u32, uN[N]) in range(u32:0, N) {
   accum | (x[i] as uN[N]) << (N-i-u32:1 as uN[N])
  }(uN[N]:0)
}

test convert_to_bits {
  let _ = assert_eq(u3:0b010, convert_to_bits(bool[3]:[false, true, false]));
  let _ = assert_eq(u3:0b011, convert_to_bits(bool[3]:[false, true, true]));
  ()
}

// Returns (found, index) given array and the element to find within the array.
//
// Note that when found is false, the index is 0 -- 0 is provided instead of a
// value like -1 to prevent out-of-bounds accesses from occurring if the index
// is used in a match expression (which will eagerly evaluate all of its arms),
// to prevent it from creating an error at simulation time if the value is
// ultimately discarded from the unselected match arm.
pub fn [BITS: u32, ELEMS: u32] find_index(
    array: uN[BITS][ELEMS], x: uN[BITS]) -> (bool, u32) {
  // Compute all the positions that are equal to our target.
  let bools: bool[ELEMS] = for (i, accum): (u32, bool[ELEMS]) in range(u32:0, ELEMS) {
    update(accum, i, array[i] == x)
  }((bool[ELEMS]:[false, ...]));

  let x: uN[ELEMS] = convert_to_bits(bools);
  let index = clz(x);
  let found: bool = index != (ELEMS as uN[ELEMS]);
  (found, index as u32 if found else u32:0)
}

test find_index {
  let haystack = u3[4]:[0b001, 0b010, 0b100, 0b111];
  let _ = assert_eq((true, u32:1), find_index(haystack, u3:0b010));
  let _ = assert_eq((true, u32:3), find_index(haystack, u3:0b111));
  let _ = assert_eq((false, u32:0), find_index(haystack, u3:0b000));
  ()
}

// Concatenates 3 values of potentially different bitwidths to a single value.
pub fn [X: u32, Y: u32, Z: u32] concat3(
    x: bits[X], y: bits[Y], z: bits[Z]) -> bits[X+Y+Z] {
  x ++ y ++ z
}

test concat3 {
  let _ = assert_eq(u12:0b111000110010,
                    concat3(u6:0b111000, u4:0b1100, u2:0b10));
  ()
}

fn divceil(x: u32, y: u32) -> u32 {
  (x-u32:1) / y + u32:1
}

test divceil {
  let _ = assert_eq(u32:3, divceil(u32:5, u32:2));
  let _ = assert_eq(u32:2, divceil(u32:4, u32:2));
  let _ = assert_eq(u32:2, divceil(u32:3, u32:2));
  let _ = assert_eq(u32:1, divceil(u32:2, u32:2));
  _
}

pub fn round_up_to_nearest(x: u32, y: u32) -> u32 {
  (divceil(x, y) * y) as u32
}

test round_up_to_nearest {
  let _ = assert_eq(u32:4, round_up_to_nearest(u32:3, u32:2));
  let _ = assert_eq(u32:4, round_up_to_nearest(u32:4, u32:2));
  _
}

pub fn [N: u32] rrot(x: bits[N], y: bits[N]) -> bits[N] {
  (x >> y) | (x << ((N as bits[N]) - y))
}

test rrot {
  let _ = assert_eq(bits[3]:0b101, rrot(bits[3]:0b011, bits[3]:1));
  let _ = assert_eq(bits[3]:0b011, rrot(bits[3]:0b110, bits[3]:1));
  _
}

pub fn [N: u32] umin(x: uN[N], y: uN[N]) -> uN[N] {
  x if x < y else y
}

test umin {
  let _ = assert_eq(u1:0, umin(u1:1, u1:0));
  let _ = assert_eq(u1:1, umin(u1:1, u1:1));
  let _ = assert_eq(u2:2, umin(u2:3, u2:2));
  ()
}

fn [N: u32] clog2(x: bits[N]) -> bits[N] {
  (N as bits[N]) - clz(x-bits[N]:1) if x >= bits[N]:1 else bits[N]:0
}

test clog2 {
  let _ = assert_eq(u32:0, clog2(u32:0));
  let _ = assert_eq(u32:0, clog2(u32:1));
  let _ = assert_eq(u32:1, clog2(u32:2));
  let _ = assert_eq(u32:2, clog2(u32:3));
  let _ = assert_eq(u32:2, clog2(u32:4));
  let _ = assert_eq(u32:3, clog2(u32:5));
  let _ = assert_eq(u32:3, clog2(u32:6));
  let _ = assert_eq(u32:3, clog2(u32:7));
  let _ = assert_eq(u32:3, clog2(u32:8));
  let _ = assert_eq(u32:4, clog2(u32:9));
  ()
}
