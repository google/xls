import std

pub fn my_lsb(x: u3) -> u1 {
  std::lsb(x)
}

pub struct Point {
  x: u32,
  y: u32,
}

pub enum MyEnum : u8 {
  FOO = 42,
  BAR = 64,
}
