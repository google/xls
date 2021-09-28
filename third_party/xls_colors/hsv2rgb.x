//
// The MIT License (MIT)
//
// Copyright (c) 2016  B. Stultiens
// Copyright 2021 The XLS Authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

// Fast HSV to RGB conversion.
//
// Range:
//   h: [0, 0x5ff], s: [0, 0xff], v: [0, 0xff]
//   r: [0,  0xff], g: [0, 0xff], b: [0, 0xff]
//
// Ported from https://www.vagrearg.org/content/hsvrgb.

pub fn slope(h_fraction: u16, s: u8, v:u8) -> u8{
  let u:u32 = (v as u32) * ((u32:255 << 8) - ((s as u16) * h_fraction) as u32);
  let u:u32 = u + (u >> 8);
  let u:u32 = u + (v as u32);
  u[16:24]
}

pub fn hsv2rgb(h: u16, s: u8, v: u8) -> (u8, u8, u8) {
  let sextant:u8 = h[8:16];
  let sextant:u8 = if sextant > u8:5 { u8:5 } else { sextant };

  let ww:u16 = (v as u16) * ((u8:255 - s) as u16);
  let ww:u16 = ww + u16:1;
  let ww:u16 = ww + ww >> 8;
  let c:u8 = ww[8:16];

  let h_fraction:u16 = h[0:8] as u16;
  let nh_fraction:u16 = u16:256 - h_fraction;
  let u:u8 = slope(nh_fraction, s, v);
  let d:u8 = slope(h_fraction, s, v);

  if s == u8:0 { (v, v, v) }
  else {
    match sextant {
      u8:0 => (v, u, c),
      u8:1 => (d, v, c),
      u8:2 => (c, v, u),
      u8:3 => (c, d, v),
      u8:4 => (u, c, v),
      u8:5 => (v, c, d),
      _ => fail!((u8:0, u8:0, u8:0)),
   }
  }
}

#![test]
fn hsv2rgb_test() {
  let _= assert_eq(hsv2rgb(u16:0, u8:0, u8:0), (u8:0, u8:0, u8:0));
  let _= assert_eq(hsv2rgb(u16:0, u8:0, u8:255), (u8:255, u8:255, u8:255));
  let _= assert_eq(hsv2rgb(u16:300, u8:0, u8:127), (u8:127, u8:127, u8:127));
  let _= assert_eq(hsv2rgb(u16:0, u8:255, u8:255), (u8:255, u8:0, u8:0));
  let _= assert_eq(hsv2rgb(u16:128, u8:255, u8:255), (u8:255, u8:127, u8:0));
  let _= assert_eq(hsv2rgb(u16:256, u8:255, u8:255), (u8:255, u8:255, u8:0));
  let _= assert_eq(hsv2rgb(u16:384, u8:255, u8:255), (u8:127, u8:255, u8:0));
  let _= assert_eq(hsv2rgb(u16:512, u8:255, u8:255), (u8:0, u8:255, u8:0));
  let _= assert_eq(hsv2rgb(u16:640, u8:255, u8:255), (u8:0, u8:255, u8:127));
  let _= assert_eq(hsv2rgb(u16:768, u8:255, u8:255), (u8:0, u8:255, u8:255));
  let _= assert_eq(hsv2rgb(u16:896, u8:255, u8:255), (u8:0, u8:127, u8:255));
  let _= assert_eq(hsv2rgb(u16:1024, u8:255, u8:255), (u8:0, u8:0, u8:255));
  let _= assert_eq(hsv2rgb(u16:1152, u8:255, u8:255), (u8:127, u8:0, u8:255));
  let _= assert_eq(hsv2rgb(u16:1280, u8:255, u8:255), (u8:255, u8:0, u8:255));
  _
}
