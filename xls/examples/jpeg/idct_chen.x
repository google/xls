// Copyright 2020 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Implementation of 2D 8x8 inverse discrete cosine transform (IDCT) -- as a
// sequence of 8-point IDCTs -- for rows and columns of the 8x8 "Minimum Coded
// Unit" (MCU) coefficients.
//
// This is a port of the go IDCT algorithm, which is ultimately a fixed-point
// version of Chen et al (1977): https://golang.org/src/image/jpeg/idct.go
//
// Note that this is not likely to be the best algorithm in hardware -- IDCT
// algorithms have a long history of trying to find factorizations that
// minimize the number of multiply operations required -- but it's a canonical
// reference for comparison to other JPEG decoding solutions (e.g. ensuring
// golden reference outputs are identical).

const COEFF_PER_MCU = u32:64;
const COEFF_PER_MCU_U8 = u8:64;
const W1 = s32:2841;  // 2048*sqrt(2)*cos(1*pi/16)
const W2 = s32:2676;  // 2048*sqrt(2)*cos(2*pi/16)
const W3 = s32:2408;  // 2048*sqrt(2)*cos(3*pi/16)
const W5 = s32:1609;  // 2048*sqrt(2)*cos(5*pi/16)
const W6 = s32:1108;  // 2048*sqrt(2)*cos(6*pi/16)
const W7 = s32:565;  // 2048*sqrt(2)*cos(7*pi/16)
const R2 = s32:181;  // 256/sqrt(2)

// Performs an 8-point 1D IDCT.
//
// This uses fixed point scaling factors that are burned in to the computation.
//
// TODO(cdleary): 2020-08-09 It should be possible to parameterize the fixed
// pointness out into types, so that we can consolidate with idct_col below as
// fixed point library comes online.
fn idct_row(s: s32[8]) -> s32[8] {
    let w1pw7 = W1 + W7;
    let w1mw7 = W1 - W7;
    let w2pw6 = W2 + W6;
    let w2mw6 = W2 - W6;
    let w3pw5 = W3 + W5;
    let w3mw5 = W3 - W5;
    let x0 = (s[0] << 11) + 128;
    let x1 = s[4] << 11;
    let x2 = s[6];
    let x3 = s[2];
    let x4 = s[1];
    let x5 = s[7];
    let x6 = s[5];
    let x7 = s[3];

    // Stage 1.
    let x8 = W7 * (x4 + x5);
    let x4 = x8 + w1mw7 * x4;
    let x5 = x8 - w1pw7 * x5;
    let x8 = W3 * (x6 + x7);
    let x6 = x8 - w3mw5 * x6;
    let x7 = x8 - w3pw5 * x7;

    // Stage 2.
    let x8 = x0 + x1;
    let x0 = x0 - x1;
    let x1 = W6 * (x3 + x2);
    let x2 = x1 - w2pw6 * x2;
    let x3 = x1 + w2mw6 * x3;
    let x1 = x4 + x6;
    let x4 = x4 - x6;
    let x6 = x5 + x7;
    let x5 = x5 - x7;

    // Stage 3.
    let x7 = x8 + x3;
    let x8 = x8 - x3;
    let x3 = x0 + x2;
    let x0 = x0 - x2;
    let x2 = (R2 * (x4 + x5) + 128) >> 8;
    let x4 = (R2 * (x4 - x5) + 128) >> 8;

    // Stage 4.
    let result = s32[8]:[
        (x7 + x1) >> 8, (x3 + x2) >> 8, (x0 + x4) >> 8, (x8 + x6) >> 8, (x8 - x6) >> 8,
        (x0 - x4) >> 8, (x3 - x2) >> 8, (x7 - x1) >> 8,
    ];
    result
}

#[test]
fn idct_row_test() {
    let input = s32[8]:[0x17, 0xffffffff, 0xfffffffe, 0, 0, 0, 0, 0];
    let got = idct_row(input);
    let want = s32[8]:[0x98, 0xa6, 0xba, 0xcb, 0xcf, 0xc7, 0xb9, 0xae];
    assert_eq(want, got)
}

// Extracts a row (8 adjacent values) from a 64-value coefficient array.
fn get_row(a: s32[COEFF_PER_MCU], rowno: u8) -> s32[8] {
    s32[8]:[
        a[8 * rowno + 0], a[8 * rowno + 1], a[8 * rowno + 2], a[8 * rowno + 3], a[8 * rowno + 4],
        a[8 * rowno + 5], a[8 * rowno + 6], a[8 * rowno + 7],
    ]
}

// Runs a row-wise IDCT over each of the 8-element rows of f.
fn idct_rows(f: s32[COEFF_PER_MCU]) -> s32[COEFF_PER_MCU] {
    let row0 = idct_row(get_row(f, 0));
    let row1 = idct_row(get_row(f, 1));
    let row2 = idct_row(get_row(f, 2));
    let row3 = idct_row(get_row(f, 3));
    let row4 = idct_row(get_row(f, 4));
    let row5 = idct_row(get_row(f, 5));
    let row6 = idct_row(get_row(f, 6));
    let row7 = idct_row(get_row(f, 7));
    row0 ++ row1 ++ row2 ++ row3 ++ row4 ++ row5 ++ row6 ++ row7
}

// Performs an 8-point 1D IDCT.
//
// This is nearly identical to the above, but uses slightly different fixed
// point.
//
// TODO(cdleary): 2020-08-09 It should be possible to parameterize the fixed
// pointness out into types, so that we can consolidate with idct_row as fixed
// point library comes online.
fn idct_col(s: s32[8]) -> s32[8] {
    let w1pw7 = W1 + W7;
    let w1mw7 = W1 - W7;
    let w2pw6 = W2 + W6;
    let w2mw6 = W2 - W6;
    let w3pw5 = W3 + W5;
    let w3mw5 = W3 - W5;

    // Prescale.
    let y0 = (s[0] << 8) + 8192;
    let y1 = s[4] << 8;
    let y2 = s[6];
    let y3 = s[2];
    let y4 = s[1];
    let y5 = s[7];
    let y6 = s[5];
    let y7 = s[3];

    // Stage 1.
    let y8 = W7 * (y4 + y5) + 4;
    let y4 = (y8 + w1mw7 * y4) >> 3;
    let y5 = (y8 - w1pw7 * y5) >> 3;
    let y8 = W3 * (y6 + y7) + 4;
    let y6 = (y8 - w3mw5 * y6) >> 3;
    let y7 = (y8 - w3pw5 * y7) >> 3;

    // Stage 2.
    let y8 = y0 + y1;
    let y0 = y0 - y1;
    let y1 = W6 * (y3 + y2) + 4;
    let y2 = (y1 - w2pw6 * y2) >> 3;
    let y3 = (y1 + w2mw6 * y3) >> 3;
    let y1 = y4 + y6;
    let y4 = y4 - y6;
    let y6 = y5 + y7;
    let y5 = y5 - y7;

    // Stage 3.
    let y7 = y8 + y3;
    let y8 = y8 - y3;
    let y3 = y0 + y2;
    let y0 = y0 - y2;
    let y2 = (R2 * (y4 + y5) + 128) >> 8;
    let y4 = (R2 * (y4 - y5) + 128) >> 8;

    // Stage 4.
    s32[8]:[
        (y7 + y1) >> 14, (y3 + y2) >> 14, (y0 + y4) >> 14, (y8 + y6) >> 14, (y8 - y6) >> 14,
        (y0 - y4) >> 14, (y3 - y2) >> 14, (y7 - y1) >> 14,
    ]
}

// Extracts a column (8 strided values) from a 64-value coefficient array.
fn get_col(a: s32[COEFF_PER_MCU], colno: u8) -> s32[8] {
    s32[8]:[
        a[8 * 0 + colno], a[8 * 1 + colno], a[8 * 2 + colno], a[8 * 3 + colno], a[8 * 4 + colno],
        a[8 * 5 + colno], a[8 * 6 + colno], a[8 * 7 + colno],
    ]
}

// Runs a column-wise IDCT over each of the 8-element columns of f.
fn idct_cols(f: s32[COEFF_PER_MCU]) -> s32[COEFF_PER_MCU] {
    let col0 = idct_col(get_col(f, 0));
    let col1 = idct_col(get_col(f, 1));
    let col2 = idct_col(get_col(f, 2));
    let col3 = idct_col(get_col(f, 3));
    let col4 = idct_col(get_col(f, 4));
    let col5 = idct_col(get_col(f, 5));
    let col6 = idct_col(get_col(f, 6));
    let col7 = idct_col(get_col(f, 7));

    // Concatenate the columns. This is awkward to do today...
    // TODO(leary): 2020-08-03 Come up with a plan for making this better.
    for (i, accum): (u8, s32[COEFF_PER_MCU]) in 0..COEFF_PER_MCU_U8 {
        let val: s32 = match i & 7 {
            0 => col0[i >> 3],
            1 => col1[i >> 3],
            2 => col2[i >> 3],
            3 => col3[i >> 3],
            4 => col4[i >> 3],
            5 => col5[i >> 3],
            6 => col6[i >> 3],
            7 => col7[i >> 3],
            _ => fail!("invalid_column_index", s32:0),
        };
        update(accum, i, val)
    }(s32[COEFF_PER_MCU]:[0, ...])
}

pub fn idct(f: s32[COEFF_PER_MCU]) -> s32[COEFF_PER_MCU] {
    let f = idct_rows(f);
    let f = idct_cols(f);
    f
}

// Test case that has row0 populated with some values.
#[test]
fn idct0_test() {
    let input = s32[COEFF_PER_MCU]:[23, -1, -2, 0, ...];
    let want = s32[COEFF_PER_MCU]:[
        2, 3, 3, 3, 3, 3, 3, 3, //
        2, 3, 3, 3, 3, 3, 3, 3, //
        2, 3, 3, 3, 3, 3, 3, 3, //
        2, 3, 3, 3, 3, 3, 3, 3, //
        2, 3, 3, 3, 3, 3, 3, 3, //
        2, 3, 3, 3, 3, 3, 3, 3, //
        2, 3, 3, 3, 3, 3, 3, 3, //
        2, 3, 3, 3, 3, 3, 3, 3, //
    ];
    assert_eq(want, idct(input))
}

// Test case that has row0 and (1, 1) populated with some values.
#[test]
fn idct1_test() {
    let input = s32[COEFF_PER_MCU]:[
        13, -7, 0, 0, 0, 0, 0, 0, //
        0, 2, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
    ];
    let want = s32[COEFF_PER_MCU]:[
        1, 1, 1, 1, 2, 2, 2, 2, //
        1, 1, 1, 1, 2, 2, 2, 2, //
        1, 1, 1, 1, 2, 2, 2, 3, //
        1, 1, 1, 1, 2, 2, 3, 3, //
        0, 1, 1, 1, 2, 2, 3, 3, //
        0, 0, 1, 1, 2, 2, 3, 3, //
        0, 0, 1, 1, 2, 3, 3, 3, //
        0, 0, 1, 1, 2, 3, 3, 3, //
    ];
    assert_eq(want, idct(input))
}

// Test case that has row0 and col0 populated with some values.
#[test]
fn idct2_test() {
    let input = s32[COEFF_PER_MCU]:[
        -166, -7, -4, -4, 0, 0, 0, 0, //
        -2, 0, 0, 0, 0, 0, 0, 0, //
        -2, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
    ];
    let want = s32[COEFF_PER_MCU]:[
        -24, -23, -21, -21, -21, -21, -21, -20, //
        -24, -22, -21, -20, -21, -21, -21, -20, //
        -23, -22, -21, -20, -20, -21, -20, -20, //
        -23, -22, -20, -20, -20, -20, -20, -19, //
        -23, -22, -20, -20, -20, -20, -20, -19, //
        -23, -22, -20, -20, -20, -20, -20, -19, //
        -23, -22, -20, -20, -20, -20, -20, -19, //
        -23, -22, -20, -20, -20, -20, -20, -20, //
    ];
    assert_eq(want, idct(input))
}

// Test case that is quite "busy" in the input space, even the highest
// frequency value is non-zero.
#[test]
fn idct3_test() {
    let input = s32[COEFF_PER_MCU]:[
        -240, 8, -11, 47, 26, -6, 0, 5, //
        28, -6, 85, 44, -4, -25, 5, 16, //
        21, 8, 32, -16, -24, 0, 30, 12, //
        -2, 18, 0, -2, 0, 7, 0, -15, //
        7, 4, 15, -24, 0, 9, 8, -6, //
        4, 9, 0, -5, -6, 0, 0, 0, //
        -4, 0, -6, 0, 0, 10, -10, -8, //
        6, 0, 0, 0, 0, 0, 0, -8, //
    ];
    let want = s32[COEFF_PER_MCU]:[
        21, -10, -26, -61, -43, -17, -22, -8, //
        5, -28, -47, -73, -11, -14, -24, -17, //
        -14, -31, -61, -45, -5, -18, -22, -34, //
        -23, -36, -49, -32, -12, -33, -33, -35, //
        -30, -39, -53, -8, -19, -31, -43, -42, //
        -41, -43, -50, -4, -15, -33, -44, -66, //
        -40, -38, -21, -14, -17, -26, -46, -52, //
        -44, -47, -9, -12, -30, -33, -38, -37, //
    ];
    assert_eq(want, idct(input))
}
