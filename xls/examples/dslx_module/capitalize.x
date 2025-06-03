// Copyright 2023 The XLS Authors
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

// Simple demonstration of string usage in DSLX.

const LOWERCASE_A = u8:97;
const LOWERCASE_Z = u8:122;

// Converts all letters in the input string to uppercase.
pub fn capitalize<N: u32>(input: u8[N]) -> u8[N] {
  let cap_distance = 'a' - 'A';

  let result = for (i, result) : (u32, u8[N]) in u32:0..N {
    let element =
        if input[i] >= LOWERCASE_A && input[i] <= LOWERCASE_Z {
          input[i] - cap_distance
        } else {
          input[i]
        };

    let result = update(result, i, element);
    result
  }(input);

  result
}

// Converts consecutive letters in the input string to alternate cases:
// uppercase, then lowercase, then uppercase, etc., as if spoken in a mocking
// manner by a somewhat anthropomorphic yellow sea sponge.
pub fn sponge_capitalize<N: u32>(input: u8[N]) -> u8[N] {
  let cap_distance = 'a' - 'A';

  let do_cap = true;
  let result = for (i, (result, do_cap)) : (u32, (u8[N], u1)) in u32:0..N {
    let input_is_cap = input[i] >= u8:65 && input[i] <= u8:90;
    let input_is_lower = input[i] >= u8:97 && input[i] <= u8:122;
    let capital = input[i] - cap_distance;
    let lower = input[i] + cap_distance;

    let element =
        if do_cap && input_is_lower {
          capital
        } else {
          if !do_cap && input_is_cap { lower }
          else { input[i] }
        };
    let do_cap =
        if input_is_lower || input_is_cap { !do_cap } else { do_cap };

    let result = update(result, i, element);
    (result, do_cap)
  }((input, do_cap));

  result.0
}

#[test]
fn cap_test() {
  assert_eq("HELLO FRIENDS", capitalize("HeLlO fRiEnDs"));
}

#[test]
fn sponge_cap_test() {
  assert_eq(
      sponge_capitalize("You shouldn't process strings in hardware"),
      "YoU sHoUlDn'T pRoCeSs StRiNgS iN hArDwArE");
}

#[test]
fn escape_sequences_test() {
  assert_eq(
      "H\t\tE\tLLO\nFR\u{beef}IENDS",
      capitalize("H\t\te\tLlO\nfR\u{beef}iEnDs"));
}
