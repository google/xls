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

// Prefix scans an array of 8 32-bit values and produces a running count of
// duplicate values in the run.
fn prefix_scan_eq(x: u32[8]) -> u3[8] {
  let (_, _, result) =
    for ((i, elem), (prior, count, result)): ((u32, u32), (u32, u3, u3[8]))
          in enumerate(x) {
      let (to_place, new_count): (u3, u3) = match (i == u32:0, prior == elem) {
        // The first iteration always places 0 and propagates seen count of 1.
        (true, _) => (u3:0, u3:1),
        // Subsequent iterations propagate seen count of previous_seen_count+1 if
        // the current element matches the prior one, and places the current seen
        // count.
        (false, true) => (count, count + u3:1),
        // If the current element doesn't match the prior one we propagate a seen
        // count of 1 and place a seen count of 0.
        (false, false) => (u3:0, u3:1),
      };
      let new_result: u3[8] = update(result, i, to_place);
    (elem, new_count, new_result)
  }((u32:0xffffffff, u3:0, u3[8]:[u3:0, ...]));
  result
}

#[test]
fn prefix_scan_eq_all_zero_test() {
  let input = u32[8]:[0, ...];
  let result = prefix_scan_eq(input);
  assert_eq(result, u3[8]:[0, 1, 2, 3, 4, 5, 6, 7])
}

#[test]
fn prefix_scan_eq_doubles_test() {
  let input = u32[8]:[0, 0, 1, 1, 2, 3, 2, 3];
  let result = prefix_scan_eq(input);
  assert_eq(result, u3[8]:[0, 1, 0, 1, 0, 0, 0, 0])
}
