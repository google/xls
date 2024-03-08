// Copyright 2024 The XLS Authors
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

import std;

#[quickcheck]
fn value_is_one_hot(shift: u3) -> bool {
    let oh = u8:1 << shift;

    // The MSB indicates that value is zero -- the value should never be zero.
    let processed: u9 = one_hot(oh, true);

    // Check the low bits are the same as the one hot value and the high bit is
    // always false (since the input is never zero).
    processed[:-1] == oh && processed[-1:] == false
}
