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

fn seemingly_can_fail(x: u32) -> u32 { if x != x { fail!("should_be_impossible", x) } else { x } }

#[quickcheck]
fn prop_effectively_identity(x: u32) -> bool { seemingly_can_fail(x) == x }

// As above, but tests exhaustive input stimulus generation for the signature.
#[quickcheck(exhaustive)]
fn prop_effectively_identity_small_space(x: u2) -> bool {
    let y = x as u32;
    seemingly_can_fail(y) == y
}
