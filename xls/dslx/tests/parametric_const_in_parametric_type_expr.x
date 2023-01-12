// Copyright 2022 The XLS Authors
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

fn min_clip<X:u32, MIN:sN[X]=sN[X]:0>(value:sN[X]) -> sN[X] {
  if (value < MIN) { MIN } else { value }
}

#[test]
fn test_min_clip() {
  const X = u32:4;
  assert_eq(min_clip<X>(s4:-1), s4:0)
}
