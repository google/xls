// Copyright 2021 The XLS Authors
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

const FORTY_TWO = u32:42;
const SIXTY_FOUR = u32:64;

// Returns the parametric value N this function is instantiated with.
fn p_id<N: u32>() -> u32 {
  N
}

fn p<N: u32>() -> u32 {
  const BASE = p_id<FORTY_TWO>();
  BASE + N
}

fn main() -> u32{
  p<SIXTY_FOUR>()
}

#[test]
fn test_main() {
  assert_eq(u32:106, main())
}
