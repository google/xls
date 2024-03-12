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

fn id<S: bool, N: u32>(x: xN[S][N]) -> xN[S][N] { x }

fn main() -> u32 {
    let ft_u32 = id(u32:42);
    let ft_s32 = id(s32:42);
    ft_u32 + ft_s32 as u32
}

#[test]
fn test_main() { assert_eq(main(), u32:84); }
