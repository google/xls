// Copyright 2026 The XLS Authors
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

struct Address { base: u32 }

impl Address {
    fn default() -> Self { Address { base: u32:0x1000 } }
}

#[derive(Default)]
struct Config<N: u32> { value: (bool, s4), address: Address, history: Address[N] }

fn main() -> Config<2> { Config<2>::default() }

#[test]
fn derived_default() {
    let config = main();
    assert_eq(config.value, (false, s4:0));
    assert_eq(config.address.base, u32:0x1000);
    assert_eq(config.history[1].base, u32:0x1000);
    assert_eq(zero!<Config<2>>().address.base, u32:0);
}
