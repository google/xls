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

// Struct instances and splats written with `Self` inside impls.

struct Counter { count: u8, step: u8 }

impl Counter {
    fn bump(self) -> Self { Self { count: self.count + self.step, step: self.step } }

    fn reset(self) -> Self { Self { count: u8:0, ..self } }
}

struct Wide<N: u32> { value: uN[N] }

impl Wide<N> {
    fn bump(self) -> Self { Self { value: self.value + uN[N]:1 } }
}

fn main(c: Counter, w: Wide<u32:12>) -> (Counter, Counter, Wide<u32:12>) {
    (c.bump(), c.bump().reset(), w.bump())
}

#[test]
fn self_struct_instance_test() {
    let c = Counter { count: u8:40, step: u8:2 };
    let w = Wide { value: u12:0xfff };
    assert_eq(
        main(c, w),
        (
            Counter { count: u8:42, step: u8:2 }, Counter { count: u8:0, step: u8:2 },
            Wide { value: u12:0 },
        ));
}
