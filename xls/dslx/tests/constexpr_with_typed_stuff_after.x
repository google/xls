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

// This is a very "implementation oriented" test.
//
// It is a contrived set of module-level constructs that:
// 1. Forces constexpr evaluation towards the top of the module.
// 2. Forces the use of type information lower in the file.
// 3. Uses the bindings made lower in the file in the test.
//
// 1 and 2 cause us to make sure constexpr eval doesn't continue to try to
// evaluate top-level module constructs past the point that constexpr eval was
// triggered, even if we rely on the interpreter to evaluate the constexprs, and
// the interpreter generally wants top-level bindings for the module.
//
// 3 makes sure that, even though the constexpr evaluation terminates the module
// scope evaluation early, eventually we evaluate the rest of the constructs and
// put them into the module-level bindings, and they can be used from
// functions/tests as expected.

const MY_CONST = u32:42;

type MyBits = bits[MY_CONST];  // This name reference forces constexpr eval.

struct Simple { x: u32 }

const SIMPLE = Simple { x: u32:2 };

// The struct access requires type information, but we forced constexpr eval
// higher up in the file.
const TWO = SIMPLE.x;

fn main() -> (u32, bits[42]) { (TWO, MyBits:0) }

#[test]
fn test_main() { assert_eq(main(), (u32:2, bits[42]:0)) }
