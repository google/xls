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

// Test to verify that constant arrays of structs are supported. Motivated by
// GitHub issue #211.

struct FooAndBar {
  foo: u32,
  bar: u32
}

const MY_FOO_BARS = FooAndBar[2] : [
  FooAndBar{ foo: u32:1, bar: u32:2 },
  FooAndBar{ foo: u32:3, bar: u32:4 },
];

// Dummy main fn for build macro happiness.
fn main() { () }
