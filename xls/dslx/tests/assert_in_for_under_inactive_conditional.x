// Copyright 2025 The XLS Authors
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

fn main() -> () { () }

fn bar() -> () { assert!(false, "Should not happen."); }

fn foo(x: bool) -> () {
    if x {
        for (_, ()) in u32:0..u32:4 {
            bar()
        }(())
    }
}

#[test]
fn test_main_no_assert_when_false() { foo(false) }
