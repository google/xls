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
import xls.tests.fuzz_test.imported_module;

#[fuzz_domain("MyStructDomain")]
struct MyStruct {
    x: u32,
    y: bool,
}

fn create_flat_domain() -> MyStructDomain {
   MyStructDomain {
     x: u32:0..10,
     y: (),
   }
}

#[fuzz_test(domains=`create_flat_domain()`)]
fn test_flat_struct_domain(s: MyStruct) -> bool {
    s.x >= u32:0 && s.x < u32:10
}

#[fuzz_domain("InnerDomain")]
struct Inner {
    y: u32,
}

#[fuzz_domain("OuterDomain")]
struct Outer {
    x: Inner,
}

fn create_nested_domain() -> OuterDomain {
   OuterDomain {
     x: InnerDomain {
       y: u32:0..10,
     },
   }
}

#[fuzz_test(domains=`create_nested_domain()`)]
fn test_nested_struct_domain(o: Outer) -> bool {
    o.x.y >= u32:0 && o.x.y < u32:10
}

fn get_domain() -> imported_module::SDomain {
   imported_module::SDomain {
     x: u32:0..10,
   }
}

#[fuzz_test(domains=`get_domain()`)]
fn test_imported_struct_domain_fn(s: imported_module::S) -> bool {
    s.x >= u32:0 && s.x < u32:10
}
