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

struct Point {
    x: u32,
    y: u8,
}

#[fuzz_test]
fn arbitrary_struct(p: Point) -> bool {
    p.x == p.x && p.y == p.y
}

#[fuzz_test(domains=`Point { x: u32:0..10, y: u8:11..20 }`)]
fn struct_range(p: Point) -> bool {
    (p.x as u8) < p.y
}

#[fuzz_test(domains=`Point { x: [u32:0, 1, 2, 3], y: u8:11..20 }`)]
fn struct_element_of(p: Point) -> bool {
    (p.x as u8) < p.y
}

#[fuzz_test(domains=`Point { x: [u32:0, 1, 2, 3] }`)]
fn struct_arbitrary_field(p: Point) -> bool {
    p.x <= u32:3
}

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
fn test_imported_struct_domain(s: imported_module::S) -> bool {
    s.x >= u32:0 && s.x < u32:10
}

struct InnerWithoutDomain {
    y: u32,
}

struct OuterWithInline {
    x: InnerWithoutDomain,
}

#[fuzz_test(domains=`OuterWithInline { x: InnerWithoutDomain { y: u32:0..99999 } }`)]
fn test_inline_nested_struct_domain(o: OuterWithInline) -> bool {
    o.x.y < u32:99999
}

#[fuzz_domain("Domain")]
struct S {
  x: u32
}

fn first_domain() -> Domain {
        Domain {
                x: u32:0..10
        }
}

#[fuzz_test(domains=`first_domain()`)]
fn first(s: S) -> bool { s.x < 11 }

fn second_domain() -> Domain {
        Domain {
                x: [u32:0, u32:1]
        }
}

#[fuzz_test(domains=`second_domain()`)]
fn second(s: S) -> bool { s.x < 2 }

// Test for type aliases inside domain structs.
#[fuzz_domain("AliasInnerDomain")]
struct AliasInner {
    y: u32,
}

type StructAlias = AliasInner;

#[fuzz_domain("AliasOuterDomain")]
struct AliasOuter {
    a: StructAlias,
}

fn create_alias_outer_domain() -> AliasOuterDomain {
   AliasOuterDomain {
     a: AliasInnerDomain {
       y: u32:0..10,
     },
   }
}

#[fuzz_test(domains=`create_alias_outer_domain()`)]
fn test_struct_domain_type_alias(s: AliasOuter) -> bool {
    s.a.y < u32:10
}

// Test for nested struct domains with tuples.
#[fuzz_domain("TupleInnerDomain")]
struct TupleInner {
    y: u32,
}

#[fuzz_domain("TupleOuterDomain")]
struct TupleOuter {
    c: (TupleInner, u8),
}

fn create_tuple_outer_domain() -> TupleOuterDomain {
   TupleOuterDomain {
     c: (TupleInnerDomain { y: u32:0..10 }, u8:0..11),
   }
}

#[fuzz_test(domains=`create_tuple_outer_domain()`)]
fn test_struct_domain_tuple(s: TupleOuter) -> bool {
    s.c.0.y < u32:10 && s.c.1 < u8:11
}

struct WideStruct { w: uN[128], x: u32 }

#[fuzz_test(domains=`WideStruct { x: u32:0..10 }`)]
fn wide(s: WideStruct) -> bool {
    true
}

#[fuzz_test(domains=`[uN[128]:5, 10, 15]`)]
fn wide_element_of(x: uN[128]) -> bool {
    x == uN[128]:5 || x == uN[128]:10 || x == uN[128]:15
}

struct WideTupleStruct { w: uN[128], x: u32 }

#[fuzz_test(domains=`WideTupleStruct { w: [uN[128]:1, 2], x: u32:0..10 }`)]
fn wide_tuple_element_of(s: WideTupleStruct) -> bool {
    true
}


