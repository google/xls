// Copyright 2023 The XLS Authors
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

struct MyStruct<WIDTH: u32> { myfield: bits[WIDTH] }

fn myfunc<FIELD_WIDTH: u32>(arg: MyStruct<FIELD_WIDTH>) -> u32 { (arg.myfield as u32) }

const WIDTH_15 = u32:15;

fn myfunc_spec1(arg: MyStruct<15>) -> u32 { (myfunc<u32:15>(arg)) }

fn myfunc_spec2(arg: MyStruct<15>) -> u32 { (myfunc<WIDTH_15>(arg)) }

fn myfunc_spec3(arg: MyStruct<15>) -> u32 { (myfunc(arg)) }

fn myfunc_spec4(arg: MyStruct<WIDTH_15>) -> u32 { (myfunc<u32:15>(arg)) }

fn myfunc_spec5(arg: MyStruct<WIDTH_15>) -> u32 { (myfunc<WIDTH_15>(arg)) }

fn myfunc_spec6(arg: MyStruct<WIDTH_15>) -> u32 { (myfunc(arg)) }
