#![feature(type_inference_v1)]
#![feature(use_syntax)]

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

use xls::dslx::tests::mod_imported_multi_const::{MOL, ONE};

fn main() -> u32 { MOL + ONE }

#[test]
fn test_main() { assert_eq(main(), u32:43); }
