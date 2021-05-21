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

// Demonstrates that the parser chokes if a ColonRef-to-StructDef is used when
// declaring a literal number; in other words, we can support
// ColonRef-as-scalar-TypeDef (pub type Foo = u8; ... mod::Foo:7;), but not
// as StructDefs.

import xls.dslx.tests.errors.invalid_colon_ref_as_literal_type_mod as imported

fn invalid_colon_ref_as_literal_type(x: u8) -> u8 {
  x + imported::ModType:7
}
