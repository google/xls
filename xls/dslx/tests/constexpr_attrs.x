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

// Tests to verify constant struct members can be used to define
// constants and array types.

import xls.dslx.tests.constexpr;

struct LocalStruct { a: bits[32] }

const LOCAL = LocalStruct { a: u32:8 };

// TODO(rspringer): 2021/03/04 Support in progress.
//const IMPORTED = constexpr::ImportedStruct { a: u32:16 };

const LOCAL_STRUCT = u32[LOCAL.a]:[u32:0, u32:1, u32:2, u32:3, ...];

//const imported_struct = u32[IMPORTED.a]:[u32:8, u32:9, u32:10, u32:11, ...];
//const imported_instance = u32[constexpr::IMPORTED_STRUCT_INSTANCE.a]:[u32:8, u32:9, u32:10,
//u32:11, ...];

// TODO(rspringer): 2021/03/04 Add a test that dereferences an attribute of an attribute,
// e.g., "u32[IMPORTED_STRUCT.a.b]:[...]".
#[test]
fn can_instantiate() {
    let local_struct_expected = u32[8]:[u32:0, u32:1, u32:2, u32:3, ...];
    assert_eq(LOCAL_STRUCT, local_struct_expected)

    //let imported_struct_expected = u32[16]:[u32:8, u32:9, u32:10, u32:11, ...];
    //assert_eq(imported_struct, imported_struct_expected);

    //let imported_instance_expected = u32[32]:[u32:8, u32:9, u32:10, u32:11, ...];
    //assert_eq(imported_instance, imported_instance_expected)
}
