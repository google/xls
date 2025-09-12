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

import xls.dslx.tests.errors.mod_simple_struct;
import xls.dslx.tests.errors.mod_simple_struct_duplicate;

fn id(x: mod_simple_struct::MyStruct) -> mod_simple_struct::MyStruct {
    x
}

fn main() {
    // Even though this is structurally the same it is nominally different, and
    // we still want to get a good (non confusing) error message.
    let other_definition = mod_simple_struct_duplicate::MyStruct{};
    id(other_definition);
}
