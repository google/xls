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

// Regression test for https://github.com/google/xls/issues/206
//
// Uses module-qualified (imported) enum member references in a match
// expression. In this case we do through an alias, just because it's more
// interesting than without an alias.

import xls.dslx.tests.mod_simple_enum as exporter;

fn main(x: exporter::EnumType) -> u32 {
    match x {
        exporter::EnumType::FIRST => u32:0,
        exporter::EnumType::SECOND => u32:1,
    }
}

#[test]
fn test_main() {
    assert_eq(u32:0, main(exporter::EnumType::FIRST));
    assert_eq(u32:1, main(exporter::EnumType::SECOND));
}
