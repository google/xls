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

import xls.dslx.tests.mod_enum_importer;

// A module that imports an enum alias and uses it.
fn main() -> mod_enum_importer::MyEnumAlias { mod_enum_importer::MyEnumAlias::FOO }

#[test]
fn test_main() { assert_eq(main(), mod_enum_importer::MyEnumAlias::FOO) }
