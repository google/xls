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

// Define some types with sv-types associated with them.

#[sv_type("sv_struct")]
pub struct my_struct { payload: bits[128], id: bits[8] }

#[sv_type("sv_enum")]
pub enum my_enum : bits[2] {
    kMyZero = 0,
    kMyOne = 1,
    kMyTwo = 2,
    kMyThree = 3,
}

#[sv_type("sv_tuple")]
pub type my_tuple = (bits[8], bits[8], bits[8]);

pub type non_sv_tuple = (bits[8], bits[8], bits[8]);
