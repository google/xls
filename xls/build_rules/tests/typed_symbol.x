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

#![feature(generics)]

fn typed_identity<T: type>(value: T) -> T { value }

fn typed_values<Z: type, A: bool, B: s8>(value: Z) -> Z {
    if A && B == s8:-1 { value } else { value }
}

fn typed_entry(value: u32) -> u32 {
    let value = typed_identity(value);
    let value = typed_values<u32, true, s8:-1>(value);
    typed_values<u32, false, s8:-1>(value)
}
