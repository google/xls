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

fn trace_binary<N: u32>(v: uN[N]) {
    trace_fmt!("{:0b}", v);
}

fn trace_binary_s<N: u32>(v: sN[N]) {
    trace_fmt!("{:0b}", v);
}

// Unsigned 16-bits
fn trace_u16(v: u16) {
    trace_binary(v);
}

// Unsigned 21-bits
fn trace_u21(v: u21) {
    trace_binary(v);
}

// Signed 32-bits
fn trace_s32(v: s32) {
    trace_binary_s(v);
}

enum foo: u21 {BAR=12345,}

fn trace_enum() {
    trace_u21(foo::BAR as u21);
}
