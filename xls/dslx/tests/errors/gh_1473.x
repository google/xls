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

import float32;

pub fn umax<N: u32>(x: uN[N], y: uN[N]) -> uN[N] { if x > y { x } else { y } }

pub fn uadd_with_overflow
    <V: u32,
     N: u32,
     M: u32,
     MAX_N_M: u32 = {umax(N, M)},
     MAX_N_M_V: u32 = {umax(MAX_N_M, V)}>
    (x: uN[N], y: uN[M]) -> (bool, uN[V]) {

    let x_extended = widening_cast<uN[MAX_N_M_V + u32:1]>(x);
    let y_extended = widening_cast<uN[MAX_N_M_V + u32:1]>(y);

    let full_result: uN[MAX_N_M_V + u32:1] = x_extended + y_extended;
    let narrowed_result = full_result as uN[V];
    let overflow_detected = or_reduce(full_result[V as s32:]);

    (overflow_detected, narrowed_result)
}

pub fn double_fraction_carry(f: float32::F32) -> (uN[float32::F32_FRACTION_SZ], u1) {
    let f = f.fraction as uN[float32::F32_FRACTION_SZ + u32:1];
    let (overflow, f_x2) = uadd_with_overflow(f, f);
    (f_x2, overflow)
}
