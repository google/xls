// Copyright 2025 The XLS Authors
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

import apfloat;
import float32;
import float64;

fn f64_to_f32(f: float64::F64) -> float32::F32 {
    apfloat::downcast<float32::F32::FRACTION_SIZE, float32::F32::EXP_SIZE>(
        f, apfloat::RoundStyle::TIES_TO_EVEN)
}

#[test]
fn f64_to_f32_test() { assert_eq(f64_to_f32(float64::one(u1:0)), float32::one(u1:0)); }
