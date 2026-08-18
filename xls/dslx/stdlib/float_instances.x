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

// Non-exported, monomorphic wrapper functions for standard library floating-point
// routines to serve as entry points for IR conversion, evaluation, and JIT wrappers.

import bfloat16;
import float32;
import float64;

// Float32 wrappers
pub fn float32_add(x: float32::F32, y: float32::F32) -> float32::F32 { float32::add(x, y) }

pub fn float32_add_lza(x: float32::F32, y: float32::F32) -> float32::F32 {
    float32::add<true>(x, y)
}

pub fn float32_sub(x: float32::F32, y: float32::F32) -> float32::F32 { float32::sub(x, y) }

pub fn float32_sub_lza(x: float32::F32, y: float32::F32) -> float32::F32 {
    float32::sub<true>(x, y)
}

// Bfloat16 wrappers
pub fn bfloat16_add(x: bfloat16::BF16, y: bfloat16::BF16) -> bfloat16::BF16 { bfloat16::add(x, y) }

pub fn bfloat16_add_lza(x: bfloat16::BF16, y: bfloat16::BF16) -> bfloat16::BF16 {
    bfloat16::add<true>(x, y)
}

pub fn bfloat16_sub(x: bfloat16::BF16, y: bfloat16::BF16) -> bfloat16::BF16 { bfloat16::sub(x, y) }

pub fn bfloat16_sub_lza(x: bfloat16::BF16, y: bfloat16::BF16) -> bfloat16::BF16 {
    bfloat16::sub<true>(x, y)
}

// Float64 wrappers
pub fn float64_add(x: float64::F64, y: float64::F64) -> float64::F64 { float64::add(x, y) }

pub fn float64_add_lza(x: float64::F64, y: float64::F64) -> float64::F64 {
    float64::add<true>(x, y)
}

pub fn float64_sub(x: float64::F64, y: float64::F64) -> float64::F64 { float64::sub(x, y) }

pub fn float64_sub_lza(x: float64::F64, y: float64::F64) -> float64::F64 {
    float64::sub<true>(x, y)
}
