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

// This file implements most of bfloat16 single-precision addition.
import xls.modules.fp.apfloat_add_2
import bfloat16

type BF16 = bfloat16::BF16;

pub fn bf16_add_2(x: BF16, y: BF16) -> BF16 {
  apfloat_add_2::add<u32:8, u32:7>(x, y)
}
