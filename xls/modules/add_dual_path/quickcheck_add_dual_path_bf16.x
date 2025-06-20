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

import bfloat16;

import xls.modules.add_dual_path.dual_path;

type BF16 = bfloat16::BF16;

#[quickcheck(test_count=10000)]
fn quickcheck_add_dual_path_bf16(x: BF16, y: BF16) -> bool {
    dual_path::add_dual_path_bf16(x, y) == bfloat16::add(x, y)
}
