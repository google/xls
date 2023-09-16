// Copyright 2023 The XLS Authors
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

// Note: we need to cause a rollover but not end up with the high bit set,
// because we flag a different error for giant values independent from the
// rollover error.
fn p<N: u32, M1: u32 = {N - (u32:1 << 31) - u32:1}>() -> uN[M1] {
  uN[M1]:0
}

fn main() {
  p<u32:0>();  // <-- cause an underflow to occur
}
