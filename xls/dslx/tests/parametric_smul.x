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

fn smul_generic<M: u32, N: u32, R: u32 = M + N>(x: sN[M], y: sN[N]) -> sN[R] {
  (x as sN[R]) * (y as sN[R])
}

fn smul_s2_s3(x: s2, y: s3) -> s5 {
  smul_generic(x, y)
}

fn smul_s3_s4(x: s3, y: s4) -> s7 {
  smul_generic(x, y)
}

#[test]
fn parametric_smul() {
  let _ = assert_eq(s5:2, smul_s2_s3(s2:-1, s3:-2));
  let _ = assert_eq(s5:6, smul_s2_s3(s2:-2, s3:-3));
  let _ = assert_eq(s5:-6, smul_s2_s3(s2:-2, s3:3));
  let _ = assert_eq(s7:-7, smul_s3_s4(s3:-1, s4:7));
  ()
}
