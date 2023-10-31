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

// We make a type alias of a parmetric struct and expect that we get an error
// where we instantiate it.
//
// See https://github.com/google/xls/issues/1069

struct MyStruct<M: u32, N: u32>{
  m: uN[M],
  n: uN[N],
}

fn f(x: bool) -> MyStruct<1, 2> {
    type S = MyStruct<1, 2>;
    if x {
        zero!<S>()
    } else {
        S {
          m: uN[1]:0,
          n: uN[3]:0,
        }
    }
}
