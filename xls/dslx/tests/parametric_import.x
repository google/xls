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

pub struct Type<A:u32, B:u32> {
  x: bits[A],
  y: bits[B]
}

pub fn zero<A:u32, B:u32>() -> Type<A, B> {
  Type<A, B>{ x: bits[A]: 0, y: bits[B]:0 }
}
