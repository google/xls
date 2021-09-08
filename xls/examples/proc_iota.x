// Copyright 2021 The XLS Authors
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

// Basic example showing how a proc network can be created and connected.

proc producer(limit: u32, c: chan out u32)(i: u32) {
  send(c, i);
  let new_i = i + u32:1;
  next(new_i)
}

proc consumer<N: u32>(a: u32[N], c: chan in u32)(i: u32) {
  let e = recv(c);
  let _ = assert_eq(a[i], e);
  let new_i = i + u32:1;
  next(new_i)
}

fn main() {
  let (p, c) = chan u32;
  spawn producer(u32:10, p)(u32:0)
  spawn consumer(range(u32:0, u32:10), c)(u32:0)
}
