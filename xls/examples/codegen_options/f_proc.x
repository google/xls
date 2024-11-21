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

fn f(x: u32) -> u32 {
  x*x+x
}

pub proc f_proc {
  input_r: chan<u32> in;
  output_w: chan<u32> out;

  init { () }

  config(r: chan<u32> in, w: chan<u32> out) {
    (r, w)
  }

  next(state: ()) {
    let (tok, data) = recv(join(), input_r);
    let data  = f(data);
    send(tok, output_w, data);
  }
}
