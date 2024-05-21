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

// https://github.com/google/xls/issues/907

proc main {
  ch0: chan<u32> in;

  init { () }

  config(c: chan<u32> in) {
    (c, )
  }
  next(st: ()) {
    // Note: the order of arguments is confused here, it should be:
    //  token, channel, predicate, value
    let (tok, _) = recv_if(join(), false, ch0, u32:0);
    ()
  }
}
