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

proc p {
    c: chan<u32> in;

    init { u32:0 }
    config(c: chan<u32> in) { (c,) }
    next(state: u32) {
        let (tok', state'): (token, u32) = match state {
            u32:0 => recv(join(), c),
            _ => recv(join(), c),
        };
        state'
    }
}
