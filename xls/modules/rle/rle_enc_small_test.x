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

import std;
import xls.modules.rle.rle_enc_test_utils as test_utils;

#[test_proc]
proc SmallTest {
    terminate: chan<bool> out;
    term_src: chan<bool> in;

    config(term: chan<bool> out) {
        let (term_s, term_r) = chan<bool>("terminate_cpy");
        spawn test_utils::CompressCount<u32:100>(term_s);
        (term, term_r)
    }

    init {  }

    next(_: ()) {
        let (tok, v) = recv(join(), term_src);
        send(tok, terminate, v);
    }
}
