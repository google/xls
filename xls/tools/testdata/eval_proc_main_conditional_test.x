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

// Proc used to test eval_proc_main.

proc test_proc {
    input: chan<u8> in;
    output: chan<u8> out;

    config(input: chan<u8> in, output: chan<u8> out) { (input, output) }

    init {  }

    next(_: ()) {
        let tok = join();
        let (tok, recv_val) = recv(tok, input);
        let do_send = recv_val != u8:42;
        send_if(tok, output, do_send, recv_val);
    }
}
