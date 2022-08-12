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
proc producer {
  p: chan out u32;

  config(p: chan out u32) {
    (p,)
  }

  next(tok: token, do_send: bool) {
    let tok = send_if(tok, p, do_send, ((do_send) as u32));
    !do_send
  }
}

proc consumer {
  c: chan in u32;

  config(c: chan in u32) {
    (c,)
  }

  next(tok: token, do_recv: bool) {
    let (tok, foo) = recv_if(tok, c, do_recv);
    !do_recv
  }
}

proc main {
    config() {
        let (p, c) = chan u32;
        spawn producer(p)(true);
        spawn consumer(c)(true);
        ()
    }
    next(tok: token) { () }
}
