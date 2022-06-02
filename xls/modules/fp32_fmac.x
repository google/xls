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
import float32
import xls.modules.apfloat_fmac

type F32 = float32::F32;

const F32_ZERO = float32::zero(false);
const F32_ONE = float32::one(false);

proc fp32_fmac {
  config(input_a: chan in F32, input_b: chan in F32,
         reset: chan in bool, output: chan out F32) {
    spawn apfloat_fmac::fmac<u32:8, u32:23>(input_a, input_b, reset, output)
        (F32_ZERO);
    ()
  }

  // Nothing to do here - the spawned fmac does all the lifting.
  next(tok: token) { () }
}

#![test_proc()]
proc smoke_test {
  input_a_p: chan out F32;
  input_b_p: chan out F32;
  reset_p: chan out bool;
  output_c: chan in F32;
  terminator: chan out bool;

  config(terminator: chan out bool) {
    let (input_a_p, input_a_c) = chan F32;
    let (input_b_p, input_b_c) = chan F32;
    let (reset_p, reset_c) = chan bool;
    let (output_p, output_c) = chan F32;
    spawn fp32_fmac(input_a_c, input_b_c, reset_c, output_p)();
    (input_a_p, input_b_p, reset_p, output_c, terminator)
  }

  next(tok: token) {
      let tok = send(tok, input_a_p, F32_ZERO);
      let tok = send(tok, input_b_p, F32_ZERO);
      let tok = send(tok, reset_p, false);
      let (tok, result) = recv(tok, output_c);
      let _ = assert_eq(result, F32_ZERO);

      let tok = send(tok, input_a_p, F32_ONE);
      let tok = send(tok, input_b_p, F32_ZERO);
      let tok = send(tok, reset_p, false);
      let (tok, result) = recv(tok, output_c);
      let _ = assert_eq(result, F32_ZERO);

      let tok = send(tok, terminator, true);
      ()
  }
}
