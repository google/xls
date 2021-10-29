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

proc fmac_32 {
  config(input_a: chan in F32, input_b: chan in F32,
         reset: chan in bool, output: chan out F32) {
    spawn apfloat_fmac::fmac<u32:8, u32:23>(input_a, input_b, reset, output)
        (float32::zero(false));
    ()
  }

  next(tok: token) { () }
}
