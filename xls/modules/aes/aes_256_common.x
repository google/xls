// Copyright 2022 The XLS Authors
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

// Common constants and types for the AES-256 cipher.
import xls.modules.aes.aes_common

pub const KEY_BITS = u32:256;
pub const KEY_WORD_BITS = u32:32;
pub const KEY_WORDS = KEY_BITS / KEY_WORD_BITS;
pub type Key = aes_common::KeyWord[KEY_WORDS];

// Until GitHub issue #629 is resolved, this MUST NOT be called in AOT-compiled
// code!
//pub fn trace_key(key: Key) {
//    let bytes0 = key[0] as u8[4];
//    let bytes1 = key[1] as u8[4];
//    let bytes2 = key[2] as u8[4];
//    let bytes3 = key[3] as u8[4];
//    let _ = trace_fmt!(
//        "0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x}",
//        bytes0[0], bytes0[1], bytes0[2], bytes0[3],
//        bytes1[0], bytes1[1], bytes1[2], bytes1[3],
//        bytes2[0], bytes2[1], bytes2[2], bytes2[3],
//        bytes3[0], bytes3[1], bytes3[2], bytes3[3]);
//    ()
//}
