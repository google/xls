// Copyright 2026 The XLS Authors
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

// DSLX 4-way priority encoder with fallback default value.

pub fn priority_encoder(req: u4, fallback: u8) -> u8 {
    if (req & u4:0b0001) != u4:0 {
        u8:100
    } else if (req & u4:0b0010) != u4:0 {
        u8:101
    } else if (req & u4:0b0100) != u4:0 {
        u8:102
    } else if (req & u4:0b1000) != u4:0 {
        u8:103
    } else {
        fallback
    }
}
