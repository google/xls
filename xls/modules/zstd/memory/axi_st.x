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

pub struct AxiStream<
    DATA_W: u32,
    DEST_W: u32,
    ID_W: u32,
    DATA_W_DIV8: u32 //= {DATA_W / u32:8}
> {
    data: uN[DATA_W],
    str: uN[DATA_W_DIV8],
    keep: uN[DATA_W_DIV8],
    id: uN[ID_W],
    dest: uN[DEST_W],
    last: u1,
}
