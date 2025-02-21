// Copyright 2023-2024 The XLS Authors
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

// AXI Stream Package

pub struct AxiStreamBundle<DATA_W: u32, DATA_W_DIV8: u32, DEST_W: u32, ID_W: u32> {
    tdata: uN[DATA_W],
    tstr: uN[DATA_W_DIV8],
    tkeep: uN[DATA_W_DIV8],
    tlast: u1,
    tid: uN[ID_W],
    tdest: uN[DEST_W],
}

pub fn simpleAxiStreamBundle<DATA_W: u32, DATA_W_DIV8: u32, DEST_W: u32, ID_W: u32>
    (data: uN[DATA_W]) -> AxiStreamBundle {
    AxiStreamBundle {
        tdata: data,
        tstr: uN[DATA_W_DIV8]:0,
        tkeep: uN[DATA_W_DIV8]:0,
        tlast: u1:0,
        tid: uN[ID_W]:0,
        tdest: uN[DEST_W]:0
    }
}

pub fn zeroAxiStreamBundle<DATA_W: u32, DATA_W_DIV8: u32, DEST_W: u32, ID_W: u32>
    () -> AxiStreamBundle {
    AxiStreamBundle {
        tdata: uN[DATA_W]:0,
        tstr: uN[DATA_W_DIV8]:0,
        tkeep: uN[DATA_W_DIV8]:0,
        tlast: u1:0,
        tid: uN[ID_W]:0,
        tdest: uN[DEST_W]:0
    }
}
