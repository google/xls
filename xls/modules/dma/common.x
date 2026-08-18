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

// Common

pub struct TransferDescBundle<ADDR_W: u32> { address: uN[ADDR_W], length: uN[ADDR_W] }

pub struct MainCtrlBundle<ADDR_W: u32> {
    start_address: uN[ADDR_W],
    line_count: uN[ADDR_W],
    line_length: uN[ADDR_W],
    line_stride: uN[ADDR_W],
}

pub fn zeroTransferDescBundle<ADDR_W: u32>() -> TransferDescBundle {
    TransferDescBundle { address: uN[ADDR_W]:0, length: uN[ADDR_W]:0 }
}

pub fn zeroMainCtrlBundle<ADDR_W: u32>() -> MainCtrlBundle {
    MainCtrlBundle {
        start_address: uN[ADDR_W]:0,
        line_count: uN[ADDR_W]:0,
        line_length: uN[ADDR_W]:0,
        line_stride: uN[ADDR_W]:0
    }
}
