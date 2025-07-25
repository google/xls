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

// Configuration

import std;
import xls.modules.dma.bus.axi_pkg;

// Top config (used in synthesis)
pub const TOP_ADDR_W = u32:32;
pub const TOP_DATA_W = u32:32;
pub const TOP_DATA_W_DIV8 = TOP_DATA_W / u32:8;
pub const TOP_DEST_W = TOP_DATA_W / u32:8;
pub const TOP_ID_W = TOP_DATA_W / u32:8;
pub const TOP_REGS_N = u32:14;
pub const TOP_STRB_W = TOP_DATA_W / u32:8;

// CSR Config
pub const CSR_DATA_W = u32:32;
pub const CSR_ADDR_W = u32:32;
pub const CSR_REGS_N = u32:14;

// Register Map
pub const CONTROL_REGISTER = uN[CSR_ADDR_W]:0x00;
pub const STATUS_REGISTER = uN[CSR_ADDR_W]:0x01;
pub const INTERRUPT_MASK_REGISTER = uN[CSR_ADDR_W]:0x02;
pub const INTERRUPT_STATUS_REGISTER = uN[CSR_ADDR_W]:0x03;
pub const READER_START_ADDRESS = uN[CSR_ADDR_W]:0x04;
pub const READER_LINE_LENGTH = uN[CSR_ADDR_W]:0x05;
pub const READER_LINE_COUNT = uN[CSR_ADDR_W]:0x06;
pub const READER_STRIDE_BETWEEN_LINES = uN[CSR_ADDR_W]:0x07;
pub const WRITER_START_ADDRESS = uN[CSR_ADDR_W]:0x08;
pub const WRITER_LINE_LENGTH = uN[CSR_ADDR_W]:0x09;
pub const WRITER_LINE_COUNT = uN[CSR_ADDR_W]:0x0a;
pub const WRITER_STRIDE_BETWEEN_LINES = uN[CSR_ADDR_W]:0x0b;
pub const VERSION_REGISTER = uN[CSR_ADDR_W]:0x0c;
pub const CONFIGURATION_REGISTER = uN[CSR_ADDR_W]:0x0d;
