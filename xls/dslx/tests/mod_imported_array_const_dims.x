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

// Structs whose member widths index into an array constant, for use from
// another module.

pub const WIDTHS = u32[3]:[40, 24, 10];

pub struct Histories { long: uN[WIDTHS[0]], short: uN[WIDTHS[2]] }

pub struct Tables<N: u32> { entries: uN[WIDTHS[1]][N] }
