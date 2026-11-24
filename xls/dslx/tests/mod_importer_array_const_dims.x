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

// Uses imported structs whose member widths are elements of an imported array
// constant (e.g. `uN[WIDTHS[0]]`).

import xls.dslx.tests.mod_imported_array_const_dims as dims;

fn main(h: dims::Histories, t: dims::Tables<u32:2>) -> (u40, u10, u24) {
    (h.long + u40:1, h.short, t.entries[1])
}

#[test]
fn test_main() {
    let h = dims::Histories { long: u40:41, short: u10:7 };
    let t = dims::Tables { entries: u24[2]:[1, 2] };
    assert_eq(main(h, t), (u40:42, u10:7, u24:2));
    assert_eq(zero!<dims::Histories>().long, u40:0);
}
