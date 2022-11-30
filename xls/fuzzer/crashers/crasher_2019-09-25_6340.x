// Copyright 2020 The XLS Authors
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

// options: {"input_is_dslx": true, "ir_converter_args": ["--top=main"], "convert_to_ir": true, "optimize_ir": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=3", "--reset_data_path=false"], "simulate": false, "simulator": null}
// args: bits[1]:0x0
// args: bits[1]:0x1
fn main(x165545: bool) -> (u57, u13, u22, u13, bool, bool, u1, u2, u13, u37) {
    let x165546: bool = !(x165545);
    let x165547: bool = !(x165546);
    let x165548: u2 = one_hot(x165546, (u1:0));
    let x165549: (u2, u2) = (x165548, x165548);
    let x165550: bool = !(x165547);
    let x165551: (bool) = (x165547,);
    let x165552: bool = clz(x165546);
    let x165553: u22 = (u22:0x200);
    let x165554: bool = one_hot_sel(x165545, [x165547]);
    let x165555: u13 = (((((((((x165545) ++ (x165550)) ++ (x165546)) ++ (x165548)) ++ (x165545)) ++ (x165548)) ++ (x165548)) ++ (x165554)) ++ (x165554)) ++ (x165554);
    let x165556: bool = -(x165547);
    let x165557: u13 = x165555;
    let x165558: u2 = (x165549).0x0;
    let x165559: u2 = (x165549).0x1;
    let x165560: u37 = (u37:0x1fffffffff);
    let x165561: u13 = clz(x165557);
    let x165562: u2 = (x165549).0x0;
    let x165563: u57 = (((((((((((((((x165555) ++ (x165552)) ++ (x165548)) ++ (x165546)) ++ (x165550)) ++ (x165552)) ++ (x165545)) ++ (x165550)) ++ (x165548)) ++ (x165546)) ++ (x165562)) ++ (x165559)) ++ (x165555)) ++ (x165554)) ++ (x165559)) ++ (x165555);
    let x165564: u13 = (((x165550 as s13)) >> (if ((x165557) >= ((u13:0x5))) { ((u13:0x5)) } else { (x165557) })) as u13;
    let x165565: bool = (x165551).0x0;
    let x165566: u14 = one_hot(x165564, (u1:1));
    let x165567: bool = (x165547) + (x165556);
    let x165568: u2 = one_hot(x165565, (u1:1));
    let x165569: u2 = one_hot(x165552, (u1:1));
    let x165570: u21 = (u21:0x0);
    let x165571: bool = (x165556) & ((x165557 as bool));
    let x165572: u38 = one_hot(x165560, (u1:0));
    let x165573: u1 = ((x165555) != ((u13:0x0))) && ((x165550) != ((bool:0x0)));
    let x165574: u14 = -(x165566);
    let x165575: u21 = x165570;
    let x165576: bool = (x165551).0x0;
    let x165577: u2 = (x165549).0x0;
    (x165563, x165561, x165553, x165555, x165545, x165571, x165573, x165577, x165555, x165560)
}


