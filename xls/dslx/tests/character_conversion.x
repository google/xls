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

fn main() -> u8[14] {
    // Declaring each ascii escape character supported.
    let newline: u8 = '\n';
    let carriage_return: u8 = '\r';
    let tab: u8 = '\t';
    let backslash: u8 = '\\';
    let null: u8 = '\0';
    let single_quote: u8 = '\'';
    let double_quote: u8 = '"';
    let random_char: u8 = 'c';
    let random_char_deduced_type = 'c';
    let dash: u8 = '-';  // negation symbol
    let dash_deduced_type = '-';  // negation symbol
    let char_escape: u8 = '_';  // underscore
    let continuation_byte: u8 = '\x80';  // continuation byte
    let null_hex_escape: u8 = '\x00';
    [
        newline, carriage_return, tab, backslash, null, single_quote, double_quote, random_char,
        random_char_deduced_type, dash, dash_deduced_type, char_escape, continuation_byte, null_hex_escape,
    ]
}

#[test]
fn conversion_test() { assert_eq(main(), "\n\r\t\\\0\'\"cc--_\x80\0") }
