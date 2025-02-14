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

#[test]
fn casts() {
    assert_eq(s1:-1, u1:1 as s1);
    assert_eq(u1:1, s1:-1 as u1);

    // TODO(tedhong): 2023-05-15 Have this fail typecheck.
    assert_eq(s2:-1, s1:-1 as s2);
    assert_eq(s2:-1, s1:-1 as s2);

    // Trivial casts.
    assert_eq(u1:1, u1:1 as u1);
    assert_eq(s1:-1, s1:-1 as s1);
    assert_eq(s1:-1, s1:-1 as s1);
}

#[test]
fn widening_casts() {
    // Casting to an s1.
    assert_eq(s1:0, widening_cast<s1>(s1:0));
    assert_eq(s1:-1, widening_cast<s1>(s1:-1));

    // Casting to an u1.
    assert_eq(u1:0, widening_cast<u1>(u1:0));
    assert_eq(u1:1, widening_cast<u1>(u1:1));

    // Casting to an s3.
    assert_eq(s3:-1, widening_cast<s3>(s1:-1));
    assert_eq(s3:0, widening_cast<s3>(s1:0));

    assert_eq(s3:-2, widening_cast<s3>(s2:-2));
    assert_eq(s3:-1, widening_cast<s3>(s2:-1));
    assert_eq(s3:0, widening_cast<s3>(s2:0));
    assert_eq(s3:1, widening_cast<s3>(s2:1));

    assert_eq(s3:-4, widening_cast<s3>(s3:-4));
    assert_eq(s3:-3, widening_cast<s3>(s3:-3));
    assert_eq(s3:-2, widening_cast<s3>(s3:-2));
    assert_eq(s3:-1, widening_cast<s3>(s3:-1));
    assert_eq(s3:0, widening_cast<s3>(s3:0));
    assert_eq(s3:1, widening_cast<s3>(s3:1));
    assert_eq(s3:2, widening_cast<s3>(s3:2));
    assert_eq(s3:3, widening_cast<s3>(s3:3));

    assert_eq(s3:0, widening_cast<s3>(u1:0));
    assert_eq(s3:1, widening_cast<s3>(u1:1));

    assert_eq(s3:0, widening_cast<s3>(u2:0));
    assert_eq(s3:1, widening_cast<s3>(u2:1));
    assert_eq(s3:2, widening_cast<s3>(u2:2));
    assert_eq(s3:3, widening_cast<s3>(u2:3));

    // Casting to an u3.
    assert_eq(u3:0, widening_cast<u3>(u1:0));
    assert_eq(u3:1, widening_cast<u3>(u1:1));

    assert_eq(u3:0, widening_cast<u3>(u2:0));
    assert_eq(u3:1, widening_cast<u3>(u2:1));
    assert_eq(u3:2, widening_cast<u3>(u2:2));
    assert_eq(u3:3, widening_cast<u3>(u2:3));

    assert_eq(u3:0, widening_cast<u3>(u3:0));
    assert_eq(u3:1, widening_cast<u3>(u3:1));
    assert_eq(u3:2, widening_cast<u3>(u3:2));
    assert_eq(u3:3, widening_cast<u3>(u3:3));
    assert_eq(u3:4, widening_cast<u3>(u3:4));
    assert_eq(u3:5, widening_cast<u3>(u3:5));
    assert_eq(u3:6, widening_cast<u3>(u3:6));
    assert_eq(u3:7, widening_cast<u3>(u3:7));
}

#[test]
fn array_casts() {
    assert_eq(uN[32]:0xff_aa_55_00, uN[8][4]:[u8:0xff, u8:0xaa, u8:0x55, u8:0x00] as u32);
    assert_eq(uN[32]:0xff_aa_55_00 as u8[4], uN[8][4]:[u8:0xff, u8:0xaa, u8:0x55, u8:0x00]);
}

#[test]
fn checked_casts() {
    // TODO(tedhong): 2023-05-25 Uncomment when assert_fail is supported.
    // https://github.com/google/xls/issues/481

    // Casting to an s1.
    assert_eq(s1:0, checked_cast<s1>(u1:0));
    //assert_fails(checked_cast<s1>(u1:1));
    assert_eq(s1:0, checked_cast<s1>(s1:0));
    assert_eq(s1:-1, checked_cast<s1>(s1:-1));

    // Casting to an u1.
    assert_eq(u1:0, checked_cast<u1>(u1:0));
    assert_eq(u1:1, checked_cast<u1>(u1:1));
    assert_eq(u1:0, checked_cast<u1>(s1:0));

    //assert_fails(checked_cast<u1>(s1:-1));

    // Casting to an s3.
    assert_eq(s3:-1, checked_cast<s3>(s1:-1));
    assert_eq(s3:0, checked_cast<s3>(s1:0));

    assert_eq(s3:-2, checked_cast<s3>(s2:-2));
    assert_eq(s3:-1, checked_cast<s3>(s2:-1));
    assert_eq(s3:0, checked_cast<s3>(s2:0));
    assert_eq(s3:1, checked_cast<s3>(s2:1));

    assert_eq(s3:-4, checked_cast<s3>(s3:-4));
    assert_eq(s3:-3, checked_cast<s3>(s3:-3));
    assert_eq(s3:-2, checked_cast<s3>(s3:-2));
    assert_eq(s3:-1, checked_cast<s3>(s3:-1));
    assert_eq(s3:0, checked_cast<s3>(s3:0));
    assert_eq(s3:1, checked_cast<s3>(s3:1));
    assert_eq(s3:2, checked_cast<s3>(s3:2));
    assert_eq(s3:3, checked_cast<s3>(s3:3));

    assert_eq(s3:0, checked_cast<s3>(u1:0));
    assert_eq(s3:1, checked_cast<s3>(u1:1));

    assert_eq(s3:0, checked_cast<s3>(u2:0));
    assert_eq(s3:1, checked_cast<s3>(u2:1));
    assert_eq(s3:2, checked_cast<s3>(u2:2));
    assert_eq(s3:3, checked_cast<s3>(u2:3));

    assert_eq(s3:0, checked_cast<s3>(u3:0));
    assert_eq(s3:1, checked_cast<s3>(u3:1));
    assert_eq(s3:2, checked_cast<s3>(u3:2));
    assert_eq(s3:3, checked_cast<s3>(u3:3));

    //assert_fails(checked_cast<s3>(u3:4));
    //assert_fails(checked_cast<s3>(u3:5));
    //assert_fails(checked_cast<s3>(u3:6));
    //assert_fails(checked_cast<s3>(u3:7));

    // Casting to an u3.
    //assert_fails(checked_cast<u3>(s1:-1));
    assert_eq(u3:0, checked_cast<u3>(s1:0));

    //assert_fails(checked_cast<u3>(s2:-2));
    //assert_fails(checked_cast<u3>(s2:-1));
    assert_eq(u3:0, checked_cast<u3>(s2:0));
    assert_eq(u3:1, checked_cast<u3>(s2:1));

    //assert_fails(checked_cast<u3>(s3:-4));
    //assert_fails(checked_cast<u3>(s3:-3));
    //assert_fails(checked_cast<u3>(s3:-2));
    //assert_fails(checked_cast<u3>(s3:-1));
    assert_eq(u3:0, checked_cast<u3>(s3:0));
    assert_eq(u3:1, checked_cast<u3>(s3:1));
    assert_eq(u3:2, checked_cast<u3>(s3:2));
    assert_eq(u3:3, checked_cast<u3>(s3:3));

    assert_eq(u3:0, checked_cast<u3>(u1:0));
    assert_eq(u3:1, checked_cast<u3>(u1:1));

    assert_eq(u3:0, checked_cast<u3>(u2:0));
    assert_eq(u3:1, checked_cast<u3>(u2:1));
    assert_eq(u3:2, checked_cast<u3>(u2:2));
    assert_eq(u3:3, checked_cast<u3>(u2:3));

    assert_eq(u3:0, checked_cast<u3>(u3:0));
    assert_eq(u3:1, checked_cast<u3>(u3:1));
    assert_eq(u3:2, checked_cast<u3>(u3:2));
    assert_eq(u3:3, checked_cast<u3>(u3:3));
    assert_eq(u3:4, checked_cast<u3>(u3:4));
    assert_eq(u3:5, checked_cast<u3>(u3:5));
    assert_eq(u3:6, checked_cast<u3>(u3:6));
    assert_eq(u3:7, checked_cast<u3>(u3:7));
}
