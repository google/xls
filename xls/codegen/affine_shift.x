pub fn main(data: bool[8], shift: u3) -> bool[8] {
    for (i, result): (u32, bool[8]) in u32:0..u32:8 {
        let value = if i < (shift as u32) {
            false
        } else {
            data[i - (shift as u32)]
        };
        update(result, i, value)
    }(bool[8]:[false, ...])
}

#[test]
fn shift_by_three_test() {
    let data = bool[8]:[true, false, true, true, false, false, true, false];
    assert_eq(
        main(data, u3:3),
        bool[8]:[false, false, false, true, false, true, true, false]
    )
}
