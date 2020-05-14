// Demonstrates use of a typedef as the element type in a multidimensional
// array.

#![cfg(let_terminator_is_semi = true)]

type TypeX = u6;

// Identity function.
fn id(x: u6[3][2]) -> u6[3][2] {
  x
}

test array_typedef {
  let a : u6[3][2] = [[TypeX:1, TypeX:2, TypeX:3],
                      [TypeX:4, TypeX:5, TypeX:6]];
  let b : TypeX[3][2] = [[TypeX:1, TypeX:2, TypeX:3],
                         [TypeX:4, TypeX:5, TypeX:6]];
  assert_eq(id(a), id(b))
}
