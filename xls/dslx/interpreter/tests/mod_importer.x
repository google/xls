import xls.dslx.interpreter.tests.mod_imported
import xls.dslx.interpreter.tests.mod_imported as mi

fn main(x: u3) -> u1 {
  mod_imported::my_lsb(x) || mi::my_lsb(x)
}

test main {
  assert_eq(u1:0b1, main(u3:0b001))
}
