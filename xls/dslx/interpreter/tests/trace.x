// Verifies that trace() can show up in a DSLX sequnce w/o the
// IR converter complaining.
fn main() -> u3 {
  let x0 = clz(u3:0b111) in
  let _ = trace(x0) in
  x0
}

test trace {
  let x0 = clz(u3:0b011) in
  let _ = trace(x0) in
  let x1 = (x0 as u8) * u8:3 in
  let _ = trace(x1) in
  ()
}
