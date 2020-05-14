// Tests for various sizes of binary operations.
// Simple test of these ops and that wide data types work...
// and as a starting point for debugging they don't.
fn main32() -> sN[32] {
  let x = sN[32]:1000 in
  let y = sN[32]:-1000 in
  let add = x + y in
  let mul = add * y in
  let shll = mul << y in
  let shra = mul >>> x in
  let shrl = mul >> x in
  let sub = shrl - y in
  sub / y
}

fn main1k() -> sN[1024] {
  let x = sN[1024]:1 in
  let y = sN[1024]:-3 in
  let add = x + y in
  let mul = add * y in
  let shll = mul << y in
  let shra = mul >>> x in
  let shrl = mul >> x in
  let sub = shrl - y in
  sub / y
}

fn main() -> sN[128] {
  let x = sN[128]:1 in
  let y = sN[128]:-3 in
  let add = x + y in
  let mul = add * y in
  let shll = mul << y in
  let shra = mul >>> x in
  let shrl = mul >> x in
  let sub = shrl - y in
  sub / y
}
