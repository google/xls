#![cfg(let_terminator_is_semi = true)]

// Performs a table-less crc32 of the input data as in Hacker's Delight:
// https://www.hackersdelight.org/hdcodetxt/crc.c.txt (roughly flavor b)

fn crc32_one_byte(byte: u8, polynomial: u32, crc: u32) -> u32 {
  let crc: u32 = crc ^ (byte as u32);
  // 8 rounds of updates.
  for (i, crc): (u32, u32) in range(u32:0, u32:8) {
    let mask: u32 = -(crc & u32:1);
    (crc >> u32:1) ^ (polynomial & mask)
  }(crc)
}
