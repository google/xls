// options: {"input_is_dslx": true, "convert_to_ir": true, "optimize_ir": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=3"], "simulate": false, "simulator": null}
// args: bits[30]:0x2e99f5a0; bits[15]:0x8d9; bits[32]:0x58490179; bits[17]:0x1228d
fn main(x17: u30, x18: u15, x19: u32, x20: u17) -> (u6, u6, u6, u6, u6, u6, u30, u6, u6, u6, u32, u6) {
    let x21: u6 = (u6:0x2) in
    let x22: u6 = (x21) * (x21) in
    let x23: u9 = (u9:0x2) in
    let x24: u6 = !(x22) in
    let x25: u6 = (x21) >>> (x21) in
    let x26: u6 = (x24) >> (x24) in
    let x27: u6 = (x21) * (x22) in
    let x28: u25 = (u25:0x2000) in
    (x21, x22, x26, x27, x25, x26, x17, x27, x25, x21, x19, x25)
}


