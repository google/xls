// options: {"input_is_dslx": true, "convert_to_ir": true, "optimize_ir": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=3"], "simulate": false, "simulator": null}
// args: bits[14]:0x3295
fn main(x81: u14) -> (u22, u14, u5, u22, u5, u5, u5, u5, u5, u14) {
    let x82: u5 = (u5:0x15) in
    let x83: u5 = !(x82) in
    let x84: u5 = ((x81) as u5) + (x82) in
    let x85: u5 = (x84) - (x82) in
    let x86: u5 = -(x83) in
    let x87: u5 = (x86) * (x83) in
    let x88: u22 = (u22:0x1fffff) in
    let x89: u5 = (x84) << (x82) in
    let x90: u13 = (u13:0xfff) in
    (x88, x81, x82, x88, x83, x86, x85, x84, x83, x81)
}


