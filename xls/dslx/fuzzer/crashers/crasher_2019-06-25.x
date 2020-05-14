// options: {"input_is_dslx": true, "convert_to_ir": true, "optimize_ir": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=3"], "simulate": false, "simulator": null}
// args: bits[10]:0xcc
fn main(x79: u10) -> (u5, u5, u5, u5, u5, u5, u5) {
    let x80: u5 = (u5:0x2) in
    let x81: u5 = (x80) << (x80) in
    let x82: u5 = ((x79) as u5) + (x80) in
    let x83: u5 = (x82) - (x80) in
    let x84: u5 = ~(x80) in
    let x85: u5 = (x82) ^ (x82) in
    (x84, x83, x81, x83, x83, x82, x80)
}


