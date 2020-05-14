// options: {"input_is_dslx": true, "convert_to_ir": true, "optimize_ir": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=3"], "simulate": false, "simulator": null}
// args: bits[6]:0x0; bits[19]:0x398dd; bits[25]:0x3fc4fe; bits[28]:0x9b743d6
fn main(x56: u6, x57: u19, x58: u25, x59: u28) -> (u22, u25, u19, u22, u6, u25, u19, u6, u25, u22, u28, u20, u6) {
    let x60: u19 = ~(x57) in
    let x61: u20 = (u20:0x80) in
    let x62: u20 = ((x56) as u20) - (x61) in
    let x63: u20 = (x61) + (x62) in
    let x64: u22 = (u22:0x1000) in
    let x65: u19 = -(x57) in
    let x66: u25 = (x58) ^ ((x61 as u25)) in
    let x67: u25 = (x58) | ((x56 as u25)) in
    let x68: u22 = ((x57 as u22)) + (x64) in
    (x68, x67, x60, x64, x56, x66, x65, x56, x67, x68, x59, x63, x56)
}


