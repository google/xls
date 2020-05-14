// options: {"input_is_dslx": true, "convert_to_ir": true, "optimize_ir": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=3"], "simulate": false, "simulator": null}
// args: bits[7]:0x1; bits[7]:0x3f
fn main(x8607: u7, x8608: u7) -> u7 {
    let x8610: u7 = (x8607) & (x8608) in
    let x8611: u7 = one_hot_sel(x8608 as u2, [x8610, x8608]) in
    x8611
}


