// options: {"input_is_dslx": true, "convert_to_ir": true, "optimize_ir": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=3"], "simulate": false, "simulator": null}
// args: bits[1]:0x0; bits[63]:0x1; bits[51]:0x200000000
fn main(x283: bool, x284: u63, x285: u51) -> u51 {
    let x286: u51 = one_hot_sel(x283, [x285]) in
    x286
}


