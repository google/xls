import float32;

pub fn double_fraction_carry(f: float32::F32) -> (uN[float32::F32_FRACTION_SZ], u1) {
    let f = f.fraction as uN[float32::F32_FRACTION_SZ + u32:1];
    let f_x2 = f + f;
    (f_x2[0+:float32::F32_FRACTION_SZ], f_x2[float32::F32_FRACTION_SZ+:u1])
}
