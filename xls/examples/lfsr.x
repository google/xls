////////////////////////////////////////////////////////////////////////////////
// Linear-feedback shift register (LFSR)
// A parametric function to compute the next value of an LFSR counter.
// Taps can be specified to have varying periods for a given number of bits.
//
// For example, lfsr(u5:4, u5:0b10100) computes the value that comes after 4
// in a 5-bit LFSR with taps on bits 2 and 4.
////////////////////////////////////////////////////////////////////////////////

fn lfsr<BIT_WIDTH: u32>(current_value: uN[BIT_WIDTH], tap_mask: uN[BIT_WIDTH]) -> uN[BIT_WIDTH] {
	// Compute the new bit from the taps
	let new_bit = for (index, xor_bit): (u32, u1) in range(u32:0, BIT_WIDTH) {
		if tap_mask[index+:u1] == u1:0 {
			xor_bit
		} else {
			xor_bit ^ current_value[index+:u1]
		}
	} (u1:0);

	// Kick the high bit and insert the new bit
	current_value[u32:0 +: uN[BIT_WIDTH - u32:1]] ++ new_bit
}

////////////////////////////////////////////////////////////////////////////////
// A series of maximal LFSRs for different bit widths.
// These are only examples and it is possible to use different bit widths and
// tap masks.
// Source: https://en.wikipedia.org/wiki/Linear-feedback_shift_register
////////////////////////////////////////////////////////////////////////////////

fn lfsr2(n: u2) -> u2 {
	lfsr(n, u2:0b11)
}

fn lfsr3(n: u3) -> u3 {
	lfsr(n, u3:0b110)
}

fn lfsr4(n: u4) -> u4 {
	lfsr(n, u4:0b1100)
}

fn lfsr5(n: u5) -> u5 {
	lfsr(n, u5:0b10100)
}

fn lfsr6(n: u6) -> u6 {
	lfsr(n, u6:0b110000)
}

fn lfsr7(n: u7) -> u7 {
	lfsr(n, u7:0b1100000)
}

fn lfsr8(n: u8) -> u8 {
	lfsr(n, u8:0b10111000)
}

fn lfsr9(n: u9) -> u9 {
	lfsr(n, u9:0b100010000)
}

fn lfsr10(n: u10) -> u10 {
	lfsr(n, u10:0b1001000000)
}

fn lfsr11(n: u11) -> u11 {
	lfsr(n, u11:0b10100000000)
}

fn lfsr12(n: u12) -> u12 {
	lfsr(n, u12:0b111000001000)
}

fn lfsr13(n: u13) -> u13 {
	lfsr(n, u13:0b1110010000000)
}

fn lfsr14(n: u14) -> u14 {
	lfsr(n, u14:0b11100000000010)
}

fn lfsr15(n: u15) -> u15 {
	lfsr(n, u15:0b110000000000000)
}

fn lfsr16(n: u16) -> u16 {
	lfsr(n, u16:0b1101000000001000)
}

fn lfsr17(n: u17) -> u17 {
	lfsr(n, u17:0b10010000000000000)
}

fn lfsr18(n: u18) -> u18 {
	lfsr(n, u18:0b100000010000000000)
}

fn lfsr19(n: u19) -> u19 {
	lfsr(n, u19:0b1110010000000000000)
}

fn lfsr20(n: u20) -> u20 {
	lfsr(n, u20:0b10010000000000000000)
}

fn lfsr21(n: u21) -> u21 {
	lfsr(n, u21:0b101000000000000000000)
}

fn lfsr22(n: u22) -> u22 {
	lfsr(n, u22:0b1100000000000000000000)
}

fn lfsr23(n: u23) -> u23 {
	lfsr(n, u23:0b10000100000000000000000)
}

fn lfsr24(n: u24) -> u24 {
	lfsr(n, u24:0b111000010000000000000000)
}

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

#[test]
fn lfsr7_test() {
    let _ = assert_eq(lfsr7(u7:0), u7:0);
    let _ = assert_eq(lfsr7(u7:1), u7:2);
    let _ = assert_eq(lfsr7(u7:2), u7:4);
    let _ = assert_eq(lfsr7(u7:4), u7:8);
    let _ = assert_eq(lfsr7(u7:8), u7:16);
    let _ = assert_eq(lfsr7(u7:16), u7:32);
    let _ = assert_eq(lfsr7(u7:32), u7:65);
    let _ = assert_eq(lfsr7(u7:65), u7:3);
    let _ = assert_eq(lfsr7(u7:3), u7:6);
    let _ = assert_eq(lfsr7(u7:6), u7:12);
    let _ = assert_eq(lfsr7(u7:12), u7:24);
    let _ = assert_eq(lfsr7(u7:24), u7:48);
    let _ = assert_eq(lfsr7(u7:48), u7:97);
    let _ = assert_eq(lfsr7(u7:97), u7:66);
    let _ = assert_eq(lfsr7(u7:66), u7:5);
    let _ = assert_eq(lfsr7(u7:5), u7:10);
    let _ = assert_eq(lfsr7(u7:10), u7:20);
    let _ = assert_eq(lfsr7(u7:20), u7:40);
    let _ = assert_eq(lfsr7(u7:40), u7:81);
    let _ = assert_eq(lfsr7(u7:81), u7:35);
    let _ = assert_eq(lfsr7(u7:35), u7:71);
    let _ = assert_eq(lfsr7(u7:71), u7:15);
    let _ = assert_eq(lfsr7(u7:15), u7:30);
    let _ = assert_eq(lfsr7(u7:30), u7:60);
    let _ = assert_eq(lfsr7(u7:60), u7:121);
    let _ = assert_eq(lfsr7(u7:121), u7:114);
    let _ = assert_eq(lfsr7(u7:114), u7:100);
    let _ = assert_eq(lfsr7(u7:100), u7:72);
    let _ = assert_eq(lfsr7(u7:72), u7:17);
    let _ = assert_eq(lfsr7(u7:17), u7:34);
    let _ = assert_eq(lfsr7(u7:34), u7:69);
    let _ = assert_eq(lfsr7(u7:69), u7:11);
    let _ = assert_eq(lfsr7(u7:11), u7:22);
    let _ = assert_eq(lfsr7(u7:22), u7:44);
    let _ = assert_eq(lfsr7(u7:44), u7:89);
    let _ = assert_eq(lfsr7(u7:89), u7:51);
    let _ = assert_eq(lfsr7(u7:51), u7:103);
    let _ = assert_eq(lfsr7(u7:103), u7:78);
    let _ = assert_eq(lfsr7(u7:78), u7:29);
    let _ = assert_eq(lfsr7(u7:29), u7:58);
    let _ = assert_eq(lfsr7(u7:58), u7:117);
    let _ = assert_eq(lfsr7(u7:117), u7:106);
    let _ = assert_eq(lfsr7(u7:106), u7:84);
    let _ = assert_eq(lfsr7(u7:84), u7:41);
    let _ = assert_eq(lfsr7(u7:41), u7:83);
    let _ = assert_eq(lfsr7(u7:83), u7:39);
    let _ = assert_eq(lfsr7(u7:39), u7:79);
    let _ = assert_eq(lfsr7(u7:79), u7:31);
    let _ = assert_eq(lfsr7(u7:31), u7:62);
    let _ = assert_eq(lfsr7(u7:62), u7:125);
    let _ = assert_eq(lfsr7(u7:125), u7:122);
    let _ = assert_eq(lfsr7(u7:122), u7:116);
    let _ = assert_eq(lfsr7(u7:116), u7:104);
    let _ = assert_eq(lfsr7(u7:104), u7:80);
    let _ = assert_eq(lfsr7(u7:80), u7:33);
    let _ = assert_eq(lfsr7(u7:33), u7:67);
    let _ = assert_eq(lfsr7(u7:67), u7:7);
    let _ = assert_eq(lfsr7(u7:7), u7:14);
    let _ = assert_eq(lfsr7(u7:14), u7:28);
    let _ = assert_eq(lfsr7(u7:28), u7:56);
    let _ = assert_eq(lfsr7(u7:56), u7:113);
    let _ = assert_eq(lfsr7(u7:113), u7:98);
    let _ = assert_eq(lfsr7(u7:98), u7:68);
    let _ = assert_eq(lfsr7(u7:68), u7:9);
    let _ = assert_eq(lfsr7(u7:9), u7:18);
    let _ = assert_eq(lfsr7(u7:18), u7:36);
    let _ = assert_eq(lfsr7(u7:36), u7:73);
    let _ = assert_eq(lfsr7(u7:73), u7:19);
    let _ = assert_eq(lfsr7(u7:19), u7:38);
    let _ = assert_eq(lfsr7(u7:38), u7:77);
    let _ = assert_eq(lfsr7(u7:77), u7:27);
    let _ = assert_eq(lfsr7(u7:27), u7:54);
    let _ = assert_eq(lfsr7(u7:54), u7:109);
    let _ = assert_eq(lfsr7(u7:109), u7:90);
    let _ = assert_eq(lfsr7(u7:90), u7:53);
    let _ = assert_eq(lfsr7(u7:53), u7:107);
    let _ = assert_eq(lfsr7(u7:107), u7:86);
    let _ = assert_eq(lfsr7(u7:86), u7:45);
    let _ = assert_eq(lfsr7(u7:45), u7:91);
    let _ = assert_eq(lfsr7(u7:91), u7:55);
    let _ = assert_eq(lfsr7(u7:55), u7:111);
    let _ = assert_eq(lfsr7(u7:111), u7:94);
    let _ = assert_eq(lfsr7(u7:94), u7:61);
    let _ = assert_eq(lfsr7(u7:61), u7:123);
    let _ = assert_eq(lfsr7(u7:123), u7:118);
    let _ = assert_eq(lfsr7(u7:118), u7:108);
    let _ = assert_eq(lfsr7(u7:108), u7:88);
    let _ = assert_eq(lfsr7(u7:88), u7:49);
    let _ = assert_eq(lfsr7(u7:49), u7:99);
    let _ = assert_eq(lfsr7(u7:99), u7:70);
    let _ = assert_eq(lfsr7(u7:70), u7:13);
    let _ = assert_eq(lfsr7(u7:13), u7:26);
    let _ = assert_eq(lfsr7(u7:26), u7:52);
    let _ = assert_eq(lfsr7(u7:52), u7:105);
    let _ = assert_eq(lfsr7(u7:105), u7:82);
    let _ = assert_eq(lfsr7(u7:82), u7:37);
    let _ = assert_eq(lfsr7(u7:37), u7:75);
    let _ = assert_eq(lfsr7(u7:75), u7:23);
    let _ = assert_eq(lfsr7(u7:23), u7:46);
    let _ = assert_eq(lfsr7(u7:46), u7:93);
    let _ = assert_eq(lfsr7(u7:93), u7:59);
    let _ = assert_eq(lfsr7(u7:59), u7:119);
    let _ = assert_eq(lfsr7(u7:119), u7:110);
    let _ = assert_eq(lfsr7(u7:110), u7:92);
    let _ = assert_eq(lfsr7(u7:92), u7:57);
    let _ = assert_eq(lfsr7(u7:57), u7:115);
    let _ = assert_eq(lfsr7(u7:115), u7:102);
    let _ = assert_eq(lfsr7(u7:102), u7:76);
    let _ = assert_eq(lfsr7(u7:76), u7:25);
    let _ = assert_eq(lfsr7(u7:25), u7:50);
    let _ = assert_eq(lfsr7(u7:50), u7:101);
    let _ = assert_eq(lfsr7(u7:101), u7:74);
    let _ = assert_eq(lfsr7(u7:74), u7:21);
    let _ = assert_eq(lfsr7(u7:21), u7:42);
    let _ = assert_eq(lfsr7(u7:42), u7:85);
    let _ = assert_eq(lfsr7(u7:85), u7:43);
    let _ = assert_eq(lfsr7(u7:43), u7:87);
    let _ = assert_eq(lfsr7(u7:87), u7:47);
    let _ = assert_eq(lfsr7(u7:47), u7:95);
    let _ = assert_eq(lfsr7(u7:95), u7:63);
    let _ = assert_eq(lfsr7(u7:63), u7:127);
    let _ = assert_eq(lfsr7(u7:127), u7:126);
    let _ = assert_eq(lfsr7(u7:126), u7:124);
    let _ = assert_eq(lfsr7(u7:124), u7:120);
    let _ = assert_eq(lfsr7(u7:120), u7:112);
    let _ = assert_eq(lfsr7(u7:112), u7:96);
    let _ = assert_eq(lfsr7(u7:96), u7:64);
    let _ = assert_eq(lfsr7(u7:64), u7:1);
	()
}

#[test]
fn lfsr8_test() {
	let _ = assert_eq(lfsr8(u8:0), u8:0);
	let _ = assert_eq(lfsr8(u8:1), u8:2);
	let _ = assert_eq(lfsr8(u8:2), u8:4);
	let _ = assert_eq(lfsr8(u8:4), u8:8);
	let _ = assert_eq(lfsr8(u8:8), u8:17);
	let _ = assert_eq(lfsr8(u8:17), u8:35);
	let _ = assert_eq(lfsr8(u8:35), u8:71);
	let _ = assert_eq(lfsr8(u8:71), u8:142);
	let _ = assert_eq(lfsr8(u8:142), u8:28);
	let _ = assert_eq(lfsr8(u8:28), u8:56);
	let _ = assert_eq(lfsr8(u8:56), u8:113);
	let _ = assert_eq(lfsr8(u8:113), u8:226);
	let _ = assert_eq(lfsr8(u8:226), u8:196);
	let _ = assert_eq(lfsr8(u8:196), u8:137);
	let _ = assert_eq(lfsr8(u8:137), u8:18);
	let _ = assert_eq(lfsr8(u8:18), u8:37);
	let _ = assert_eq(lfsr8(u8:37), u8:75);
	let _ = assert_eq(lfsr8(u8:75), u8:151);
	let _ = assert_eq(lfsr8(u8:151), u8:46);
	let _ = assert_eq(lfsr8(u8:46), u8:92);
	let _ = assert_eq(lfsr8(u8:92), u8:184);
	let _ = assert_eq(lfsr8(u8:184), u8:112);
	let _ = assert_eq(lfsr8(u8:112), u8:224);
	let _ = assert_eq(lfsr8(u8:224), u8:192);
	let _ = assert_eq(lfsr8(u8:192), u8:129);
	let _ = assert_eq(lfsr8(u8:129), u8:3);
	let _ = assert_eq(lfsr8(u8:3), u8:6);
	let _ = assert_eq(lfsr8(u8:6), u8:12);
	let _ = assert_eq(lfsr8(u8:12), u8:25);
	let _ = assert_eq(lfsr8(u8:25), u8:50);
	let _ = assert_eq(lfsr8(u8:50), u8:100);
	let _ = assert_eq(lfsr8(u8:100), u8:201);
	let _ = assert_eq(lfsr8(u8:201), u8:146);
	let _ = assert_eq(lfsr8(u8:146), u8:36);
	let _ = assert_eq(lfsr8(u8:36), u8:73);
	let _ = assert_eq(lfsr8(u8:73), u8:147);
	let _ = assert_eq(lfsr8(u8:147), u8:38);
	let _ = assert_eq(lfsr8(u8:38), u8:77);
	let _ = assert_eq(lfsr8(u8:77), u8:155);
	let _ = assert_eq(lfsr8(u8:155), u8:55);
	let _ = assert_eq(lfsr8(u8:55), u8:110);
	let _ = assert_eq(lfsr8(u8:110), u8:220);
	let _ = assert_eq(lfsr8(u8:220), u8:185);
	let _ = assert_eq(lfsr8(u8:185), u8:114);
	let _ = assert_eq(lfsr8(u8:114), u8:228);
	let _ = assert_eq(lfsr8(u8:228), u8:200);
	let _ = assert_eq(lfsr8(u8:200), u8:144);
	let _ = assert_eq(lfsr8(u8:144), u8:32);
	let _ = assert_eq(lfsr8(u8:32), u8:65);
	let _ = assert_eq(lfsr8(u8:65), u8:130);
	let _ = assert_eq(lfsr8(u8:130), u8:5);
	let _ = assert_eq(lfsr8(u8:5), u8:10);
	let _ = assert_eq(lfsr8(u8:10), u8:21);
	let _ = assert_eq(lfsr8(u8:21), u8:43);
	let _ = assert_eq(lfsr8(u8:43), u8:86);
	let _ = assert_eq(lfsr8(u8:86), u8:173);
	let _ = assert_eq(lfsr8(u8:173), u8:91);
	let _ = assert_eq(lfsr8(u8:91), u8:182);
	let _ = assert_eq(lfsr8(u8:182), u8:109);
	let _ = assert_eq(lfsr8(u8:109), u8:218);
	let _ = assert_eq(lfsr8(u8:218), u8:181);
	let _ = assert_eq(lfsr8(u8:181), u8:107);
	let _ = assert_eq(lfsr8(u8:107), u8:214);
	let _ = assert_eq(lfsr8(u8:214), u8:172);
	let _ = assert_eq(lfsr8(u8:172), u8:89);
	let _ = assert_eq(lfsr8(u8:89), u8:178);
	let _ = assert_eq(lfsr8(u8:178), u8:101);
	let _ = assert_eq(lfsr8(u8:101), u8:203);
	let _ = assert_eq(lfsr8(u8:203), u8:150);
	let _ = assert_eq(lfsr8(u8:150), u8:44);
	let _ = assert_eq(lfsr8(u8:44), u8:88);
	let _ = assert_eq(lfsr8(u8:88), u8:176);
	let _ = assert_eq(lfsr8(u8:176), u8:97);
	let _ = assert_eq(lfsr8(u8:97), u8:195);
	let _ = assert_eq(lfsr8(u8:195), u8:135);
	let _ = assert_eq(lfsr8(u8:135), u8:15);
	let _ = assert_eq(lfsr8(u8:15), u8:31);
	let _ = assert_eq(lfsr8(u8:31), u8:62);
	let _ = assert_eq(lfsr8(u8:62), u8:125);
	let _ = assert_eq(lfsr8(u8:125), u8:251);
	let _ = assert_eq(lfsr8(u8:251), u8:246);
	let _ = assert_eq(lfsr8(u8:246), u8:237);
	let _ = assert_eq(lfsr8(u8:237), u8:219);
	let _ = assert_eq(lfsr8(u8:219), u8:183);
	let _ = assert_eq(lfsr8(u8:183), u8:111);
	let _ = assert_eq(lfsr8(u8:111), u8:222);
	let _ = assert_eq(lfsr8(u8:222), u8:189);
	let _ = assert_eq(lfsr8(u8:189), u8:122);
	let _ = assert_eq(lfsr8(u8:122), u8:245);
	let _ = assert_eq(lfsr8(u8:245), u8:235);
	let _ = assert_eq(lfsr8(u8:235), u8:215);
	let _ = assert_eq(lfsr8(u8:215), u8:174);
	let _ = assert_eq(lfsr8(u8:174), u8:93);
	let _ = assert_eq(lfsr8(u8:93), u8:186);
	let _ = assert_eq(lfsr8(u8:186), u8:116);
	let _ = assert_eq(lfsr8(u8:116), u8:232);
	let _ = assert_eq(lfsr8(u8:232), u8:209);
	let _ = assert_eq(lfsr8(u8:209), u8:162);
	let _ = assert_eq(lfsr8(u8:162), u8:68);
	let _ = assert_eq(lfsr8(u8:68), u8:136);
	let _ = assert_eq(lfsr8(u8:136), u8:16);
	let _ = assert_eq(lfsr8(u8:16), u8:33);
	let _ = assert_eq(lfsr8(u8:33), u8:67);
	let _ = assert_eq(lfsr8(u8:67), u8:134);
	let _ = assert_eq(lfsr8(u8:134), u8:13);
	let _ = assert_eq(lfsr8(u8:13), u8:27);
	let _ = assert_eq(lfsr8(u8:27), u8:54);
	let _ = assert_eq(lfsr8(u8:54), u8:108);
	let _ = assert_eq(lfsr8(u8:108), u8:216);
	let _ = assert_eq(lfsr8(u8:216), u8:177);
	let _ = assert_eq(lfsr8(u8:177), u8:99);
	let _ = assert_eq(lfsr8(u8:99), u8:199);
	let _ = assert_eq(lfsr8(u8:199), u8:143);
	let _ = assert_eq(lfsr8(u8:143), u8:30);
	let _ = assert_eq(lfsr8(u8:30), u8:60);
	let _ = assert_eq(lfsr8(u8:60), u8:121);
	let _ = assert_eq(lfsr8(u8:121), u8:243);
	let _ = assert_eq(lfsr8(u8:243), u8:231);
	let _ = assert_eq(lfsr8(u8:231), u8:206);
	let _ = assert_eq(lfsr8(u8:206), u8:156);
	let _ = assert_eq(lfsr8(u8:156), u8:57);
	let _ = assert_eq(lfsr8(u8:57), u8:115);
	let _ = assert_eq(lfsr8(u8:115), u8:230);
	let _ = assert_eq(lfsr8(u8:230), u8:204);
	let _ = assert_eq(lfsr8(u8:204), u8:152);
	let _ = assert_eq(lfsr8(u8:152), u8:49);
	let _ = assert_eq(lfsr8(u8:49), u8:98);
	let _ = assert_eq(lfsr8(u8:98), u8:197);
	let _ = assert_eq(lfsr8(u8:197), u8:139);
	let _ = assert_eq(lfsr8(u8:139), u8:22);
	let _ = assert_eq(lfsr8(u8:22), u8:45);
	let _ = assert_eq(lfsr8(u8:45), u8:90);
	let _ = assert_eq(lfsr8(u8:90), u8:180);
	let _ = assert_eq(lfsr8(u8:180), u8:105);
	let _ = assert_eq(lfsr8(u8:105), u8:210);
	let _ = assert_eq(lfsr8(u8:210), u8:164);
	let _ = assert_eq(lfsr8(u8:164), u8:72);
	let _ = assert_eq(lfsr8(u8:72), u8:145);
	let _ = assert_eq(lfsr8(u8:145), u8:34);
	let _ = assert_eq(lfsr8(u8:34), u8:69);
	let _ = assert_eq(lfsr8(u8:69), u8:138);
	let _ = assert_eq(lfsr8(u8:138), u8:20);
	let _ = assert_eq(lfsr8(u8:20), u8:41);
	let _ = assert_eq(lfsr8(u8:41), u8:82);
	let _ = assert_eq(lfsr8(u8:82), u8:165);
	let _ = assert_eq(lfsr8(u8:165), u8:74);
	let _ = assert_eq(lfsr8(u8:74), u8:149);
	let _ = assert_eq(lfsr8(u8:149), u8:42);
	let _ = assert_eq(lfsr8(u8:42), u8:84);
	let _ = assert_eq(lfsr8(u8:84), u8:169);
	let _ = assert_eq(lfsr8(u8:169), u8:83);
	let _ = assert_eq(lfsr8(u8:83), u8:167);
	let _ = assert_eq(lfsr8(u8:167), u8:78);
	let _ = assert_eq(lfsr8(u8:78), u8:157);
	let _ = assert_eq(lfsr8(u8:157), u8:59);
	let _ = assert_eq(lfsr8(u8:59), u8:119);
	let _ = assert_eq(lfsr8(u8:119), u8:238);
	let _ = assert_eq(lfsr8(u8:238), u8:221);
	let _ = assert_eq(lfsr8(u8:221), u8:187);
	let _ = assert_eq(lfsr8(u8:187), u8:118);
	let _ = assert_eq(lfsr8(u8:118), u8:236);
	let _ = assert_eq(lfsr8(u8:236), u8:217);
	let _ = assert_eq(lfsr8(u8:217), u8:179);
	let _ = assert_eq(lfsr8(u8:179), u8:103);
	let _ = assert_eq(lfsr8(u8:103), u8:207);
	let _ = assert_eq(lfsr8(u8:207), u8:158);
	let _ = assert_eq(lfsr8(u8:158), u8:61);
	let _ = assert_eq(lfsr8(u8:61), u8:123);
	let _ = assert_eq(lfsr8(u8:123), u8:247);
	let _ = assert_eq(lfsr8(u8:247), u8:239);
	let _ = assert_eq(lfsr8(u8:239), u8:223);
	let _ = assert_eq(lfsr8(u8:223), u8:191);
	let _ = assert_eq(lfsr8(u8:191), u8:126);
	let _ = assert_eq(lfsr8(u8:126), u8:253);
	let _ = assert_eq(lfsr8(u8:253), u8:250);
	let _ = assert_eq(lfsr8(u8:250), u8:244);
	let _ = assert_eq(lfsr8(u8:244), u8:233);
	let _ = assert_eq(lfsr8(u8:233), u8:211);
	let _ = assert_eq(lfsr8(u8:211), u8:166);
	let _ = assert_eq(lfsr8(u8:166), u8:76);
	let _ = assert_eq(lfsr8(u8:76), u8:153);
	let _ = assert_eq(lfsr8(u8:153), u8:51);
	let _ = assert_eq(lfsr8(u8:51), u8:102);
	let _ = assert_eq(lfsr8(u8:102), u8:205);
	let _ = assert_eq(lfsr8(u8:205), u8:154);
	let _ = assert_eq(lfsr8(u8:154), u8:53);
	let _ = assert_eq(lfsr8(u8:53), u8:106);
	let _ = assert_eq(lfsr8(u8:106), u8:212);
	let _ = assert_eq(lfsr8(u8:212), u8:168);
	let _ = assert_eq(lfsr8(u8:168), u8:81);
	let _ = assert_eq(lfsr8(u8:81), u8:163);
	let _ = assert_eq(lfsr8(u8:163), u8:70);
	let _ = assert_eq(lfsr8(u8:70), u8:140);
	let _ = assert_eq(lfsr8(u8:140), u8:24);
	let _ = assert_eq(lfsr8(u8:24), u8:48);
	let _ = assert_eq(lfsr8(u8:48), u8:96);
	let _ = assert_eq(lfsr8(u8:96), u8:193);
	let _ = assert_eq(lfsr8(u8:193), u8:131);
	let _ = assert_eq(lfsr8(u8:131), u8:7);
	let _ = assert_eq(lfsr8(u8:7), u8:14);
	let _ = assert_eq(lfsr8(u8:14), u8:29);
	let _ = assert_eq(lfsr8(u8:29), u8:58);
	let _ = assert_eq(lfsr8(u8:58), u8:117);
	let _ = assert_eq(lfsr8(u8:117), u8:234);
	let _ = assert_eq(lfsr8(u8:234), u8:213);
	let _ = assert_eq(lfsr8(u8:213), u8:170);
	let _ = assert_eq(lfsr8(u8:170), u8:85);
	let _ = assert_eq(lfsr8(u8:85), u8:171);
	let _ = assert_eq(lfsr8(u8:171), u8:87);
	let _ = assert_eq(lfsr8(u8:87), u8:175);
	let _ = assert_eq(lfsr8(u8:175), u8:95);
	let _ = assert_eq(lfsr8(u8:95), u8:190);
	let _ = assert_eq(lfsr8(u8:190), u8:124);
	let _ = assert_eq(lfsr8(u8:124), u8:249);
	let _ = assert_eq(lfsr8(u8:249), u8:242);
	let _ = assert_eq(lfsr8(u8:242), u8:229);
	let _ = assert_eq(lfsr8(u8:229), u8:202);
	let _ = assert_eq(lfsr8(u8:202), u8:148);
	let _ = assert_eq(lfsr8(u8:148), u8:40);
	let _ = assert_eq(lfsr8(u8:40), u8:80);
	let _ = assert_eq(lfsr8(u8:80), u8:161);
	let _ = assert_eq(lfsr8(u8:161), u8:66);
	let _ = assert_eq(lfsr8(u8:66), u8:132);
	let _ = assert_eq(lfsr8(u8:132), u8:9);
	let _ = assert_eq(lfsr8(u8:9), u8:19);
	let _ = assert_eq(lfsr8(u8:19), u8:39);
	let _ = assert_eq(lfsr8(u8:39), u8:79);
	let _ = assert_eq(lfsr8(u8:79), u8:159);
	let _ = assert_eq(lfsr8(u8:159), u8:63);
	let _ = assert_eq(lfsr8(u8:63), u8:127);
	let _ = assert_eq(lfsr8(u8:127), u8:255);
	let _ = assert_eq(lfsr8(u8:255), u8:254);
	let _ = assert_eq(lfsr8(u8:254), u8:252);
	let _ = assert_eq(lfsr8(u8:252), u8:248);
	let _ = assert_eq(lfsr8(u8:248), u8:240);
	let _ = assert_eq(lfsr8(u8:240), u8:225);
	let _ = assert_eq(lfsr8(u8:225), u8:194);
	let _ = assert_eq(lfsr8(u8:194), u8:133);
	let _ = assert_eq(lfsr8(u8:133), u8:11);
	let _ = assert_eq(lfsr8(u8:11), u8:23);
	let _ = assert_eq(lfsr8(u8:23), u8:47);
	let _ = assert_eq(lfsr8(u8:47), u8:94);
	let _ = assert_eq(lfsr8(u8:94), u8:188);
	let _ = assert_eq(lfsr8(u8:188), u8:120);
	let _ = assert_eq(lfsr8(u8:120), u8:241);
	let _ = assert_eq(lfsr8(u8:241), u8:227);
	let _ = assert_eq(lfsr8(u8:227), u8:198);
	let _ = assert_eq(lfsr8(u8:198), u8:141);
	let _ = assert_eq(lfsr8(u8:141), u8:26);
	let _ = assert_eq(lfsr8(u8:26), u8:52);
	let _ = assert_eq(lfsr8(u8:52), u8:104);
	let _ = assert_eq(lfsr8(u8:104), u8:208);
	let _ = assert_eq(lfsr8(u8:208), u8:160);
	let _ = assert_eq(lfsr8(u8:160), u8:64);
	let _ = assert_eq(lfsr8(u8:64), u8:128);
	let _ = assert_eq(lfsr8(u8:128), u8:1);
	()
}

