/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016  B. Stultiens
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include "fast_hsv2rgb.h"

#if defined(HSV_USE_ASSEMBLY) && !defined(__AVR_ARCH__)
#warning "Only AVR assembly is implemented. Other architectures use C fallback."
#undef HSV_USE_ASSEMBLY
#endif

void fast_hsv2rgb_32bit(uint16_t h, uint8_t s, uint8_t v, uint8_t *r, uint8_t *g , uint8_t *b)
{
#ifndef HSV_USE_ASSEMBLY
	HSV_MONOCHROMATIC_TEST(s, v, r, g, b);	// Exit with grayscale if s == 0

	uint8_t sextant = h >> 8;

	HSV_SEXTANT_TEST(sextant);		// Optional: Limit hue sextants to defined space

	HSV_POINTER_SWAP(sextant, r, g, b);	// Swap pointers depending which sextant we are in

	*g = v;		// Top level

	// Perform actual calculations

	/*
	 * Bottom level: v * (1.0 - s)
	 * --> (v * (255 - s) + error_corr + 1) / 256
	 */
	uint16_t ww;		// Intermediate result
	ww = v * (255 - s);	// We don't use ~s to prevent size-promotion side effects
	ww += 1;		// Error correction
	ww += ww >> 8;		// Error correction
	*b = ww >> 8;

	uint8_t h_fraction = h & 0xff;	// 0...255
	uint32_t d;			// Intermediate result

	if(!(sextant & 1)) {
		// *r = ...slope_up...;
		d = v * (uint32_t)((255 << 8) - (uint16_t)(s * (256 - h_fraction)));
		d += d >> 8;	// Error correction
		d += v;		// Error correction
		*r = d >> 16;
	} else {
		// *r = ...slope_down...;
		d = v * (uint32_t)((255 << 8) - (uint16_t)(s * h_fraction));
		d += d >> 8;	// Error correction
		d += v;		// Error correction
		*r = d >> 16;
	}

#else /* HSV_USE_ASSEMBLY */

#ifdef __AVR_ARCH__
	/*
	 * Function arguments passed in registers:
	 *   h = r25:r24
	 *   s = r22
	 *   v = r20
	 *   *r = r19:r18
	 *   *g = r17:r16
	 *   *b = r15:r14
	 */
	asm volatile (
		MOVW("r27", "r26", "r19", "r18")	// r -> X
		MOVW("r29", "r28", "r17", "r16")	// g -> Y
		MOVW("r31", "r30", "r15", "r14")	// b -> Z

		"cpse	r22, __zero_reg__\n\t"	// if(!s) --> monochromatic
		"rjmp	.Lneedcalc%=\n\t"

		"st	X, r20\n\t"		// *r = *g = *b = v;
		"st	Y, r20\n\t"
		"st	Z, r20\n\t"
		"rjmp	.Lendoffunc%=\n"	// return

	".Lneedcalc%=:\n\t"

		"cpi	r25, lo8(6)\n\t"	// if(hi8(h) > 5) hi8(h) = 5;
		"brlo	.Linrange%=\n\t"
		"ldi	r25,lo8(5)\n"
	".Linrange%=:\n\t"

		"sbrs	r25, 1\n\t"		// if(sextant & 2) swapptr(r, b);
		"rjmp	.Lsextno1%=\n\t"
		MOVW("r19", "r18", "r27", "r26")
		MOVW("r27", "r26", "r31", "r30")
		MOVW("r31", "r30", "r19", "r18")
	"\n"
	".Lsextno1%=:\n\t"

		"sbrs	r25, 2\n\t"		// if(sextant & 4) swapptr(g, b);
		"rjmp	.Lsextno2%=\n\t"
		MOVW("r19", "r18", "r29", "r28")
		MOVW("r29", "r28", "r31", "r30")
		MOVW("r31", "r30", "r19", "r18")
	"\n"
	".Lsextno2%=:\n\t"

		"ldi	r18, lo8(6)\n\t"
		"and	r18, r25\n\t"		// if(!(sextant & 6))
		"brne	.Lsext2345%=\n\t"

		"sbrc	r25, 0\n\t"		// if(!(sextant & 6) && !(sextant & 1)) --> doswasp
		"rjmp	.Ldoneswap%=\n"
	".Lsext0%=:\n\t"
		MOVW("r19", "r18", "r27", "r26")
		MOVW("r27", "r26", "r29", "r28")
		MOVW("r29", "r28", "r19", "r18")
		"rjmp	.Ldoneswap%=\n"

	".Lsext2345%=:\n\t"
		"sbrc	r25, 0\n\t"		// if((sextant & 6) && (sextant & 1)) --> doswap
		"rjmp	.Lsext0%=\n"
	".Ldoneswap%=:\n\t"

		/* Top level assignment first to free up Y register (r29:r28) */
		"st	Y, r20\n\t"		// *g = v

		"ldi	r18, 0\n\t"		// Temporary zero reg (r1 is used by mul)
		"ldi	r19, 1\n\t"		// Temporary one reg

		/*
		 * Do bottom level next so we may use Z register (r31:r30).
		 *
		 *	Bottom level: v * (1.0 - s)
		 *	--> (v * (255 - s) + error_corr + 1) / 256
		 *	1 bb = ~s;
		 *	2 ww = v * bb;
		 *	3 ww += 1;
		 *	4 ww += ww >> 8;	// error_corr for division 1/256 instead of 1/255
		 *	5 *b = ww >> 8;
		 */
		"mov	r23, r22\n\t"		// 1 use copy of s
		"com	r23\n\t"		// 1
		MUL("r23", "r20", "a")		// 2 r1:r0 = v *  ~s
		"add	r0, r19\n\t"		// 3 r1:r0 += 1
		"adc	r1, r18\n\t"		// 3
		"add	r0, r1\n\t"		// 4 r1:r0 += r1:r0 >> 8
		"adc	r1, r18\n\t"		// 4
		"st	Z, r1\n\t"		// 5 *b = r1:r0 >> 8

		/* All that is left are the slopes */

		"sbrc	r25, 0\n\t"		// if(sextant & 1) --> slope down
		"rjmp	.Lslopedown%=\n\t"

		/* Slope up:
		 *	d = v * ((255 << 8) - (s * (256 - h_fraction)));
		 *	d += d >> 8;	// Error correction
		 *	d += v;		// Error correction
		 *	*r = d >> 16;
		 */
		"ldi	r28, 0\n\t"		// 0 r19:r28 = 256 (0x100)
		"sub	r28, r24\n\t"		// 0 256 - h_fraction
		"sbc	r19, r18\n\t"
		MUL("r22", "r28", "b")		// r1:r0 = s * lo8(256 - h_fraction)
		"sbrc	r19, 0\n\t"		// if(256 - h_fraction == 0x100)
		"add	r1, r22\n\t"		//   r1:r0 += s << 8
		"rjmp	.Lslopecommon%=\n\t"	// r1:r0 holds inner multiplication

		/*
		 * Slope down:
		 *	d = v * ((255 << 8) - (s * h_fraction));
		 *	d += d >> 8;	// Error correction
		 *	d += v;		// Error correction
		 *	*r = d >> 16;
		 */
	"\n"
	".Lslopedown%=:\n\t"
		MUL("r22", "r24", "b")		// r1:r0 = s * h_fraction
	"\n"
	".Lslopecommon%=:\n\t"
		"ldi	r31, 255\n\t"		// Z = 255 << 8
		"mov	r30, r18\n\t"
		"sub	r30, r0\n\t"		// Z = (255 << 8) - (s * (...))
		"sbc	r31, r1\n\t"

		/*
		 * Multiply r20 * r31:r30:
		 * r29:r28	= r20 * r31	lo8 * hi8
		 * r1:r0	= r20 * r30	lo8 * lo8
		 * Sum:
		 *	r29 : r28 : (0)
		 *	(0) : r1  : r0
		 *	--------------- +
		 *	r29 : r28 : r0
		 */
		MUL("r31", "r20", "c")
		MOVW("r29", "r28", "r1", "r0")	// Y = hi8(Z) * v
		MUL("r30", "r20", "d")		// r1:r0 = lo8(Z) * v
		"add	r28, r1\n\t"
		"adc	r29, r18\n\t"		// r29:r28:r0 = Y * v
		"add	r0, r28\n\t"		// Error correction r29:r28:r0 += 0:r29:r28
		"adc	r28, r29\n\t"
		"adc	r29, r18\n\t"
		"add	r0, r20\n\t"		// Error correction r29:r28:r0 += v
		"adc	r28, r18\n\t"
		"adc	r29, r18\n\t"
		"st	X, r29\n\t"		// *r = slope result >> 16

		"clr	__zero_reg__\n"		// Restore zero reg

	".Lendoffunc%=:\n"
	:
	:
	: "r31", "r30", "r29", "r28", "r27", "r26", "r25", "r24", "r23", "r22", "r19", "r18"
#ifndef __AVR_HAVE_MUL__
	 , "r17", "r16"
#endif
	);

#else /* __AVR_ARCH__ */
#error "No assembly version implemented for architecture"
#endif
#endif
}
