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
#ifndef __HSV_FAST_HSV2RGB_H__
#define __HSV_FAST_HSV2RGB_H__

#include <stdint.h>

#define HSV_HUE_SEXTANT		256
#define HSV_HUE_STEPS		(6 * HSV_HUE_SEXTANT)

#define HSV_HUE_MIN		0
#define HSV_HUE_MAX		(HSV_HUE_STEPS - 1)
#define HSV_SAT_MIN		0
#define HSV_SAT_MAX		255
#define HSV_VAL_MIN		0
#define HSV_VAL_MAX		255

/* Options: */
#define HSV_USE_SEXTANT_TEST	/* Limit the hue to 0...360 degrees */
#define HSV_USE_ASSEMBLY	/* Optimize code using assembly */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The "used" attribute is required in the assembly version because Arduino
 * uses lto. With lto, the function is inlined into the rest of the code, which
 * eliminates the pointers to r, g, and b. However, the code requires pointers
 * because we need to do pointer swapping.
 * Adding the attribute to the function here forces gcc/linker to add the
 * function separately to the code and it must be called, just like other
 * library functions.
 */
#ifdef HSV_USE_ASSEMBLY
#define HSVFUNC_ATTRUSED	__attribute__((used))
#else
#define HSVFUNC_ATTRUSED
#endif

void fast_hsv2rgb_8bit(uint16_t h, uint8_t s, uint8_t v, uint8_t *r, uint8_t *g , uint8_t *b) HSVFUNC_ATTRUSED;
void fast_hsv2rgb_32bit(uint16_t h, uint8_t s, uint8_t v, uint8_t *r, uint8_t *g , uint8_t *b) HSVFUNC_ATTRUSED;

#ifdef __cplusplus
}
#endif


/*
 * Macros that are common to all implementations
 */
#define HSV_MONOCHROMATIC_TEST(s,v,r,g,b) \
	do { \
		if(!(s)) { \
			 *(r) = *(g) = *(b) = (v); \
			return; \
		} \
	} while(0)

#ifdef HSV_USE_SEXTANT_TEST
#define HSV_SEXTANT_TEST(sextant) \
	do { \
		if((sextant) > 5) { \
			(sextant) = 5; \
		} \
	} while(0)
#else
#define HSV_SEXTANT_TEST(sextant) do { ; } while(0)
#endif

/*
 * Pointer swapping:
 * 	sext.	r g b	r<>b	g<>b	r <> g	result
 *	0 0 0	v u c			!u v c	u v c
 *	0 0 1	d v c				d v c
 *	0 1 0	c v u	u v c			u v c
 *	0 1 1	c d v	v d c		d v c	d v c
 *	1 0 0	u c v		u v c		u v c
 *	1 0 1	v c d		v d c	d v c	d v c
 *
 * if(sextant & 2)
 * 	r <-> b
 *
 * if(sextant & 4)
 * 	g <-> b
 *
 * if(!(sextant & 6) {
 * 	if(!(sextant & 1))
 * 		r <-> g
 * } else {
 * 	if(sextant & 1)
 * 		r <-> g
 * }
 */
#define HSV_SWAPPTR(a,b)	do { uint8_t *tmp = (a); (a) = (b); (b) = tmp; } while(0)
#define HSV_POINTER_SWAP(sextant,r,g,b) \
	do { \
		if((sextant) & 2) { \
			HSV_SWAPPTR((r), (b)); \
		} \
		if((sextant) & 4) { \
			HSV_SWAPPTR((g), (b)); \
		} \
		if(!((sextant) & 6)) { \
			if(!((sextant) & 1)) { \
				HSV_SWAPPTR((r), (g)); \
			} \
		} else { \
			if((sextant) & 1) { \
				HSV_SWAPPTR((r), (g)); \
			} \
		} \
	} while(0)


#if defined(HSV_USE_ASSEMBLY)

#if defined(__AVR_HAVE_MUL__)
/* Multiply instruction available */
#define MUL(ra,rb,pfx) \
		"mul	" ra ", " rb "\n\t"
#else /* defined(__AVR_HAVE_MUL__) */
/*
 * Small AVR cores (f.x. ATtiny and ATmega*u2) do not have a mul instruction.
 * It is available from the avr4 architecture.
 *
 * Multiply: r1:r0 = ra * rb (clobbers r19, r17, r16)
 * Algorithm:
 * uint16_t mul(uint8_t ra, uint8_t rb)
 * {
 *	r1:r0   = 0
 *	r19     = ra
 *	r17:r16 = rb
 *	do {
 *		if(r19 & 1)
 *			r1:r0 += r17:r16
 *		r17:r16 += r17:r16
 *		r19 >>= 1;
 *	} while(r19)
 *	return r1:r0
 * }
 */
#define MUL(ra,rb,pfx) \
	"\n" \
	".L" pfx "mul_" ra "_" rb "_%=:\n\t" \
		"clr	r0\n\t" \
		"clr	r1\n\t" \
		"mov	r19, " ra "\n\t" \
		"mov	r16, " rb "\n\t" \
		"clr	r17\n" \
	".L" pfx "mul_loop_" ra "_" rb "_%=:\n\t" \
		"sbrc	r19, 0\n\t" \
		"add	r0, r16\n\t" \
		"sbrc	r19, 0\n\t" \
		"adc	r1, r17\n\t" \
		"add	r16, r16\n\t" \
		"adc	r17, r17\n\t" \
		"lsr	r19\n\t" \
		"brne	.L" pfx "mul_loop_" ra "_" rb "_%=\n\t"
#endif /* defined(__AVR_HAVE_MUL__) */

#if defined(__AVR_HAVE_MOVW__)
#define MOVW(rdh,rdl,rsh,rsl) \
		"movw	" rdl ", " rsl "\n\t"
#else /* defined(__AVR_HAVE_MOVW__) */
/*
 * The avr2 (and avr1) architecture is missing the movw instruction
 * (ATtiny22/26/1* and AT90s*). All others should have it.
 */
#define MOVW(rdh,rdl,rsh,rsl) \
		"mov	" rdl ", " rsl "\n\t" \
		"mov	" rdh ", " rsh "\n\t"
#endif /* defined(__AVR_HAVE_MOVW__) */
#endif /* defined(HSV_USE_ASSEMBLY) */

#endif
