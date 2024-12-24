# DSLX Pattern Exhaustiveness

Document contact: [cdleary](https://github.com/cdleary), first written 2025-02-03

## Overview

DSLX supports pattern matching for a limited pattern grammar/syntax. This document describes the approach we take to checking the exhaustiveness of pattern matches.

## Pattern Matching

The pattern matching syntax supports the following:

* `_` -- a wildcard pattern.
* `name` -- a name pattern -- this can either bind a name, if it is currently unbound, or can be used to check for equality if it is bound.

  (Aside: This is admittedly a little scary, but it is identical to Rust, and the defined-but-did-not-use warning usually flags any misuse.)
* `range` -- a range pattern.
* `colon::reference` -- a colon-ref pattern -- equality to some externally-defined entity.
* `..` -- usable within a tuple, discards an arbitrary number of elements
* `()` -- tuples and nesting of tuple patterns within tuple patterns.

Notably there is not support for structure pattern matching or arrays; these are desirable to match expressive ability of Rust but are not yet implemented.

## Exhaustiveness Checking

Exhaustiveness checking works by projecting types into a sparse N-dimensional array.

At the start of exhaustiveness checking we use the type being matched on to make an N-dimensional array with the extent in each dimension being determined by the type; e.g. if we're matching on a `u32` we make an interval that represents the range `[0, 2^32-1]`.

As we add patterns, we subtract out the space that the pattern covers from the overall space.

If we manage to subtract the entire space, then we know that the pattern is exhaustive.

If we don't manage to subtract the entire space, then we know that the pattern is not exhaustive.

### Monolithic N-dimensional Space

The exhaustiveness checker works by representing the entire space of values that can be matched on as a single N-dimensional space. This is conceptually simple but it flattens all values being matched on into a single N-dimensional sparse array, which means we discard all hierarchy present in the value being matched upon. It seems worth noting that this approach is monolithic and flattens the entire value space down to leaves, razing the tuple structure of the value being matched upon.

### Subtracting from N-dimensional Space

Note that there are two ways you could conceptually approach the idea of exhaustiveness -- additive and subtractive. In an additive approach you might unify spaces until you arrived at a space that was equal to the full space. Instead, our data structure subtracts, as it's easier to distribute subtraction over disjoint pieces. For example, consider a 2D rectangle where we subtract out the center of the rectangle, leaving just the left hand side and left hand side disjointly. We store that in our `NdRegion` as two disjoint contiguous `NdIntervals`, and then when we go to subtract out a new pattern (which can be represented as a `NdInterval` as a single pattern always occupies contiguous space) we can just subtract the new "user added interval" from both pieces of the space in "brute force" fashion without worrying about merging or which sub-space might be affected. This definitely seems to simplify things. The canonical paper on different approaches for this is Luc Maranget's "Warnings for pattern matching" -- admittedly I didn't do a lot of research for this implementation, though, the N-dimensional interval space was intuitive enough, so there may be better approaches we want to try over time.

### Semantically Load-Bearing

To be clear, exhaustiveness checking is semantically load-bearing; i.e. it is not just a lint, but a semantic check. Once exhaustive pattern match is proven it means we can turn the final arm of a match statement into an "else" condition, e.g. for a [priority select](https://google.github.io/xls/ir_semantics/#priority_sel) IR lowering in the `default` case.

It was discussed among XLS steering folks whether to make this an opt-in language feature for purposes of landing, but consensus was that it was useful enough to enable by default -- this is one of the top reported pain points for DSLX writing -- just with warnings for redundant patterns default-off for now so that real-world code bases have time to transition.

## Structure

The code is initially structured as follows:

* `DeduceMatch` -- the main entry point for pattern matching and exhaustiveness checking from the type system, we call into the `MatchExhaustivenessChecker` for each pattern encountered in the match statement.
* `match_exhaustiveness_checker.h` -- the main class for checking exhaustiveness.

  The deduce rule feeds this object pattern by pattern to check whether a pattern has led us to the point of exhaustion. This streaming pattern-at-a-time interface allows us also give a helpful warning when a pattern is fully redundant with previous patterns, or if we have similarly added patterns even though we've passed the point of exhaustion.

  DSLX types and values of particular types are translated into intervals and points at this level to subtract from the `NdRegion` that we maintain to determine exhaustiveness.
* `nd_region.h` -- an N-dimensional region type `NdRegion` for representing the space we're whittling down, with patterns, towards exhaustiveness. Note that an `NdRegion` is a collection of disjoint `NdInterval`s that we subtract from in "brute force" fashion as user-written patterns are introduced to the space, as described above.

  Each pattern can conceptually be translated into an `NdInterval` as it represents some contiguous space of values in the overall N-dimensional space.
* `interp_value_interval.h` -- an interval type `InterpValueInterval` for representing the range of DSLX values for a given type; provides some basic facilities for 1D interval arithmetic and queries like `Contains`, `Intersects`, `Covers`, etc.

## Notes on Wrinkles

**Enums** in DSLX are conceptually a set of names that live in some underlying bit space representation, and may be sparse within that space. That is, there's nothing wrong with making `enum E : u8 { A = u8:5, B = u8:10 }`. The language contract is that there can never be an enum value that takes on an out-of-defined-namespace value. As a result, we project the enum namespace into a dense unsigned bit space and make intervals over that dense space. i.e. for the enum `E` above we would require two values to cover the entire space. Effectively, we don't care what the underlying bit representation is for the purpose of pattern matching exhaustion.

Empty enums (i.e. enums with no defined values in its namespace) are similar to **zero-bit values** in that they have no real representable values. These bit-space could be defined to be trivially exhaustive, or impossible to match on -- we choose the latter for now because it's more convenient for implementation, we can call it a one-bit space and any value with this type trivially that one value (so we need to have one pattern covering it, but the pattern matches by definition).

**Tokens**: Note that there is also a question of tokens which are zero-bit-like, but I imagine we don't want to re-bind tokens through a pattern match so we can observe its linear dataflow in a given function.

**Arrays**: We don't currently support any interesting pattern syntax for arrays, and they can conceptually create large spaces of values in the flattening process, so this initial change for exhaustiveness makes them disallowed in matched expressions until support can be added more comprehensively. It's not for very serious reasons, however, they could be flattened in a similar fashion to tuples.

**Zero-Element Ranges** are possible to write in DSLX, and so in building up intervals we have to keep a maybe-interval concept until we have resolved fully that there are nonzero values in the interval. These wind up being N-dimensional intervals with zero volume so they never subtract out any space in the exhaustion process.
