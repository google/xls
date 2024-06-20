# Dictionary coder implementation in DSLX

Dictionary coder is a compression algorithm that compresses data by replacing
sequences of symbols in original data by pointers to sequences of symbols
stored within a "dictionary" data structure that is known by the decoder.
One example of such algorithms is LZ4.

This module implements encoder and decoder blocks that implement LZ4 algorithm.

## LZ77 algorithms

LZ4 belongs to a broader class of LZ77-like encoders. LZ77 encoders usually
have no preset dictionary and they build dictionary as they compress data. Each
input symbol is thus processed in the context of its own dictionary, and these
dictionaries are in general different for any two symbols. The decoders perform
a reverse process: they almost always start with an empty dictionary and build
it as they are decoding the data, thus obtaining perfect reconstruction not
only of the original raw data that was encoded, but also of the dictionary that
should match the encoder's dictionary at every step.

Within the class of LZ77 algorithms, the dictionary is the buffer which contains
up to `N` past raw symbols that have been already consumed by the encoder, or
emitted by the encoder. Usually it is implemented as a circular buffer, and is
called simply a *history buffer (HB)*.

LZ77 encoder emits two types of tokens (we call them _tokens_ to differentiate
them from _symbols_ which always refer to the original uncompressed data):
- *UNMATCHED_SYMBOL*, which contains the original symbol as-is and which the
  decoder simply copies to the output, resulting in no data size reduction.
- *MATCH*, which tells the decoder to copy a string of symbols from the history
  buffer to the output. As these symbols are not present in the token itself,
  this may result in a significant data size reduction. *MATCH* token contains
  of an *offset-count* pair
  - *Offset* tells the decoder how far back in the history buffer the string
    begins (usually 0 means "start with the last character", 1 is "the one
    before the last", etc.).
  - *Count* tells the decoder how many symbols to copy from the history buffer
    to the output. It's logical that count of `0` results in an empty string,
    thus specific byte-level encoding of the token can instead
    encode `count - 1` value.

## LZ4 algorithm

What puts LZ4 algorithm aside other LZ77-compatible algorithms (such as ALDC)
is how the encoder finds matches in its history buffer. Instead of performing
exhaustive search which is inefficient for algorithms running on the CPU, and
complicated for ASIC implementations (e.g. silicon implementations of ALDC
may use highly-custom CAM memory cores), it performs non-exhaustive search
using a hash table.

LZ4 algorithm uses two random-addressable memory blocks:
1. History buffer (also called *HB RAM*), which is the dictionary itself. HB
   stores raw data symbols, usually using a circular-buffer addressing scheme.
   - HB address bus is *MATCH_OFFSET_WIDTH* bits wide, thus it contains
     up to `(1 << MATCH_OFFSET_WIDTH)` symbols.
   - HB data word contains a single symbol, thus it is *SYMBOL_WIDTH* bits
     wide.
   - The size of the HB RAM is `(1 << MATCH_OFFSET_WIDTH) * SYMBOL_WIDTH` bits.
2. Hash table (also called *HT*). It works as follows:
   - HT address is *HASH_WIDTH* bits wide, this value can be configured to
     slightly tune a footprint-vs-compression ratio tradeoff. The
     address into HT is usually created by hashing a small string that contains
     a small number of symbols (*HASH_SYMBOLS*) extracted in-order from the
     incoming data.
   - HT data word contains a pointer (an address) into the HB RAM at which that
     string of symbols may be found.
   - The size of the HT is `(1 << HASH_WIDTH) * SYMBOL_WIDTH` bits.

In addition, because LZ4 wants to look ahead into the input data to hash it, it
needs a small FIFO with parallel output of all the bits to store *HASH_SYMBOLS*
input symbols and feed them to the hash function.

The flow of operation of LZ4 algorithm is depicted on the flowchart:
![LZ4 encoder flowchart](lz4_encoder_flowchart.png)

A textual expression of it which gives a bit more context:
* __(1)__ Consume one input symbol.
  * __(2)__ If it's an EOF, end processing.
  * Otherwise:
    * __(3)__ Push it to the HB (dropping the oldest symbol from the HB).
    * __(4)__ Push it to the FIFO (dropping the oldest symbol from the FIFO).
* __(5)__ Designate the current oldest symbol in the FIFO as *current_symbol*,
  calculate *current_ptr* - the location of this symbol in the HB.
* __(6)__ If we do not have an existing matching string that we're trying to
  grow, try to start a new matching string:
  * __(7)__ Compute hash of the data contained in the FIFO.
  * __(8)__ Load *candidate_ptr* pointer from the HT.
  * __(9)__ Store *current_ptr* to the HT.
  * __(10)__ Initiate a new matching string: calculate offset from a difference
    of *candidate_ptr* and *current_ptr*, set length to zero.
  * Here, *candidate_ptr* points to the string in history buffer, the
    beginning of which *potentially* matches the current input string which
    starts with *current_symbol* (the first symbols of the current input
    string are stored in the FIFO, and we've just hashed them and performed a
    HT lookup to find a similar sequence of symbols in the HB).
  * The match is not guaranteed since there can be a hash collision - the
    hash is the same, but actual symbols pointed by *candidate_ptr* differ,
    so we need to check each of them if it matches the one in the FIFO.
    In addition to that, we'd like to grow a matching string longer than
    what's contained in the FIFO.
  * As shown below, the same procedure is used to check matching of the
    beginning of the string, as well as of its continuation.
* Here, *current_ptr* points to the *current_symbol* - the next input symbol
  for which no output token has been emitted, while *candidate_ptr* points to
  an old symbol in the HB which we'd like to compare with *current_symbol*.
  * __(11)__ We load a symbol from the HB pointed by *candidate_ptr*, it
    becomes a *candidate_symbol*.
  * __(12)__ The *candidate_symbol* is compared with *current_symbol*.
  * __(13)__ If it's a match, continue this matching string:
    * __(14)__ Increment the match length by 1.
    * Go to step __(1)__ .
  * If it's not a match and we've been already growing a matching string:
    * __(15)__ Emit a *MATCH* token for the current matching string.
    * __(16)__ Terminate current matching string.
    * Go to step __(5)__ - this re-processes current input symbol one more time
      (this is done for two reasons: no output token has been emitted for
      that symbol yet, and it must be emitted at some point, and also
      this symbol may be able to start a new matching string on its own).
  * Otherwise:
    * __(17)__ Emit an *UNMATCHED_SYMBOL* token for the current symbol.
    * Go to step __(1)__.

## DSLX implementation

LZ4 encoder in DSLX is implemented as an FSM-like which changes states at most
once per "tick". It has a form of a _proc_ module.

Compared to the reference flowchart depicted above, this FSM is a bit more
complicated as it has to take into account some corner cases:
* Prefilling the FIFO with data before beginning the processing.
* Correctl handling of current symbol when it runs into an EOF.
* Draining the FIFO after EOF has been observed.
* Allowing "warm" restart of the algorithm after EOF to allow compression of
  *dependent* blocks - that is, without resetting the HT and the HB between the
  blocks (this allows tokens from the curent block to refer HB symbols of the
  previous block).
* System-level integration considerations to allow this block to be chained
  with other XLS-based data pre-/postprocessors:
  * *MATCHED_SYMBOL* tokens. Whenever the encoder adds a symbol to the matching
    string, it emits a *MATCHED_SYMBOL* token, which can be used by the
    postprocessing blocks to reconstruct symbols encoded by *MATCH* tokens
    without having to decode *MATCH*es (such decoding would require a full-size
    history buffer RAM). These tokens should not be stored into the final
    encoded block.
  * Support for a limited form of in-band control signaling via
    *marker tokens*:
    * *END* marker that signals end of block (EOF condition).
    * *ERROR_* family of markers that allow passing error codes between
      processing blocks. These tokens abort the encoding.
    * *RESET* marker that performs a *cold* restart of the encoder (clearing HT
      RAM) and other blocks in the chain, also clearing all error conditions.

One example of a post-processor module can be a *block writer* proc that
gathers tokens produced by the encoder and encodes them using a standardized
byte-oriented *LZ4 Block Format*. Implementation of alternative encoding
schemes may be of interest as well, as the header format used by the standard
LZ4 requires one to know the number of unmatched symbols between two matches
before those symbols are emitted, making bufferless stream processing
difficult.

### Data format

#### PlainData

The encoder consumes a stream of raw symbols, intermixed with control markers.
This is represented using a parametrized `PlainData` DSLX structure:
```rust
pub struct PlainData<DATA_WIDTH: u32> {
    is_marker: bool,
    data: uN[DATA_WIDTH],
    mark: Mark,
}
```

* *is_marker* tells whether this object is a symbol or a marker.
* *data* communicates a symbol whenever *is_marker* is not set.
* *mark* communicates a control mark whenever *is_marker* is set.

#### Token

The encoder produces a stream of tokens. There are four types of them:
* *MATCH* is an *offset-length* pair that represents a sequence of
  symbols that is the same as the specified sequence within HB.
* *UNMATCHED_SYMBOL* represents a symbol for which no match was found in the
  HB. It will be encoded as a raw symbol in the final piece of encoded data.
* *MATCHED_SYMBOL* is a symbol that is encoded within the next *MATCH* token.
  Its intended use is to allow easy postprocessing of a stream of tokens
  without a need for a full-fledged and heavy *MATCH* decoder.
* *MARKER* contains a control mark code.

Tokens are represented using following enum and structure in DSLX:
```rust
pub enum TokenKind : u2 {
    UNMATCHED_SYMBOL = 0,
    MATCHED_SYMBOL = 1,
    MATCH = 2,
    MARKER = 3,
}

pub struct Token<
    SYMBOL_WIDTH: u32, MATCH_OFFSET_WIDTH: u32, MATCH_LENGTH_WIDTH: u32
>{
    kind: TokenKind,
    symbol: uN[SYMBOL_WIDTH],
    match_offset: uN[MATCH_OFFSET_WIDTH],
    match_length: uN[MATCH_LENGTH_WIDTH],
    mark: Mark
}
```

* *kind* specifies one of the four token kinds.
* *symbol* contains symbol value for *UNMATCHED_SYMBOL* and *MATCHED_SYMBOL*
  tokens.
* *match_offset* and *match_length* are valid only for a *MATCH* token:
  * Length is the length of the string that has to be copied from the history
    buffer, minus one. That is, *match_offset=0* specifies a string of 1
    symbol, *match_offset=3* a string of 4 symbols, etc.
  * Offset points to the beginning of the string that has to be copied from
    HB.
    * An offset of 0 means that the first symbol to be copied is the last
      symbol written to the HB - the last symbol emitted when processing the
      previous token.
    * An offset of 1 means starting with the symbol preceding the one for
      offset 0, thus with the second newest symbol in the HB.
  * *MATCH* token may specify *length > offset* - in this case the decoder will
    have to copy not only old symbols, but also symbols produced when handling
    the current *MATCH* token, generating a repetitive sequence of characters,
    which resembles the behavior of a multisymbol *Run-Length Encoder*.

### Special considerations (encoder)

As previously mentioned, in response to the sequence of input symbols, the
encoder emits a sequence of compressed data tokens. There are certain rules
that the input data stream must obey for the encoder to generate a correct
sequence:
1. The blocks are delimited by *END* markers. The encoder buffers a few
   symbols internally. To ensure that all buffers are flushed and all symbols
   are emitted, an input data block *must* be followed by the *END* marker
   even if that's the last piece of data to be encoded.
2. A block consisting of no symbols and termianted with an *END* marker is
   treated as an empty (zero length) block.
3. *END* marker terminates the block, but does not reset the encoder's history.
   Therefore, the block that follows the *END* marker is treated as a
   *depdendent* block and generated *MATCH* tokens may refer to the data from
   the previous block. To introduce an *independent* block, encoder's state
   must be reset using a *RESET* marker. Therefore, a correct way to terminate
   one block and begin a new independent block will be to feed two markers to
   the encoder: an *END* followed by a *RESET*.
4. When the encoder is initialized, it enters the state that is equivalent
   to the one entered after receiving a *RESET* marker. Therefore, it is not
   necessary to issue a *RESET* marker before the first block - it is
   implicitly treated as an independent block.

The output token stream is formatted as follows:
1. Tokens are emitted in order that matches the order of input symbols and
   markers.
2. Each incoming *END* marker produces one *END* marker token.
3. Each incoming *RESET* marker produces one *RESET* marker token.
4. Each incoming symbol produces one *symbolic* token:
   either an *UNMATCHED_SYMBOL* or a *MATCHED_SYMBOL*. The symbol contained
   within the token matches the input symbol.
5. Each sequence of one or more *MATCHED_SYMBOL* tokens is a matching string.
   Such a sequence is always followed by a single *MATCH* token, in which the
   *match offset* and *match length* fields encode the same payload as the
   preceding *MATCHED_SYMBOL* tokens.
6. Emission of *x_SYMBOL* and *MATCH* tokens is subject to internal buffering,
   and can be delayed with respect to the input data consumed by the encoder.
7. *END* marker flushes the internal buffers and makes the encoder to emit all
   the *x_SYMBOL* and *MATCH* tokens that have not been emitted yet.
8. *RESET* marker immediately clears the internal buffers without flushing,
   which may result in loss of data if it is not immediately preceded by the
   *END* marker.
9. The encoder does not emit single-symbol matches - they are emitted directly
   as an *UNMATCHED_SYMBOL*.


#### Encoding examples

**Example 1**

Sequence without repetitions, thus no matches are emitted.

Input:
```
 0: 'A'
 1: 'B'
 2: 'C'
 3: 'D'
 4: 'E'
 5: 'F'
 6: Mark::END
```

Output:
```
 0: UNMATCHED_SYMBOL  'A'
 1: UNMATCHED_SYMBOL  'B'
 2: UNMATCHED_SYMBOL  'C'
 3: UNMATCHED_SYMBOL  'D'
 4: UNMATCHED_SYMBOL  'E'
 5: UNMATCHED_SYMBOL  'F'
 6: MARKER            Mark::END
```

**Example 2**

Single symbol repeated, single match (RLE-like behavior).

Input:
```
 0: 'A'
 1: 'A'
 2: 'A'
 3: 'A'
 4: 'A'
 5: 'A'
 6: Mark::END
```

Output:
```
 0: UNMATCHED_SYMBOL  'A'
 1: MATCHED_SYMBOL    'A'
 2: MATCHED_SYMBOL    'A'
 3: MATCHED_SYMBOL    'A'
 4: MATCHED_SYMBOL    'A'
 5: MATCHED_SYMBOL    'A'
 6: MATCH             offset=0 length=4  (length of matching string minus one)
 6: MARKER            Mark::END
```

**Example 3**

A string repeated, single match.

Input:
```
 0: 'A'
 1: 'B'
 2: 'C'
 3: 'A'
 4: 'B'
 5: 'C'
 6: 'A'
 7: 'B'
 8: 'C'
 9: Mark::END
```

Output:
```
 0: UNMATCHED_SYMBOL  'A'
 1: UNMATCHED_SYMBOL  'B'
 2: UNMATCHED_SYMBOL  'C'
 3: MATCHED_SYMBOL    'A'
 4: MATCHED_SYMBOL    'B'
 5: MATCHED_SYMBOL    'C'
 6: MATCHED_SYMBOL    'A'
 7: MATCHED_SYMBOL    'B'
 8: MATCHED_SYMBOL    'C'
 9: MATCH             offset=2 length=5
10: MARKER            Mark::END
```

**Example 4**

Several matches of different lengths.

Input:
```
 0: 'A'
 1: 'E'
 2: 'T'
 3: 'H'
 4: 'E'
 5: 'R'
 6: 'I'
 7: 'S'
 8: 'A'
 9: 'E'
10: 'T'
11: 'E'
12: 'R'
13: 'N'
14: 'I'
15: Mark::END
```

Output:
```
 0: UNMATCHED_SYMBOL  'A'
 1: UNMATCHED_SYMBOL  'E'
 2: UNMATCHED_SYMBOL  'T'
 3: UNMATCHED_SYMBOL  'H'
 4: UNMATCHED_SYMBOL  'E'
 5: UNMATCHED_SYMBOL  'R'
 6: UNMATCHED_SYMBOL  'I'
 7: UNMATCHED_SYMBOL  'S'
 8: MATCHED_SYMBOL    'A'
 9: MATCHED_SYMBOL    'E'
10: MATCHED_SYMBOL    'T'
11: MATCH             offset=7 length=2
12: MATCHED_SYMBOL    'E'
13: MATCHED_SYMBOL    'R'
14: MATCH             offset=6 length=1
15: UNMATCHED_SYMBOL  'N'
16: UNMATCHED_SYMBOL  'I'
17: MARKER            Mark::END
```

#### Rewriting matches

It is possible to use *MATCHED_SYMBOL* tokens emitted by the decoder to
rewrite (modify, or replace by *UNMATCHED_SYMBOL*) matches produced by
the encoder. If *MATCHED_SYMBOL* tokens are also preserved, this allows one to
implement a sequence of token post-processors that each perform their own
kind of operation.

Considering the fact that the tokens can only be accessed sequentially (no
random access unless they are buffered somewhere), the set of operations that
can be performed on the matches is limited. Since the attributes of the
matching string (offset and length) are communicated within the *MATCH* token
that is seen only after observing all the *MATCHED_SYMBOL* tokens, all the
permutations of the matching strings use the *end* of the matching string as
a reference and require a FIFO-like buffering of matched symbols.

**Operation 1 - unmatching the match**

The decoded plaintext will not change if the post-processor:
1. Identifies a complete matching string, that is a string of *MATCHED_SYMBOL*
   tokens preceded by a token of any other type, and followed by a *MATCH*
   token.
2. Changes the type of the *MATCHED_SYMBOL* tokens to *UNMATCHED_SYMBOL*.
3. Removes the terminating *MATCH* token.

Since this operation will usually be performed only on matches that fulfill
certain condition (e.g. "*unmatch the matches shorter than 4 symbols*"), and
that condition can only be evaluated after the *MATCH* token has been observed,
the data for the complete match to be rewritten has to be buffered. However
if, while processing the match, it becomes known that the match doesn't have to
be rewritten, then there is no need for further buffering of its contents.

For example, if the goal is to remove all matches shorter than 4 symbols,
while keeping matches of 4 or more symbols intact (a requirement prescribed by
the LZ4 block format), the post-processor will have to buffer up to 4
*MATCHED_SYMBOL* tokens in a FIFO. If there are more than 4 *MATCHED_SYMBOL*
tokens, the FIFO is flushed and the match is passed through as-is. Otherwise,
the symbols from the FIFO are emitted as *UNMATCHED_SYMBOL*-s and the final
*MATCH* token is omitted.

**Operation 2 - splitting the match**

The match of length *N* can be split into two smaller matches of lengths *N-M*
and *M* (*0 < M < N*) if the post-processor:
1. Identifies a complete matching string, that is a string of *MATCHED_SYMBOL*
   tokens preceded by a token of any other type, and followed by a *MATCH*
   token, where the *MATCH* token specifies *offset=K* and *length=N-1*.
2. Emits first *N-M* symbol tokens as-is.
3. Emits a *MATCH* token with:
   - *offset = K*
   - *length = N-M-1*
4. Emits next *M* symbol tokens as-is.
5. Emits a *MATCH* token with:
   - *offset = K*
   - *length = M-1*

Like operation 1, to know the parameters of the match (e.g. the offset that is
emitted within the *MATCH* token at step *3* above), the post-processor has
to buffer symbols of the matching string. However, unlike operation 1,
since the symbols for the first sub-match are emitted completely intact up to
the newly-inserted *MATCH* token, the post-processor needs to buffer only the
symbols from the second sub-match of length *M*.

A practical post-processor is thus possible if there is a known upper limit on
*M* (the length of the second sub-match), since that also sets an upper limit
on the size of the internal buffer.

For example, LZ4 block format spec stipulates that:
1. The last 5 bytes of input are always literals.
2. The last match must start at least 12 bytes before the end of block.

Both of these conditions can be fulfilled by a post-processor that ensures that
the last 12 symbols of a block are emitted as *UNMATCHED_SYMBOL*-s.

It can be realized by:
1. Delaying the token stream by passing it through a FIFO of 15 elements (the
   worst case: 12 matched or unmatched symbol tokens and at most 3 match
   tokens) until the *END* marker is observed.
2. If some match crosses the boundary of the FIFO, splitting the match at the
   FIFO boundary, emitting the first half as a match and unmatching the second
   half (up to 14 symbols long, fully stored within the FIFO).
3. Unmatching all the matches remaining in the FIFO, if any (symbols of those
   matching strings are fully contained in the FIFO).
4. Flushing the FIFO, which at this point contains only *UNMATCHED_SYMBOL*
   tokens.


### FSM

A state diagram of the FSM is displayed below:
![LZ4 encoder states](lz4_encoder_states.png)

* **RESET** - the initial state of the decoder. It initializes other state
  variables, resulting in a *cold* block start and jumps to
  **HASH_TABLE_CLEAR**.
  * To facilitate testing in presence of
    [issue #1042](https://github.com/google/xls/issues/1042), there is a proc
    parameter that allows bypassing **HASH_TABLE_CLEAR** and jumping
    directly into the **RESTART** state. It should not be used in real
    implementations as it will make encoding of an independent block to depend
    on the contents of the preceding data blocks, thus making encoding
    non-deterministic and potentially allowing for data leaks across blocks.
* **HASH_TABLE_CLEAR** - encoder iterates here, clearing one word of HT RAM
  per tick.
* **RESTART** - the encoder clears a small set of state variables, resulting in
  a *warm* start, allowing to preserve necessary state (HB, HT) between
  depending data blocks.
* **FIFO_PREFILL** - the encoder fills the input FIFO with symbols.
  * If an *END* token is observed, the FSM will transition into either
    *EMIT_END* or to *FIFO_DRAIN* - this depends on whether the FIFO already
    has multiple symbols in it and thus whether extra ticks are needed to drain
    all of them.
  * Handles input steps of a flowchart: __1, 4__.
* **START_MATCH_0** - roughly corresponds to the *Starting a new match*,
  *Read potential matching symbol from HB*, *Check for match*,
  *Growing the match* parts of the flowchart.
  * Handles input steps: __1, 4__.
  * Handles match steps: __5, 7, 8, 10, 11, 12, 13__.
  * Step __14__ belongs here, but is skipped as an optimization, since step
    __10__ can initialize match variables properly from the start.
* **START_MATCH_1** - "upper" counterpart of **START_MATCH_0**, necessary
  because two accesses to the same (single-port) RAM can not be done in the
  same tick.
  * Handles step __9__, writing to the HT RAM.
  * Handles step __3__, writing to the HB RAM.
  * May emit *UNMATCHED_SYMBOL* token, step __17__.
* **CONTINUE_MATCH_0** - mostly the same as **START_MATCH_0** except that it
  does not start a new match.
  * Handles input steps: __1, 4__.
  * Handles match steps __5, 11, 12, 16__.
* **CONTINUE_MATCH_1** - "upper" counterpart of **CONTINUE_MATCH_0**.
  * Handles step __3__, writing to the HB RAM.
  * May emit *MATCH* token, step __16__.
* **FIFO_DRAIN** - the encoder loops here, draining symbols from the FIFO
  and emitting **UNMATCHED_SYMBOL* tokens for them.
  * Symbols are also written to the HB to make them visible in case a new
    block is started after a *warm* restart.
* **EMIT_END** - the encoder emits a single *END* token and transitions into
  a **RESTART** state.
* **ERROR** - state that is entered whenever the error condition is
  encountered. This can happen if e.g. *ERROR* marker is received from another
  block that precedes the encoder, or if an unknown (unsupported) marker is
  received.
  * The encoder receives and discards incoming symbols, with an exception of
    a *RESET* command marker that is replicated on the output (so  that other
    processing blocks can be reset) and makes FSM transition to the **RESET**
    state.
