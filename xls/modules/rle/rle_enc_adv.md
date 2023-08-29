# Multisymbol Run-length Encoder

The encoder uses Run Length Encoding (RLE) to compress the input stream of
repeating symbols to the output stream that contains the symbols and
the number of its consecutive occurrences in the input stream.

Overall, we can break down the data processing into four stages: reduction, alignment, compression, and output generation. The division of responsibility allowed the specialized blocks to efficiently process data and made it possible test each functionality separately.

The first block is responsible for taking the input and reducing it to pairs of symbols and the number of its occurrences, as in the [basic RLE implementation](https://github.com/google/xls/pull/974). The second element of the encoder shifts the previously emitted pairs and adjusts them for further processing. Both of these elements have an empty state. The next block takes the prepared data and combines it with the information about previously processed symbols. The last element is responsible for adjusting the width of the output data to the receiver interface.


Input width is defined using the `INPUT_WIDTH` parameter and output width is defined with the `OUTPUT_WIDTH` parameter.
Both the input and the output channels use additional `last` flag
that indicates whether the packet ends the transmission. After sending
the last packet the encoder dumps all the data to the output stream.

## Encoder processing pipeline detailed breakdown.

### Initial conditions
- input width is 4 symbols wide,
- output width is 2 pairs wide,
- symbol counter is 2 bits wide.

### Process
1. Reduce step - this process takes incoming symbols and symbol_valid
and reduces them into symbol count pairs. This step is stateless.

Example:

|||
|-----|-------|
|input|output |
|[(A, True), (A, True), (A, True), (A, True)]|[.., .., (A, 3), (A, 1)]|
|input|output |
|[(A, True), (A, True), (A, False), (A, True)]|[.., .., .., (A, 3)]|
|input|output |
|[(A, True), (B, True), (C, True), (D, True)]|[(A, 1), (B, 1), (C, 1), (D, 1)]|

2. Realign step - this process moves pairs emitted from the reduce step
so that they are aligned to the left, it also calculates propagation
distance for the first pair.

Example:

||||
|-----|-------|--------------------|
|input|output |propagation distance|
|[.., (A, 2), .., (B, 2)]|[(A, 2), (B, 2), .., ..]| 0|
|input|output |propagation distance|
|[.., .., (A, 3), (A, 1)]|[(A, 3), (A, 1), .., ..]| 1|

3. Core step - this step is stateful. It takes align pairs from
the realign step, and combines them with its state to create multiple
symbol/count pairs output. State is represented by following tuple
`<symbol, count, last>`. It contains symbol and count from last pair
received from the realign step, or current sum of repeating symbol spanning
multiple input widths. 

Example:

|||||
|------|-----|-------|----------|
|state |input|output |next state|
|(A, 2)| [(A, 2), (B, 2), .., ..]|[(A, 3), (A, 1), .., ..]| (B, 2)|
|state |input|output |next state|
|(A, 1)| [(A, 1), (B, 2), .., ..]|[(A, 2), .., .., ..]| (B, 2)|
|state |input|output |next state|
|(A, 1)| [(A, 1), .., .., ..]|[.., .., .., ..]| (A, 2)|

4. Adjust Width step - this step takes output from the core step.
If output can handle more or equal number of pairs as
input number of symbols. This step does nothing.
If the output is narrower than the input,
this step will serialize symbol counter pairs.

Example:

|||||
|-----|-----|-------|-----------| 
|state|input|output | next state|
|[]|[(A, 3), (A, 2), .., ..]|[(A, 3), (A, 2)]|[]|
|state|input|output | next state|
|[]|[(A, 1), (B, 1), (C, 1), (D, 1)]|[(A, 1), (B, 1)]|[(C, 1), (D, 1)]|
|state|input|output | next state|
|[(C, 1), (D, 1)]|ignored|[(C, 1), (D, 1)]|[]|