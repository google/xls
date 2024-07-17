# ZSTD decoder

The ZSTD decoder decompresses the correctly formed ZSTD frames and blocks.
It implements the [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878.html) decompression algorithm.
Overview of the decoder architecture is presented on the diagram below.
The decoder comprises:
* frame decoder,
* block dispatcher,
* 3 types of processing units: RAW, RLE, and compressed,
* command aggregator,
* history buffer,
* repacketizer.

Incoming ZSTD frames are processed in the following order:
1. magic number is detected,
2. frame header is parsed,
3. ZSTD data blocks are being redirected to correct processing unit based on the block header,
4. processing unit results are aggregated in correct order into a stream
and routed to the history buffer,
5. data block outputs are assembled based on the history buffer contents and update history,
6. decoded data is processed by repacketizer in order to prepare the final output of the decoder,
7. (optional) calculated checksum is compared against frame checksum.

![](img/ZSTD_decoder.png)

## ZSTD decoder architecture

### Top level Proc
This state machine is responsible for receiving encoded ZSTD frames, buffering the input and passing it to decoder's internal components based on the state of the proc.
The states defined for the processing of ZSTD frame are as follows:

```mermaid
stateDiagram
    direction LR

    [*] --> DECODE_MAGIC_NUMBER

    DECODE_MAGIC_NUMBER --> DECODE_MAGIC_NUMBER: Not enough data
    DECODE_MAGIC_NUMBER --> DECODE_FRAME_HEADER: Got magic number
    DECODE_MAGIC_NUMBER --> ERROR: Corrupted

    DECODE_FRAME_HEADER --> DECODE_FRAME_HEADER: Not enough data
    DECODE_FRAME_HEADER --> DECODE_BLOCK_HEADER: Header decoded
    DECODE_FRAME_HEADER --> ERROR: Unsupported window size
    DECODE_FRAME_HEADER --> ERROR: Corrupted

    DECODE_BLOCK_HEADER --> DECODE_BLOCK_HEADER: Not enough data
    DECODE_BLOCK_HEADER --> FEED_BLOCK_DECODER: Feed raw data
    DECODE_BLOCK_HEADER --> FEED_BLOCK_DECODER: Feed RLE data
    DECODE_BLOCK_HEADER --> FEED_BLOCK_DECODER: Feed compressed data
    DECODE_BLOCK_HEADER --> ERROR: Corrupted

    state if_decode_checksum <<choice>>
    state if_block_done <<choice>>

    FEED_BLOCK_DECODER --> if_decode_checksum: Is the checksum available?
    if_decode_checksum --> DECODE_CHECKSUM: True
    if_decode_checksum --> DECODE_MAGIC_NUMBER: False
    FEED_BLOCK_DECODER --> if_block_done: Is the block decoding done?
    if_block_done --> DECODE_BLOCK_HEADER: Decode next block
    if_block_done --> FEED_BLOCK_DECODER: Continue feeding

    DECODE_CHECKSUM --> DECODE_MAGIC_NUMBER: Frame decoded

    ERROR --> [*]
```

After going through initial stages of decoding magic number and frame header, decoder starts the block division process.
It decodes block headers to calculate how many bytes must be sent to the block dispatcher and when the current frame's last data block is being processed.
Knowing that, it starts feeding the block decoder with data required for decoding current block.
After transmitting all data required for current block, it loops around to the block header decoding state and when next block header is not found it decodes checksum when it was requested in frame header or finishes ZSTD frame decoding and loops around to magic number decoding.

### ZSTD frame header decoder
This part of the design starts with detecting the ZSTD magic number.
Then it parses and decodes the frame header's content and checks the header's correctness.
If the frame header has the checksum option enabled, this will enable `DECODE_CHECKSUM` stage at the end of the frame decoding where the frame's checksum will be computed and compared with the checksum embedded at the end of the frame stream.

### Block dispatcher (demux)
At this stage, block headers are parsed and removed from the block data stream.
Based on parse values, it directs the block data stream to either RAW, RLE or compressed block sections.
For this task it uses an 8 byte native interface: a 64-bit data bus and a 64-bit length field that contains the number of correct bits on the data bus.
It also attaches a unique block ID value to each processed data block.
The IDs are sequential starting from 0 and are reset only after receiving and processing the current frame's last data block.

### RAW
This proc passes the received data directly to its output channel.
It preserves the block ID and attaches a tag, stating that the data contains literals and should be placed in the history buffer unchanged, to each data output.

### RLE decoder
This proc receives a tuple (s, N), where s is an 8 bit symbol and N is an accompanying `symbol_count`.
The proc produces `N*s` repeats of the given symbol.
This step preserves the block ID and attaches the literals tag to all its outputs.

### Compressed block decoder
This part of the design is responsible for decoding the compressed data blocks.
It ingests the bytes stream, internally translates and interprets incoming data.
Only this part of the design creates data chunks tagged both with `literals` and/or `copy`.
This step preserves the block ID.
More in depth description can be found in [Compressed block decoder architecture](#compressed-block-decoder-architecture) paragraph of this doc.

### Commands aggregator (mux)
This stage takes the output from either RAW, RLE or Command constructor and sends it to the History buffer and command execution stage.
This stage orders streams based on the ID value assigned by the block dispatcher.
It is expected that single base decoders (RAW, RLE, compressed block decoder) will be continuously transmitting a single ID to the point of sending the `last` signal which marks the last packet of currently decoded block.
That ID can change only when mux receives the `last` signal or `last` and `last_block` signals.

It works as a priority mux that waits for a stream with the expected ID.
It continues to read that stream until the `last` signal is set, then it switches to the next stream ID.

The command aggregator starts by waiting for `ID = 0`, after receiving the `last` signal it expects `ID = 1` and so on.
Only when both `last` and `last_block` are set the command aggregator will wait for `ID = 0`.

### History buffer and command execution
This stage receives data which is tagged either `literals` or `copy`.
This stage will show the following behavior, depending on the tag:
* `literals`
    * Packet contents placed as newest in the history buffer,
    * Packet contents copied to the decoder's output,
* `copy`
    * Wait for all previous writes to be completed,
    * Copy `copy_length` literals starting `offset _length` from the newest in history buffer to the decoder's output,
    * Copy `copy_length` literals starting `offset _length` from the newest in history buffer to the history buffer as the newest.

### Compressed block decoder architecture
This part of the design is responsible for processing the compressed blocks up to the `literals`/`copy` command sequence.
This sequence is then processed by the history buffer to generate expected data output.
Overview of the architecture is provided on the diagram below.
The architecture is split into 2 paths: literals path and sequence path.
Architecture is split into 3 paths: literals path, FSE encoded Huffman trees and sequence path.
Literals path uses Hufman trees to decode some types of compressed blocks: Compressed and Treeless blocks.

![](img/ZSTD_compressed_block_decoder.png)

#### Compressed block dispatcher
This proc parses literals section headers to calculate block compression format, Huffmman tree size (if applicable based on compression format), compressed and regenerated sizes for literals.
If compressed block format is `Compressed_Literals_Block`, dispatcher reads Huffman tree header byte from Huffman bitstream, and directs expected number of bytes to the Huffman tree decoder.
Following this step, the proc sends an appropriate number of bytes to the literals decoder dispatcher.

After sending literals to literals decompression, it redirects the remaining bytes to the sequence parsing stages.

#### Command Constructor
This stage takes literals length, offset length and copy length.
When `literals length` is greater than 0, it will send a request to the literals buffer to obtain `literals length` literals and then send them to the history buffer.
Then based on the offset and copy length it either creates a match command using the provided offset and match lengths, or uses repeated offset and updates the repeated offset memory.
Formed commands are sent to the Commands aggregator (mux).

### Literals path architecture

![](img/ZSTD_compressed_block_literals_decoder.png)

#### Literals decoder dispatcher
This proc parses and consumes the literals section header.
Based on the received values it passes the remaining bytes to RAW/RLE/Huffman tree/Huffman code decoders.
It also controls the 4 stream operation mode [4-stream mode in RFC](https://www.rfc-editor.org/rfc/rfc8878.html#name-jump_table).

All packets sent to the Huffman bitstream buffer will be tagged either `in_progress` or `finished`.
If the compressed literals use the 4 streams encoding, the dispatcher will send the `finished` tag 4 times, each time a fully compressed stream is sent to the bitstream buffer.

#### RAW Literals
This stage simply passes the incoming bytes as literals to the literals buffer.

#### RLE Literals
This stage works similarly to the [RLE stage](#rle-decoder) for RLE data blocks.

#### Huffman bitstream buffer
This stage takes data from the literals decoder dispatcher and stores it in the buffer memory.
Once the data with the `finished` tag set is received, this stage sends a tuple containing (start, end) positions for the current bitstream to the Huffman codes decoder.
This stage receives a response from the Huffman codes decoder when decoding is done and all bits got processed.
Upon receiving this message, the buffer will reclaim free space.

#### Huffman codes decoder
This stage receives bitstream pointers from the Huffman bitstream buffer and Huffman tree configuration from the Huffman tree builder.
It accesses the bitstream buffers memory to retrieve bitstream data in reversed byte order and runs it through an array of comparators to decode Huffman code to correct literals values.

#### Literals buffer
This stage receives data either from RAW, RLE or Huffman decoder and stores it.
Upon receiving the literals copy command from the Command Constructor for `N` number of bytes, it provides a reply with `N` literals.

### FSE Huffman decoder architecture

![](img/ZSTD_compressed_block_Huffman_decoder.png)

#### Huffman tree decoder dispatcher
This stage parses and consumes the Huffman tree description header.
Based on the value of the Huffman descriptor header, it passes the tree description to the FSE decoder or to direct weight extraction.

#### FSE weight decoder
This stage performs multiple functions.
1. It decodes and builds the FSE distribution table.
2. It stores all remaining bitstream data.
3. After receiving the last byte, it translates the bitstream to Huffman weights using 2 interleaved FSE streams.

#### Direct weight decoder
This stage takes the incoming bytes and translates them to the stream of Huffman tree weights.
The first byte of the transfer defines the number of symbols to be decoded.

#### Weight aggregator
This stage receives tree weights either from the FSE decoder or the direct decoder and transfers them to Huffman tree builder.
This stage also resolves the number of bits of the final weight and the max number of bits required in the tree representation.
This stage will emit the weights and number of symbols of the same weight before the current symbol for all possible byte values.

#### Huffman tree builder
This stage takes `max_number_of_bits` (maximal length of Huffman code) as the first value, then the number of symbols with lower weight for each possible weight (11 bytes), followed by a tuple (number of preceding symbols with the same weight, symbol's_weight).
It's expected to receive weights for all possible byte values in the correct order.
Based on this information, this stage will configure the Huffman codes decoder.

### Sequence path architecture

![](img/ZSTD_compressed_block_sequence_decoder.png)

#### Sequence Header parser and dispatcher
This stage parses and consumes `Sequences_Section_Header`.
Based on the parsed data, it redirects FSE description to the FSE table decoder and triggers Literals FSE, Offset FSE or Match FSE decoder to reconfigure its values based on the FSE table decoder.
After parsing the FSE tables, this stage buffers bitstream and starts sending bytes, starting from the last one received as per ZSTD format.
Bytes are sent to all decoders at the same time.
This stage monitors and triggers sequence decoding phases starting from initialization, followed by decode and state advance.
FSE decoders send each other the number of bits they read.

#### Literals FSE decoder
This stage reconfigures its FSE table when triggered from [sequence header parse and dispatcher](#sequence-header-parser-and-dispatcher).
It initializes its state as the first FSE decoder.
In the decode phase, this stage is the last one to decode extra raw bits from the bitstream, and the number of ingested bits is transmitted to all other decoders.
This stage is the first stage to get a new FSE state from the bitstream, and it transmits the number of bits it used.

#### Offset FSE decoder
This stage reconfigures its FSE table when triggered from [sequence header parse and dispatcher](#sequence-header-parser-and-dispatcher).
It initializes its state as the second FSE decoder.
In the decode phase, this stage is the first one to decode extra raw bits from bitstream, and the number of ingested bits is transmitted to all other decoders.
This stage is the last decoder to update its FSE state after the decode phase, and it transmits the number of used bits to other decoders.

#### Match FSE decoder
This stage reconfigures its FSE table when triggered from [sequence header parse and dispatcher](#sequence-header-parser-and-dispatcher).
It initializes its state as the last FSE decoder.
In the decode phase, this stage is the second one to decode extra raw bits from the bitstream, and the number of ingested bits is transmitted to all other decoders.
This stage is the second stage to update its state after the decode phase, and the number of used bits is sent to all other decoders.

### Repacketizer
This proc is used at the end of the processing flow in the ZSTD decoder.
It gathers the output of `SequenceExecutor` proc and processes it to form final output packets of the ZSTD decoder.
Input packets coming from the `SequenceExecutor` consist of:

* data - bit vector of constant length
* length - field describing how many bits in bit vector are valid
* last - flag which marks the last packet in currently decoded ZSTD frame.

It is not guaranteed that all bits in data bit vectors in packets received from `SequenceExecutor` are valid as those can include padding bits which were added in previous decoding steps and now have to be removed.
Repacketizer buffers input packets, removes the padding bits and forms new packets with all bits of the bit vector valid, meaning that all bits are decoded data.
Newly formed packets are then sent out to the output of the whole ZSTD decoder.

## Testing methodology

Testing of the `ZSTD decoder` is carried out on two levels:

* Decoder components
* Integrated decoder

Each component of the decoder is tested individually in DSLX tests.
Testing on the DSLX level allows the creation of small test cases that test for both positive and negative outcomes of a given part of the design.
When need be, those test cases can be also modified by the user to better understand how the component operates.

Tests of the integrated ZSTD decoder are written in C++.
The objective of those is to verify the functionality of the decoder as a whole.
Testing setup for the ZSTD decoder is based on comparing the simulated decoding results against the decoding of the reference library.
Currently, due to the restrictions from the ZSTD frame generator, it is possible to test only the positive cases (decoding valid ZSTD frames).

### Failure points

#### User-facing decoder errors

The design will fail the tests under the following conditions:

* Straightforward failures:
  * Top Level State Machine transitions to `ERROR` state
  * Simulation encounters `assert!()` or `fail!()` statements
  * The decoding result from the simulation has a different size than the results from the reference library
  * The decoding result from the simulation has different contents than the results from the reference library
* Caveats:
  * Timeout occurred while waiting for a valid `Magic Number` to start the decoding process
  * Other timeouts occurring while waiting on channel operations (To be fixed)

Currently, all mentioned conditions lead to an eventual test failure.
Most of those cases are handled properly while some are yet to be reworked to finish faster or to provide more information about the error.
For example, in case of transitioning to the `ERROR` state, the test will timeout on channel operations waiting to read from the decoder output.
In case of waiting for a valid `Magic Number`, the decoder will transition to an `ERROR` state without registering the correct `Magic Number` on the input channel which will lead to a similar timeout.

Those cases should be handled in a way that allows for early failure of the test.
It can be done through a Proc parameter enabled for tests that change the behavior of the logic, e.g. launching `assert!()` when the decoder enters the `ERROR` state.
Another idea is to use a special output channel for signaling internal states and errors to monitor the decoder for the errors encountered during decoding.
For example, in an invalid `Magic Number`, the test case should expect a certain type of error reported on this channel at the very beginning of the simulation.

#### Failures in ZSTD Decoder components

It is important to note that some of the errors (e.g. errors in magic number or frame header decoding) are easy to trigger in the integration test cases by manual modification of the generated ZSTD frames.
However, the majority of the errors require modification of the deeper parts of the raw ZSTD frame which is significantly harder.
Because of that, it is better to rely on DSLX tests for the individual components where inputs for the test cases are smaller, easier to understand and modify when needed.

The components of the ZSTD decoder can fail on `assert!()` and `fail!()` statements or propagate specific error states to the Top Level Proc and cause it to transition to the `ERROR` state.
The following enumeration will describe how to trigger each possible ZSTD Decoder error.

The `ERROR` state can be encountered under the following conditions when running Top Level Proc C++ tests but also in DSLX tests for the specific components:
* Corrupted data on the `Magic Number` decoding stage
  * Provide data for the decoding with the first 4 bytes not being the valid `Magic Number` (0xFD2FB528)
* Corrupted data during frame header decoding
  * Set the `Reserved bit` in the frame header descriptor
* Unsupported Window Size during frame header decoding
  * Set `Window Size` in frame header to value greater than `max window size` calculated from current `WINDOW_LOG_MAX` (by default in Top Level Proc tests `Window Size` must be greater than `0x78000000` to trigger the error)
* Corrupted data during Block Header decoding
  * Set the `Block Type` of any block in the ZSTD frame to `RESERVED`

The `assert!()` or `fail!()` will occur in:
* Buffer
  * Add data to the buffer with `buffer_append()` when it is already full or unable to fit the whole length of the data
  * Fetch data from the buffer with `buffer_pop()` when it is empty or have not enough data
* DecoderDemux
  * Receive more than one `raw` or `compressed` block in a single `BlockDataPacket`
* RawBlockDecoder
  * Receive `BlockDataPacket` with `ID` different than the previous packet which did not have the `last` flag set
* DecoderMux
  * At the beginning of the simulation or after receiving `ExtendedBlockDataPacket` with `last` and `last_block` (decoding new ZSTD frame) set receive on channels `raw_r`, `rle_r` and `cmp_r` `ExtendedBlockDataPackets` without any of those having `ID==0`
  * Receive `ExtendedBlockDataPacket` with a smaller `ID` than any of the previously processed packets during the current ZSTD frame decoding
* SequenceExecutor
  * Receive `SequenceExecutorPacket` with `msg_type==SEQUENCE` and `content` field with value: `0`

There are also several `impossible cases` covered by `assert!()` and `fail!()`:

* Frame header decoder
  * `Window Descriptor` does not exist after checking that it is available in the frame header
  * `Frame Content Size` does not exist after checking that it is available in the frame header
  * `Dictionary ID Flag` has an illegal value
  * `Frame Content Size Flag` has an illegal value
* DecoderDemux
  * Data packet has a different `Block Type` than `RAW`, `RLE` or `COMPRESSED`
* SequenceExecutor
  * Proc transitions to `SEQUENCE_READ` state after receiving `SequenceExecutorPacket` with `msg_type` different than `SEQUENCE` or the message was invalid
* Top Level Proc
  * Block header type is different than `RAW`, `RLE`, `COMPRESSED`
  * There is not enough data to feed the `BlockDecoder`, even though the previous check indicated a valid amount of data in the buffer

### Testing against [libzstd](https://github.com/facebook/zstd)

Design is verified by comparing decoding results to the reference library `libzstd`.
ZSTD frames used for testing are generated with [decodecorpus](https://github.com/facebook/zstd/blob/dev/tests/decodecorpus.c) utility.
The generated frame is then decoded with `libzstd`.

#### Positive test cases

If the results of decoding with `libzstd` are valid, the test runs the same encoded frame through the simulation of DSLX design.
The output of the simulation is gathered and compared with the results of `libzstd` in terms of its size and contents.

Encoded ZSTD frame is generated with the function `GenerateFrame(int seed, BlockType btype)` from [data_generator](https://github.com/antmicro/xls/blob/52186-zstd-top/xls/modules/zstd/data_generator.cc) library.
This function takes as arguments the seed for the generator and enum which codes the type of blocks that should be generated in a given frame.
The available block types are:

* RAW
* RLE
* COMPRESSED
* RANDOM

The function returns a vector of bytes representing a valid encoded ZSTD frame.
Such generated frame can be passed to `ParseAndCompareWithZstd(std::vector<uint8_t> frame)` which is responsible for decoding the frame, running simulation and comparing the results.

Tests are available in the `zstd_dec_test.cc` file and can be launched with the following Bazel command:

```
bazel test //xls/modules/zstd:zstd_dec_cc_test
```

#### Negative test cases

Currently, `decodecorpus` does not support generating ZSTD frames with subtle errors that trigger failure points provided in the ZSTD Decoder.
Because of that, it is not possible to efficiently provide valuable negative tests for the integrated ZSTD Decoder.

The alternatives for writing negative tests include:

* Generating a well-known valid ZSTD frame from a specific generator seed and then tweaking the raw bits in this frame to trigger the error response from the decoder
* Using [FuzzTest](https://github.com/google/fuzztest) to create multiple randomized test cases for the decoder and then compare `libzstd` decoder failure with `ZSTD Decoder` failure.

### Known Limitations

* **[WIP]** Bugs in the current flow cause failures in some of the test cases of decoding ZSTD frame with RLE block types
* **[WIP]** Compressed block type is not supported
* Checksum is not being verified

