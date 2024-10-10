# Memory

This directory contains procs responsible for issuing read and write
transactions compatible with AXI subordinates. The AXI bus was selected
because it consists of five streams, which closely resemble the streams
to which XLS channels are translated.

Although XLS channels do not fully conform to the AXI standard, with proper
I/O configuration, DSLX procs using XLS channels can successfully interface
and communicate with AXI4 peripherals.

The signals used to form AXI channels are represented as individual fields
in dedicated structures. However, the XLS toolchain generates these fields
as a flattened bit array. As a result, the Verilog code produced from these
structures will not match the expected AXI bus signature. To interface with
other AXI peripherals, additional Verilog wrappers may be needed to split
the flattened bit vector into individual signals.

# Data Structures

As noted, the procs in this directory use channels with dedicated
structures to represent AXI bus signals in DSLX. For instance, the AXI read
interface comprises the AR channel, which is used to provide a read address,
and the length of data to read, as well as the R channel, which is used
to receive the read data. These channels are represented by the `AxiAr` and
`AxiR` structures in the `axi.x` file, which contains AXI4 bus definitions.
The structures used to represent AXI4 Stream interface can be found in
the `axi_st.x` file.

# Main components

The primary components of this directory are `MemReader` and `MemWriter`,
which facilitate issuing read and write transactions on the AXI bus.
They provide a convenient interface for the DSLX side of the design,
managing the complexities of AXI transactions.

The `MemReader` includes several procs that can be used individually:

- `AxiReader`: Handles the creation of AXI transactions, managing unaligned
  addresses and issuing additional transactions when crossing the 4KB boundary
  or when the read request is longer than maximum possible burst size on the AXI bus

- `AxiStreamDownscaler`: An optional proc available in `MemReaderAdv`,
  enabling DSLX designs to connect to a wider AXI bus.

- `AxiStreamRemoveEmpty`: Removes empty data bits, identified by zeroed bits in
  the `tkeep` and `tstr` signals of the AXI Stream.

The `MemWriter` proc is organised in a similar manner, it consists of:

- `AxiWriter`: Handles the creation of AXI write transactions, managing unaligned
  addresses and issuing additional transactions when crossing the 4KB boundary,
  or when the write request is longer than maximum possible burst size on the AXI bus

- `AxiStreamAddEmpty`: Adds empty data bits in the stream of data to write.
  It is used to shift the data in the stream to facilitate writes to unaligned
  addresses.

# Usage

The list below shows the usage of the `MemReader` proc:

1. Send a `MemReaderReq` to the `req_in_r` channel, providing the information
   about the absolute address from which to read the data, and the length
   of data to read.

2. Wait for the response on the `resp_s` channel. The received packet
   consists of:

   - the `status` of the operation indicating if the read was successful
   - `data` read from the bus
   - `length` of the valid `data` in bytes
   - `last` flag used for marking the last `packet` with data

The list below shows the usage of the `MemWriter` proc:

1. Send a `MemWriterReq` to the `req_in_r` channel, providing the information
   about the absolute address to which data should be sent, and length of the
   transaction in bytes.

2. Provide the data to write using the `MemWriterDataPacket` structure, which
   should be send to `data_in_r` channel. The structure consists of

   - the actual `data`
   - `length` of the `data` in bytes
   - `last` flag used for marking the last `packet` with data.

3. Wait for the response submitted on the `resp_s` channel, which indicates
   if the write operation was successful or an error occurred.

# Cocotb Simulation

This directory also contains Verilog simulations of the created modules,
which test their interaction with RAM attached to the AXI bus. These Verilog
simulations provide insight into the design's latency and achievable throughput.

The simulation interacts with verilog file generated from the particular DSLX proc
through a verilog wrapper. The wrapper is used to create an interface that is
compliant with the AXI specification so that the cocotb testbench can interact
with the DUT with the help of an extension tailored for handling the AXI bus.

## Usage

1. Run the simulation with the following command:

```
bazel run -c opt //xls/modules/zstd/memory:<module_name>_cocotb_test  -- --logtostderr
```

2. Observe simulation results, e.g. for `mem_writer_cocotb_test`:

```
*************************************************************************************************************************************************************
** TEST                                                                                                 STATUS  SIM TIME (ns)  REAL TIME (s)  RATIO (ns/s) **
*************************************************************************************************************************************************************
** mem_writer_cocotb_test.ram_test_single_burst_1_transfer                                               PASS     1970000.00           0.05   40004933.01  **
** mem_writer_cocotb_test.ram_test_single_burst_2_transfers                                              PASS     2140000.00           0.04   52208013.80  **
** mem_writer_cocotb_test.ram_test_single_burst_almost_max_burst_transfer                                PASS    42620000.00           1.00   42734572.11  **
** mem_writer_cocotb_test.ram_test_single_burst_max_burst_transfer                                       PASS    43380000.00           1.03   42245987.95  **
** mem_writer_cocotb_test.ram_test_multiburst_2_full_bursts                                              PASS    85940000.00           2.00   42978720.13  **
** mem_writer_cocotb_test.ram_test_multiburst_1_full_burst_and_single_transfer                           PASS    44510000.00           1.02   43487911.16  **
** mem_writer_cocotb_test.ram_test_multiburst_crossing_4kb_boundary                                      PASS     3740000.00           0.06   60190612.91  **
** mem_writer_cocotb_test.ram_test_multiburst_crossing_4kb_boundary_with_perfectly_aligned_full_bursts   PASS    21440000.00           0.50   42469371.00  **
** mem_writer_cocotb_test.ram_test_multiburst_crossing_4kb_boundary_with_2_full_bursts_and_1_transfer    PASS    87070000.00           2.01   43348812.05  **
** mem_writer_cocotb_test.ram_test_random                                                                PASS   4491230000.00         109.05   41184670.96  **
*************************************************************************************************************************************************************
** TESTS=10 PASS=10 FAIL=0 SKIP=0                                                                               4824040000.01         116.82   41296261.92  **
*************************************************************************************************************************************************************
```
