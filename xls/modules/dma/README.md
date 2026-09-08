# Direct Memory Access

## Overview

Direct Memory Access enables AXI communication between any AXI capable memory and AXI-Stream capable Generic Physical Function (GPF), e.g. an encoding accelerator.
The system is designed to use the AXI interface as the main System Bus.
It is expected that the DMA, an accessible memory module (Main Memory) and a Host Application Module (Host) are also connected to the System Bus.
The host application is responsible for accessing DMA Control and Status Registers (CSR) and issuing a processing request.
The processing request defines the number and addresses of memory transfers.
After receiving a valid request, the DMA accesses the Main Memory to read a block of data and sends it to the GPF.
A FIFO queue is put in place so that data traffic can flow to the GPF over AXI-Stream, regardless of the pending System Bus transactions.
Once the GPF finishes its job, a return data path is used to send the processed data back to the Main Memory.

## Top-level

The main controller defines the following independent parameters:

|     Parameter     | Parameter Identifier | Default value |
| :---------------: | :------------------: | :-----------: |
| Address Bus Width |       `ADDR_W`       |      32       |
|  Data Bus Width   |       `DATA_W`       |      32       |
|  Number of CSRs   |       `REGS_N`       |      14       |

Full list is under development, e.g. it is expected that FIFO depth will also be a top-level configurable parameter.

## Submodules

### Control and Status Registers

The Control and Status Registers implementation is centered around a parameterizable memory array, which targets the 32-bit width and 14 registers.
It is expected that CSRs will be synthesized as flip flops and since most of the 32 bits are unused, they will be removed from the design by optimization tools in the conversion flow.

The implementation is divided into 2 parts: the array, which can be accessed with generic request/response interfaces, and the wrapper, which provides the AXI functionality.
Only a subset of AXI features and properties are supported and usage should be limited to:
* transactions of length 1
* 32-bit data payloads
* aligned data transfers
* all bits in the transfer are valid

It is currently assumed that the CSRs are located at the 0x0000 offset of the system memory map.

#### CSR Features

Most of the control and configuration communication occurs between CSRs and the Address Generators (AG).
Transfers begin upon write to the `Control Register` and this information is passed to the corresponding AG.
In response, the AG sets the busy bit in the `Status Register` and manages the frontend transactions.
Only once all transactions were performed, does the AG signal `done` via a write to the `Interrupt status register` and the busy bit is cleared.
Further changes to the behavior of the DMA will be enabled after implementation of `loop mode` and `external frame synchronization`.
Details of all registers and their respective bits' meaning are described below.

#### Future development

All features that do not yet perform exactly as described in the table below are listed here:
* `Control register`
  * `writer/reader sync disable`, synchronization is not implemented. DSLX DMA works as if the bit was always set
* the `Interrupt mask register` has no effect on the core
* the interrupt bits in the `Interrupt status register` are set when the transfer is done, but there is no interrupt signal to the outside world. Also, it is cleared by writing '0', not '1'.
* The following registers are implemented, but they operate as generic R/W registers with no dedicated function:
  * `Version register`
  * `Configuration register`

#### Register table

**Register table follows [the FastVDMA project](https://antmicro.github.io/fastvdma/RegisterMap.html).**

**This DMA targets only AXI interfaces, so the `configuration register` should be unused, writes to it are ignored.**

Register layout is shown in the table below:

| Address | Role                        |
| ------- | --------------------------- |
| `0x00`  | Control register            |
| `0x04`  | Status register             |
| `0x08`  | Interrupt mask register     |
| `0x0c`  | Interrupt status register   |
| `0x10`  | Reader start address        |
| `0x14`  | Reader line length          |
| `0x18`  | Reader line count           |
| `0x1c`  | Reader stride between lines |
| `0x20`  | Writer start address        |
| `0x24`  | Writer line length          |
| `0x28`  | Writer line count           |
| `0x2c`  | Writer stride between lines |
| `0x30`  | Version register            |
| `0x34`  | Configuration register      |


#### Detailed register description

Provided descriptions are implementation targets, compare with previous section to make sure that the corresponding feature is already working.

##### Control register (0x00)

| Bit  | Name                | Description                                                                                         |
| ---- | ------------------- | --------------------------------------------------------------------------------------------------- |
| 0    | Writer start        | Write `1` to start write frontend (This bit automatically resets itself to `0` if not in loop mode) |
| 1    | Reader start        | Write `1` to start read frontend (This bit automatically resets itself to `0` if not in loop mode)  |
| 2    | Writer sync disable | Write `1` to disable waiting for external writer synchronization (rising edge on `writerSync`)      |
| 3    | Reader sync disable | Write `1` to disable waiting for external reader synchronization (rising edge on `readerSync`)      |
| 4    | Writer loop mode    | Write `1` to automatically start next write frontend transfer after finishing the current one       |
| 5    | Reader loop mode    | Write `1` to automatically start next read frontend transfer after finishing the current one        |
| 6-31 | -                   | Unused                                                                                              |

---

##### Status register (0x04)

| Bit  | Name        | Description                                                |
| ---- | ----------- | ---------------------------------------------------------- |
| 0    | Writer busy | Reads as `1` when write frontend is busy transferring data |
| 1    | Reader busy | Reads as `1` when read frontend is busy transferring data  |
| 2-31 | -           | Unused                                                     |

---

##### Interrupt mask register (0x08)

| Bit  | Name        | Description                          |
| ---- | ----------- | ------------------------------------ |
| 0    | Writer mask | Write `1` to enable writer interrupt |
| 1    | Reader mask | Write `1` to enable reader interrupt |
| 2-31 | -           | Unused                               |

---

##### Interrupt status register (0x0c)

| Bit  | Name             | Description                                                                      |
| ---- | ---------------- | -------------------------------------------------------------------------------- |
| 0    | Writer interrupt | Reads as `1` if writer interrupt has occurred, write `1` to clear that interrupt |
| 1    | Reader interrupt | Reads as `1` if reader interrupt has occurred, write `1` to clear that interrupt |
| 2-31 | -                | Unused                                                                           |

---

##### Reader start address (0x10)

| Bit  | Name          | Description                                                                |
| ---- | ------------- | -------------------------------------------------------------------------- |
| 0-31 | Start address | Reader start address (set to `0` if reader frontend is a stream interface) |

---

##### Reader line length (0x14)

| Bit  | Name        | Description                                              |
| ---- | ----------- | -------------------------------------------------------- |
| 0-31 | Line length | Reader line length (as number of reader data bus widths) |

---

##### Reader line count (0x18)

| Bit  | Name       | Description       |
| ---- | ---------- | ----------------- |
| 0-31 | Line count | Reader line count |

---

##### Reader stride between lines (0x1c)

| Bit  | Name   | Description                                                         |
| ---- | ------ | ------------------------------------------------------------------- |
| 0-31 | Stride | Gap between consecutive lines (as number of reader data bus widths) |

---

##### Writer start address (0x20)

| Bit  | Name          | Description                                                                |
| ---- | ------------- | -------------------------------------------------------------------------- |
| 0-31 | Start address | Writer start address (set to `0` if writer frontend is a stream interface) |

---

##### Writer line length (0x24)

| Bit  | Name        | Description                                              |
| ---- | ----------- | -------------------------------------------------------- |
| 0-31 | Line length | Writer line length (as number of writer data bus widths) |

---

##### Writer line count (0x28)

| Bit  | Name       | Description       |
| ---- | ---------- | ----------------- |
| 0-31 | Line count | Writer line count |

##### Writer stride between lines (0x2c)

| Bit  | Name   | Description                                                         |
| ---- | ------ | ------------------------------------------------------------------- |
| 0-31 | Stride | Gap between consecutive lines (as number of writer data bus widths) |

---

##### Version register (0x30)

| Bit  | Name             | Description           |
| ---- | ---------------- | --------------------- |
| 0-31 | Version register | Holds the DMA version |

---

##### Configuration register (0x34)

| Bit  | Name                   | Description                          |
| ---- | ---------------------- | ------------------------------------ |
| 0-31 | Configuration register | Reader, writer and control bus types |

### FIFO

FIFOs serve a crucial role in buffering input and output data of the GPF.
The DSLX implementation is divided into 3 processes: a memory wrapper, a reader and a writer.
The memory wrapper spawns the dual port RAM model (`examples/ram.x`), which has 3 channels per port.
These channels are used to signal: a request, a read response and a write response. In FIFO, however, ports are used either for writes or reads, so one of the responses is left unused, e.g. a port used for reads does not need the write response.
This motivates the choice to handle the 2 unused channels inside of the wrapper and use the 4 remaining channels as IO.

The reader process is connected to port 0 of the RAM and is responsible for fetching data from the memory and streaming it to the GPF.
Only a subset of AXI Stream features and properties are supported, and usage should be limited to:
* transactions of length 1
* 32-bit data payloads
* aligned data transfers
* all bits in the transfer are valid

The reader process manages the state of the read pointer, which holds the address to the "First Out" data in the memory.
Anytime a read is performed, the pointer is incremented, however, it is also compared with the write pointer to ensure that the queue is not empty (reading before a write occurred).

Similarly, the writer process manages the state of the write pointer, which holds the address to the "Last In" data in the memory. Anytime a write is performed, the pointer is incremented, however, it is also compared with the read pointer to ensure that old data is not overwritten (writing before data could be read).

Pointer synchronization for these 2 processes is performed via dedicated channels.

### Main Controller

The Main Controller is responsible for coordinating the data flow of the DMA Core and comprises:
* Control and Status Registers
* Address Generators (one for writes, one for reads)
* Frontends (one for writes, one for reads)

Changes in the CSR configuration issue new processing requests: the type and number of AXI transactions needed to perform the task are calculated.

The flow can be summarized as follows:
* After power-up, the controller defaults to an `IDLE` state, which means that no transfers are processed
* The only way to change the state is to set the `ControlRegister.start` bit
  * `StatusRegister.busy` bit is set accordingly until the Controller performs all transfers
* The number and type of transactions is determined by the `Transfer configuration registers` settings

```
  // Main controller pseudo-code

  for device in {reader, writer}:

    while( !ControlRegister.start ){}

    update_configuration()
    Set StatusRegister.busy

    do {

      Until all transfers are completed:
        Access System Bus, Convert to AXI Stream, Access the FIFO

    } while ( ControlRegister.loop_mode )

    Clear StatusRegister.busy
    Trigger an interrupt

```

#### Address Generator

The address generator is responsible for calculating all addresses required for the DMA transfer.
The calculation is performed based on the configuration registers:
* `Reader/Writer start address`,
* `Reader/Writer line length`,
* `Reader/Writer line count`,
* `Reader/Writer stride between lines`

Let's denote:
* Starting address of the first transfer is A
* A memory element is an interval of length D bytes
  * D is also equal to the data bus width expressed in bytes (DATA_W)
* Line is composed of a number L of memory elements (line length)
* Stride S is the increment between lines (empty gaps)
* A complete DMA transfer reads/writes C lines

Example #1:

```
    A := 0x1000
    C := 4
    D := 4
    L := 1
    S := 0
```

Resulting transfers T (in a byte aligned memory)

    T([0x1000, 0x1003])
    T([0x1004, 0x1007])
    T([0x1008, 0x100b])
    T([0x100c, 0x100f])

Example #2:

```
    A := 0x1000
    C := 4
    D := 4
    L := 2
    S := 1
```

Resulting transfers T (in a byte aligned memory)

```
  T([0x1000, 0x1003]) // This is Line 1
  T([0x1004, 0x1007]) // This is Line 1
  // stride is 1, so skip [0x1008, 0x100b]
  T([0x100c, 0x100f])
  T([0x1010, 0x1013])
  T([0x1018, 0x101b])
  T([0x101c, 0x101f])
  T([0x1024, 0x1027])
  T([0x1028, 0x102b])
```

In pseudo-code, the function of the address generator can be represented as:

```
function address_generator(A,C,D,L,S):
    tab_address = []
    for c in range(C):
        for k in range(L):
            a = A + D*(k + c*(L+S))
            tab_address.append(a)
    return tab_address
```


#### Frontend

Frontend is responsible for converting addresses from the address generator into valid AXI/AXI-Stream transactions.
The frontend communicates with the `Address Generator` via `start` and `done` channels to signal start and end of transactions.
The `Address Generator` sends a message on the `start` channel and `(Address, Length)` tuple, then awaits for a message on the `done` channel.
`done` will be asserted only once the `Frontend` ensures that the current transaction is completed.

#### Interrupt Controller

Work on the interrupt controller has not yet begun.

## Implementation of AMBA interfaces in DSLX

Specifications of AMBA interfaces (AXI, AXI Stream, AHB, etc.) were designed and ratified with an HDL implementation in mind, but there are distinct differences between modeling in DSLX and HDLs like Verilog.
First of all, DSLX automatically infers valid/ready handshakes from the description of channels and calls to `send()` and `recv()` functions (and their conditional and/or non-blocking versions).
The side effect of this automatic translation is that valid/ready signals, which are mandatory in the AXI specification, are not explicitly visible in the DSLX implementation and are later given obfuscated names.
Thus, conformance to the specification is hard to establish at the DSLX level and since there is no automatic way to find inferred signals, it requires manual work from designers to create testbenches at Verilog level.
The inability to assess specification conformance is a problem that needs to be solved or it may result in compromised interoperability or compatibility of DSLX implementations with existing HDL designs.

### AXI

AXI relies in total on 5 channels for enabling read and write transactions: 3 are used to complete writes and 2 to complete reads.
In case of writes, the flow consists of the following steps:
* Setup the transaction, which is confirmed with a handshake on AW channel
* Send the payload, which is confirmed with a handshake on W channel
* Receive the error status, which is confirmed with a handshake on B channel.
If we only consider the simplest transactions, i.e. a burst of length 1 and data aligned to the bus width, the DSLX implementation may look like this on the AXI subordinate side:

```
next(...) {
  let (tok, aw_payload) = recv(tok, aw_ch);
  let (tok, w_payload) = recv(tok, w_ch);
  let tok = send(tok, b_ch, OKAY);
  ...
  let result = process(aw_payload, w_payload);
}
```

and on the AXI Manager side:

```
next(...) {
    let (aw,w) = consume(data);
    ...
    let tok = send(tok, aw_ch, aw);
    let tok = send(tok, w_ch, w);
    let (tok, b_resp) = recv(tok, b_ch);
}
```

It is expected that this style of description yields an AXI compliant handshake, however, designer has no control over it and may not adjust code to handle corner cases.

## Software based control

The Host manages the DMA via CSRs.

The recommended flow is to:
* select mode of operation by setting bits in the `Configuration Register`
  * currently only AXI/AXI-Stream is supported
* set memory block configuration in the `Writer_*` and `Reader_*` registers
* enable interrupts
* issue processing begin request via write to the `Control register`
* await for an interrupt and handle it with an Interrupt Service Routine

The ISR is expected to:
  * Read `Interrupt status register` to identify the interrupt trigger

The `dma_irq` pin of the DMA Module is set to `1` to signal that there is a pending interrupt.
The Host is responsible for reading the Interrupt Status Register and clearing the interrupt bit.
The interrupts occur either when DMA is done with the workload or an error occurred.
