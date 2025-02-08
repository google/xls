// verilog_lint: waive-start struct-union-name-style
package memory_proc;
  // DSLX Type: pub type MemWord = u32;
  typedef logic [31:0] MemWord;

  // DSLX Type: pub struct MemReq {
  //     is_write: bool,
  //     address: u32,
  //     wdata: MemWord,
  // }
  typedef struct packed {
    logic is_write;
    logic [31:0] address;
    MemWord wdata;
  } MemReq;
endpackage
// verilog_lint: waive-end struct-union-name-style
