module __mem_reader__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__MemReaderInternalNoFsm_0__16_128_16_8_8_2_2_16_64_8_next(
  input wire clk,
  input wire rst,
  input wire [96:0] mem_reader__axi_st_out_data,
  input wire mem_reader__axi_st_out_vld,
  input wire mem_reader__reader_err_data,
  input wire mem_reader__reader_err_vld,
  input wire mem_reader__reader_req_rdy,
  input wire [31:0] mem_reader__req_r_data,
  input wire mem_reader__req_r_vld,
  input wire mem_reader__resp_s_rdy,
  output wire mem_reader__axi_st_out_rdy,
  output wire mem_reader__reader_err_rdy,
  output wire [31:0] mem_reader__reader_req_data,
  output wire mem_reader__reader_req_vld,
  output wire mem_reader__req_r_rdy,
  output wire [81:0] mem_reader__resp_s_data,
  output wire mem_reader__resp_s_vld
);
  wire [96:0] __mem_reader__axi_st_out_data_reg_init = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  wire [96:0] __mem_reader__axi_st_out_data_skid_reg_init = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  wire [31:0] __mem_reader__req_r_data_reg_init = {16'h0000, 16'h0000};
  wire [31:0] __mem_reader__req_r_data_skid_reg_init = {16'h0000, 16'h0000};
  wire [31:0] __mem_reader__reader_req_data_reg_init = {16'h0000, 16'h0000};
  wire [31:0] __mem_reader__reader_req_data_skid_reg_init = {16'h0000, 16'h0000};
  wire [81:0] __mem_reader__resp_s_data_reg_init = {1'h0, 64'h0000_0000_0000_0000, 16'h0000, 1'h0};
  wire [81:0] __mem_reader__resp_s_data_skid_reg_init = {1'h0, 64'h0000_0000_0000_0000, 16'h0000, 1'h0};
  wire [31:0] literal_6282 = {16'h0000, 16'h0000};
  wire [96:0] literal_6252 = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  reg ____state;
  reg [2:0] p0_acc__4_squeezed;
  reg p0_bit_slice_6278;
  reg p0_bit_slice_6279;
  reg p0_bit_slice_6285;
  reg p0_bit_slice_6294;
  reg p0_nor_6295;
  reg p0_and_6313;
  reg p0_or_6320;
  reg [63:0] p0_and_6314;
  reg p0_or_6315;
  reg [2:0] p1_acc__6_squeezed_squeezed;
  reg p1_bit_slice_6285;
  reg p1_bit_slice_6294;
  reg p1_nor_6295;
  reg p1_and_6313;
  reg p1_or_6320;
  reg [63:0] p1_and_6314;
  reg p1_or_6315;
  reg p0_valid;
  reg p1_valid;
  reg __mem_reader__reader_err_data_reg;
  reg __mem_reader__reader_err_data_skid_reg;
  reg __mem_reader__reader_err_data_valid_reg;
  reg __mem_reader__reader_err_data_valid_skid_reg;
  reg [96:0] __mem_reader__axi_st_out_data_reg;
  reg [96:0] __mem_reader__axi_st_out_data_skid_reg;
  reg __mem_reader__axi_st_out_data_valid_reg;
  reg __mem_reader__axi_st_out_data_valid_skid_reg;
  reg [31:0] __mem_reader__req_r_data_reg;
  reg [31:0] __mem_reader__req_r_data_skid_reg;
  reg __mem_reader__req_r_data_valid_reg;
  reg __mem_reader__req_r_data_valid_skid_reg;
  reg [31:0] __mem_reader__reader_req_data_reg;
  reg [31:0] __mem_reader__reader_req_data_skid_reg;
  reg __mem_reader__reader_req_data_valid_reg;
  reg __mem_reader__reader_req_data_valid_skid_reg;
  reg [81:0] __mem_reader__resp_s_data_reg;
  reg [81:0] __mem_reader__resp_s_data_skid_reg;
  reg __mem_reader__resp_s_data_valid_reg;
  reg __mem_reader__resp_s_data_valid_skid_reg;
  wire [31:0] mem_reader__req_r_data_select;
  wire mem_reader__axi_st_out_data_valid_or;
  wire [31:0] mem_reader__req_r_select;
  wire st_received;
  wire [96:0] mem_reader__axi_st_out_data_select;
  wire [15:0] req_length__3;
  wire [96:0] mem_reader__axi_st_out_select;
  wire mem_reader__resp_s_data_from_skid_rdy;
  wire ne_6293;
  wire [7:0] st_str;
  wire [7:0] st_keep;
  wire [7:0] or_6259;
  wire p2_stage_done;
  wire p2_not_valid;
  wire nor_6302;
  wire acc__3_squeezed_squeezed_const_msb_bits;
  wire p1_enable;
  wire mem_reader__reader_req_not_pred;
  wire mem_reader__reader_req_data_from_skid_rdy;
  wire [2:0] add_6370;
  wire [1:0] acc__10;
  wire p1_data_enable;
  wire p1_not_valid;
  wire p0_all_active_inputs_valid;
  wire acc__3_squeezed_squeezed_const_msb_bits__2;
  wire [2:0] acc__7_squeezed_squeezed;
  wire [1:0] ____state__next_value_predicates;
  wire [1:0] acc__2_squeezed;
  wire p0_enable;
  wire p0_stage_done;
  wire [3:0] acc__7_squeezed;
  wire [2:0] one_hot_6296;
  wire [1:0] add_6270;
  wire mem_reader__reader_err_data_valid_or;
  wire p0_data_enable;
  wire [3:0] add_6375;
  wire [2:0] add_6346;
  wire acc__3_squeezed_squeezed_const_msb_bits__1;
  wire [1:0] acc__3_squeezed_squeezed;
  wire error;
  wire st_last__1;
  wire mem_reader__reader_err_data_from_skid_rdy;
  wire and_6447;
  wire mem_reader__axi_st_out_data_from_skid_rdy;
  wire mem_reader__req_r_data_from_skid_rdy;
  wire and_6449;
  wire __mem_reader__reader_req_vld_buf;
  wire [3:0] acc__8_squeezed;
  wire __mem_reader__resp_s_vld_buf;
  wire [2:0] acc__5_squeezed_squeezed;
  wire [2:0] acc__3_squeezed;
  wire nor_6295;
  wire mem_reader__reader_err_data_data_valid_load_en;
  wire mem_reader__reader_err_data_to_is_not_rdy;
  wire mem_reader__axi_st_out_data_data_valid_load_en;
  wire mem_reader__req_r_data_data_valid_load_en;
  wire mem_reader__req_r_data_to_is_not_rdy;
  wire mem_reader__reader_req_data_data_valid_load_en;
  wire mem_reader__reader_req_data_to_is_not_rdy;
  wire mem_reader__resp_s_data_data_valid_load_en;
  wire mem_reader__resp_s_data_to_is_not_rdy;
  wire ____state__at_most_one_next_value;
  wire [2:0] add_6349;
  wire [2:0] add_6276;
  wire nor_6306;
  wire and_6313;
  wire [63:0] st_data;
  wire [1:0] concat_6441;
  wire next_active;
  wire mem_reader__reader_err_data_data_is_sent_to;
  wire mem_reader__reader_err_data_skid_data_load_en;
  wire mem_reader__reader_err_data_skid_valid_set_zero;
  wire mem_reader__axi_st_out_data_data_is_sent_to;
  wire mem_reader__axi_st_out_data_skid_data_load_en;
  wire mem_reader__axi_st_out_data_skid_valid_set_zero;
  wire mem_reader__req_r_data_data_is_sent_to;
  wire mem_reader__req_r_data_skid_data_load_en;
  wire mem_reader__req_r_data_skid_valid_set_zero;
  wire mem_reader__reader_req_data_data_is_sent_to;
  wire mem_reader__reader_req_data_skid_data_load_en;
  wire mem_reader__reader_req_data_skid_valid_set_zero;
  wire mem_reader__resp_s_data_data_is_sent_to;
  wire mem_reader__resp_s_data_skid_data_load_en;
  wire mem_reader__resp_s_data_skid_valid_set_zero;
  wire [31:0] mem_reader__reader_req_data_select;
  wire mem_reader__reader_req_data_valid_or;
  wire [81:0] mem_reader__resp_s_data_select;
  wire mem_reader__resp_s_data_valid_or;
  wire [2:0] acc__6_squeezed_squeezed;
  wire [2:0] acc__4_squeezed;
  wire bit_slice_6278;
  wire bit_slice_6279;
  wire bit_slice_6285;
  wire bit_slice_6294;
  wire or_6320;
  wire [63:0] and_6314;
  wire or_6315;
  wire one_hot_sel_6442;
  wire or_6443;
  wire mem_reader__reader_err_data_data_valid_load_en__1;
  wire mem_reader__reader_err_data_skid_valid_load_en;
  wire mem_reader__axi_st_out_data_data_valid_load_en__1;
  wire mem_reader__axi_st_out_data_skid_valid_load_en;
  wire mem_reader__req_r_data_data_valid_load_en__1;
  wire mem_reader__req_r_data_skid_valid_load_en;
  wire mem_reader__reader_req_data_data_valid_load_en__1;
  wire mem_reader__reader_req_data_skid_valid_load_en;
  wire [81:0] __mem_reader__resp_s_data_buf;
  wire mem_reader__resp_s_data_data_valid_load_en__1;
  wire mem_reader__resp_s_data_skid_valid_load_en;
  assign mem_reader__req_r_data_select = __mem_reader__req_r_data_valid_skid_reg ? __mem_reader__req_r_data_skid_reg : __mem_reader__req_r_data_reg;
  assign mem_reader__axi_st_out_data_valid_or = __mem_reader__axi_st_out_data_valid_reg | __mem_reader__axi_st_out_data_valid_skid_reg;
  assign mem_reader__req_r_select = ~____state ? mem_reader__req_r_data_select : literal_6282;
  assign st_received = ____state & mem_reader__axi_st_out_data_valid_or;
  assign mem_reader__axi_st_out_data_select = __mem_reader__axi_st_out_data_valid_skid_reg ? __mem_reader__axi_st_out_data_skid_reg : __mem_reader__axi_st_out_data_reg;
  assign req_length__3 = mem_reader__req_r_select[15:0];
  assign mem_reader__axi_st_out_select = st_received ? mem_reader__axi_st_out_data_select : literal_6252;
  assign mem_reader__resp_s_data_from_skid_rdy = ~__mem_reader__resp_s_data_valid_skid_reg;
  assign ne_6293 = req_length__3 != 16'h0000;
  assign st_str = mem_reader__axi_st_out_select[32:25];
  assign st_keep = mem_reader__axi_st_out_select[24:17];
  assign or_6259 = st_str | st_keep;
  assign p2_stage_done = p1_valid & (~p1_or_6320 | mem_reader__resp_s_data_from_skid_rdy);
  assign p2_not_valid = ~p1_valid;
  assign nor_6302 = ~(____state | ~ne_6293);
  assign acc__3_squeezed_squeezed_const_msb_bits = 1'h0;
  assign p1_enable = p2_stage_done | p2_not_valid;
  assign mem_reader__reader_req_not_pred = ~nor_6302;
  assign mem_reader__reader_req_data_from_skid_rdy = ~__mem_reader__reader_req_data_valid_skid_reg;
  assign add_6370 = p1_acc__6_squeezed_squeezed + 3'h1;
  assign acc__10 = {acc__3_squeezed_squeezed_const_msb_bits, or_6259[0]};
  assign p1_data_enable = p1_enable & p0_valid;
  assign p1_not_valid = ~p0_valid;
  assign p0_all_active_inputs_valid = ____state | __mem_reader__req_r_data_valid_reg | __mem_reader__req_r_data_valid_skid_reg;
  assign acc__3_squeezed_squeezed_const_msb_bits__2 = 1'h0;
  assign acc__7_squeezed_squeezed = p1_bit_slice_6285 ? add_6370 : p1_acc__6_squeezed_squeezed;
  assign ____state__next_value_predicates = {~____state, ____state};
  assign acc__2_squeezed = or_6259[1] ? (or_6259[0] ? 2'h2 : 2'h1) : acc__10;
  assign p0_enable = p1_data_enable | p1_not_valid;
  assign p0_stage_done = p0_all_active_inputs_valid & (mem_reader__reader_req_not_pred | mem_reader__reader_req_data_from_skid_rdy);
  assign acc__7_squeezed = {acc__3_squeezed_squeezed_const_msb_bits__2, acc__7_squeezed_squeezed};
  assign one_hot_6296 = {____state__next_value_predicates[1:0] == 2'h0, ____state__next_value_predicates[1] && !____state__next_value_predicates[0], ____state__next_value_predicates[0]};
  assign add_6270 = acc__2_squeezed + 2'h1;
  assign mem_reader__reader_err_data_valid_or = __mem_reader__reader_err_data_valid_reg | __mem_reader__reader_err_data_valid_skid_reg;
  assign p0_data_enable = p0_enable & p0_stage_done;
  assign add_6375 = acc__7_squeezed + 4'h1;
  assign add_6346 = p0_acc__4_squeezed + 3'h1;
  assign acc__3_squeezed_squeezed_const_msb_bits__1 = 1'h0;
  assign acc__3_squeezed_squeezed = or_6259[2] ? add_6270 : acc__2_squeezed;
  assign error = ____state & mem_reader__reader_err_data_valid_or;
  assign st_last__1 = mem_reader__axi_st_out_select[0:0];
  assign mem_reader__reader_err_data_from_skid_rdy = ~__mem_reader__reader_err_data_valid_skid_reg;
  assign and_6447 = ____state & p0_data_enable;
  assign mem_reader__axi_st_out_data_from_skid_rdy = ~__mem_reader__axi_st_out_data_valid_skid_reg;
  assign mem_reader__req_r_data_from_skid_rdy = ~__mem_reader__req_r_data_valid_skid_reg;
  assign and_6449 = ~____state & p0_data_enable;
  assign __mem_reader__reader_req_vld_buf = p0_all_active_inputs_valid & p0_enable & nor_6302;
  assign acc__8_squeezed = p1_bit_slice_6294 ? add_6375 : acc__7_squeezed;
  assign __mem_reader__resp_s_vld_buf = p1_valid & p1_or_6320;
  assign acc__5_squeezed_squeezed = p0_bit_slice_6278 ? add_6346 : p0_acc__4_squeezed;
  assign acc__3_squeezed = {acc__3_squeezed_squeezed_const_msb_bits__1, acc__3_squeezed_squeezed};
  assign nor_6295 = ~(~____state | error | ~st_received);
  assign mem_reader__reader_err_data_data_valid_load_en = mem_reader__reader_err_vld & mem_reader__reader_err_data_from_skid_rdy;
  assign mem_reader__reader_err_data_to_is_not_rdy = ~and_6447;
  assign mem_reader__axi_st_out_data_data_valid_load_en = mem_reader__axi_st_out_vld & mem_reader__axi_st_out_data_from_skid_rdy;
  assign mem_reader__req_r_data_data_valid_load_en = mem_reader__req_r_vld & mem_reader__req_r_data_from_skid_rdy;
  assign mem_reader__req_r_data_to_is_not_rdy = ~and_6449;
  assign mem_reader__reader_req_data_data_valid_load_en = __mem_reader__reader_req_vld_buf & mem_reader__reader_req_data_from_skid_rdy;
  assign mem_reader__reader_req_data_to_is_not_rdy = ~mem_reader__reader_req_rdy;
  assign mem_reader__resp_s_data_data_valid_load_en = __mem_reader__resp_s_vld_buf & mem_reader__resp_s_data_from_skid_rdy;
  assign mem_reader__resp_s_data_to_is_not_rdy = ~mem_reader__resp_s_rdy;
  assign ____state__at_most_one_next_value = ~____state == one_hot_6296[1] & ____state == one_hot_6296[0];
  assign add_6349 = acc__5_squeezed_squeezed + 3'h1;
  assign add_6276 = acc__3_squeezed + 3'h1;
  assign nor_6306 = ~(____state | ne_6293);
  assign and_6313 = ____state & error;
  assign st_data = mem_reader__axi_st_out_select[96:33];
  assign concat_6441 = {and_6449, and_6447};
  assign next_active = ~(st_received & st_last__1) & ~error;
  assign mem_reader__reader_err_data_data_is_sent_to = __mem_reader__reader_err_data_valid_reg & and_6447 & mem_reader__reader_err_data_from_skid_rdy;
  assign mem_reader__reader_err_data_skid_data_load_en = __mem_reader__reader_err_data_valid_reg & mem_reader__reader_err_data_data_valid_load_en & mem_reader__reader_err_data_to_is_not_rdy;
  assign mem_reader__reader_err_data_skid_valid_set_zero = __mem_reader__reader_err_data_valid_skid_reg & and_6447;
  assign mem_reader__axi_st_out_data_data_is_sent_to = __mem_reader__axi_st_out_data_valid_reg & and_6447 & mem_reader__axi_st_out_data_from_skid_rdy;
  assign mem_reader__axi_st_out_data_skid_data_load_en = __mem_reader__axi_st_out_data_valid_reg & mem_reader__axi_st_out_data_data_valid_load_en & mem_reader__reader_err_data_to_is_not_rdy;
  assign mem_reader__axi_st_out_data_skid_valid_set_zero = __mem_reader__axi_st_out_data_valid_skid_reg & and_6447;
  assign mem_reader__req_r_data_data_is_sent_to = __mem_reader__req_r_data_valid_reg & and_6449 & mem_reader__req_r_data_from_skid_rdy;
  assign mem_reader__req_r_data_skid_data_load_en = __mem_reader__req_r_data_valid_reg & mem_reader__req_r_data_data_valid_load_en & mem_reader__req_r_data_to_is_not_rdy;
  assign mem_reader__req_r_data_skid_valid_set_zero = __mem_reader__req_r_data_valid_skid_reg & and_6449;
  assign mem_reader__reader_req_data_data_is_sent_to = __mem_reader__reader_req_data_valid_reg & mem_reader__reader_req_rdy & mem_reader__reader_req_data_from_skid_rdy;
  assign mem_reader__reader_req_data_skid_data_load_en = __mem_reader__reader_req_data_valid_reg & mem_reader__reader_req_data_data_valid_load_en & mem_reader__reader_req_data_to_is_not_rdy;
  assign mem_reader__reader_req_data_skid_valid_set_zero = __mem_reader__reader_req_data_valid_skid_reg & mem_reader__reader_req_rdy;
  assign mem_reader__resp_s_data_data_is_sent_to = __mem_reader__resp_s_data_valid_reg & mem_reader__resp_s_rdy & mem_reader__resp_s_data_from_skid_rdy;
  assign mem_reader__resp_s_data_skid_data_load_en = __mem_reader__resp_s_data_valid_reg & mem_reader__resp_s_data_data_valid_load_en & mem_reader__resp_s_data_to_is_not_rdy;
  assign mem_reader__resp_s_data_skid_valid_set_zero = __mem_reader__resp_s_data_valid_skid_reg & mem_reader__resp_s_rdy;
  assign mem_reader__reader_req_data_select = __mem_reader__reader_req_data_valid_skid_reg ? __mem_reader__reader_req_data_skid_reg : __mem_reader__reader_req_data_reg;
  assign mem_reader__reader_req_data_valid_or = __mem_reader__reader_req_data_valid_reg | __mem_reader__reader_req_data_valid_skid_reg;
  assign mem_reader__resp_s_data_select = __mem_reader__resp_s_data_valid_skid_reg ? __mem_reader__resp_s_data_skid_reg : __mem_reader__resp_s_data_reg;
  assign mem_reader__resp_s_data_valid_or = __mem_reader__resp_s_data_valid_reg | __mem_reader__resp_s_data_valid_skid_reg;
  assign acc__6_squeezed_squeezed = p0_bit_slice_6279 ? add_6349 : acc__5_squeezed_squeezed;
  assign acc__4_squeezed = or_6259[3] ? add_6276 : acc__3_squeezed;
  assign bit_slice_6278 = or_6259[4];
  assign bit_slice_6279 = or_6259[5];
  assign bit_slice_6285 = or_6259[6];
  assign bit_slice_6294 = or_6259[7];
  assign or_6320 = nor_6306 | nor_6295 | and_6313;
  assign and_6314 = st_data & {64{nor_6295}};
  assign or_6315 = nor_6295 & st_last__1 | nor_6306;
  assign one_hot_sel_6442 = next_active & concat_6441[0] | ne_6293 & concat_6441[1];
  assign or_6443 = and_6449 | and_6447;
  assign mem_reader__reader_err_data_data_valid_load_en__1 = mem_reader__reader_err_data_data_is_sent_to | mem_reader__reader_err_data_data_valid_load_en;
  assign mem_reader__reader_err_data_skid_valid_load_en = mem_reader__reader_err_data_skid_data_load_en | mem_reader__reader_err_data_skid_valid_set_zero;
  assign mem_reader__axi_st_out_data_data_valid_load_en__1 = mem_reader__axi_st_out_data_data_is_sent_to | mem_reader__axi_st_out_data_data_valid_load_en;
  assign mem_reader__axi_st_out_data_skid_valid_load_en = mem_reader__axi_st_out_data_skid_data_load_en | mem_reader__axi_st_out_data_skid_valid_set_zero;
  assign mem_reader__req_r_data_data_valid_load_en__1 = mem_reader__req_r_data_data_is_sent_to | mem_reader__req_r_data_data_valid_load_en;
  assign mem_reader__req_r_data_skid_valid_load_en = mem_reader__req_r_data_skid_data_load_en | mem_reader__req_r_data_skid_valid_set_zero;
  assign mem_reader__reader_req_data_data_valid_load_en__1 = mem_reader__reader_req_data_data_is_sent_to | mem_reader__reader_req_data_data_valid_load_en;
  assign mem_reader__reader_req_data_skid_valid_load_en = mem_reader__reader_req_data_skid_data_load_en | mem_reader__reader_req_data_skid_valid_set_zero;
  assign __mem_reader__resp_s_data_buf = {p1_and_6313, p1_and_6314, {12'h000, acc__8_squeezed & {4{p1_nor_6295}}}, p1_or_6315};
  assign mem_reader__resp_s_data_data_valid_load_en__1 = mem_reader__resp_s_data_data_is_sent_to | mem_reader__resp_s_data_data_valid_load_en;
  assign mem_reader__resp_s_data_skid_valid_load_en = mem_reader__resp_s_data_skid_data_load_en | mem_reader__resp_s_data_skid_valid_set_zero;
  always @ (posedge clk) begin
    if (rst) begin
      ____state <= 1'h0;
      p0_acc__4_squeezed <= 3'h0;
      p0_bit_slice_6278 <= 1'h0;
      p0_bit_slice_6279 <= 1'h0;
      p0_bit_slice_6285 <= 1'h0;
      p0_bit_slice_6294 <= 1'h0;
      p0_nor_6295 <= 1'h0;
      p0_and_6313 <= 1'h0;
      p0_or_6320 <= 1'h0;
      p0_and_6314 <= 64'h0000_0000_0000_0000;
      p0_or_6315 <= 1'h0;
      p1_acc__6_squeezed_squeezed <= 3'h0;
      p1_bit_slice_6285 <= 1'h0;
      p1_bit_slice_6294 <= 1'h0;
      p1_nor_6295 <= 1'h0;
      p1_and_6313 <= 1'h0;
      p1_or_6320 <= 1'h0;
      p1_and_6314 <= 64'h0000_0000_0000_0000;
      p1_or_6315 <= 1'h0;
      p0_valid <= 1'h0;
      p1_valid <= 1'h0;
      __mem_reader__reader_err_data_reg <= 1'h0;
      __mem_reader__reader_err_data_skid_reg <= 1'h0;
      __mem_reader__reader_err_data_valid_reg <= 1'h0;
      __mem_reader__reader_err_data_valid_skid_reg <= 1'h0;
      __mem_reader__axi_st_out_data_reg <= __mem_reader__axi_st_out_data_reg_init;
      __mem_reader__axi_st_out_data_skid_reg <= __mem_reader__axi_st_out_data_skid_reg_init;
      __mem_reader__axi_st_out_data_valid_reg <= 1'h0;
      __mem_reader__axi_st_out_data_valid_skid_reg <= 1'h0;
      __mem_reader__req_r_data_reg <= __mem_reader__req_r_data_reg_init;
      __mem_reader__req_r_data_skid_reg <= __mem_reader__req_r_data_skid_reg_init;
      __mem_reader__req_r_data_valid_reg <= 1'h0;
      __mem_reader__req_r_data_valid_skid_reg <= 1'h0;
      __mem_reader__reader_req_data_reg <= __mem_reader__reader_req_data_reg_init;
      __mem_reader__reader_req_data_skid_reg <= __mem_reader__reader_req_data_skid_reg_init;
      __mem_reader__reader_req_data_valid_reg <= 1'h0;
      __mem_reader__reader_req_data_valid_skid_reg <= 1'h0;
      __mem_reader__resp_s_data_reg <= __mem_reader__resp_s_data_reg_init;
      __mem_reader__resp_s_data_skid_reg <= __mem_reader__resp_s_data_skid_reg_init;
      __mem_reader__resp_s_data_valid_reg <= 1'h0;
      __mem_reader__resp_s_data_valid_skid_reg <= 1'h0;
    end else begin
      ____state <= or_6443 ? one_hot_sel_6442 : ____state;
      p0_acc__4_squeezed <= p0_data_enable ? acc__4_squeezed : p0_acc__4_squeezed;
      p0_bit_slice_6278 <= p0_data_enable ? bit_slice_6278 : p0_bit_slice_6278;
      p0_bit_slice_6279 <= p0_data_enable ? bit_slice_6279 : p0_bit_slice_6279;
      p0_bit_slice_6285 <= p0_data_enable ? bit_slice_6285 : p0_bit_slice_6285;
      p0_bit_slice_6294 <= p0_data_enable ? bit_slice_6294 : p0_bit_slice_6294;
      p0_nor_6295 <= p0_data_enable ? nor_6295 : p0_nor_6295;
      p0_and_6313 <= p0_data_enable ? and_6313 : p0_and_6313;
      p0_or_6320 <= p0_data_enable ? or_6320 : p0_or_6320;
      p0_and_6314 <= p0_data_enable ? and_6314 : p0_and_6314;
      p0_or_6315 <= p0_data_enable ? or_6315 : p0_or_6315;
      p1_acc__6_squeezed_squeezed <= p1_data_enable ? acc__6_squeezed_squeezed : p1_acc__6_squeezed_squeezed;
      p1_bit_slice_6285 <= p1_data_enable ? p0_bit_slice_6285 : p1_bit_slice_6285;
      p1_bit_slice_6294 <= p1_data_enable ? p0_bit_slice_6294 : p1_bit_slice_6294;
      p1_nor_6295 <= p1_data_enable ? p0_nor_6295 : p1_nor_6295;
      p1_and_6313 <= p1_data_enable ? p0_and_6313 : p1_and_6313;
      p1_or_6320 <= p1_data_enable ? p0_or_6320 : p1_or_6320;
      p1_and_6314 <= p1_data_enable ? p0_and_6314 : p1_and_6314;
      p1_or_6315 <= p1_data_enable ? p0_or_6315 : p1_or_6315;
      p0_valid <= p0_enable ? p0_stage_done : p0_valid;
      p1_valid <= p1_enable ? p0_valid : p1_valid;
      __mem_reader__reader_err_data_reg <= mem_reader__reader_err_data_data_valid_load_en ? mem_reader__reader_err_data : __mem_reader__reader_err_data_reg;
      __mem_reader__reader_err_data_skid_reg <= mem_reader__reader_err_data_skid_data_load_en ? __mem_reader__reader_err_data_reg : __mem_reader__reader_err_data_skid_reg;
      __mem_reader__reader_err_data_valid_reg <= mem_reader__reader_err_data_data_valid_load_en__1 ? mem_reader__reader_err_vld : __mem_reader__reader_err_data_valid_reg;
      __mem_reader__reader_err_data_valid_skid_reg <= mem_reader__reader_err_data_skid_valid_load_en ? mem_reader__reader_err_data_from_skid_rdy : __mem_reader__reader_err_data_valid_skid_reg;
      __mem_reader__axi_st_out_data_reg <= mem_reader__axi_st_out_data_data_valid_load_en ? mem_reader__axi_st_out_data : __mem_reader__axi_st_out_data_reg;
      __mem_reader__axi_st_out_data_skid_reg <= mem_reader__axi_st_out_data_skid_data_load_en ? __mem_reader__axi_st_out_data_reg : __mem_reader__axi_st_out_data_skid_reg;
      __mem_reader__axi_st_out_data_valid_reg <= mem_reader__axi_st_out_data_data_valid_load_en__1 ? mem_reader__axi_st_out_vld : __mem_reader__axi_st_out_data_valid_reg;
      __mem_reader__axi_st_out_data_valid_skid_reg <= mem_reader__axi_st_out_data_skid_valid_load_en ? mem_reader__axi_st_out_data_from_skid_rdy : __mem_reader__axi_st_out_data_valid_skid_reg;
      __mem_reader__req_r_data_reg <= mem_reader__req_r_data_data_valid_load_en ? mem_reader__req_r_data : __mem_reader__req_r_data_reg;
      __mem_reader__req_r_data_skid_reg <= mem_reader__req_r_data_skid_data_load_en ? __mem_reader__req_r_data_reg : __mem_reader__req_r_data_skid_reg;
      __mem_reader__req_r_data_valid_reg <= mem_reader__req_r_data_data_valid_load_en__1 ? mem_reader__req_r_vld : __mem_reader__req_r_data_valid_reg;
      __mem_reader__req_r_data_valid_skid_reg <= mem_reader__req_r_data_skid_valid_load_en ? mem_reader__req_r_data_from_skid_rdy : __mem_reader__req_r_data_valid_skid_reg;
      __mem_reader__reader_req_data_reg <= mem_reader__reader_req_data_data_valid_load_en ? mem_reader__req_r_select : __mem_reader__reader_req_data_reg;
      __mem_reader__reader_req_data_skid_reg <= mem_reader__reader_req_data_skid_data_load_en ? __mem_reader__reader_req_data_reg : __mem_reader__reader_req_data_skid_reg;
      __mem_reader__reader_req_data_valid_reg <= mem_reader__reader_req_data_data_valid_load_en__1 ? __mem_reader__reader_req_vld_buf : __mem_reader__reader_req_data_valid_reg;
      __mem_reader__reader_req_data_valid_skid_reg <= mem_reader__reader_req_data_skid_valid_load_en ? mem_reader__reader_req_data_from_skid_rdy : __mem_reader__reader_req_data_valid_skid_reg;
      __mem_reader__resp_s_data_reg <= mem_reader__resp_s_data_data_valid_load_en ? __mem_reader__resp_s_data_buf : __mem_reader__resp_s_data_reg;
      __mem_reader__resp_s_data_skid_reg <= mem_reader__resp_s_data_skid_data_load_en ? __mem_reader__resp_s_data_reg : __mem_reader__resp_s_data_skid_reg;
      __mem_reader__resp_s_data_valid_reg <= mem_reader__resp_s_data_data_valid_load_en__1 ? __mem_reader__resp_s_vld_buf : __mem_reader__resp_s_data_valid_reg;
      __mem_reader__resp_s_data_valid_skid_reg <= mem_reader__resp_s_data_skid_valid_load_en ? mem_reader__resp_s_data_from_skid_rdy : __mem_reader__resp_s_data_valid_skid_reg;
    end
  end
  assign mem_reader__axi_st_out_rdy = mem_reader__axi_st_out_data_from_skid_rdy;
  assign mem_reader__reader_err_rdy = mem_reader__reader_err_data_from_skid_rdy;
  assign mem_reader__reader_req_data = mem_reader__reader_req_data_select;
  assign mem_reader__reader_req_vld = mem_reader__reader_req_data_valid_or;
  assign mem_reader__req_r_rdy = mem_reader__req_r_data_from_skid_rdy;
  assign mem_reader__resp_s_data = mem_reader__resp_s_data_select;
  assign mem_reader__resp_s_vld = mem_reader__resp_s_data_valid_or;
endmodule


module __xls_modules_zstd_memory_axi_reader__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiReaderNoFsm_0__16_128_16_8_8_5_4_12_next(
  input wire clk,
  input wire rst,
  input wire mem_reader__axi_ar_s_rdy,
  input wire mem_reader__rconf_rdy,
  input wire mem_reader__reader_err_rdy,
  input wire [31:0] mem_reader__reader_req_data,
  input wire mem_reader__reader_req_vld,
  input wire mem_reader__rresp_data,
  input wire mem_reader__rresp_vld,
  output wire [51:0] mem_reader__axi_ar_s_data,
  output wire mem_reader__axi_ar_s_vld,
  output wire [47:0] mem_reader__rconf_data,
  output wire mem_reader__rconf_vld,
  output wire mem_reader__reader_err_data,
  output wire mem_reader__reader_err_vld,
  output wire mem_reader__reader_req_rdy,
  output wire mem_reader__rresp_rdy
);
  wire [31:0] __mem_reader__reader_req_data_reg_init = {16'h0000, 16'h0000};
  wire [31:0] __mem_reader__reader_req_data_skid_reg_init = {16'h0000, 16'h0000};
  wire [51:0] __mem_reader__axi_ar_s_data_reg_init = {8'h00, 16'h0000, 4'h0, 8'h00, 3'h0, 2'h0, 4'h0, 3'h0, 4'h0};
  wire [51:0] __mem_reader__axi_ar_s_data_skid_reg_init = {8'h00, 16'h0000, 4'h0, 8'h00, 3'h0, 2'h0, 4'h0, 3'h0, 4'h0};
  wire [47:0] __mem_reader__rconf_data_reg_init = {16'h0000, 16'h0000, 8'h00, 4'h0, 4'h0};
  wire [47:0] __mem_reader__rconf_data_skid_reg_init = {16'h0000, 16'h0000, 8'h00, 4'h0, 4'h0};
  wire [31:0] literal_6687 = {16'h0000, 16'h0000};
  reg [15:0] ____state_1;
  reg [3:0] p0_aligned_offset;
  reg [12:0] p0_sel_6589;
  reg [15:0] ____state_2;
  reg [15:0] p1_tran_len;
  reg [3:0] p1_bit_slice_6610;
  reg [7:0] p1_bit_slice_6611;
  reg ____state_0;
  reg [15:0] p2_next_len;
  reg [15:0] p2_next_addr;
  reg ____state_0_full;
  reg ____state_1_full;
  reg ____state_2_full;
  reg p0_valid;
  reg p1_valid;
  reg p2_valid;
  reg __mem_reader__axi_ar_s_data_has_been_sent_reg;
  reg __mem_reader__rconf_data_has_been_sent_reg;
  reg __mem_reader__reader_err_data_has_been_sent_reg;
  reg __mem_reader__rresp_data_reg;
  reg __mem_reader__rresp_data_skid_reg;
  reg __mem_reader__rresp_data_valid_reg;
  reg __mem_reader__rresp_data_valid_skid_reg;
  reg [31:0] __mem_reader__reader_req_data_reg;
  reg [31:0] __mem_reader__reader_req_data_skid_reg;
  reg __mem_reader__reader_req_data_valid_reg;
  reg __mem_reader__reader_req_data_valid_skid_reg;
  reg [51:0] __mem_reader__axi_ar_s_data_reg;
  reg [51:0] __mem_reader__axi_ar_s_data_skid_reg;
  reg __mem_reader__axi_ar_s_data_valid_reg;
  reg __mem_reader__axi_ar_s_data_valid_skid_reg;
  reg [47:0] __mem_reader__rconf_data_reg;
  reg [47:0] __mem_reader__rconf_data_skid_reg;
  reg __mem_reader__rconf_data_valid_reg;
  reg __mem_reader__rconf_data_valid_skid_reg;
  reg __mem_reader__reader_err_data_reg;
  reg __mem_reader__reader_err_data_skid_reg;
  reg __mem_reader__reader_err_data_valid_reg;
  reg __mem_reader__reader_err_data_valid_skid_reg;
  wire mem_reader__rresp_data_select;
  wire mem_reader__rresp_select;
  wire and_6681;
  wire mem_reader__reader_err_not_pred;
  wire mem_reader__reader_err_data_from_skid_rdy;
  wire mem_reader__axi_ar_s_data_from_skid_rdy;
  wire mem_reader__rconf_data_from_skid_rdy;
  wire [15:0] next_len_or_exit;
  wire p3_all_active_inputs_valid;
  wire or_8419;
  wire eq_6677;
  wire p3_stage_done;
  wire p3_not_valid;
  wire p2_stage_valid;
  wire p2_all_active_outputs_ready;
  wire nor_6678;
  wire and_6679;
  wire nor_6680;
  wire p2_enable;
  wire p2_stage_done;
  wire [2:0] ____state_1__next_value_predicates;
  wire [2:0] ____state_2__next_value_predicates;
  wire p2_data_enable;
  wire p2_not_valid;
  wire [3:0] one_hot_6684;
  wire [3:0] one_hot_6685;
  wire [15:0] concat_6604;
  wire [3:0] aligned_offset;
  wire p1_enable;
  wire p1_stage_valid;
  wire __mem_reader__axi_ar_s_vld_buf;
  wire __mem_reader__axi_ar_s_data_not_has_been_sent;
  wire __mem_reader__rconf_data_not_has_been_sent;
  wire __mem_reader__reader_err_vld_buf;
  wire __mem_reader__reader_err_data_not_has_been_sent;
  wire [31:0] mem_reader__reader_req_data_select;
  wire p1_data_enable;
  wire p1_not_valid;
  wire mem_reader__rresp_data_from_skid_rdy;
  wire and_6815;
  wire mem_reader__reader_req_data_from_skid_rdy;
  wire and_6816;
  wire __mem_reader__axi_ar_s_data_valid_and_not_has_been_sent;
  wire [3:0] tran_len_mod;
  wire __mem_reader__rconf_data_valid_and_not_has_been_sent;
  wire __mem_reader__reader_err_data_valid_and_not_has_been_sent;
  wire [15:0] tran_len;
  wire [12:0] bytes_to_4k_squeezed;
  wire [12:0] bytes_to_max_burst_squeezed;
  wire and_6795;
  wire and_6796;
  wire [31:0] mem_reader__reader_req_select;
  wire p0_enable;
  wire and_6804;
  wire and_6805;
  wire mem_reader__rresp_data_data_valid_load_en;
  wire mem_reader__rresp_data_to_is_not_rdy;
  wire mem_reader__reader_req_data_data_valid_load_en;
  wire mem_reader__reader_req_data_to_is_not_rdy;
  wire rest;
  wire [7:0] add_6634;
  wire mem_reader__axi_ar_s_data_data_valid_load_en;
  wire mem_reader__axi_ar_s_data_to_is_not_rdy;
  wire [3:0] add_6635;
  wire [3:0] MAX_LANE__1;
  wire mem_reader__rconf_data_data_valid_load_en;
  wire mem_reader__rconf_data_to_is_not_rdy;
  wire mem_reader__reader_err_data_data_valid_load_en;
  wire mem_reader__reader_err_data_to_is_not_rdy;
  wire ____state_1__at_most_one_next_value;
  wire ____state_2__at_most_one_next_value;
  wire [15:0] adjusted_tran_len;
  wire [2:0] concat_6797;
  wire [15:0] req_addr;
  wire p0_data_enable;
  wire or_6799;
  wire [2:0] concat_6806;
  wire [15:0] req_len;
  wire or_6808;
  wire __mem_reader__axi_ar_s_data_valid_and_all_active_outputs_ready;
  wire __mem_reader__reader_err_data_valid_and_all_active_outputs_ready;
  wire mem_reader__rresp_data_data_is_sent_to;
  wire mem_reader__rresp_data_skid_data_load_en;
  wire mem_reader__rresp_data_skid_valid_set_zero;
  wire mem_reader__reader_req_data_data_is_sent_to;
  wire mem_reader__reader_req_data_skid_data_load_en;
  wire mem_reader__reader_req_data_skid_valid_set_zero;
  wire [15:0] aligned_addr;
  wire [7:0] ar_len;
  wire mem_reader__axi_ar_s_data_data_is_sent_to;
  wire mem_reader__axi_ar_s_data_skid_data_load_en;
  wire mem_reader__axi_ar_s_data_skid_valid_set_zero;
  wire [15:0] next_len;
  wire [3:0] high_lane;
  wire mem_reader__rconf_data_data_is_sent_to;
  wire mem_reader__rconf_data_skid_data_load_en;
  wire mem_reader__rconf_data_skid_valid_set_zero;
  wire mem_reader__reader_err_data_data_is_sent_to;
  wire mem_reader__reader_err_data_skid_data_load_en;
  wire mem_reader__reader_err_data_skid_valid_set_zero;
  wire [51:0] mem_reader__axi_ar_s_data_select;
  wire mem_reader__axi_ar_s_data_valid_or;
  wire [47:0] mem_reader__rconf_data_select;
  wire mem_reader__rconf_data_valid_or;
  wire mem_reader__reader_err_data_select;
  wire mem_reader__reader_err_data_valid_or;
  wire [15:0] next_addr;
  wire [3:0] bit_slice_6610;
  wire [7:0] bit_slice_6611;
  wire [12:0] sel_6589;
  wire nand_6705;
  wire or_6792;
  wire [15:0] one_hot_sel_6798;
  wire or_6801;
  wire [15:0] one_hot_sel_6807;
  wire or_6810;
  wire __mem_reader__axi_ar_s_data_not_stage_load;
  wire __mem_reader__axi_ar_s_data_has_been_sent_reg_load_en;
  wire __mem_reader__rconf_data_has_been_sent_reg_load_en;
  wire __mem_reader__reader_err_data_not_stage_load;
  wire __mem_reader__reader_err_data_has_been_sent_reg_load_en;
  wire mem_reader__rresp_data_data_valid_load_en__1;
  wire mem_reader__rresp_data_skid_valid_load_en;
  wire mem_reader__reader_req_data_data_valid_load_en__1;
  wire mem_reader__reader_req_data_skid_valid_load_en;
  wire [51:0] __mem_reader__axi_ar_s_data_buf;
  wire mem_reader__axi_ar_s_data_data_valid_load_en__1;
  wire mem_reader__axi_ar_s_data_skid_valid_load_en;
  wire [47:0] __mem_reader__rconf_data_buf;
  wire mem_reader__rconf_data_data_valid_load_en__1;
  wire mem_reader__rconf_data_skid_valid_load_en;
  wire __mem_reader__reader_err_data_buf;
  wire mem_reader__reader_err_data_data_valid_load_en__1;
  wire mem_reader__reader_err_data_skid_valid_load_en;
  assign mem_reader__rresp_data_select = __mem_reader__rresp_data_valid_skid_reg ? __mem_reader__rresp_data_skid_reg : __mem_reader__rresp_data_reg;
  assign mem_reader__rresp_select = ____state_0 ? mem_reader__rresp_data_select : 1'h0;
  assign and_6681 = ____state_0 & mem_reader__rresp_select;
  assign mem_reader__reader_err_not_pred = ~and_6681;
  assign mem_reader__reader_err_data_from_skid_rdy = ~__mem_reader__reader_err_data_valid_skid_reg;
  assign mem_reader__axi_ar_s_data_from_skid_rdy = ~__mem_reader__axi_ar_s_data_valid_skid_reg;
  assign mem_reader__rconf_data_from_skid_rdy = ~__mem_reader__rconf_data_valid_skid_reg;
  assign next_len_or_exit = p2_next_len & {16{~mem_reader__rresp_select}};
  assign p3_all_active_inputs_valid = (~____state_0 | __mem_reader__rresp_data_valid_reg | __mem_reader__rresp_data_valid_skid_reg) & (____state_0 | __mem_reader__reader_req_data_valid_reg | __mem_reader__reader_req_data_valid_skid_reg);
  assign or_8419 = mem_reader__reader_err_not_pred | mem_reader__reader_err_data_from_skid_rdy | __mem_reader__reader_err_data_has_been_sent_reg;
  assign eq_6677 = next_len_or_exit == 16'h0000;
  assign p3_stage_done = p2_valid & p3_all_active_inputs_valid & or_8419;
  assign p3_not_valid = ~p2_valid;
  assign p2_stage_valid = ____state_0_full & p1_valid;
  assign p2_all_active_outputs_ready = (~____state_0 | mem_reader__axi_ar_s_data_from_skid_rdy | __mem_reader__axi_ar_s_data_has_been_sent_reg) & (~____state_0 | mem_reader__rconf_data_from_skid_rdy | __mem_reader__rconf_data_has_been_sent_reg);
  assign nor_6678 = ~(~____state_0 | eq_6677);
  assign and_6679 = ____state_0 & eq_6677;
  assign nor_6680 = ~(~____state_0 | mem_reader__rresp_select);
  assign p2_enable = p3_stage_done | p3_not_valid;
  assign p2_stage_done = p2_stage_valid & p2_all_active_outputs_ready;
  assign ____state_1__next_value_predicates = {~____state_0, nor_6678, and_6679};
  assign ____state_2__next_value_predicates = {~____state_0, nor_6680, and_6681};
  assign p2_data_enable = p2_enable & p2_stage_done;
  assign p2_not_valid = ~p1_valid;
  assign one_hot_6684 = {____state_1__next_value_predicates[2:0] == 3'h0, ____state_1__next_value_predicates[2] && ____state_1__next_value_predicates[1:0] == 2'h0, ____state_1__next_value_predicates[1] && !____state_1__next_value_predicates[0], ____state_1__next_value_predicates[0]};
  assign one_hot_6685 = {____state_2__next_value_predicates[2:0] == 3'h0, ____state_2__next_value_predicates[2] && ____state_2__next_value_predicates[1:0] == 2'h0, ____state_2__next_value_predicates[1] && !____state_2__next_value_predicates[0], ____state_2__next_value_predicates[0]};
  assign concat_6604 = {3'h0, p0_sel_6589};
  assign aligned_offset = ____state_1[3:0];
  assign p1_enable = p2_data_enable | p2_not_valid;
  assign p1_stage_valid = ____state_2_full & p0_valid;
  assign __mem_reader__axi_ar_s_vld_buf = p2_stage_valid & p2_enable & ____state_0;
  assign __mem_reader__axi_ar_s_data_not_has_been_sent = ~__mem_reader__axi_ar_s_data_has_been_sent_reg;
  assign __mem_reader__rconf_data_not_has_been_sent = ~__mem_reader__rconf_data_has_been_sent_reg;
  assign __mem_reader__reader_err_vld_buf = p3_all_active_inputs_valid & p2_valid & and_6681;
  assign __mem_reader__reader_err_data_not_has_been_sent = ~__mem_reader__reader_err_data_has_been_sent_reg;
  assign mem_reader__reader_req_data_select = __mem_reader__reader_req_data_valid_skid_reg ? __mem_reader__reader_req_data_skid_reg : __mem_reader__reader_req_data_reg;
  assign p1_data_enable = p1_enable & p1_stage_valid;
  assign p1_not_valid = ~p0_valid;
  assign mem_reader__rresp_data_from_skid_rdy = ~__mem_reader__rresp_data_valid_skid_reg;
  assign and_6815 = ____state_0 & p3_stage_done;
  assign mem_reader__reader_req_data_from_skid_rdy = ~__mem_reader__reader_req_data_valid_skid_reg;
  assign and_6816 = ~____state_0 & p3_stage_done;
  assign __mem_reader__axi_ar_s_data_valid_and_not_has_been_sent = __mem_reader__axi_ar_s_vld_buf & __mem_reader__axi_ar_s_data_not_has_been_sent;
  assign tran_len_mod = p1_tran_len[3:0];
  assign __mem_reader__rconf_data_valid_and_not_has_been_sent = __mem_reader__axi_ar_s_vld_buf & __mem_reader__rconf_data_not_has_been_sent;
  assign __mem_reader__reader_err_data_valid_and_not_has_been_sent = __mem_reader__reader_err_vld_buf & __mem_reader__reader_err_data_not_has_been_sent;
  assign tran_len = ____state_2 < concat_6604 ? ____state_2 : concat_6604;
  assign bytes_to_4k_squeezed = 13'h1000 - {1'h0, ____state_1[11:0]};
  assign bytes_to_max_burst_squeezed = 13'h1000 - {9'h000, aligned_offset};
  assign and_6795 = nor_6678 & p3_stage_done;
  assign and_6796 = and_6679 & p3_stage_done;
  assign mem_reader__reader_req_select = ~____state_0 ? mem_reader__reader_req_data_select : literal_6687;
  assign p0_enable = p1_data_enable | p1_not_valid;
  assign and_6804 = nor_6680 & p3_stage_done;
  assign and_6805 = and_6681 & p3_stage_done;
  assign mem_reader__rresp_data_data_valid_load_en = mem_reader__rresp_vld & mem_reader__rresp_data_from_skid_rdy;
  assign mem_reader__rresp_data_to_is_not_rdy = ~and_6815;
  assign mem_reader__reader_req_data_data_valid_load_en = mem_reader__reader_req_vld & mem_reader__reader_req_data_from_skid_rdy;
  assign mem_reader__reader_req_data_to_is_not_rdy = ~and_6816;
  assign rest = p1_bit_slice_6610 != 4'h0;
  assign add_6634 = p1_bit_slice_6611 + 8'hff;
  assign mem_reader__axi_ar_s_data_data_valid_load_en = __mem_reader__axi_ar_s_data_valid_and_not_has_been_sent & mem_reader__axi_ar_s_data_from_skid_rdy;
  assign mem_reader__axi_ar_s_data_to_is_not_rdy = ~mem_reader__axi_ar_s_rdy;
  assign add_6635 = p0_aligned_offset + tran_len_mod;
  assign MAX_LANE__1 = 4'hf;
  assign mem_reader__rconf_data_data_valid_load_en = __mem_reader__rconf_data_valid_and_not_has_been_sent & mem_reader__rconf_data_from_skid_rdy;
  assign mem_reader__rconf_data_to_is_not_rdy = ~mem_reader__rconf_rdy;
  assign mem_reader__reader_err_data_data_valid_load_en = __mem_reader__reader_err_data_valid_and_not_has_been_sent & mem_reader__reader_err_data_from_skid_rdy;
  assign mem_reader__reader_err_data_to_is_not_rdy = ~mem_reader__reader_err_rdy;
  assign ____state_1__at_most_one_next_value = ~____state_0 == one_hot_6684[2] & nor_6678 == one_hot_6684[1] & and_6679 == one_hot_6684[0];
  assign ____state_2__at_most_one_next_value = ~____state_0 == one_hot_6685[2] & nor_6680 == one_hot_6685[1] & and_6681 == one_hot_6685[0];
  assign adjusted_tran_len = {12'h000, p0_aligned_offset} + tran_len;
  assign concat_6797 = {and_6816, and_6795, and_6796};
  assign req_addr = mem_reader__reader_req_select[31:16];
  assign p0_data_enable = p0_enable & ____state_1_full;
  assign or_6799 = and_6816 | and_6795 | and_6796;
  assign concat_6806 = {and_6816, and_6804, and_6805};
  assign req_len = mem_reader__reader_req_select[15:0];
  assign or_6808 = and_6816 | and_6804 | and_6805;
  assign __mem_reader__axi_ar_s_data_valid_and_all_active_outputs_ready = __mem_reader__axi_ar_s_vld_buf & p2_all_active_outputs_ready;
  assign __mem_reader__reader_err_data_valid_and_all_active_outputs_ready = __mem_reader__reader_err_vld_buf & or_8419;
  assign mem_reader__rresp_data_data_is_sent_to = __mem_reader__rresp_data_valid_reg & and_6815 & mem_reader__rresp_data_from_skid_rdy;
  assign mem_reader__rresp_data_skid_data_load_en = __mem_reader__rresp_data_valid_reg & mem_reader__rresp_data_data_valid_load_en & mem_reader__rresp_data_to_is_not_rdy;
  assign mem_reader__rresp_data_skid_valid_set_zero = __mem_reader__rresp_data_valid_skid_reg & and_6815;
  assign mem_reader__reader_req_data_data_is_sent_to = __mem_reader__reader_req_data_valid_reg & and_6816 & mem_reader__reader_req_data_from_skid_rdy;
  assign mem_reader__reader_req_data_skid_data_load_en = __mem_reader__reader_req_data_valid_reg & mem_reader__reader_req_data_data_valid_load_en & mem_reader__reader_req_data_to_is_not_rdy;
  assign mem_reader__reader_req_data_skid_valid_set_zero = __mem_reader__reader_req_data_valid_skid_reg & and_6816;
  assign aligned_addr = {____state_1[15:4], 4'h0};
  assign ar_len = rest ? p1_bit_slice_6611 : add_6634;
  assign mem_reader__axi_ar_s_data_data_is_sent_to = __mem_reader__axi_ar_s_data_valid_reg & mem_reader__axi_ar_s_rdy & mem_reader__axi_ar_s_data_from_skid_rdy;
  assign mem_reader__axi_ar_s_data_skid_data_load_en = __mem_reader__axi_ar_s_data_valid_reg & mem_reader__axi_ar_s_data_data_valid_load_en & mem_reader__axi_ar_s_data_to_is_not_rdy;
  assign mem_reader__axi_ar_s_data_skid_valid_set_zero = __mem_reader__axi_ar_s_data_valid_skid_reg & mem_reader__axi_ar_s_rdy;
  assign next_len = ____state_2 - p1_tran_len;
  assign high_lane = add_6635 + MAX_LANE__1;
  assign mem_reader__rconf_data_data_is_sent_to = __mem_reader__rconf_data_valid_reg & mem_reader__rconf_rdy & mem_reader__rconf_data_from_skid_rdy;
  assign mem_reader__rconf_data_skid_data_load_en = __mem_reader__rconf_data_valid_reg & mem_reader__rconf_data_data_valid_load_en & mem_reader__rconf_data_to_is_not_rdy;
  assign mem_reader__rconf_data_skid_valid_set_zero = __mem_reader__rconf_data_valid_skid_reg & mem_reader__rconf_rdy;
  assign mem_reader__reader_err_data_data_is_sent_to = __mem_reader__reader_err_data_valid_reg & mem_reader__reader_err_rdy & mem_reader__reader_err_data_from_skid_rdy;
  assign mem_reader__reader_err_data_skid_data_load_en = __mem_reader__reader_err_data_valid_reg & mem_reader__reader_err_data_data_valid_load_en & mem_reader__reader_err_data_to_is_not_rdy;
  assign mem_reader__reader_err_data_skid_valid_set_zero = __mem_reader__reader_err_data_valid_skid_reg & mem_reader__reader_err_rdy;
  assign mem_reader__axi_ar_s_data_select = __mem_reader__axi_ar_s_data_valid_skid_reg ? __mem_reader__axi_ar_s_data_skid_reg : __mem_reader__axi_ar_s_data_reg;
  assign mem_reader__axi_ar_s_data_valid_or = __mem_reader__axi_ar_s_data_valid_reg | __mem_reader__axi_ar_s_data_valid_skid_reg;
  assign mem_reader__rconf_data_select = __mem_reader__rconf_data_valid_skid_reg ? __mem_reader__rconf_data_skid_reg : __mem_reader__rconf_data_reg;
  assign mem_reader__rconf_data_valid_or = __mem_reader__rconf_data_valid_reg | __mem_reader__rconf_data_valid_skid_reg;
  assign mem_reader__reader_err_data_select = __mem_reader__reader_err_data_valid_skid_reg ? __mem_reader__reader_err_data_skid_reg : __mem_reader__reader_err_data_reg;
  assign mem_reader__reader_err_data_valid_or = __mem_reader__reader_err_data_valid_reg | __mem_reader__reader_err_data_valid_skid_reg;
  assign next_addr = ____state_1 + p1_tran_len;
  assign bit_slice_6610 = adjusted_tran_len[3:0];
  assign bit_slice_6611 = adjusted_tran_len[11:4];
  assign sel_6589 = bytes_to_4k_squeezed < bytes_to_max_burst_squeezed ? bytes_to_4k_squeezed : bytes_to_max_burst_squeezed;
  assign nand_6705 = ~(____state_0 & eq_6677);
  assign or_6792 = p2_data_enable | p3_stage_done;
  assign one_hot_sel_6798 = 16'h0000 & {16{concat_6797[0]}} | p2_next_addr & {16{concat_6797[1]}} | req_addr & {16{concat_6797[2]}};
  assign or_6801 = p0_data_enable | or_6799;
  assign one_hot_sel_6807 = 16'h0000 & {16{concat_6806[0]}} | p2_next_len & {16{concat_6806[1]}} | req_len & {16{concat_6806[2]}};
  assign or_6810 = p1_data_enable | or_6808;
  assign __mem_reader__axi_ar_s_data_not_stage_load = ~__mem_reader__axi_ar_s_data_valid_and_all_active_outputs_ready;
  assign __mem_reader__axi_ar_s_data_has_been_sent_reg_load_en = mem_reader__axi_ar_s_data_data_valid_load_en | __mem_reader__axi_ar_s_data_valid_and_all_active_outputs_ready;
  assign __mem_reader__rconf_data_has_been_sent_reg_load_en = mem_reader__rconf_data_data_valid_load_en | __mem_reader__axi_ar_s_data_valid_and_all_active_outputs_ready;
  assign __mem_reader__reader_err_data_not_stage_load = ~__mem_reader__reader_err_data_valid_and_all_active_outputs_ready;
  assign __mem_reader__reader_err_data_has_been_sent_reg_load_en = mem_reader__reader_err_data_data_valid_load_en | __mem_reader__reader_err_data_valid_and_all_active_outputs_ready;
  assign mem_reader__rresp_data_data_valid_load_en__1 = mem_reader__rresp_data_data_is_sent_to | mem_reader__rresp_data_data_valid_load_en;
  assign mem_reader__rresp_data_skid_valid_load_en = mem_reader__rresp_data_skid_data_load_en | mem_reader__rresp_data_skid_valid_set_zero;
  assign mem_reader__reader_req_data_data_valid_load_en__1 = mem_reader__reader_req_data_data_is_sent_to | mem_reader__reader_req_data_data_valid_load_en;
  assign mem_reader__reader_req_data_skid_valid_load_en = mem_reader__reader_req_data_skid_data_load_en | mem_reader__reader_req_data_skid_valid_set_zero;
  assign __mem_reader__axi_ar_s_data_buf = {8'h00, aligned_addr, 4'h0, ar_len, 3'h4, 2'h1, 4'h0, 3'h0, 4'h0};
  assign mem_reader__axi_ar_s_data_data_valid_load_en__1 = mem_reader__axi_ar_s_data_data_is_sent_to | mem_reader__axi_ar_s_data_data_valid_load_en;
  assign mem_reader__axi_ar_s_data_skid_valid_load_en = mem_reader__axi_ar_s_data_skid_data_load_en | mem_reader__axi_ar_s_data_skid_valid_set_zero;
  assign __mem_reader__rconf_data_buf = {aligned_addr, next_len, ar_len, p0_aligned_offset, high_lane};
  assign mem_reader__rconf_data_data_valid_load_en__1 = mem_reader__rconf_data_data_is_sent_to | mem_reader__rconf_data_data_valid_load_en;
  assign mem_reader__rconf_data_skid_valid_load_en = mem_reader__rconf_data_skid_data_load_en | mem_reader__rconf_data_skid_valid_set_zero;
  assign __mem_reader__reader_err_data_buf = 1'h0;
  assign mem_reader__reader_err_data_data_valid_load_en__1 = mem_reader__reader_err_data_data_is_sent_to | mem_reader__reader_err_data_data_valid_load_en;
  assign mem_reader__reader_err_data_skid_valid_load_en = mem_reader__reader_err_data_skid_data_load_en | mem_reader__reader_err_data_skid_valid_set_zero;
  always @ (posedge clk) begin
    if (rst) begin
      ____state_1 <= 16'h0000;
      p0_aligned_offset <= 4'h0;
      p0_sel_6589 <= 13'h0000;
      ____state_2 <= 16'h0000;
      p1_tran_len <= 16'h0000;
      p1_bit_slice_6610 <= 4'h0;
      p1_bit_slice_6611 <= 8'h00;
      ____state_0 <= 1'h0;
      p2_next_len <= 16'h0000;
      p2_next_addr <= 16'h0000;
      ____state_0_full <= 1'h1;
      ____state_1_full <= 1'h1;
      ____state_2_full <= 1'h1;
      p0_valid <= 1'h0;
      p1_valid <= 1'h0;
      p2_valid <= 1'h0;
      __mem_reader__axi_ar_s_data_has_been_sent_reg <= 1'h0;
      __mem_reader__rconf_data_has_been_sent_reg <= 1'h0;
      __mem_reader__reader_err_data_has_been_sent_reg <= 1'h0;
      __mem_reader__rresp_data_reg <= 1'h0;
      __mem_reader__rresp_data_skid_reg <= 1'h0;
      __mem_reader__rresp_data_valid_reg <= 1'h0;
      __mem_reader__rresp_data_valid_skid_reg <= 1'h0;
      __mem_reader__reader_req_data_reg <= __mem_reader__reader_req_data_reg_init;
      __mem_reader__reader_req_data_skid_reg <= __mem_reader__reader_req_data_skid_reg_init;
      __mem_reader__reader_req_data_valid_reg <= 1'h0;
      __mem_reader__reader_req_data_valid_skid_reg <= 1'h0;
      __mem_reader__axi_ar_s_data_reg <= __mem_reader__axi_ar_s_data_reg_init;
      __mem_reader__axi_ar_s_data_skid_reg <= __mem_reader__axi_ar_s_data_skid_reg_init;
      __mem_reader__axi_ar_s_data_valid_reg <= 1'h0;
      __mem_reader__axi_ar_s_data_valid_skid_reg <= 1'h0;
      __mem_reader__rconf_data_reg <= __mem_reader__rconf_data_reg_init;
      __mem_reader__rconf_data_skid_reg <= __mem_reader__rconf_data_skid_reg_init;
      __mem_reader__rconf_data_valid_reg <= 1'h0;
      __mem_reader__rconf_data_valid_skid_reg <= 1'h0;
      __mem_reader__reader_err_data_reg <= 1'h0;
      __mem_reader__reader_err_data_skid_reg <= 1'h0;
      __mem_reader__reader_err_data_valid_reg <= 1'h0;
      __mem_reader__reader_err_data_valid_skid_reg <= 1'h0;
    end else begin
      ____state_1 <= or_6799 ? one_hot_sel_6798 : ____state_1;
      p0_aligned_offset <= p0_data_enable ? aligned_offset : p0_aligned_offset;
      p0_sel_6589 <= p0_data_enable ? sel_6589 : p0_sel_6589;
      ____state_2 <= or_6808 ? one_hot_sel_6807 : ____state_2;
      p1_tran_len <= p1_data_enable ? tran_len : p1_tran_len;
      p1_bit_slice_6610 <= p1_data_enable ? bit_slice_6610 : p1_bit_slice_6610;
      p1_bit_slice_6611 <= p1_data_enable ? bit_slice_6611 : p1_bit_slice_6611;
      ____state_0 <= p3_stage_done ? nand_6705 : ____state_0;
      p2_next_len <= p2_data_enable ? next_len : p2_next_len;
      p2_next_addr <= p2_data_enable ? next_addr : p2_next_addr;
      ____state_0_full <= or_6792 ? p3_stage_done : ____state_0_full;
      ____state_1_full <= or_6801 ? or_6799 : ____state_1_full;
      ____state_2_full <= or_6810 ? or_6808 : ____state_2_full;
      p0_valid <= p0_enable ? ____state_1_full : p0_valid;
      p1_valid <= p1_enable ? p1_stage_valid : p1_valid;
      p2_valid <= p2_enable ? p2_stage_done : p2_valid;
      __mem_reader__axi_ar_s_data_has_been_sent_reg <= __mem_reader__axi_ar_s_data_has_been_sent_reg_load_en ? __mem_reader__axi_ar_s_data_not_stage_load : __mem_reader__axi_ar_s_data_has_been_sent_reg;
      __mem_reader__rconf_data_has_been_sent_reg <= __mem_reader__rconf_data_has_been_sent_reg_load_en ? __mem_reader__axi_ar_s_data_not_stage_load : __mem_reader__rconf_data_has_been_sent_reg;
      __mem_reader__reader_err_data_has_been_sent_reg <= __mem_reader__reader_err_data_has_been_sent_reg_load_en ? __mem_reader__reader_err_data_not_stage_load : __mem_reader__reader_err_data_has_been_sent_reg;
      __mem_reader__rresp_data_reg <= mem_reader__rresp_data_data_valid_load_en ? mem_reader__rresp_data : __mem_reader__rresp_data_reg;
      __mem_reader__rresp_data_skid_reg <= mem_reader__rresp_data_skid_data_load_en ? __mem_reader__rresp_data_reg : __mem_reader__rresp_data_skid_reg;
      __mem_reader__rresp_data_valid_reg <= mem_reader__rresp_data_data_valid_load_en__1 ? mem_reader__rresp_vld : __mem_reader__rresp_data_valid_reg;
      __mem_reader__rresp_data_valid_skid_reg <= mem_reader__rresp_data_skid_valid_load_en ? mem_reader__rresp_data_from_skid_rdy : __mem_reader__rresp_data_valid_skid_reg;
      __mem_reader__reader_req_data_reg <= mem_reader__reader_req_data_data_valid_load_en ? mem_reader__reader_req_data : __mem_reader__reader_req_data_reg;
      __mem_reader__reader_req_data_skid_reg <= mem_reader__reader_req_data_skid_data_load_en ? __mem_reader__reader_req_data_reg : __mem_reader__reader_req_data_skid_reg;
      __mem_reader__reader_req_data_valid_reg <= mem_reader__reader_req_data_data_valid_load_en__1 ? mem_reader__reader_req_vld : __mem_reader__reader_req_data_valid_reg;
      __mem_reader__reader_req_data_valid_skid_reg <= mem_reader__reader_req_data_skid_valid_load_en ? mem_reader__reader_req_data_from_skid_rdy : __mem_reader__reader_req_data_valid_skid_reg;
      __mem_reader__axi_ar_s_data_reg <= mem_reader__axi_ar_s_data_data_valid_load_en ? __mem_reader__axi_ar_s_data_buf : __mem_reader__axi_ar_s_data_reg;
      __mem_reader__axi_ar_s_data_skid_reg <= mem_reader__axi_ar_s_data_skid_data_load_en ? __mem_reader__axi_ar_s_data_reg : __mem_reader__axi_ar_s_data_skid_reg;
      __mem_reader__axi_ar_s_data_valid_reg <= mem_reader__axi_ar_s_data_data_valid_load_en__1 ? __mem_reader__axi_ar_s_data_valid_and_not_has_been_sent : __mem_reader__axi_ar_s_data_valid_reg;
      __mem_reader__axi_ar_s_data_valid_skid_reg <= mem_reader__axi_ar_s_data_skid_valid_load_en ? mem_reader__axi_ar_s_data_from_skid_rdy : __mem_reader__axi_ar_s_data_valid_skid_reg;
      __mem_reader__rconf_data_reg <= mem_reader__rconf_data_data_valid_load_en ? __mem_reader__rconf_data_buf : __mem_reader__rconf_data_reg;
      __mem_reader__rconf_data_skid_reg <= mem_reader__rconf_data_skid_data_load_en ? __mem_reader__rconf_data_reg : __mem_reader__rconf_data_skid_reg;
      __mem_reader__rconf_data_valid_reg <= mem_reader__rconf_data_data_valid_load_en__1 ? __mem_reader__rconf_data_valid_and_not_has_been_sent : __mem_reader__rconf_data_valid_reg;
      __mem_reader__rconf_data_valid_skid_reg <= mem_reader__rconf_data_skid_valid_load_en ? mem_reader__rconf_data_from_skid_rdy : __mem_reader__rconf_data_valid_skid_reg;
      __mem_reader__reader_err_data_reg <= mem_reader__reader_err_data_data_valid_load_en ? __mem_reader__reader_err_data_buf : __mem_reader__reader_err_data_reg;
      __mem_reader__reader_err_data_skid_reg <= mem_reader__reader_err_data_skid_data_load_en ? __mem_reader__reader_err_data_reg : __mem_reader__reader_err_data_skid_reg;
      __mem_reader__reader_err_data_valid_reg <= mem_reader__reader_err_data_data_valid_load_en__1 ? __mem_reader__reader_err_data_valid_and_not_has_been_sent : __mem_reader__reader_err_data_valid_reg;
      __mem_reader__reader_err_data_valid_skid_reg <= mem_reader__reader_err_data_skid_valid_load_en ? mem_reader__reader_err_data_from_skid_rdy : __mem_reader__reader_err_data_valid_skid_reg;
    end
  end
  assign mem_reader__axi_ar_s_data = mem_reader__axi_ar_s_data_select;
  assign mem_reader__axi_ar_s_vld = mem_reader__axi_ar_s_data_valid_or;
  assign mem_reader__rconf_data = mem_reader__rconf_data_select;
  assign mem_reader__rconf_vld = mem_reader__rconf_data_valid_or;
  assign mem_reader__reader_err_data = mem_reader__reader_err_data_select;
  assign mem_reader__reader_err_vld = mem_reader__reader_err_data_valid_or;
  assign mem_reader__reader_req_rdy = mem_reader__reader_req_data_from_skid_rdy;
  assign mem_reader__rresp_rdy = mem_reader__rresp_data_from_skid_rdy;
endmodule


module __xls_modules_zstd_memory_axi_stream_downscaler__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiStreamDownscaler_0__8_8_128_16_64_8_2_2_next(
  input wire clk,
  input wire rst,
  input wire [176:0] mem_reader__axi_st_in_data,
  input wire mem_reader__axi_st_in_vld,
  input wire mem_reader__axi_st_remove_rdy,
  output wire mem_reader__axi_st_in_rdy,
  output wire [96:0] mem_reader__axi_st_remove_data,
  output wire mem_reader__axi_st_remove_vld
);
  wire [176:0] __mem_reader__axi_st_in_data_reg_init = {128'h0000_0000_0000_0000_0000_0000_0000_0000, 16'h0000, 16'h0000, 8'h00, 8'h00, 1'h0};
  wire [176:0] __mem_reader__axi_st_in_data_skid_reg_init = {128'h0000_0000_0000_0000_0000_0000_0000_0000, 16'h0000, 16'h0000, 8'h00, 8'h00, 1'h0};
  wire [96:0] __mem_reader__axi_st_remove_data_reg_init = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  wire [96:0] __mem_reader__axi_st_remove_data_skid_reg_init = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  wire [176:0] literal_7480 = {128'h0000_0000_0000_0000_0000_0000_0000_0000, 16'h0000, 16'h0000, 8'h00, 8'h00, 1'h0};
  reg ____state_6;
  reg [127:0] ____state_0;
  reg [15:0] ____state_1;
  reg [15:0] ____state_2;
  reg ____state_5;
  reg [7:0] ____state_3;
  reg [7:0] ____state_4;
  reg [176:0] __mem_reader__axi_st_in_data_reg;
  reg [176:0] __mem_reader__axi_st_in_data_skid_reg;
  reg __mem_reader__axi_st_in_data_valid_reg;
  reg __mem_reader__axi_st_in_data_valid_skid_reg;
  reg [96:0] __mem_reader__axi_st_remove_data_reg;
  reg [96:0] __mem_reader__axi_st_remove_data_skid_reg;
  reg __mem_reader__axi_st_remove_data_valid_reg;
  reg __mem_reader__axi_st_remove_data_valid_skid_reg;
  wire do_recv;
  wire [1:0] ____state_0__next_value_predicates;
  wire [1:0] ____state_6__next_value_predicates;
  wire [176:0] mem_reader__axi_st_in_data_select;
  wire __mem_reader__axi_st_remove_vld_buf;
  wire mem_reader__axi_st_remove_data_from_skid_rdy;
  wire [2:0] one_hot_7494;
  wire [2:0] one_hot_7495;
  wire [176:0] mem_reader__axi_st_in_select;
  wire p0_stage_done;
  wire mem_reader__axi_st_in_data_from_skid_rdy;
  wire and_7613;
  wire [127:0] tuple_index_7486;
  wire [15:0] tuple_index_7488;
  wire [15:0] tuple_index_7490;
  wire and_7555;
  wire and_7556;
  wire in_data_last;
  wire mem_reader__axi_st_in_data_data_valid_load_en;
  wire mem_reader__axi_st_in_data_to_is_not_rdy;
  wire [7:0] tuple_index_7502;
  wire [7:0] tuple_index_7504;
  wire mem_reader__axi_st_remove_data_data_valid_load_en;
  wire mem_reader__axi_st_remove_data_to_is_not_rdy;
  wire ____state_0__at_most_one_next_value;
  wire ____state_6__at_most_one_next_value;
  wire [1:0] concat_7557;
  wire [1:0] concat_7588;
  wire unexpand_for_next_value_605_6_case_1;
  wire unexpand_for_next_value_605_6_case_0;
  wire mem_reader__axi_st_in_data_data_is_sent_to;
  wire mem_reader__axi_st_in_data_skid_data_load_en;
  wire mem_reader__axi_st_in_data_skid_valid_set_zero;
  wire [63:0] data;
  wire [7:0] str;
  wire [7:0] keep;
  wire [7:0] id;
  wire [7:0] dest;
  wire last;
  wire mem_reader__axi_st_remove_data_data_is_sent_to;
  wire mem_reader__axi_st_remove_data_skid_data_load_en;
  wire mem_reader__axi_st_remove_data_skid_valid_set_zero;
  wire [96:0] mem_reader__axi_st_remove_data_select;
  wire mem_reader__axi_st_remove_data_valid_or;
  wire or_8385;
  wire [127:0] one_hot_sel_7558;
  wire [15:0] one_hot_sel_7564;
  wire [15:0] one_hot_sel_7570;
  wire [7:0] one_hot_sel_7576;
  wire [7:0] one_hot_sel_7582;
  wire nor_7524;
  wire one_hot_sel_7589;
  wire mem_reader__axi_st_in_data_data_valid_load_en__1;
  wire mem_reader__axi_st_in_data_skid_valid_load_en;
  wire [96:0] out_data;
  wire mem_reader__axi_st_remove_data_data_valid_load_en__1;
  wire mem_reader__axi_st_remove_data_skid_valid_load_en;
  assign do_recv = ~____state_6;
  assign ____state_0__next_value_predicates = {____state_6, do_recv};
  assign ____state_6__next_value_predicates = {do_recv, ____state_6};
  assign mem_reader__axi_st_in_data_select = __mem_reader__axi_st_in_data_valid_skid_reg ? __mem_reader__axi_st_in_data_skid_reg : __mem_reader__axi_st_in_data_reg;
  assign __mem_reader__axi_st_remove_vld_buf = ____state_6 | __mem_reader__axi_st_in_data_valid_reg | __mem_reader__axi_st_in_data_valid_skid_reg;
  assign mem_reader__axi_st_remove_data_from_skid_rdy = ~__mem_reader__axi_st_remove_data_valid_skid_reg;
  assign one_hot_7494 = {____state_0__next_value_predicates[1:0] == 2'h0, ____state_0__next_value_predicates[1] && !____state_0__next_value_predicates[0], ____state_0__next_value_predicates[0]};
  assign one_hot_7495 = {____state_6__next_value_predicates[1:0] == 2'h0, ____state_6__next_value_predicates[1] && !____state_6__next_value_predicates[0], ____state_6__next_value_predicates[0]};
  assign mem_reader__axi_st_in_select = do_recv ? mem_reader__axi_st_in_data_select : literal_7480;
  assign p0_stage_done = __mem_reader__axi_st_remove_vld_buf & mem_reader__axi_st_remove_data_from_skid_rdy;
  assign mem_reader__axi_st_in_data_from_skid_rdy = ~__mem_reader__axi_st_in_data_valid_skid_reg;
  assign and_7613 = do_recv & p0_stage_done;
  assign tuple_index_7486 = mem_reader__axi_st_in_select[176:49];
  assign tuple_index_7488 = mem_reader__axi_st_in_select[48:33];
  assign tuple_index_7490 = mem_reader__axi_st_in_select[32:17];
  assign and_7555 = ____state_6 & p0_stage_done;
  assign and_7556 = do_recv & p0_stage_done;
  assign in_data_last = ____state_6 ? ____state_5 : mem_reader__axi_st_in_select[0:0];
  assign mem_reader__axi_st_in_data_data_valid_load_en = mem_reader__axi_st_in_vld & mem_reader__axi_st_in_data_from_skid_rdy;
  assign mem_reader__axi_st_in_data_to_is_not_rdy = ~and_7613;
  assign tuple_index_7502 = mem_reader__axi_st_in_select[16:9];
  assign tuple_index_7504 = mem_reader__axi_st_in_select[8:1];
  assign mem_reader__axi_st_remove_data_data_valid_load_en = __mem_reader__axi_st_remove_vld_buf & mem_reader__axi_st_remove_data_from_skid_rdy;
  assign mem_reader__axi_st_remove_data_to_is_not_rdy = ~mem_reader__axi_st_remove_rdy;
  assign ____state_0__at_most_one_next_value = ____state_6 == one_hot_7494[1] & do_recv == one_hot_7494[0];
  assign ____state_6__at_most_one_next_value = do_recv == one_hot_7495[1] & ____state_6 == one_hot_7495[0];
  assign concat_7557 = {and_7555, and_7556};
  assign concat_7588 = {and_7556, and_7555};
  assign unexpand_for_next_value_605_6_case_1 = 1'h0;
  assign unexpand_for_next_value_605_6_case_0 = 1'h1;
  assign mem_reader__axi_st_in_data_data_is_sent_to = __mem_reader__axi_st_in_data_valid_reg & and_7613 & mem_reader__axi_st_in_data_from_skid_rdy;
  assign mem_reader__axi_st_in_data_skid_data_load_en = __mem_reader__axi_st_in_data_valid_reg & mem_reader__axi_st_in_data_data_valid_load_en & mem_reader__axi_st_in_data_to_is_not_rdy;
  assign mem_reader__axi_st_in_data_skid_valid_set_zero = __mem_reader__axi_st_in_data_valid_skid_reg & and_7613;
  assign data = ____state_6 ? ____state_0[127:64] : tuple_index_7486[63:0];
  assign str = ____state_6 ? ____state_1[15:8] : tuple_index_7488[7:0];
  assign keep = ____state_6 ? ____state_2[15:8] : tuple_index_7490[7:0];
  assign id = ____state_6 ? ____state_3 : tuple_index_7502;
  assign dest = ____state_6 ? ____state_4 : tuple_index_7504;
  assign last = ____state_6 & in_data_last;
  assign mem_reader__axi_st_remove_data_data_is_sent_to = __mem_reader__axi_st_remove_data_valid_reg & mem_reader__axi_st_remove_rdy & mem_reader__axi_st_remove_data_from_skid_rdy;
  assign mem_reader__axi_st_remove_data_skid_data_load_en = __mem_reader__axi_st_remove_data_valid_reg & mem_reader__axi_st_remove_data_data_valid_load_en & mem_reader__axi_st_remove_data_to_is_not_rdy;
  assign mem_reader__axi_st_remove_data_skid_valid_set_zero = __mem_reader__axi_st_remove_data_valid_skid_reg & mem_reader__axi_st_remove_rdy;
  assign mem_reader__axi_st_remove_data_select = __mem_reader__axi_st_remove_data_valid_skid_reg ? __mem_reader__axi_st_remove_data_skid_reg : __mem_reader__axi_st_remove_data_reg;
  assign mem_reader__axi_st_remove_data_valid_or = __mem_reader__axi_st_remove_data_valid_reg | __mem_reader__axi_st_remove_data_valid_skid_reg;
  assign or_8385 = ~p0_stage_done | ____state_0__at_most_one_next_value | rst;
  assign one_hot_sel_7558 = tuple_index_7486 & {128{concat_7557[0]}} | 128'h0000_0000_0000_0000_0000_0000_0000_0000 & {128{concat_7557[1]}};
  assign one_hot_sel_7564 = tuple_index_7488 & {16{concat_7557[0]}} | 16'h0000 & {16{concat_7557[1]}};
  assign one_hot_sel_7570 = tuple_index_7490 & {16{concat_7557[0]}} | 16'h0000 & {16{concat_7557[1]}};
  assign one_hot_sel_7576 = tuple_index_7502 & {8{concat_7557[0]}} | 8'h00 & {8{concat_7557[1]}};
  assign one_hot_sel_7582 = tuple_index_7504 & {8{concat_7557[0]}} | 8'h00 & {8{concat_7557[1]}};
  assign nor_7524 = ~(____state_6 | ~in_data_last);
  assign one_hot_sel_7589 = unexpand_for_next_value_605_6_case_1 & concat_7588[0] | unexpand_for_next_value_605_6_case_0 & concat_7588[1];
  assign mem_reader__axi_st_in_data_data_valid_load_en__1 = mem_reader__axi_st_in_data_data_is_sent_to | mem_reader__axi_st_in_data_data_valid_load_en;
  assign mem_reader__axi_st_in_data_skid_valid_load_en = mem_reader__axi_st_in_data_skid_data_load_en | mem_reader__axi_st_in_data_skid_valid_set_zero;
  assign out_data = {data, str, keep, id, dest, last};
  assign mem_reader__axi_st_remove_data_data_valid_load_en__1 = mem_reader__axi_st_remove_data_data_is_sent_to | mem_reader__axi_st_remove_data_data_valid_load_en;
  assign mem_reader__axi_st_remove_data_skid_valid_load_en = mem_reader__axi_st_remove_data_skid_data_load_en | mem_reader__axi_st_remove_data_skid_valid_set_zero;
  always @ (posedge clk) begin
    if (rst) begin
      ____state_6 <= 1'h0;
      ____state_0 <= 128'h0000_0000_0000_0000_0000_0000_0000_0000;
      ____state_1 <= 16'h0000;
      ____state_2 <= 16'h0000;
      ____state_5 <= 1'h0;
      ____state_3 <= 8'h00;
      ____state_4 <= 8'h00;
      __mem_reader__axi_st_in_data_reg <= __mem_reader__axi_st_in_data_reg_init;
      __mem_reader__axi_st_in_data_skid_reg <= __mem_reader__axi_st_in_data_skid_reg_init;
      __mem_reader__axi_st_in_data_valid_reg <= 1'h0;
      __mem_reader__axi_st_in_data_valid_skid_reg <= 1'h0;
      __mem_reader__axi_st_remove_data_reg <= __mem_reader__axi_st_remove_data_reg_init;
      __mem_reader__axi_st_remove_data_skid_reg <= __mem_reader__axi_st_remove_data_skid_reg_init;
      __mem_reader__axi_st_remove_data_valid_reg <= 1'h0;
      __mem_reader__axi_st_remove_data_valid_skid_reg <= 1'h0;
    end else begin
      ____state_6 <= p0_stage_done ? one_hot_sel_7589 : ____state_6;
      ____state_0 <= p0_stage_done ? one_hot_sel_7558 : ____state_0;
      ____state_1 <= p0_stage_done ? one_hot_sel_7564 : ____state_1;
      ____state_2 <= p0_stage_done ? one_hot_sel_7570 : ____state_2;
      ____state_5 <= p0_stage_done ? nor_7524 : ____state_5;
      ____state_3 <= p0_stage_done ? one_hot_sel_7576 : ____state_3;
      ____state_4 <= p0_stage_done ? one_hot_sel_7582 : ____state_4;
      __mem_reader__axi_st_in_data_reg <= mem_reader__axi_st_in_data_data_valid_load_en ? mem_reader__axi_st_in_data : __mem_reader__axi_st_in_data_reg;
      __mem_reader__axi_st_in_data_skid_reg <= mem_reader__axi_st_in_data_skid_data_load_en ? __mem_reader__axi_st_in_data_reg : __mem_reader__axi_st_in_data_skid_reg;
      __mem_reader__axi_st_in_data_valid_reg <= mem_reader__axi_st_in_data_data_valid_load_en__1 ? mem_reader__axi_st_in_vld : __mem_reader__axi_st_in_data_valid_reg;
      __mem_reader__axi_st_in_data_valid_skid_reg <= mem_reader__axi_st_in_data_skid_valid_load_en ? mem_reader__axi_st_in_data_from_skid_rdy : __mem_reader__axi_st_in_data_valid_skid_reg;
      __mem_reader__axi_st_remove_data_reg <= mem_reader__axi_st_remove_data_data_valid_load_en ? out_data : __mem_reader__axi_st_remove_data_reg;
      __mem_reader__axi_st_remove_data_skid_reg <= mem_reader__axi_st_remove_data_skid_data_load_en ? __mem_reader__axi_st_remove_data_reg : __mem_reader__axi_st_remove_data_skid_reg;
      __mem_reader__axi_st_remove_data_valid_reg <= mem_reader__axi_st_remove_data_data_valid_load_en__1 ? __mem_reader__axi_st_remove_vld_buf : __mem_reader__axi_st_remove_data_valid_reg;
      __mem_reader__axi_st_remove_data_valid_skid_reg <= mem_reader__axi_st_remove_data_skid_valid_load_en ? mem_reader__axi_st_remove_data_from_skid_rdy : __mem_reader__axi_st_remove_data_valid_skid_reg;
    end
  end
  assign mem_reader__axi_st_in_rdy = mem_reader__axi_st_in_data_from_skid_rdy;
  assign mem_reader__axi_st_remove_data = mem_reader__axi_st_remove_data_select;
  assign mem_reader__axi_st_remove_vld = mem_reader__axi_st_remove_data_valid_or;
endmodule


module __xls_modules_zstd_memory_axi_stream_remove_empty__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiStreamRemoveEmpty_0__AxiStreamRemoveEmptyInternal_0__64_8_7_8_8_next(
  input wire clk,
  input wire rst,
  input wire mem_reader__axi_st_out_rdy,
  input wire [87:0] mem_reader__continuous_stream_data,
  input wire mem_reader__continuous_stream_vld,
  output wire [96:0] mem_reader__axi_st_out_data,
  output wire mem_reader__axi_st_out_vld,
  output wire mem_reader__continuous_stream_rdy
);
  wire [87:0] __mem_reader__continuous_stream_data_reg_init = {64'h0000_0000_0000_0000, 7'h00, 8'h00, 8'h00, 1'h0};
  wire [87:0] __mem_reader__continuous_stream_data_skid_reg_init = {64'h0000_0000_0000_0000, 7'h00, 8'h00, 8'h00, 1'h0};
  wire [96:0] __mem_reader__axi_st_out_data_reg_init = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  wire [96:0] __mem_reader__axi_st_out_data_skid_reg_init = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  wire [87:0] literal_7669 = {64'h0000_0000_0000_0000, 7'h00, 8'h00, 8'h00, 1'h0};
  reg ____state_2;
  reg [6:0] ____state_1;
  reg [63:0] ____state_0;
  reg [7:0] ____state_3;
  reg [7:0] ____state_4;
  reg [87:0] __mem_reader__continuous_stream_data_reg;
  reg [87:0] __mem_reader__continuous_stream_data_skid_reg;
  reg __mem_reader__continuous_stream_data_valid_reg;
  reg __mem_reader__continuous_stream_data_valid_skid_reg;
  reg [96:0] __mem_reader__axi_st_out_data_reg;
  reg [96:0] __mem_reader__axi_st_out_data_skid_reg;
  reg __mem_reader__axi_st_out_data_valid_reg;
  reg __mem_reader__axi_st_out_data_valid_skid_reg;
  wire do_recv;
  wire [87:0] mem_reader__continuous_stream_data_select;
  wire [87:0] mem_reader__continuous_stream_select;
  wire [6:0] tuple_index_7673;
  wire [6:0] MAX_LEN;
  wire [6:0] len;
  wire [6:0] empty_input_bytes;
  wire uge_7679;
  wire tuple_index_7680;
  wire exact_transfer;
  wire nor_7686;
  wire or_7684;
  wire or_7682;
  wire do_send;
  wire and_7687;
  wire and_7688;
  wire mem_reader__axi_st_out_data_from_skid_rdy;
  wire [2:0] ____state_0__next_value_predicates;
  wire [1:0] ____state_3__next_value_predicates;
  wire or_8452;
  wire [6:0] sum_len;
  wire [3:0] one_hot_7698;
  wire [2:0] one_hot_7699;
  wire [63:0] tuple_index_7689;
  wire [6:0] add_7697;
  wire p0_stage_done;
  wire [7:0] MAX_MASK;
  wire [63:0] data;
  wire [6:0] MAX_LEN__1;
  wire mem_reader__continuous_stream_data_from_skid_rdy;
  wire and_7809;
  wire __mem_reader__axi_st_out_vld_buf;
  wire and_7763;
  wire and_7765;
  wire [6:0] empty_state_bytes;
  wire mem_reader__continuous_stream_data_data_valid_load_en;
  wire mem_reader__continuous_stream_data_to_is_not_rdy;
  wire [7:0] sum_mask;
  wire [7:0] MAX_MASK__1;
  wire [7:0] tuple_index_7704;
  wire [7:0] tuple_index_7706;
  wire mem_reader__axi_st_out_data_data_valid_load_en;
  wire mem_reader__axi_st_out_data_to_is_not_rdy;
  wire ____state_0__at_most_one_next_value;
  wire ____state_3__at_most_one_next_value;
  wire [2:0] concat_7767;
  wire [63:0] combined_state_data;
  wire [6:0] overflow_len;
  wire [1:0] concat_7783;
  wire mem_reader__continuous_stream_data_data_is_sent_to;
  wire mem_reader__continuous_stream_data_skid_data_load_en;
  wire mem_reader__continuous_stream_data_skid_valid_set_zero;
  wire mem_reader__axi_st_out_data_data_is_sent_to;
  wire mem_reader__axi_st_out_data_skid_data_load_en;
  wire mem_reader__axi_st_out_data_skid_valid_set_zero;
  wire [96:0] mem_reader__axi_st_out_data_select;
  wire mem_reader__axi_st_out_data_valid_or;
  wire or_8397;
  wire or_8401;
  wire [63:0] one_hot_sel_7768;
  wire and_7797;
  wire [6:0] one_hot_sel_7776;
  wire nor_7733;
  wire [7:0] one_hot_sel_7784;
  wire and_7804;
  wire [7:0] one_hot_sel_7791;
  wire mem_reader__continuous_stream_data_data_valid_load_en__1;
  wire mem_reader__continuous_stream_data_skid_valid_load_en;
  wire [96:0] data__1;
  wire mem_reader__axi_st_out_data_data_valid_load_en__1;
  wire mem_reader__axi_st_out_data_skid_valid_load_en;
  assign do_recv = ~____state_2;
  assign mem_reader__continuous_stream_data_select = __mem_reader__continuous_stream_data_valid_skid_reg ? __mem_reader__continuous_stream_data_skid_reg : __mem_reader__continuous_stream_data_reg;
  assign mem_reader__continuous_stream_select = do_recv ? mem_reader__continuous_stream_data_select : literal_7669;
  assign tuple_index_7673 = mem_reader__continuous_stream_select[23:17];
  assign MAX_LEN = 7'h40;
  assign len = tuple_index_7673 & {7{do_recv}};
  assign empty_input_bytes = MAX_LEN - len;
  assign uge_7679 = empty_input_bytes >= ____state_1;
  assign tuple_index_7680 = mem_reader__continuous_stream_select[0:0];
  assign exact_transfer = empty_input_bytes == ____state_1;
  assign nor_7686 = ~(____state_2 | uge_7679);
  assign or_7684 = ____state_2 | tuple_index_7680 | exact_transfer;
  assign or_7682 = ____state_2 | uge_7679;
  assign do_send = nor_7686 | or_7684;
  assign and_7687 = or_7682 & ~(____state_2 | tuple_index_7680 | exact_transfer);
  assign and_7688 = or_7682 & or_7684;
  assign mem_reader__axi_st_out_data_from_skid_rdy = ~__mem_reader__axi_st_out_data_valid_skid_reg;
  assign ____state_0__next_value_predicates = {nor_7686, and_7687, and_7688};
  assign ____state_3__next_value_predicates = {nor_7686, and_7688};
  assign or_8452 = ____state_2 | __mem_reader__continuous_stream_data_valid_reg | __mem_reader__continuous_stream_data_valid_skid_reg;
  assign sum_len = ____state_1 + len;
  assign one_hot_7698 = {____state_0__next_value_predicates[2:0] == 3'h0, ____state_0__next_value_predicates[2] && ____state_0__next_value_predicates[1:0] == 2'h0, ____state_0__next_value_predicates[1] && !____state_0__next_value_predicates[0], ____state_0__next_value_predicates[0]};
  assign one_hot_7699 = {____state_3__next_value_predicates[1:0] == 2'h0, ____state_3__next_value_predicates[1] && !____state_3__next_value_predicates[0], ____state_3__next_value_predicates[0]};
  assign tuple_index_7689 = mem_reader__continuous_stream_select[87:24];
  assign add_7697 = ____state_1 + tuple_index_7673;
  assign p0_stage_done = or_8452 & (~do_send | mem_reader__axi_st_out_data_from_skid_rdy);
  assign MAX_MASK = 8'hff;
  assign data = tuple_index_7689 & {64{do_recv}};
  assign MAX_LEN__1 = 7'h40;
  assign mem_reader__continuous_stream_data_from_skid_rdy = ~__mem_reader__continuous_stream_data_valid_skid_reg;
  assign and_7809 = do_recv & p0_stage_done;
  assign __mem_reader__axi_st_out_vld_buf = or_8452 & do_send;
  assign and_7763 = nor_7686 & p0_stage_done;
  assign and_7765 = and_7688 & p0_stage_done;
  assign empty_state_bytes = MAX_LEN__1 - ____state_1;
  assign mem_reader__continuous_stream_data_data_valid_load_en = mem_reader__continuous_stream_vld & mem_reader__continuous_stream_data_from_skid_rdy;
  assign mem_reader__continuous_stream_data_to_is_not_rdy = ~and_7809;
  assign sum_mask = ~(sum_len[6:3] >= 4'h8 ? 8'h00 : MAX_MASK << sum_len[6:3]);
  assign MAX_MASK__1 = 8'hff;
  assign tuple_index_7704 = mem_reader__continuous_stream_select[16:9];
  assign tuple_index_7706 = mem_reader__continuous_stream_select[8:1];
  assign mem_reader__axi_st_out_data_data_valid_load_en = __mem_reader__axi_st_out_vld_buf & mem_reader__axi_st_out_data_from_skid_rdy;
  assign mem_reader__axi_st_out_data_to_is_not_rdy = ~mem_reader__axi_st_out_rdy;
  assign ____state_0__at_most_one_next_value = nor_7686 == one_hot_7698[2] & and_7687 == one_hot_7698[1] & and_7688 == one_hot_7698[0];
  assign ____state_3__at_most_one_next_value = nor_7686 == one_hot_7699[1] & and_7688 == one_hot_7699[0];
  assign concat_7767 = {and_7763, and_7687 & p0_stage_done, and_7765};
  assign combined_state_data = ____state_0 | (____state_1 >= 7'h40 ? 64'h0000_0000_0000_0000 : data << ____state_1);
  assign overflow_len = {~add_7697[6], add_7697[5:0]};
  assign concat_7783 = {and_7763, and_7765};
  assign mem_reader__continuous_stream_data_data_is_sent_to = __mem_reader__continuous_stream_data_valid_reg & and_7809 & mem_reader__continuous_stream_data_from_skid_rdy;
  assign mem_reader__continuous_stream_data_skid_data_load_en = __mem_reader__continuous_stream_data_valid_reg & mem_reader__continuous_stream_data_data_valid_load_en & mem_reader__continuous_stream_data_to_is_not_rdy;
  assign mem_reader__continuous_stream_data_skid_valid_set_zero = __mem_reader__continuous_stream_data_valid_skid_reg & and_7809;
  assign mem_reader__axi_st_out_data_data_is_sent_to = __mem_reader__axi_st_out_data_valid_reg & mem_reader__axi_st_out_rdy & mem_reader__axi_st_out_data_from_skid_rdy;
  assign mem_reader__axi_st_out_data_skid_data_load_en = __mem_reader__axi_st_out_data_valid_reg & mem_reader__axi_st_out_data_data_valid_load_en & mem_reader__axi_st_out_data_to_is_not_rdy;
  assign mem_reader__axi_st_out_data_skid_valid_set_zero = __mem_reader__axi_st_out_data_valid_skid_reg & mem_reader__axi_st_out_rdy;
  assign mem_reader__axi_st_out_data_select = __mem_reader__axi_st_out_data_valid_skid_reg ? __mem_reader__axi_st_out_data_skid_reg : __mem_reader__axi_st_out_data_reg;
  assign mem_reader__axi_st_out_data_valid_or = __mem_reader__axi_st_out_data_valid_reg | __mem_reader__axi_st_out_data_valid_skid_reg;
  assign or_8397 = ~p0_stage_done | ____state_0__at_most_one_next_value | rst;
  assign or_8401 = ~p0_stage_done | ____state_3__at_most_one_next_value | rst;
  assign one_hot_sel_7768 = 64'h0000_0000_0000_0000 & {64{concat_7767[0]}} | combined_state_data & {64{concat_7767[1]}} | (empty_state_bytes >= 7'h40 ? 64'h0000_0000_0000_0000 : tuple_index_7689 >> empty_state_bytes) & {64{concat_7767[2]}};
  assign and_7797 = (nor_7686 | and_7687 | and_7688) & p0_stage_done;
  assign one_hot_sel_7776 = 7'h00 & {7{concat_7767[0]}} | sum_len & {7{concat_7767[1]}} | overflow_len & {7{concat_7767[2]}};
  assign nor_7733 = ~(____state_2 | uge_7679 | ~tuple_index_7680);
  assign one_hot_sel_7784 = 8'h00 & {8{concat_7783[0]}} | tuple_index_7704 & {8{concat_7783[1]}};
  assign and_7804 = (nor_7686 | and_7688) & p0_stage_done;
  assign one_hot_sel_7791 = 8'h00 & {8{concat_7783[0]}} | tuple_index_7706 & {8{concat_7783[1]}};
  assign mem_reader__continuous_stream_data_data_valid_load_en__1 = mem_reader__continuous_stream_data_data_is_sent_to | mem_reader__continuous_stream_data_data_valid_load_en;
  assign mem_reader__continuous_stream_data_skid_valid_load_en = mem_reader__continuous_stream_data_skid_data_load_en | mem_reader__continuous_stream_data_skid_valid_set_zero;
  assign data__1 = {combined_state_data, nor_7686 ? MAX_MASK__1 : sum_mask, nor_7686 ? MAX_MASK__1 : sum_mask, ____state_2 ? ____state_3 : tuple_index_7704, ____state_2 ? ____state_4 : tuple_index_7706, or_7682 & (____state_2 | tuple_index_7680)};
  assign mem_reader__axi_st_out_data_data_valid_load_en__1 = mem_reader__axi_st_out_data_data_is_sent_to | mem_reader__axi_st_out_data_data_valid_load_en;
  assign mem_reader__axi_st_out_data_skid_valid_load_en = mem_reader__axi_st_out_data_skid_data_load_en | mem_reader__axi_st_out_data_skid_valid_set_zero;
  always @ (posedge clk) begin
    if (rst) begin
      ____state_2 <= 1'h0;
      ____state_1 <= 7'h00;
      ____state_0 <= 64'h0000_0000_0000_0000;
      ____state_3 <= 8'h00;
      ____state_4 <= 8'h00;
      __mem_reader__continuous_stream_data_reg <= __mem_reader__continuous_stream_data_reg_init;
      __mem_reader__continuous_stream_data_skid_reg <= __mem_reader__continuous_stream_data_skid_reg_init;
      __mem_reader__continuous_stream_data_valid_reg <= 1'h0;
      __mem_reader__continuous_stream_data_valid_skid_reg <= 1'h0;
      __mem_reader__axi_st_out_data_reg <= __mem_reader__axi_st_out_data_reg_init;
      __mem_reader__axi_st_out_data_skid_reg <= __mem_reader__axi_st_out_data_skid_reg_init;
      __mem_reader__axi_st_out_data_valid_reg <= 1'h0;
      __mem_reader__axi_st_out_data_valid_skid_reg <= 1'h0;
    end else begin
      ____state_2 <= p0_stage_done ? nor_7733 : ____state_2;
      ____state_1 <= and_7797 ? one_hot_sel_7776 : ____state_1;
      ____state_0 <= and_7797 ? one_hot_sel_7768 : ____state_0;
      ____state_3 <= and_7804 ? one_hot_sel_7784 : ____state_3;
      ____state_4 <= and_7804 ? one_hot_sel_7791 : ____state_4;
      __mem_reader__continuous_stream_data_reg <= mem_reader__continuous_stream_data_data_valid_load_en ? mem_reader__continuous_stream_data : __mem_reader__continuous_stream_data_reg;
      __mem_reader__continuous_stream_data_skid_reg <= mem_reader__continuous_stream_data_skid_data_load_en ? __mem_reader__continuous_stream_data_reg : __mem_reader__continuous_stream_data_skid_reg;
      __mem_reader__continuous_stream_data_valid_reg <= mem_reader__continuous_stream_data_data_valid_load_en__1 ? mem_reader__continuous_stream_vld : __mem_reader__continuous_stream_data_valid_reg;
      __mem_reader__continuous_stream_data_valid_skid_reg <= mem_reader__continuous_stream_data_skid_valid_load_en ? mem_reader__continuous_stream_data_from_skid_rdy : __mem_reader__continuous_stream_data_valid_skid_reg;
      __mem_reader__axi_st_out_data_reg <= mem_reader__axi_st_out_data_data_valid_load_en ? data__1 : __mem_reader__axi_st_out_data_reg;
      __mem_reader__axi_st_out_data_skid_reg <= mem_reader__axi_st_out_data_skid_data_load_en ? __mem_reader__axi_st_out_data_reg : __mem_reader__axi_st_out_data_skid_reg;
      __mem_reader__axi_st_out_data_valid_reg <= mem_reader__axi_st_out_data_data_valid_load_en__1 ? __mem_reader__axi_st_out_vld_buf : __mem_reader__axi_st_out_data_valid_reg;
      __mem_reader__axi_st_out_data_valid_skid_reg <= mem_reader__axi_st_out_data_skid_valid_load_en ? mem_reader__axi_st_out_data_from_skid_rdy : __mem_reader__axi_st_out_data_valid_skid_reg;
    end
  end
  assign mem_reader__axi_st_out_data = mem_reader__axi_st_out_data_select;
  assign mem_reader__axi_st_out_vld = mem_reader__axi_st_out_data_valid_or;
  assign mem_reader__continuous_stream_rdy = mem_reader__continuous_stream_data_from_skid_rdy;
endmodule


module __xls_modules_zstd_memory_axi_stream_remove_empty__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiStreamRemoveEmpty_0__RemoveEmptyBytes_0__64_8_7_8_10_8_next(
  input wire clk,
  input wire rst,
  input wire [96:0] mem_reader__axi_st_remove_data,
  input wire mem_reader__axi_st_remove_vld,
  input wire mem_reader__continuous_stream_rdy,
  output wire mem_reader__axi_st_remove_rdy,
  output wire [87:0] mem_reader__continuous_stream_data,
  output wire mem_reader__continuous_stream_vld
);
  wire [96:0] __mem_reader__axi_st_remove_data_reg_init = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  wire [96:0] __mem_reader__axi_st_remove_data_skid_reg_init = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  wire [87:0] __mem_reader__continuous_stream_data_reg_init = {64'h0000_0000_0000_0000, 7'h00, 8'h00, 8'h00, 1'h0};
  wire [87:0] __mem_reader__continuous_stream_data_skid_reg_init = {64'h0000_0000_0000_0000, 7'h00, 8'h00, 8'h00, 1'h0};
  reg [15:0] p0_data__2_squeezed;
  reg p0_bit_slice_7899;
  reg [23:0] p0_bit_slice_7901;
  reg [7:0] p0_bit_slice_7902;
  reg [1:0] p0_offset__17_squeezed;
  reg p0_bit_slice_7912;
  reg [7:0] p0_bit_slice_7916;
  reg [2:0] p0_offset__16;
  reg [1:0] p0_len__3_squeezed_squeezed;
  reg p0_bit_slice_7920;
  reg [7:0] p0_bit_slice_7921;
  reg p0_bit_slice_7922;
  reg [7:0] p0_bit_slice_7923;
  reg p0_bit_slice_7924;
  reg [7:0] p0_bit_slice_7925;
  reg p0_bit_slice_7926;
  reg [7:0] p0_frame_id;
  reg [7:0] p0_frame_dest;
  reg p0_frame_last;
  reg [31:0] p1_data__4_squeezed;
  reg p1_bit_slice_7920;
  reg [39:0] p1_bit_slice_7997;
  reg p1_bit_slice_7922;
  reg [47:0] p1_bit_slice_8011;
  reg [7:0] p1_bit_slice_7923;
  reg [2:0] p1_offset__11;
  reg [2:0] p1_len__5_squeezed_squeezed;
  reg p1_bit_slice_7924;
  reg [7:0] p1_bit_slice_7925;
  reg p1_bit_slice_7926;
  reg [7:0] p1_frame_id;
  reg [7:0] p1_frame_dest;
  reg p1_frame_last;
  reg [7:0] p2_bit_slice_7925;
  reg [2:0] p2_offset__8;
  reg [2:0] p2_len__7_squeezed_squeezed;
  reg [55:0] p2_data__7_squeezed;
  reg p2_bit_slice_7926;
  reg [7:0] p2_frame_id;
  reg [7:0] p2_frame_dest;
  reg p2_frame_last;
  reg p0_valid;
  reg p1_valid;
  reg p2_valid;
  reg [96:0] __mem_reader__axi_st_remove_data_reg;
  reg [96:0] __mem_reader__axi_st_remove_data_skid_reg;
  reg __mem_reader__axi_st_remove_data_valid_reg;
  reg __mem_reader__axi_st_remove_data_valid_skid_reg;
  reg [87:0] __mem_reader__continuous_stream_data_reg;
  reg [87:0] __mem_reader__continuous_stream_data_skid_reg;
  reg __mem_reader__continuous_stream_data_valid_reg;
  reg __mem_reader__continuous_stream_data_valid_skid_reg;
  wire [96:0] mem_reader__axi_st_remove_data_select;
  wire mem_reader__continuous_stream_data_from_skid_rdy;
  wire [7:0] data__7_squeezed_const_msb_bits__4;
  wire [7:0] frame_str;
  wire p3_stage_done;
  wire p3_not_valid;
  wire [39:0] data__4_squeezed__1;
  wire p2_enable;
  wire len__4_squeezed_const_msb_bits;
  wire p2_data_enable;
  wire p2_not_valid;
  wire [7:0] data__7_squeezed_const_msb_bits__5;
  wire [39:0] data__5_squeezed;
  wire [7:0] data__7_squeezed_const_msb_bits__2;
  wire [63:0] frame_data;
  wire p1_enable;
  wire [47:0] data__5_squeezed__1;
  wire [7:0] data__7_squeezed_const_msb_bits__1;
  wire [2:0] len__2_squeezed_const_lsb_bits__3;
  wire [23:0] data__2_squeezed__1;
  wire [2:0] len__2_squeezed_const_lsb_bits__7;
  wire len__4_squeezed_const_msb_bits__3;
  wire [7:0] data__7_squeezed_const_msb_bits;
  wire [2:0] len__2_squeezed_const_lsb_bits;
  wire [1:0] offset__18_squeezed;
  wire p1_data_enable;
  wire p1_not_valid;
  wire [2:0] len__2_squeezed_const_lsb_bits__4;
  wire len__4_squeezed_const_msb_bits__4;
  wire [2:0] add_7994;
  wire [2:0] len__3_squeezed;
  wire [7:0] data__7_squeezed_const_msb_bits__8;
  wire bit_slice_7899;
  wire [1:0] add_7900;
  wire len__4_squeezed_const_msb_bits__1;
  wire p0_enable;
  wire mem_reader__axi_st_remove_data_valid_or;
  wire [7:0] data__7_squeezed_const_msb_bits__7;
  wire [3:0] len__7_squeezed;
  wire [2:0] add_8057;
  wire [7:0] data__7_squeezed_const_msb_bits__6;
  wire [47:0] data__6_squeezed;
  wire [63:0] shrl_8060;
  wire [7:0] data__7_squeezed_const_msb_bits__3;
  wire [23:0] data__3_squeezed;
  wire [63:0] shrl_7981;
  wire [2:0] len__2_squeezed_const_lsb_bits__1;
  wire [2:0] offset__13;
  wire [2:0] len__2_squeezed_const_lsb_bits__2;
  wire [2:0] add_8002;
  wire [63:0] shrl_7882;
  wire [2:0] len__2_squeezed_const_lsb_bits__5;
  wire len__4_squeezed_const_msb_bits__2;
  wire [1:0] offset__17_squeezed;
  wire [1:0] len__10;
  wire mem_reader__axi_st_remove_data_from_skid_rdy;
  wire p0_data_enable;
  wire [63:0] data__7;
  wire [3:0] add_8100;
  wire [2:0] len__6_squeezed_squeezed;
  wire [55:0] data__6_squeezed__1;
  wire [31:0] data__3_squeezed__1;
  wire [2:0] len__4_squeezed;
  wire [15:0] data__9;
  wire [2:0] offset__17;
  wire [1:0] len__2_squeezed;
  wire mem_reader__axi_st_remove_data_data_valid_load_en;
  wire mem_reader__axi_st_remove_data_to_is_not_rdy;
  wire [3:0] len__8_squeezed;
  wire [2:0] len__2_squeezed_const_lsb_bits__6;
  wire mem_reader__continuous_stream_data_data_valid_load_en;
  wire mem_reader__continuous_stream_data_to_is_not_rdy;
  wire [2:0] add_8066;
  wire [2:0] add_8067;
  wire [63:0] shrl_7993;
  wire [63:0] shrl_8008;
  wire [2:0] add_8009;
  wire [2:0] add_8010;
  wire [63:0] shrl_7898;
  wire bit_slice_7912;
  wire [2:0] add_7913;
  wire [1:0] add_7918;
  wire mem_reader__axi_st_remove_data_data_is_sent_to;
  wire mem_reader__axi_st_remove_data_skid_data_load_en;
  wire mem_reader__axi_st_remove_data_skid_valid_set_zero;
  wire [63:0] data__8;
  wire [6:0] len__8;
  wire mem_reader__continuous_stream_data_data_is_sent_to;
  wire mem_reader__continuous_stream_data_skid_data_load_en;
  wire mem_reader__continuous_stream_data_skid_valid_set_zero;
  wire [87:0] mem_reader__continuous_stream_data_select;
  wire mem_reader__continuous_stream_data_valid_or;
  wire [2:0] offset__8;
  wire [2:0] len__7_squeezed_squeezed;
  wire [55:0] data__7_squeezed;
  wire [31:0] data__4_squeezed;
  wire [39:0] bit_slice_7997;
  wire [47:0] bit_slice_8011;
  wire [2:0] offset__11;
  wire [2:0] len__5_squeezed_squeezed;
  wire [15:0] data__2_squeezed;
  wire [23:0] bit_slice_7901;
  wire [7:0] bit_slice_7902;
  wire [7:0] bit_slice_7916;
  wire [2:0] offset__16;
  wire [1:0] len__3_squeezed_squeezed;
  wire bit_slice_7920;
  wire [7:0] bit_slice_7921;
  wire bit_slice_7922;
  wire [7:0] bit_slice_7923;
  wire bit_slice_7924;
  wire [7:0] bit_slice_7925;
  wire bit_slice_7926;
  wire [7:0] frame_id;
  wire [7:0] frame_dest;
  wire frame_last;
  wire mem_reader__axi_st_remove_data_data_valid_load_en__1;
  wire mem_reader__axi_st_remove_data_skid_valid_load_en;
  wire [87:0] continuous_stream;
  wire mem_reader__continuous_stream_data_data_valid_load_en__1;
  wire mem_reader__continuous_stream_data_skid_valid_load_en;
  assign mem_reader__axi_st_remove_data_select = __mem_reader__axi_st_remove_data_valid_skid_reg ? __mem_reader__axi_st_remove_data_skid_reg : __mem_reader__axi_st_remove_data_reg;
  assign mem_reader__continuous_stream_data_from_skid_rdy = ~__mem_reader__continuous_stream_data_valid_skid_reg;
  assign data__7_squeezed_const_msb_bits__4 = 8'h00;
  assign frame_str = mem_reader__axi_st_remove_data_select[32:25];
  assign p3_stage_done = p2_valid & mem_reader__continuous_stream_data_from_skid_rdy;
  assign p3_not_valid = ~p2_valid;
  assign data__4_squeezed__1 = {data__7_squeezed_const_msb_bits__4, p1_data__4_squeezed};
  assign p2_enable = p3_stage_done | p3_not_valid;
  assign len__4_squeezed_const_msb_bits = 1'h0;
  assign p2_data_enable = p2_enable & p1_valid;
  assign p2_not_valid = ~p1_valid;
  assign data__7_squeezed_const_msb_bits__5 = 8'h00;
  assign data__5_squeezed = p1_bit_slice_7920 ? data__4_squeezed__1 | p1_bit_slice_7997 : data__4_squeezed__1;
  assign data__7_squeezed_const_msb_bits__2 = 8'h00;
  assign frame_data = mem_reader__axi_st_remove_data_select[96:33];
  assign p1_enable = p2_data_enable | p2_not_valid;
  assign data__5_squeezed__1 = {data__7_squeezed_const_msb_bits__5, data__5_squeezed};
  assign data__7_squeezed_const_msb_bits__1 = 8'h00;
  assign len__2_squeezed_const_lsb_bits__3 = 3'h0;
  assign data__2_squeezed__1 = {data__7_squeezed_const_msb_bits__2, p0_data__2_squeezed};
  assign len__2_squeezed_const_lsb_bits__7 = 3'h0;
  assign len__4_squeezed_const_msb_bits__3 = 1'h0;
  assign data__7_squeezed_const_msb_bits = 8'h00;
  assign len__2_squeezed_const_lsb_bits = 3'h0;
  assign offset__18_squeezed = frame_str[1] ? {len__4_squeezed_const_msb_bits, ~frame_str[0]} : (frame_str[0] ? 2'h1 : 2'h2);
  assign p1_data_enable = p1_enable & p0_valid;
  assign p1_not_valid = ~p0_valid;
  assign len__2_squeezed_const_lsb_bits__4 = 3'h0;
  assign len__4_squeezed_const_msb_bits__4 = 1'h0;
  assign add_7994 = p0_offset__16 + 3'h1;
  assign len__3_squeezed = {len__4_squeezed_const_msb_bits__3, p0_len__3_squeezed_squeezed};
  assign data__7_squeezed_const_msb_bits__8 = 8'h00;
  assign bit_slice_7899 = frame_str[2];
  assign add_7900 = offset__18_squeezed + 2'h1;
  assign len__4_squeezed_const_msb_bits__1 = 1'h0;
  assign p0_enable = p1_data_enable | p1_not_valid;
  assign mem_reader__axi_st_remove_data_valid_or = __mem_reader__axi_st_remove_data_valid_reg | __mem_reader__axi_st_remove_data_valid_skid_reg;
  assign data__7_squeezed_const_msb_bits__7 = 8'h00;
  assign len__7_squeezed = {len__4_squeezed_const_msb_bits__4, p2_len__7_squeezed_squeezed};
  assign add_8057 = p1_len__5_squeezed_squeezed + 3'h1;
  assign data__7_squeezed_const_msb_bits__6 = 8'h00;
  assign data__6_squeezed = p1_bit_slice_7922 ? data__5_squeezed__1 | p1_bit_slice_8011 : data__5_squeezed__1;
  assign shrl_8060 = {data__7_squeezed_const_msb_bits__1, p1_bit_slice_7923, 48'h0000_0000_0000} >> {p1_offset__11, len__2_squeezed_const_lsb_bits__3};
  assign data__7_squeezed_const_msb_bits__3 = 8'h00;
  assign data__3_squeezed = p0_bit_slice_7899 ? data__2_squeezed__1 | p0_bit_slice_7901 : data__2_squeezed__1;
  assign shrl_7981 = {32'h0000_0000, p0_bit_slice_7902, 24'h00_0000} >> {p0_offset__17_squeezed, len__2_squeezed_const_lsb_bits__7};
  assign len__2_squeezed_const_lsb_bits__1 = 3'h0;
  assign offset__13 = p0_bit_slice_7920 ? p0_offset__16 : add_7994;
  assign len__2_squeezed_const_lsb_bits__2 = 3'h0;
  assign add_8002 = len__3_squeezed + 3'h1;
  assign shrl_7882 = {48'h0000_0000_0000, frame_data[15:8], data__7_squeezed_const_msb_bits} >> {~frame_str[0], len__2_squeezed_const_lsb_bits};
  assign len__2_squeezed_const_lsb_bits__5 = 3'h0;
  assign len__4_squeezed_const_msb_bits__2 = 1'h0;
  assign offset__17_squeezed = bit_slice_7899 ? offset__18_squeezed : add_7900;
  assign len__10 = {len__4_squeezed_const_msb_bits__1, frame_str[0]};
  assign mem_reader__axi_st_remove_data_from_skid_rdy = ~__mem_reader__axi_st_remove_data_valid_skid_reg;
  assign p0_data_enable = p0_enable & mem_reader__axi_st_remove_data_valid_or;
  assign data__7 = {data__7_squeezed_const_msb_bits__7, p2_data__7_squeezed};
  assign add_8100 = len__7_squeezed + 4'h1;
  assign len__6_squeezed_squeezed = p1_bit_slice_7922 ? add_8057 : p1_len__5_squeezed_squeezed;
  assign data__6_squeezed__1 = {data__7_squeezed_const_msb_bits__6, data__6_squeezed};
  assign data__3_squeezed__1 = {data__7_squeezed_const_msb_bits__3, data__3_squeezed};
  assign len__4_squeezed = p0_bit_slice_7912 ? add_8002 : len__3_squeezed;
  assign data__9 = {data__7_squeezed_const_msb_bits__8, frame_data[7:0]} & {16{frame_str[0]}};
  assign offset__17 = {len__4_squeezed_const_msb_bits__2, offset__17_squeezed};
  assign len__2_squeezed = frame_str[1] ? (frame_str[0] ? 2'h2 : 2'h1) : len__10;
  assign mem_reader__axi_st_remove_data_data_valid_load_en = mem_reader__axi_st_remove_vld & mem_reader__axi_st_remove_data_from_skid_rdy;
  assign mem_reader__axi_st_remove_data_to_is_not_rdy = ~p0_data_enable;
  assign len__8_squeezed = p2_bit_slice_7926 ? add_8100 : len__7_squeezed;
  assign len__2_squeezed_const_lsb_bits__6 = 3'h0;
  assign mem_reader__continuous_stream_data_data_valid_load_en = p2_valid & mem_reader__continuous_stream_data_from_skid_rdy;
  assign mem_reader__continuous_stream_data_to_is_not_rdy = ~mem_reader__continuous_stream_rdy;
  assign add_8066 = p1_offset__11 + 3'h1;
  assign add_8067 = len__6_squeezed_squeezed + 3'h1;
  assign shrl_7993 = {24'h00_0000, p0_bit_slice_7916, 32'h0000_0000} >> {p0_offset__16, len__2_squeezed_const_lsb_bits__1};
  assign shrl_8008 = {16'h0000, p0_bit_slice_7921, 40'h00_0000_0000} >> {offset__13, len__2_squeezed_const_lsb_bits__2};
  assign add_8009 = offset__13 + 3'h1;
  assign add_8010 = len__4_squeezed + 3'h1;
  assign shrl_7898 = {40'h00_0000_0000, frame_data[23:16], 16'h0000} >> {offset__18_squeezed, len__2_squeezed_const_lsb_bits__5};
  assign bit_slice_7912 = frame_str[3];
  assign add_7913 = offset__17 + 3'h1;
  assign add_7918 = len__2_squeezed + 2'h1;
  assign mem_reader__axi_st_remove_data_data_is_sent_to = __mem_reader__axi_st_remove_data_valid_reg & p0_data_enable & mem_reader__axi_st_remove_data_from_skid_rdy;
  assign mem_reader__axi_st_remove_data_skid_data_load_en = __mem_reader__axi_st_remove_data_valid_reg & mem_reader__axi_st_remove_data_data_valid_load_en & mem_reader__axi_st_remove_data_to_is_not_rdy;
  assign mem_reader__axi_st_remove_data_skid_valid_set_zero = __mem_reader__axi_st_remove_data_valid_skid_reg & p0_data_enable;
  assign data__8 = p2_bit_slice_7926 ? data__7 | {p2_bit_slice_7925, 56'h00_0000_0000_0000} >> {p2_offset__8, len__2_squeezed_const_lsb_bits__4} : data__7;
  assign len__8 = {len__8_squeezed, len__2_squeezed_const_lsb_bits__6};
  assign mem_reader__continuous_stream_data_data_is_sent_to = __mem_reader__continuous_stream_data_valid_reg & mem_reader__continuous_stream_rdy & mem_reader__continuous_stream_data_from_skid_rdy;
  assign mem_reader__continuous_stream_data_skid_data_load_en = __mem_reader__continuous_stream_data_valid_reg & mem_reader__continuous_stream_data_data_valid_load_en & mem_reader__continuous_stream_data_to_is_not_rdy;
  assign mem_reader__continuous_stream_data_skid_valid_set_zero = __mem_reader__continuous_stream_data_valid_skid_reg & mem_reader__continuous_stream_rdy;
  assign mem_reader__continuous_stream_data_select = __mem_reader__continuous_stream_data_valid_skid_reg ? __mem_reader__continuous_stream_data_skid_reg : __mem_reader__continuous_stream_data_reg;
  assign mem_reader__continuous_stream_data_valid_or = __mem_reader__continuous_stream_data_valid_reg | __mem_reader__continuous_stream_data_valid_skid_reg;
  assign offset__8 = p1_bit_slice_7924 ? p1_offset__11 : add_8066;
  assign len__7_squeezed_squeezed = p1_bit_slice_7924 ? add_8067 : len__6_squeezed_squeezed;
  assign data__7_squeezed = p1_bit_slice_7924 ? data__6_squeezed__1 | shrl_8060[55:0] : data__6_squeezed__1;
  assign data__4_squeezed = p0_bit_slice_7912 ? data__3_squeezed__1 | shrl_7981[31:0] : data__3_squeezed__1;
  assign bit_slice_7997 = shrl_7993[39:0];
  assign bit_slice_8011 = shrl_8008[47:0];
  assign offset__11 = p0_bit_slice_7922 ? offset__13 : add_8009;
  assign len__5_squeezed_squeezed = p0_bit_slice_7920 ? add_8010 : len__4_squeezed;
  assign data__2_squeezed = frame_str[1] ? data__9 | shrl_7882[15:0] : data__9;
  assign bit_slice_7901 = shrl_7898[23:0];
  assign bit_slice_7902 = frame_data[31:24];
  assign bit_slice_7916 = frame_data[39:32];
  assign offset__16 = bit_slice_7912 ? offset__17 : add_7913;
  assign len__3_squeezed_squeezed = bit_slice_7899 ? add_7918 : len__2_squeezed;
  assign bit_slice_7920 = frame_str[4];
  assign bit_slice_7921 = frame_data[47:40];
  assign bit_slice_7922 = frame_str[5];
  assign bit_slice_7923 = frame_data[55:48];
  assign bit_slice_7924 = frame_str[6];
  assign bit_slice_7925 = frame_data[63:56];
  assign bit_slice_7926 = frame_str[7];
  assign frame_id = mem_reader__axi_st_remove_data_select[16:9];
  assign frame_dest = mem_reader__axi_st_remove_data_select[8:1];
  assign frame_last = mem_reader__axi_st_remove_data_select[0:0];
  assign mem_reader__axi_st_remove_data_data_valid_load_en__1 = mem_reader__axi_st_remove_data_data_is_sent_to | mem_reader__axi_st_remove_data_data_valid_load_en;
  assign mem_reader__axi_st_remove_data_skid_valid_load_en = mem_reader__axi_st_remove_data_skid_data_load_en | mem_reader__axi_st_remove_data_skid_valid_set_zero;
  assign continuous_stream = {data__8, len__8, p2_frame_id, p2_frame_dest, p2_frame_last};
  assign mem_reader__continuous_stream_data_data_valid_load_en__1 = mem_reader__continuous_stream_data_data_is_sent_to | mem_reader__continuous_stream_data_data_valid_load_en;
  assign mem_reader__continuous_stream_data_skid_valid_load_en = mem_reader__continuous_stream_data_skid_data_load_en | mem_reader__continuous_stream_data_skid_valid_set_zero;
  always @ (posedge clk) begin
    if (rst) begin
      p0_data__2_squeezed <= 16'h0000;
      p0_bit_slice_7899 <= 1'h0;
      p0_bit_slice_7901 <= 24'h00_0000;
      p0_bit_slice_7902 <= 8'h00;
      p0_offset__17_squeezed <= 2'h0;
      p0_bit_slice_7912 <= 1'h0;
      p0_bit_slice_7916 <= 8'h00;
      p0_offset__16 <= 3'h0;
      p0_len__3_squeezed_squeezed <= 2'h0;
      p0_bit_slice_7920 <= 1'h0;
      p0_bit_slice_7921 <= 8'h00;
      p0_bit_slice_7922 <= 1'h0;
      p0_bit_slice_7923 <= 8'h00;
      p0_bit_slice_7924 <= 1'h0;
      p0_bit_slice_7925 <= 8'h00;
      p0_bit_slice_7926 <= 1'h0;
      p0_frame_id <= 8'h00;
      p0_frame_dest <= 8'h00;
      p0_frame_last <= 1'h0;
      p1_data__4_squeezed <= 32'h0000_0000;
      p1_bit_slice_7920 <= 1'h0;
      p1_bit_slice_7997 <= 40'h00_0000_0000;
      p1_bit_slice_7922 <= 1'h0;
      p1_bit_slice_8011 <= 48'h0000_0000_0000;
      p1_bit_slice_7923 <= 8'h00;
      p1_offset__11 <= 3'h0;
      p1_len__5_squeezed_squeezed <= 3'h0;
      p1_bit_slice_7924 <= 1'h0;
      p1_bit_slice_7925 <= 8'h00;
      p1_bit_slice_7926 <= 1'h0;
      p1_frame_id <= 8'h00;
      p1_frame_dest <= 8'h00;
      p1_frame_last <= 1'h0;
      p2_bit_slice_7925 <= 8'h00;
      p2_offset__8 <= 3'h0;
      p2_len__7_squeezed_squeezed <= 3'h0;
      p2_data__7_squeezed <= 56'h00_0000_0000_0000;
      p2_bit_slice_7926 <= 1'h0;
      p2_frame_id <= 8'h00;
      p2_frame_dest <= 8'h00;
      p2_frame_last <= 1'h0;
      p0_valid <= 1'h0;
      p1_valid <= 1'h0;
      p2_valid <= 1'h0;
      __mem_reader__axi_st_remove_data_reg <= __mem_reader__axi_st_remove_data_reg_init;
      __mem_reader__axi_st_remove_data_skid_reg <= __mem_reader__axi_st_remove_data_skid_reg_init;
      __mem_reader__axi_st_remove_data_valid_reg <= 1'h0;
      __mem_reader__axi_st_remove_data_valid_skid_reg <= 1'h0;
      __mem_reader__continuous_stream_data_reg <= __mem_reader__continuous_stream_data_reg_init;
      __mem_reader__continuous_stream_data_skid_reg <= __mem_reader__continuous_stream_data_skid_reg_init;
      __mem_reader__continuous_stream_data_valid_reg <= 1'h0;
      __mem_reader__continuous_stream_data_valid_skid_reg <= 1'h0;
    end else begin
      p0_data__2_squeezed <= p0_data_enable ? data__2_squeezed : p0_data__2_squeezed;
      p0_bit_slice_7899 <= p0_data_enable ? bit_slice_7899 : p0_bit_slice_7899;
      p0_bit_slice_7901 <= p0_data_enable ? bit_slice_7901 : p0_bit_slice_7901;
      p0_bit_slice_7902 <= p0_data_enable ? bit_slice_7902 : p0_bit_slice_7902;
      p0_offset__17_squeezed <= p0_data_enable ? offset__17_squeezed : p0_offset__17_squeezed;
      p0_bit_slice_7912 <= p0_data_enable ? bit_slice_7912 : p0_bit_slice_7912;
      p0_bit_slice_7916 <= p0_data_enable ? bit_slice_7916 : p0_bit_slice_7916;
      p0_offset__16 <= p0_data_enable ? offset__16 : p0_offset__16;
      p0_len__3_squeezed_squeezed <= p0_data_enable ? len__3_squeezed_squeezed : p0_len__3_squeezed_squeezed;
      p0_bit_slice_7920 <= p0_data_enable ? bit_slice_7920 : p0_bit_slice_7920;
      p0_bit_slice_7921 <= p0_data_enable ? bit_slice_7921 : p0_bit_slice_7921;
      p0_bit_slice_7922 <= p0_data_enable ? bit_slice_7922 : p0_bit_slice_7922;
      p0_bit_slice_7923 <= p0_data_enable ? bit_slice_7923 : p0_bit_slice_7923;
      p0_bit_slice_7924 <= p0_data_enable ? bit_slice_7924 : p0_bit_slice_7924;
      p0_bit_slice_7925 <= p0_data_enable ? bit_slice_7925 : p0_bit_slice_7925;
      p0_bit_slice_7926 <= p0_data_enable ? bit_slice_7926 : p0_bit_slice_7926;
      p0_frame_id <= p0_data_enable ? frame_id : p0_frame_id;
      p0_frame_dest <= p0_data_enable ? frame_dest : p0_frame_dest;
      p0_frame_last <= p0_data_enable ? frame_last : p0_frame_last;
      p1_data__4_squeezed <= p1_data_enable ? data__4_squeezed : p1_data__4_squeezed;
      p1_bit_slice_7920 <= p1_data_enable ? p0_bit_slice_7920 : p1_bit_slice_7920;
      p1_bit_slice_7997 <= p1_data_enable ? bit_slice_7997 : p1_bit_slice_7997;
      p1_bit_slice_7922 <= p1_data_enable ? p0_bit_slice_7922 : p1_bit_slice_7922;
      p1_bit_slice_8011 <= p1_data_enable ? bit_slice_8011 : p1_bit_slice_8011;
      p1_bit_slice_7923 <= p1_data_enable ? p0_bit_slice_7923 : p1_bit_slice_7923;
      p1_offset__11 <= p1_data_enable ? offset__11 : p1_offset__11;
      p1_len__5_squeezed_squeezed <= p1_data_enable ? len__5_squeezed_squeezed : p1_len__5_squeezed_squeezed;
      p1_bit_slice_7924 <= p1_data_enable ? p0_bit_slice_7924 : p1_bit_slice_7924;
      p1_bit_slice_7925 <= p1_data_enable ? p0_bit_slice_7925 : p1_bit_slice_7925;
      p1_bit_slice_7926 <= p1_data_enable ? p0_bit_slice_7926 : p1_bit_slice_7926;
      p1_frame_id <= p1_data_enable ? p0_frame_id : p1_frame_id;
      p1_frame_dest <= p1_data_enable ? p0_frame_dest : p1_frame_dest;
      p1_frame_last <= p1_data_enable ? p0_frame_last : p1_frame_last;
      p2_bit_slice_7925 <= p2_data_enable ? p1_bit_slice_7925 : p2_bit_slice_7925;
      p2_offset__8 <= p2_data_enable ? offset__8 : p2_offset__8;
      p2_len__7_squeezed_squeezed <= p2_data_enable ? len__7_squeezed_squeezed : p2_len__7_squeezed_squeezed;
      p2_data__7_squeezed <= p2_data_enable ? data__7_squeezed : p2_data__7_squeezed;
      p2_bit_slice_7926 <= p2_data_enable ? p1_bit_slice_7926 : p2_bit_slice_7926;
      p2_frame_id <= p2_data_enable ? p1_frame_id : p2_frame_id;
      p2_frame_dest <= p2_data_enable ? p1_frame_dest : p2_frame_dest;
      p2_frame_last <= p2_data_enable ? p1_frame_last : p2_frame_last;
      p0_valid <= p0_enable ? mem_reader__axi_st_remove_data_valid_or : p0_valid;
      p1_valid <= p1_enable ? p0_valid : p1_valid;
      p2_valid <= p2_enable ? p1_valid : p2_valid;
      __mem_reader__axi_st_remove_data_reg <= mem_reader__axi_st_remove_data_data_valid_load_en ? mem_reader__axi_st_remove_data : __mem_reader__axi_st_remove_data_reg;
      __mem_reader__axi_st_remove_data_skid_reg <= mem_reader__axi_st_remove_data_skid_data_load_en ? __mem_reader__axi_st_remove_data_reg : __mem_reader__axi_st_remove_data_skid_reg;
      __mem_reader__axi_st_remove_data_valid_reg <= mem_reader__axi_st_remove_data_data_valid_load_en__1 ? mem_reader__axi_st_remove_vld : __mem_reader__axi_st_remove_data_valid_reg;
      __mem_reader__axi_st_remove_data_valid_skid_reg <= mem_reader__axi_st_remove_data_skid_valid_load_en ? mem_reader__axi_st_remove_data_from_skid_rdy : __mem_reader__axi_st_remove_data_valid_skid_reg;
      __mem_reader__continuous_stream_data_reg <= mem_reader__continuous_stream_data_data_valid_load_en ? continuous_stream : __mem_reader__continuous_stream_data_reg;
      __mem_reader__continuous_stream_data_skid_reg <= mem_reader__continuous_stream_data_skid_data_load_en ? __mem_reader__continuous_stream_data_reg : __mem_reader__continuous_stream_data_skid_reg;
      __mem_reader__continuous_stream_data_valid_reg <= mem_reader__continuous_stream_data_data_valid_load_en__1 ? p2_valid : __mem_reader__continuous_stream_data_valid_reg;
      __mem_reader__continuous_stream_data_valid_skid_reg <= mem_reader__continuous_stream_data_skid_valid_load_en ? mem_reader__continuous_stream_data_from_skid_rdy : __mem_reader__continuous_stream_data_valid_skid_reg;
    end
  end
  assign mem_reader__axi_st_remove_rdy = mem_reader__axi_st_remove_data_from_skid_rdy;
  assign mem_reader__continuous_stream_data = mem_reader__continuous_stream_data_select;
  assign mem_reader__continuous_stream_vld = mem_reader__continuous_stream_data_valid_or;
endmodule


module mem_reader_adv__1(
  input wire clk,
  input wire rst
);

endmodule


module fifo_for_depth_1_ty__bits_128___bits_16___bits_16___bits_8___bits_8___bits_1___with_bypass_register_push(
  input wire clk,
  input wire rst,
  input wire push_valid,
  input wire pop_ready,
  input wire [176:0] push_data,
  output wire push_ready,
  output wire pop_valid,
  output wire [176:0] pop_data
);
  wire [176:0] buf__1_init[0:1];
  assign buf__1_init[0] = {128'h0000_0000_0000_0000_0000_0000_0000_0000, 16'h0000, 16'h0000, 8'h00, 8'h00, 1'h0};
  assign buf__1_init[1] = {128'h0000_0000_0000_0000_0000_0000_0000_0000, 16'h0000, 16'h0000, 8'h00, 8'h00, 1'h0};
  reg [1:0] head;
  reg [1:0] tail;
  reg [1:0] slots;
  reg [176:0] buf__1[0:1];
  wire is_full_bool;
  wire [1:0] literal_8465;
  wire can_do_push;
  wire and_8488;
  wire eq_8493;
  wire ne_8477;
  wire and_8494;
  wire or_8491;
  wire [2:0] add_8485;
  wire [2:0] long_buf_size_lit;
  wire [2:0] add_8480;
  wire popped;
  wire [1:0] sub_8506;
  wire [1:0] add_8508;
  wire [2:0] umod_8486;
  wire [2:0] umod_8481;
  wire pushed;
  wire [1:0] next_head_if_push;
  wire did_push_occur;
  wire [1:0] next_tail_if_pop;
  wire did_pop_occur;
  wire [1:0] sel_8510;
  wire [176:0] array_update_8517[0:1];
  assign is_full_bool = slots == 2'h1;
  assign literal_8465 = 2'h1;
  assign can_do_push = ~is_full_bool | pop_ready;
  assign and_8488 = pop_ready & push_valid;
  assign eq_8493 = head == tail;
  assign ne_8477 = head != tail;
  assign and_8494 = eq_8493 & and_8488;
  assign or_8491 = ne_8477 | push_valid;
  assign add_8485 = {1'h0, head} + {1'h0, literal_8465};
  assign long_buf_size_lit = 3'h2;
  assign add_8480 = {1'h0, tail} + {1'h0, literal_8465};
  assign popped = pop_ready & or_8491;
  assign sub_8506 = slots - literal_8465;
  assign add_8508 = slots + literal_8465;
  assign umod_8486 = add_8485 % long_buf_size_lit;
  assign umod_8481 = add_8480 % long_buf_size_lit;
  assign pushed = ~is_full_bool & push_valid;
  assign next_head_if_push = umod_8486[1:0];
  assign did_push_occur = (can_do_push | and_8488) & push_valid & ~and_8494 & ~is_full_bool;
  assign next_tail_if_pop = umod_8481[1:0];
  assign did_pop_occur = (ne_8477 | and_8488) & pop_ready & ~and_8494;
  assign sel_8510 = pushed ? (popped ? slots : add_8508) : (popped ? sub_8506 : slots);
  assign array_update_8517[0] = head == 2'h0 ? push_data : buf__1[0];
  assign array_update_8517[1] = head == 2'h1 ? push_data : buf__1[1];
  always @ (posedge clk) begin
    if (rst) begin
      head <= 2'h0;
      tail <= 2'h0;
      slots <= 2'h0;
      buf__1[0] <= buf__1_init[0];
      buf__1[1] <= buf__1_init[1];
    end else begin
      head <= did_push_occur ? next_head_if_push : head;
      tail <= did_pop_occur ? next_tail_if_pop : tail;
      slots <= sel_8510;
      buf__1[0] <= did_push_occur ? array_update_8517[0] : buf__1[0];
      buf__1[1] <= did_push_occur ? array_update_8517[1] : buf__1[1];
    end
  end
  assign push_ready = ~is_full_bool;
  assign pop_valid = or_8491;
  assign pop_data = eq_8493 ? push_data : buf__1[tail > 2'h1 ? 1'h1 : tail[0:0]];
endmodule


module fifo_for_depth_1_ty__bits_64___bits_8___bits_8___bits_8___bits_8___bits_1___with_bypass_register_push(
  input wire clk,
  input wire rst,
  input wire push_valid,
  input wire pop_ready,
  input wire [96:0] push_data,
  output wire push_ready,
  output wire pop_valid,
  output wire [96:0] pop_data
);
  wire [96:0] buf__1_init[0:1];
  assign buf__1_init[0] = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  assign buf__1_init[1] = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  reg [1:0] head;
  reg [1:0] tail;
  reg [1:0] slots;
  reg [96:0] buf__1[0:1];
  wire is_full_bool;
  wire [1:0] literal_8522;
  wire can_do_push;
  wire and_8545;
  wire eq_8550;
  wire ne_8534;
  wire and_8551;
  wire or_8548;
  wire [2:0] add_8542;
  wire [2:0] long_buf_size_lit;
  wire [2:0] add_8537;
  wire popped;
  wire [1:0] sub_8563;
  wire [1:0] add_8565;
  wire [2:0] umod_8543;
  wire [2:0] umod_8538;
  wire pushed;
  wire [1:0] next_head_if_push;
  wire did_push_occur;
  wire [1:0] next_tail_if_pop;
  wire did_pop_occur;
  wire [1:0] sel_8567;
  wire [96:0] array_update_8574[0:1];
  assign is_full_bool = slots == 2'h1;
  assign literal_8522 = 2'h1;
  assign can_do_push = ~is_full_bool | pop_ready;
  assign and_8545 = pop_ready & push_valid;
  assign eq_8550 = head == tail;
  assign ne_8534 = head != tail;
  assign and_8551 = eq_8550 & and_8545;
  assign or_8548 = ne_8534 | push_valid;
  assign add_8542 = {1'h0, head} + {1'h0, literal_8522};
  assign long_buf_size_lit = 3'h2;
  assign add_8537 = {1'h0, tail} + {1'h0, literal_8522};
  assign popped = pop_ready & or_8548;
  assign sub_8563 = slots - literal_8522;
  assign add_8565 = slots + literal_8522;
  assign umod_8543 = add_8542 % long_buf_size_lit;
  assign umod_8538 = add_8537 % long_buf_size_lit;
  assign pushed = ~is_full_bool & push_valid;
  assign next_head_if_push = umod_8543[1:0];
  assign did_push_occur = (can_do_push | and_8545) & push_valid & ~and_8551 & ~is_full_bool;
  assign next_tail_if_pop = umod_8538[1:0];
  assign did_pop_occur = (ne_8534 | and_8545) & pop_ready & ~and_8551;
  assign sel_8567 = pushed ? (popped ? slots : add_8565) : (popped ? sub_8563 : slots);
  assign array_update_8574[0] = head == 2'h0 ? push_data : buf__1[0];
  assign array_update_8574[1] = head == 2'h1 ? push_data : buf__1[1];
  always @ (posedge clk) begin
    if (rst) begin
      head <= 2'h0;
      tail <= 2'h0;
      slots <= 2'h0;
      buf__1[0] <= buf__1_init[0];
      buf__1[1] <= buf__1_init[1];
    end else begin
      head <= did_push_occur ? next_head_if_push : head;
      tail <= did_pop_occur ? next_tail_if_pop : tail;
      slots <= sel_8567;
      buf__1[0] <= did_push_occur ? array_update_8574[0] : buf__1[0];
      buf__1[1] <= did_push_occur ? array_update_8574[1] : buf__1[1];
    end
  end
  assign push_ready = ~is_full_bool;
  assign pop_valid = or_8548;
  assign pop_data = eq_8550 ? push_data : buf__1[tail > 2'h1 ? 1'h1 : tail[0:0]];
endmodule


module fifo_for_depth_1_ty__bits_64___bits_8___bits_8___bits_8___bits_8___bits_1___with_bypass_register_push___1(
  input wire clk,
  input wire rst,
  input wire push_valid,
  input wire pop_ready,
  input wire [96:0] push_data,
  output wire push_ready,
  output wire pop_valid,
  output wire [96:0] pop_data
);
  wire [96:0] buf__1_init[0:1];
  assign buf__1_init[0] = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  assign buf__1_init[1] = {64'h0000_0000_0000_0000, 8'h00, 8'h00, 8'h00, 8'h00, 1'h0};
  reg [1:0] head;
  reg [1:0] tail;
  reg [1:0] slots;
  reg [96:0] buf__1[0:1];
  wire is_full_bool;
  wire [1:0] literal_8579;
  wire can_do_push;
  wire and_8602;
  wire eq_8607;
  wire ne_8591;
  wire and_8608;
  wire or_8605;
  wire [2:0] add_8599;
  wire [2:0] long_buf_size_lit;
  wire [2:0] add_8594;
  wire popped;
  wire [1:0] sub_8620;
  wire [1:0] add_8622;
  wire [2:0] umod_8600;
  wire [2:0] umod_8595;
  wire pushed;
  wire [1:0] next_head_if_push;
  wire did_push_occur;
  wire [1:0] next_tail_if_pop;
  wire did_pop_occur;
  wire [1:0] sel_8624;
  wire [96:0] array_update_8631[0:1];
  assign is_full_bool = slots == 2'h1;
  assign literal_8579 = 2'h1;
  assign can_do_push = ~is_full_bool | pop_ready;
  assign and_8602 = pop_ready & push_valid;
  assign eq_8607 = head == tail;
  assign ne_8591 = head != tail;
  assign and_8608 = eq_8607 & and_8602;
  assign or_8605 = ne_8591 | push_valid;
  assign add_8599 = {1'h0, head} + {1'h0, literal_8579};
  assign long_buf_size_lit = 3'h2;
  assign add_8594 = {1'h0, tail} + {1'h0, literal_8579};
  assign popped = pop_ready & or_8605;
  assign sub_8620 = slots - literal_8579;
  assign add_8622 = slots + literal_8579;
  assign umod_8600 = add_8599 % long_buf_size_lit;
  assign umod_8595 = add_8594 % long_buf_size_lit;
  assign pushed = ~is_full_bool & push_valid;
  assign next_head_if_push = umod_8600[1:0];
  assign did_push_occur = (can_do_push | and_8602) & push_valid & ~and_8608 & ~is_full_bool;
  assign next_tail_if_pop = umod_8595[1:0];
  assign did_pop_occur = (ne_8591 | and_8602) & pop_ready & ~and_8608;
  assign sel_8624 = pushed ? (popped ? slots : add_8622) : (popped ? sub_8620 : slots);
  assign array_update_8631[0] = head == 2'h0 ? push_data : buf__1[0];
  assign array_update_8631[1] = head == 2'h1 ? push_data : buf__1[1];
  always @ (posedge clk) begin
    if (rst) begin
      head <= 2'h0;
      tail <= 2'h0;
      slots <= 2'h0;
      buf__1[0] <= buf__1_init[0];
      buf__1[1] <= buf__1_init[1];
    end else begin
      head <= did_push_occur ? next_head_if_push : head;
      tail <= did_pop_occur ? next_tail_if_pop : tail;
      slots <= sel_8624;
      buf__1[0] <= did_push_occur ? array_update_8631[0] : buf__1[0];
      buf__1[1] <= did_push_occur ? array_update_8631[1] : buf__1[1];
    end
  end
  assign push_ready = ~is_full_bool;
  assign pop_valid = or_8605;
  assign pop_data = eq_8607 ? push_data : buf__1[tail > 2'h1 ? 1'h1 : tail[0:0]];
endmodule


module fifo_for_depth_1_ty__bits_64___bits_7___bits_8___bits_8___bits_1___with_bypass_register_push(
  input wire clk,
  input wire rst,
  input wire push_valid,
  input wire pop_ready,
  input wire [87:0] push_data,
  output wire push_ready,
  output wire pop_valid,
  output wire [87:0] pop_data
);
  wire [87:0] buf__1_init[0:1];
  assign buf__1_init[0] = {64'h0000_0000_0000_0000, 7'h00, 8'h00, 8'h00, 1'h0};
  assign buf__1_init[1] = {64'h0000_0000_0000_0000, 7'h00, 8'h00, 8'h00, 1'h0};
  reg [1:0] head;
  reg [1:0] tail;
  reg [1:0] slots;
  reg [87:0] buf__1[0:1];
  wire is_full_bool;
  wire [1:0] literal_8636;
  wire can_do_push;
  wire and_8659;
  wire eq_8664;
  wire ne_8648;
  wire and_8665;
  wire or_8662;
  wire [2:0] add_8656;
  wire [2:0] long_buf_size_lit;
  wire [2:0] add_8651;
  wire popped;
  wire [1:0] sub_8677;
  wire [1:0] add_8679;
  wire [2:0] umod_8657;
  wire [2:0] umod_8652;
  wire pushed;
  wire [1:0] next_head_if_push;
  wire did_push_occur;
  wire [1:0] next_tail_if_pop;
  wire did_pop_occur;
  wire [1:0] sel_8681;
  wire [87:0] array_update_8688[0:1];
  assign is_full_bool = slots == 2'h1;
  assign literal_8636 = 2'h1;
  assign can_do_push = ~is_full_bool | pop_ready;
  assign and_8659 = pop_ready & push_valid;
  assign eq_8664 = head == tail;
  assign ne_8648 = head != tail;
  assign and_8665 = eq_8664 & and_8659;
  assign or_8662 = ne_8648 | push_valid;
  assign add_8656 = {1'h0, head} + {1'h0, literal_8636};
  assign long_buf_size_lit = 3'h2;
  assign add_8651 = {1'h0, tail} + {1'h0, literal_8636};
  assign popped = pop_ready & or_8662;
  assign sub_8677 = slots - literal_8636;
  assign add_8679 = slots + literal_8636;
  assign umod_8657 = add_8656 % long_buf_size_lit;
  assign umod_8652 = add_8651 % long_buf_size_lit;
  assign pushed = ~is_full_bool & push_valid;
  assign next_head_if_push = umod_8657[1:0];
  assign did_push_occur = (can_do_push | and_8659) & push_valid & ~and_8665 & ~is_full_bool;
  assign next_tail_if_pop = umod_8652[1:0];
  assign did_pop_occur = (ne_8648 | and_8659) & pop_ready & ~and_8665;
  assign sel_8681 = pushed ? (popped ? slots : add_8679) : (popped ? sub_8677 : slots);
  assign array_update_8688[0] = head == 2'h0 ? push_data : buf__1[0];
  assign array_update_8688[1] = head == 2'h1 ? push_data : buf__1[1];
  always @ (posedge clk) begin
    if (rst) begin
      head <= 2'h0;
      tail <= 2'h0;
      slots <= 2'h0;
      buf__1[0] <= buf__1_init[0];
      buf__1[1] <= buf__1_init[1];
    end else begin
      head <= did_push_occur ? next_head_if_push : head;
      tail <= did_pop_occur ? next_tail_if_pop : tail;
      slots <= sel_8681;
      buf__1[0] <= did_push_occur ? array_update_8688[0] : buf__1[0];
      buf__1[1] <= did_push_occur ? array_update_8688[1] : buf__1[1];
    end
  end
  assign push_ready = ~is_full_bool;
  assign pop_valid = or_8662;
  assign pop_data = eq_8664 ? push_data : buf__1[tail > 2'h1 ? 1'h1 : tail[0:0]];
endmodule


module fifo_for_depth_1_ty__bits_16___bits_16___bits_8___bits_4___bits_4___with_bypass_register_push(
  input wire clk,
  input wire rst,
  input wire push_valid,
  input wire pop_ready,
  input wire [47:0] push_data,
  output wire push_ready,
  output wire pop_valid,
  output wire [47:0] pop_data
);
  wire [47:0] buf__1_init[0:1];
  assign buf__1_init[0] = {16'h0000, 16'h0000, 8'h00, 4'h0, 4'h0};
  assign buf__1_init[1] = {16'h0000, 16'h0000, 8'h00, 4'h0, 4'h0};
  reg [1:0] head;
  reg [1:0] tail;
  reg [1:0] slots;
  reg [47:0] buf__1[0:1];
  wire is_full_bool;
  wire [1:0] literal_8693;
  wire can_do_push;
  wire and_8716;
  wire eq_8721;
  wire ne_8705;
  wire and_8722;
  wire or_8719;
  wire [2:0] add_8713;
  wire [2:0] long_buf_size_lit;
  wire [2:0] add_8708;
  wire popped;
  wire [1:0] sub_8734;
  wire [1:0] add_8736;
  wire [2:0] umod_8714;
  wire [2:0] umod_8709;
  wire pushed;
  wire [1:0] next_head_if_push;
  wire did_push_occur;
  wire [1:0] next_tail_if_pop;
  wire did_pop_occur;
  wire [1:0] sel_8738;
  wire [47:0] array_update_8745[0:1];
  assign is_full_bool = slots == 2'h1;
  assign literal_8693 = 2'h1;
  assign can_do_push = ~is_full_bool | pop_ready;
  assign and_8716 = pop_ready & push_valid;
  assign eq_8721 = head == tail;
  assign ne_8705 = head != tail;
  assign and_8722 = eq_8721 & and_8716;
  assign or_8719 = ne_8705 | push_valid;
  assign add_8713 = {1'h0, head} + {1'h0, literal_8693};
  assign long_buf_size_lit = 3'h2;
  assign add_8708 = {1'h0, tail} + {1'h0, literal_8693};
  assign popped = pop_ready & or_8719;
  assign sub_8734 = slots - literal_8693;
  assign add_8736 = slots + literal_8693;
  assign umod_8714 = add_8713 % long_buf_size_lit;
  assign umod_8709 = add_8708 % long_buf_size_lit;
  assign pushed = ~is_full_bool & push_valid;
  assign next_head_if_push = umod_8714[1:0];
  assign did_push_occur = (can_do_push | and_8716) & push_valid & ~and_8722 & ~is_full_bool;
  assign next_tail_if_pop = umod_8709[1:0];
  assign did_pop_occur = (ne_8705 | and_8716) & pop_ready & ~and_8722;
  assign sel_8738 = pushed ? (popped ? slots : add_8736) : (popped ? sub_8734 : slots);
  assign array_update_8745[0] = head == 2'h0 ? push_data : buf__1[0];
  assign array_update_8745[1] = head == 2'h1 ? push_data : buf__1[1];
  always @ (posedge clk) begin
    if (rst) begin
      head <= 2'h0;
      tail <= 2'h0;
      slots <= 2'h0;
      buf__1[0] <= buf__1_init[0];
      buf__1[1] <= buf__1_init[1];
    end else begin
      head <= did_push_occur ? next_head_if_push : head;
      tail <= did_pop_occur ? next_tail_if_pop : tail;
      slots <= sel_8738;
      buf__1[0] <= did_push_occur ? array_update_8745[0] : buf__1[0];
      buf__1[1] <= did_push_occur ? array_update_8745[1] : buf__1[1];
    end
  end
  assign push_ready = ~is_full_bool;
  assign pop_valid = or_8719;
  assign pop_data = eq_8721 ? push_data : buf__1[tail > 2'h1 ? 1'h1 : tail[0:0]];
endmodule


module fifo_for_depth_1_ty_bits_1__with_bypass_register_push(
  input wire clk,
  input wire rst,
  input wire push_valid,
  input wire pop_ready,
  input wire push_data,
  output wire push_ready,
  output wire pop_valid,
  output wire pop_data
);
  wire buf__1_init[0:1];
  assign buf__1_init[0] = 1'h0;
  assign buf__1_init[1] = 1'h0;
  reg [1:0] head;
  reg [1:0] tail;
  reg [1:0] slots;
  reg buf__1[0:1];
  wire is_full_bool;
  wire [1:0] literal_8750;
  wire can_do_push;
  wire and_8773;
  wire eq_8778;
  wire ne_8762;
  wire and_8779;
  wire or_8776;
  wire [2:0] add_8770;
  wire [2:0] long_buf_size_lit;
  wire [2:0] add_8765;
  wire popped;
  wire [1:0] sub_8791;
  wire [1:0] add_8793;
  wire [2:0] umod_8771;
  wire [2:0] umod_8766;
  wire pushed;
  wire [1:0] next_head_if_push;
  wire did_push_occur;
  wire [1:0] next_tail_if_pop;
  wire did_pop_occur;
  wire [1:0] sel_8795;
  wire array_update_8802[0:1];
  assign is_full_bool = slots == 2'h1;
  assign literal_8750 = 2'h1;
  assign can_do_push = ~is_full_bool | pop_ready;
  assign and_8773 = pop_ready & push_valid;
  assign eq_8778 = head == tail;
  assign ne_8762 = head != tail;
  assign and_8779 = eq_8778 & and_8773;
  assign or_8776 = ne_8762 | push_valid;
  assign add_8770 = {1'h0, head} + {1'h0, literal_8750};
  assign long_buf_size_lit = 3'h2;
  assign add_8765 = {1'h0, tail} + {1'h0, literal_8750};
  assign popped = pop_ready & or_8776;
  assign sub_8791 = slots - literal_8750;
  assign add_8793 = slots + literal_8750;
  assign umod_8771 = add_8770 % long_buf_size_lit;
  assign umod_8766 = add_8765 % long_buf_size_lit;
  assign pushed = ~is_full_bool & push_valid;
  assign next_head_if_push = umod_8771[1:0];
  assign did_push_occur = (can_do_push | and_8773) & push_valid & ~and_8779 & ~is_full_bool;
  assign next_tail_if_pop = umod_8766[1:0];
  assign did_pop_occur = (ne_8762 | and_8773) & pop_ready & ~and_8779;
  assign sel_8795 = pushed ? (popped ? slots : add_8793) : (popped ? sub_8791 : slots);
  always @ (posedge clk) begin
    if (rst) begin
      head <= 2'h0;
      tail <= 2'h0;
      slots <= 2'h0;
      buf__1[0] <= buf__1_init[0];
      buf__1[1] <= buf__1_init[1];
    end else begin
      head <= did_push_occur ? next_head_if_push : head;
      tail <= did_pop_occur ? next_tail_if_pop : tail;
      slots <= sel_8795;
      buf__1[0] <= did_push_occur ? array_update_8802[0] : buf__1[0];
      buf__1[1] <= did_push_occur ? array_update_8802[1] : buf__1[1];
    end
  end
  assign push_ready = ~is_full_bool;
  assign pop_valid = or_8776;
  assign pop_data = eq_8778 ? push_data : buf__1[tail > 2'h1 ? 1'h1 : tail[0:0]];
  genvar array_update_8802__index;
  generate
    for (array_update_8802__index = 0; array_update_8802__index < 2; array_update_8802__index = array_update_8802__index + 1) begin : array_update_8802__gen
      assign array_update_8802[array_update_8802__index] = head == array_update_8802__index ? push_data : buf__1[array_update_8802__index];
    end
  endgenerate
endmodule


module fifo_for_depth_1_ty__bits_16___bits_16___with_bypass_register_push(
  input wire clk,
  input wire rst,
  input wire push_valid,
  input wire pop_ready,
  input wire [31:0] push_data,
  output wire push_ready,
  output wire pop_valid,
  output wire [31:0] pop_data
);
  wire [31:0] buf__1_init[0:1];
  assign buf__1_init[0] = {16'h0000, 16'h0000};
  assign buf__1_init[1] = {16'h0000, 16'h0000};
  reg [1:0] head;
  reg [1:0] tail;
  reg [1:0] slots;
  reg [31:0] buf__1[0:1];
  wire is_full_bool;
  wire [1:0] literal_8807;
  wire can_do_push;
  wire and_8830;
  wire eq_8835;
  wire ne_8819;
  wire and_8836;
  wire or_8833;
  wire [2:0] add_8827;
  wire [2:0] long_buf_size_lit;
  wire [2:0] add_8822;
  wire popped;
  wire [1:0] sub_8848;
  wire [1:0] add_8850;
  wire [2:0] umod_8828;
  wire [2:0] umod_8823;
  wire pushed;
  wire [1:0] next_head_if_push;
  wire did_push_occur;
  wire [1:0] next_tail_if_pop;
  wire did_pop_occur;
  wire [1:0] sel_8852;
  wire [31:0] array_update_8859[0:1];
  assign is_full_bool = slots == 2'h1;
  assign literal_8807 = 2'h1;
  assign can_do_push = ~is_full_bool | pop_ready;
  assign and_8830 = pop_ready & push_valid;
  assign eq_8835 = head == tail;
  assign ne_8819 = head != tail;
  assign and_8836 = eq_8835 & and_8830;
  assign or_8833 = ne_8819 | push_valid;
  assign add_8827 = {1'h0, head} + {1'h0, literal_8807};
  assign long_buf_size_lit = 3'h2;
  assign add_8822 = {1'h0, tail} + {1'h0, literal_8807};
  assign popped = pop_ready & or_8833;
  assign sub_8848 = slots - literal_8807;
  assign add_8850 = slots + literal_8807;
  assign umod_8828 = add_8827 % long_buf_size_lit;
  assign umod_8823 = add_8822 % long_buf_size_lit;
  assign pushed = ~is_full_bool & push_valid;
  assign next_head_if_push = umod_8828[1:0];
  assign did_push_occur = (can_do_push | and_8830) & push_valid & ~and_8836 & ~is_full_bool;
  assign next_tail_if_pop = umod_8823[1:0];
  assign did_pop_occur = (ne_8819 | and_8830) & pop_ready & ~and_8836;
  assign sel_8852 = pushed ? (popped ? slots : add_8850) : (popped ? sub_8848 : slots);
  assign array_update_8859[0] = head == 2'h0 ? push_data : buf__1[0];
  assign array_update_8859[1] = head == 2'h1 ? push_data : buf__1[1];
  always @ (posedge clk) begin
    if (rst) begin
      head <= 2'h0;
      tail <= 2'h0;
      slots <= 2'h0;
      buf__1[0] <= buf__1_init[0];
      buf__1[1] <= buf__1_init[1];
    end else begin
      head <= did_push_occur ? next_head_if_push : head;
      tail <= did_pop_occur ? next_tail_if_pop : tail;
      slots <= sel_8852;
      buf__1[0] <= did_push_occur ? array_update_8859[0] : buf__1[0];
      buf__1[1] <= did_push_occur ? array_update_8859[1] : buf__1[1];
    end
  end
  assign push_ready = ~is_full_bool;
  assign pop_valid = or_8833;
  assign pop_data = eq_8835 ? push_data : buf__1[tail > 2'h1 ? 1'h1 : tail[0:0]];
endmodule


module fifo_for_depth_1_ty_bits_1__with_bypass_register_push___1(
  input wire clk,
  input wire rst,
  input wire push_valid,
  input wire pop_ready,
  input wire push_data,
  output wire push_ready,
  output wire pop_valid,
  output wire pop_data
);
  wire buf__1_init[0:1];
  assign buf__1_init[0] = 1'h0;
  assign buf__1_init[1] = 1'h0;
  reg [1:0] head;
  reg [1:0] tail;
  reg [1:0] slots;
  reg buf__1[0:1];
  wire is_full_bool;
  wire [1:0] literal_8864;
  wire can_do_push;
  wire and_8887;
  wire eq_8892;
  wire ne_8876;
  wire and_8893;
  wire or_8890;
  wire [2:0] add_8884;
  wire [2:0] long_buf_size_lit;
  wire [2:0] add_8879;
  wire popped;
  wire [1:0] sub_8905;
  wire [1:0] add_8907;
  wire [2:0] umod_8885;
  wire [2:0] umod_8880;
  wire pushed;
  wire [1:0] next_head_if_push;
  wire did_push_occur;
  wire [1:0] next_tail_if_pop;
  wire did_pop_occur;
  wire [1:0] sel_8909;
  wire array_update_8916[0:1];
  assign is_full_bool = slots == 2'h1;
  assign literal_8864 = 2'h1;
  assign can_do_push = ~is_full_bool | pop_ready;
  assign and_8887 = pop_ready & push_valid;
  assign eq_8892 = head == tail;
  assign ne_8876 = head != tail;
  assign and_8893 = eq_8892 & and_8887;
  assign or_8890 = ne_8876 | push_valid;
  assign add_8884 = {1'h0, head} + {1'h0, literal_8864};
  assign long_buf_size_lit = 3'h2;
  assign add_8879 = {1'h0, tail} + {1'h0, literal_8864};
  assign popped = pop_ready & or_8890;
  assign sub_8905 = slots - literal_8864;
  assign add_8907 = slots + literal_8864;
  assign umod_8885 = add_8884 % long_buf_size_lit;
  assign umod_8880 = add_8879 % long_buf_size_lit;
  assign pushed = ~is_full_bool & push_valid;
  assign next_head_if_push = umod_8885[1:0];
  assign did_push_occur = (can_do_push | and_8887) & push_valid & ~and_8893 & ~is_full_bool;
  assign next_tail_if_pop = umod_8880[1:0];
  assign did_pop_occur = (ne_8876 | and_8887) & pop_ready & ~and_8893;
  assign sel_8909 = pushed ? (popped ? slots : add_8907) : (popped ? sub_8905 : slots);
  always @ (posedge clk) begin
    if (rst) begin
      head <= 2'h0;
      tail <= 2'h0;
      slots <= 2'h0;
      buf__1[0] <= buf__1_init[0];
      buf__1[1] <= buf__1_init[1];
    end else begin
      head <= did_push_occur ? next_head_if_push : head;
      tail <= did_pop_occur ? next_tail_if_pop : tail;
      slots <= sel_8909;
      buf__1[0] <= did_push_occur ? array_update_8916[0] : buf__1[0];
      buf__1[1] <= did_push_occur ? array_update_8916[1] : buf__1[1];
    end
  end
  assign push_ready = ~is_full_bool;
  assign pop_valid = or_8890;
  assign pop_data = eq_8892 ? push_data : buf__1[tail > 2'h1 ? 1'h1 : tail[0:0]];
  genvar array_update_8916__index;
  generate
    for (array_update_8916__index = 0; array_update_8916__index < 2; array_update_8916__index = array_update_8916__index + 1) begin : array_update_8916__gen
      assign array_update_8916[array_update_8916__index] = head == array_update_8916__index ? push_data : buf__1[array_update_8916__index];
    end
  endgenerate
endmodule


module mem_reader_adv(
  input wire clk,
  input wire rst,
  input wire mem_reader__axi_ar_s_rdy,
  input wire [139:0] mem_reader__axi_r_r_data,
  input wire mem_reader__axi_r_r_vld,
  input wire [31:0] mem_reader__req_r_data,
  input wire mem_reader__req_r_vld,
  input wire mem_reader__resp_s_rdy,
  output wire [51:0] mem_reader__axi_ar_s_data,
  output wire mem_reader__axi_ar_s_vld,
  output wire mem_reader__axi_r_r_rdy,
  output wire mem_reader__req_r_rdy,
  output wire [81:0] mem_reader__resp_s_data,
  output wire mem_reader__resp_s_vld
);
  wire instantiation_output_8276;
  wire instantiation_output_8328;
  wire [31:0] instantiation_output_8333;
  wire instantiation_output_8334;
  wire instantiation_output_8347;
  wire [81:0] instantiation_output_8351;
  wire instantiation_output_8352;
  wire [51:0] instantiation_output_8242;
  wire instantiation_output_8243;
  wire [47:0] instantiation_output_8307;
  wire instantiation_output_8308;
  wire instantiation_output_8320;
  wire instantiation_output_8321;
  wire instantiation_output_8341;
  wire instantiation_output_8366;
  wire instantiation_output_8250;
  wire [176:0] instantiation_output_8255;
  wire instantiation_output_8256;
  wire instantiation_output_8315;
  wire instantiation_output_8358;
  wire instantiation_output_8359;
  wire instantiation_output_8263;
  wire [96:0] instantiation_output_8281;
  wire instantiation_output_8282;
  wire [96:0] instantiation_output_8268;
  wire instantiation_output_8269;
  wire instantiation_output_8302;
  wire instantiation_output_8289;
  wire [87:0] instantiation_output_8294;
  wire instantiation_output_8295;
  wire instantiation_output_8924;
  wire [176:0] instantiation_output_8925;
  wire instantiation_output_8926;
  wire instantiation_output_8931;
  wire [96:0] instantiation_output_8932;
  wire instantiation_output_8933;
  wire instantiation_output_8938;
  wire [96:0] instantiation_output_8939;
  wire instantiation_output_8940;
  wire instantiation_output_8945;
  wire [87:0] instantiation_output_8946;
  wire instantiation_output_8947;
  wire instantiation_output_8952;
  wire [47:0] instantiation_output_8953;
  wire instantiation_output_8954;
  wire instantiation_output_8959;
  wire instantiation_output_8960;
  wire instantiation_output_8961;
  wire instantiation_output_8966;
  wire [31:0] instantiation_output_8967;
  wire instantiation_output_8968;
  wire instantiation_output_8973;
  wire instantiation_output_8974;
  wire instantiation_output_8975;

  // ===== Instantiations
  __mem_reader__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__MemReaderInternalNoFsm_0__16_128_16_8_8_2_2_16_64_8_next __mem_reader__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__MemReaderInternalNoFsm_0__16_128_16_8_8_2_2_16_64_8_next_inst0 (
    .rst(rst),
    .mem_reader__axi_st_out_data(instantiation_output_8932),
    .mem_reader__axi_st_out_vld(instantiation_output_8933),
    .mem_reader__reader_err_data(instantiation_output_8960),
    .mem_reader__reader_err_vld(instantiation_output_8961),
    .mem_reader__reader_req_rdy(instantiation_output_8966),
    .mem_reader__req_r_data(mem_reader__req_r_data),
    .mem_reader__req_r_vld(mem_reader__req_r_vld),
    .mem_reader__resp_s_rdy(mem_reader__resp_s_rdy),
    .mem_reader__axi_st_out_rdy(instantiation_output_8276),
    .mem_reader__reader_err_rdy(instantiation_output_8328),
    .mem_reader__reader_req_data(instantiation_output_8333),
    .mem_reader__reader_req_vld(instantiation_output_8334),
    .mem_reader__req_r_rdy(instantiation_output_8347),
    .mem_reader__resp_s_data(instantiation_output_8351),
    .mem_reader__resp_s_vld(instantiation_output_8352),
    .clk(clk)
  );
  __xls_modules_zstd_memory_axi_reader__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiReaderNoFsm_0__16_128_16_8_8_5_4_12_next __xls_modules_zstd_memory_axi_reader__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiReaderNoFsm_0__16_128_16_8_8_5_4_12_next_inst1 (
    .rst(rst),
    .mem_reader__axi_ar_s_rdy(mem_reader__axi_ar_s_rdy),
    .mem_reader__rconf_rdy(instantiation_output_8952),
    .mem_reader__reader_err_rdy(instantiation_output_8959),
    .mem_reader__reader_req_data(instantiation_output_8967),
    .mem_reader__reader_req_vld(instantiation_output_8968),
    .mem_reader__rresp_data(instantiation_output_8974),
    .mem_reader__rresp_vld(instantiation_output_8975),
    .mem_reader__axi_ar_s_data(instantiation_output_8242),
    .mem_reader__axi_ar_s_vld(instantiation_output_8243),
    .mem_reader__rconf_data(instantiation_output_8307),
    .mem_reader__rconf_vld(instantiation_output_8308),
    .mem_reader__reader_err_data(instantiation_output_8320),
    .mem_reader__reader_err_vld(instantiation_output_8321),
    .mem_reader__reader_req_rdy(instantiation_output_8341),
    .mem_reader__rresp_rdy(instantiation_output_8366),
    .clk(clk)
  );
  __xls_modules_zstd_memory_axi_reader__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiReaderNoFsm_0__AxiReaderInternalR_0__16_128_16_8_8_5_4_12_next __xls_modules_zstd_memory_axi_reader__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiReaderNoFsm_0__AxiReaderInternalR_0__16_128_16_8_8_5_4_12_next_inst2 (
    .rst(rst),
    .mem_reader__axi_r_r_data(mem_reader__axi_r_r_data),
    .mem_reader__axi_r_r_vld(mem_reader__axi_r_r_vld),
    .mem_reader__axi_st_in_rdy(instantiation_output_8924),
    .mem_reader__rconf_data(instantiation_output_8953),
    .mem_reader__rconf_vld(instantiation_output_8954),
    .mem_reader__rresp_rdy(instantiation_output_8973),
    .mem_reader__axi_r_r_rdy(instantiation_output_8250),
    .mem_reader__axi_st_in_data(instantiation_output_8255),
    .mem_reader__axi_st_in_vld(instantiation_output_8256),
    .mem_reader__rconf_rdy(instantiation_output_8315),
    .mem_reader__rresp_data(instantiation_output_8358),
    .mem_reader__rresp_vld(instantiation_output_8359),
    .clk(clk)
  );
  __xls_modules_zstd_memory_axi_stream_downscaler__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiStreamDownscaler_0__8_8_128_16_64_8_2_2_next __xls_modules_zstd_memory_axi_stream_downscaler__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiStreamDownscaler_0__8_8_128_16_64_8_2_2_next_inst3 (
    .rst(rst),
    .mem_reader__axi_st_in_data(instantiation_output_8925),
    .mem_reader__axi_st_in_vld(instantiation_output_8926),
    .mem_reader__axi_st_remove_rdy(instantiation_output_8938),
    .mem_reader__axi_st_in_rdy(instantiation_output_8263),
    .mem_reader__axi_st_remove_data(instantiation_output_8281),
    .mem_reader__axi_st_remove_vld(instantiation_output_8282),
    .clk(clk)
  );
  __xls_modules_zstd_memory_axi_stream_remove_empty__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiStreamRemoveEmpty_0__AxiStreamRemoveEmptyInternal_0__64_8_7_8_8_next __xls_modules_zstd_memory_axi_stream_remove_empty__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiStreamRemoveEmpty_0__AxiStreamRemoveEmptyInternal_0__64_8_7_8_8_next_inst4 (
    .rst(rst),
    .mem_reader__axi_st_out_rdy(instantiation_output_8931),
    .mem_reader__continuous_stream_data(instantiation_output_8946),
    .mem_reader__continuous_stream_vld(instantiation_output_8947),
    .mem_reader__axi_st_out_data(instantiation_output_8268),
    .mem_reader__axi_st_out_vld(instantiation_output_8269),
    .mem_reader__continuous_stream_rdy(instantiation_output_8302),
    .clk(clk)
  );
  __xls_modules_zstd_memory_axi_stream_remove_empty__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiStreamRemoveEmpty_0__RemoveEmptyBytes_0__64_8_7_8_10_8_next __xls_modules_zstd_memory_axi_stream_remove_empty__MemReaderAdvNoFsmInst__MemReaderAdvNoFsm_0__AxiStreamRemoveEmpty_0__RemoveEmptyBytes_0__64_8_7_8_10_8_next_inst5 (
    .rst(rst),
    .mem_reader__axi_st_remove_data(instantiation_output_8939),
    .mem_reader__axi_st_remove_vld(instantiation_output_8940),
    .mem_reader__continuous_stream_rdy(instantiation_output_8945),
    .mem_reader__axi_st_remove_rdy(instantiation_output_8289),
    .mem_reader__continuous_stream_data(instantiation_output_8294),
    .mem_reader__continuous_stream_vld(instantiation_output_8295),
    .clk(clk)
  );
  mem_reader_adv__1 mem_reader_adv__1_inst6 (
    .rst(rst),
    .clk(clk)
  );
  fifo_for_depth_1_ty__bits_128___bits_16___bits_16___bits_8___bits_8___bits_1___with_bypass_register_push materialized_fifo_fifo_mem_reader__axi_st_in_ (
    .rst(rst),
    .push_data(instantiation_output_8255),
    .push_valid(instantiation_output_8256),
    .pop_ready(instantiation_output_8263),
    .push_ready(instantiation_output_8924),
    .pop_data(instantiation_output_8925),
    .pop_valid(instantiation_output_8926),
    .clk(clk)
  );
  fifo_for_depth_1_ty__bits_64___bits_8___bits_8___bits_8___bits_8___bits_1___with_bypass_register_push materialized_fifo_fifo_mem_reader__axi_st_out_ (
    .rst(rst),
    .push_data(instantiation_output_8268),
    .push_valid(instantiation_output_8269),
    .pop_ready(instantiation_output_8276),
    .push_ready(instantiation_output_8931),
    .pop_data(instantiation_output_8932),
    .pop_valid(instantiation_output_8933),
    .clk(clk)
  );
  fifo_for_depth_1_ty__bits_64___bits_8___bits_8___bits_8___bits_8___bits_1___with_bypass_register_push___1 materialized_fifo_fifo_mem_reader__axi_st_remove_ (
    .rst(rst),
    .push_data(instantiation_output_8281),
    .push_valid(instantiation_output_8282),
    .pop_ready(instantiation_output_8289),
    .push_ready(instantiation_output_8938),
    .pop_data(instantiation_output_8939),
    .pop_valid(instantiation_output_8940),
    .clk(clk)
  );
  fifo_for_depth_1_ty__bits_64___bits_7___bits_8___bits_8___bits_1___with_bypass_register_push materialized_fifo_fifo_mem_reader__continuous_stream_ (
    .rst(rst),
    .push_data(instantiation_output_8294),
    .push_valid(instantiation_output_8295),
    .pop_ready(instantiation_output_8302),
    .push_ready(instantiation_output_8945),
    .pop_data(instantiation_output_8946),
    .pop_valid(instantiation_output_8947),
    .clk(clk)
  );
  fifo_for_depth_1_ty__bits_16___bits_16___bits_8___bits_4___bits_4___with_bypass_register_push materialized_fifo_fifo_mem_reader__rconf_ (
    .rst(rst),
    .push_data(instantiation_output_8307),
    .push_valid(instantiation_output_8308),
    .pop_ready(instantiation_output_8315),
    .push_ready(instantiation_output_8952),
    .pop_data(instantiation_output_8953),
    .pop_valid(instantiation_output_8954),
    .clk(clk)
  );
  fifo_for_depth_1_ty_bits_1__with_bypass_register_push materialized_fifo_fifo_mem_reader__reader_err_ (
    .rst(rst),
    .push_data(instantiation_output_8320),
    .push_valid(instantiation_output_8321),
    .pop_ready(instantiation_output_8328),
    .push_ready(instantiation_output_8959),
    .pop_data(instantiation_output_8960),
    .pop_valid(instantiation_output_8961),
    .clk(clk)
  );
  fifo_for_depth_1_ty__bits_16___bits_16___with_bypass_register_push materialized_fifo_fifo_mem_reader__reader_req_ (
    .rst(rst),
    .push_data(instantiation_output_8333),
    .push_valid(instantiation_output_8334),
    .pop_ready(instantiation_output_8341),
    .push_ready(instantiation_output_8966),
    .pop_data(instantiation_output_8967),
    .pop_valid(instantiation_output_8968),
    .clk(clk)
  );
  fifo_for_depth_1_ty_bits_1__with_bypass_register_push___1 materialized_fifo_fifo_mem_reader__rresp_ (
    .rst(rst),
    .push_data(instantiation_output_8358),
    .push_valid(instantiation_output_8359),
    .pop_ready(instantiation_output_8366),
    .push_ready(instantiation_output_8973),
    .pop_data(instantiation_output_8974),
    .pop_valid(instantiation_output_8975),
    .clk(clk)
  );
  assign mem_reader__axi_ar_s_data = instantiation_output_8242;
  assign mem_reader__axi_ar_s_vld = instantiation_output_8243;
  assign mem_reader__axi_r_r_rdy = instantiation_output_8250;
  assign mem_reader__req_r_rdy = instantiation_output_8347;
  assign mem_reader__resp_s_data = instantiation_output_8351;
  assign mem_reader__resp_s_vld = instantiation_output_8352;
endmodule
