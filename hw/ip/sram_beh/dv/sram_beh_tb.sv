// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// sram_beh_tb — wraps sram_beh_banked (DUT) alongside a flat "magic" array
// REF of the same logical shape, both driven by the identical 1R1W stimulus.
// The REF models the macro's read-before-write + ReadLat semantics so a clean
// banked tiling must match it bit-for-bit. Picks a config that forces real
// banking in BOTH dimensions (NW>1 and ND>1).

module sram_beh_tb #(
  parameter int LWidth   = 256,   // > MaxWidth  -> NW = 2
  parameter int LDepth   = 96,    // > MaxDepth  -> ND = 3
  parameter int MaxWidth = 144,
  parameter int MaxDepth = 32,
  parameter int ReadLat  = 1,
  localparam int AddrW = (LDepth <= 1) ? 1 : $clog2(LDepth)
) (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               w_en_i,
  input  logic [AddrW-1:0]    w_addr_i,
  input  logic [LWidth-1:0]   w_data_i,
  input  logic               r_en_i,
  input  logic [AddrW-1:0]    r_addr_i,
  output logic               mismatch_o
);

  logic [LWidth-1:0] dut_rd, ref_rd;

  sram_beh_banked #(.LWidth(LWidth), .LDepth(LDepth),
                    .MaxWidth(MaxWidth), .MaxDepth(MaxDepth),
                    .ReadLat(ReadLat)) u_dut (
    .clk_i, .rst_ni,
    .w_en_i, .w_addr_i, .w_data_i,
    .r_en_i, .r_addr_i, .r_data_o(dut_rd)
  );

  // ── flat reference: a single magic array with matching pipe semantics ──
  logic [LWidth-1:0] refmem [LDepth];
  logic [LWidth-1:0] ref_pipe [ReadLat];
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < ReadLat; i++) ref_pipe[i] <= '0;
    end else begin
      if (w_en_i) refmem[w_addr_i] <= w_data_i;
      // read-before-write: sample memory before this cycle's write commits
      ref_pipe[0] <= r_en_i ? refmem[r_addr_i] : ref_pipe[0];
      for (int i = 1; i < ReadLat; i++) ref_pipe[i] <= ref_pipe[i-1];
    end
  end
  assign ref_rd = ref_pipe[ReadLat-1];

  assign mismatch_o = (dut_rd !== ref_rd);

endmodule

// Named configs (ReadLat 1 and 2). Both shapes force NW>1 and ND>1.
/* verilator lint_off DECLFILENAME */
module sram_beh_tb_lat1 (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               w_en_i,
  input  logic [6:0]         w_addr_i,
  input  logic [255:0]       w_data_i,
  input  logic               r_en_i,
  input  logic [6:0]         r_addr_i,
  output logic               mismatch_o
);
  sram_beh_tb #(.ReadLat(1)) u (.*);
endmodule

module sram_beh_tb_lat2 (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               w_en_i,
  input  logic [6:0]         w_addr_i,
  input  logic [255:0]       w_data_i,
  input  logic               r_en_i,
  input  logic [6:0]         r_addr_i,
  output logic               mismatch_o
);
  sram_beh_tb #(.ReadLat(2)) u (.*);
endmodule
/* verilator lint_on DECLFILENAME */
