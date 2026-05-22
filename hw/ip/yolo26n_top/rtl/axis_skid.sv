// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// axis_skid — 2-entry AXI4-Stream skid buffer (a.k.a. register slice).
//
// Fully registers TVALID/TREADY/TDATA/TUSER/TLAST across the cut. Sustains
// full throughput (one beat per cycle in steady state). Used at the chip
// boundary so PD sees flops driving / capturing every AXIS pin.

module axis_skid #(
  parameter int unsigned DATA_W = 64,
  parameter int unsigned USER_W = 1
) (
  input  logic              clk_i,
  input  logic              rst_ni,

  input  logic              s_tvalid_i,
  output logic              s_tready_o,
  input  logic [DATA_W-1:0] s_tdata_i,
  input  logic [USER_W-1:0] s_tuser_i,
  input  logic              s_tlast_i,

  output logic              m_tvalid_o,
  input  logic              m_tready_i,
  output logic [DATA_W-1:0] m_tdata_o,
  output logic [USER_W-1:0] m_tuser_o,
  output logic              m_tlast_o
);

  // Primary register
  logic              p_valid_q;
  logic [DATA_W-1:0] p_data_q;
  logic [USER_W-1:0] p_user_q;
  logic              p_last_q;

  // Skid register (fills only when downstream stalls while a new beat arrives)
  logic              s_valid_q;
  logic [DATA_W-1:0] s_data_q;
  logic [USER_W-1:0] s_user_q;
  logic              s_last_q;

  assign s_tready_o = ~s_valid_q;
  assign m_tvalid_o = p_valid_q;
  assign m_tdata_o  = p_data_q;
  assign m_tuser_o  = p_user_q;
  assign m_tlast_o  = p_last_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      p_valid_q <= 1'b0;
      s_valid_q <= 1'b0;
      p_data_q  <= '0;
      p_user_q  <= '0;
      p_last_q  <= 1'b0;
      s_data_q  <= '0;
      s_user_q  <= '0;
      s_last_q  <= 1'b0;
    end else begin
      // primary advances when consumer accepts or it's empty
      if (m_tready_i || !p_valid_q) begin
        if (s_valid_q) begin
          p_valid_q <= 1'b1;
          p_data_q  <= s_data_q;
          p_user_q  <= s_user_q;
          p_last_q  <= s_last_q;
          s_valid_q <= 1'b0;
        end else begin
          p_valid_q <= s_tvalid_i;
          p_data_q  <= s_tdata_i;
          p_user_q  <= s_tuser_i;
          p_last_q  <= s_tlast_i;
        end
      end else if (s_tvalid_i && s_tready_o) begin
        // primary busy and stalled; capture upstream into skid
        s_valid_q <= 1'b1;
        s_data_q  <= s_tdata_i;
        s_user_q  <= s_tuser_i;
        s_last_q  <= s_tlast_i;
      end
    end
  end

endmodule
