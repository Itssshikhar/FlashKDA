#pragma once

#include "utils.cuh"

// K1 publishes each workspace slot as a byte image of its shared-memory
// layout. K2 restores the same byte image with matching bulk copies.

template <int D, int CHUNK = 16> struct K1Layouts {
  using QKLayout =
      decltype(make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), LayoutRight{}));
  using GLayout =
      decltype(make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), LayoutRight{}));
  using MMALayout =
      decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
                             make_shape(Int<CHUNK>{}, Int<D>{}), LayoutLeft{}));
  using BetaSmemLayout = Layout<Shape<Int<32>>, Stride<Int<1>>>;
  using GTotalLayout = Layout<Shape<Int<D>>, Stride<Int<1>>>;
  using LMLayout = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
      make_shape(Int<CHUNK>{}, Int<CHUNK>{}), LayoutLeft{}));
  using TransposedLMLayout = decltype(tile_to_shape(
      GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
      make_shape(Int<CHUNK>{}, Int<CHUNK>{}), LayoutRight{}));

  using TMABetaSmemLayout = BetaSmemLayout; // 1D TMA, no dummy dim
  using TMAQKLayout =
      decltype(prepend(QKLayout{})); // adds a leading size-1 mode to layout
  using TMAGLayout = decltype(prepend(GLayout{}));
  using TMAGTotalSmemLayout = decltype(prepend(GTotalLayout{}));
};

template <class Layouts> struct SharedStorageK1 {
  using BF16 = cutlass::bfloat16_t;
  using QKLayout = typename Layouts::QKLayout;
  using GLayout = typename Layouts::GLayout;
  using BetaSmemLayout = typename Layouts::BetaSmemLayout;
  using GTotalLayout = typename Layouts::GTotalLayout;
  using LMLayout = typename Layouts::LMLayout;
  using MMALayout = typename Layouts::MMALayout;

  // Phase A: q, k, g alive
  // Phase B: k_decayed, q_decayed, k_inv, L, INV, Mqk alive
  // These don't overlap → union saves ~14KB shared memory
  union {
    struct {
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> q;
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> k;
      alignas(128) cute::ArrayEngine<float, cute::cosize_v<GLayout>> g;
    };
    struct {
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed;
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_inv;
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> L;
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> Mqk;
    };
  };

  alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
  alignas(16) cute::ArrayEngine<float, 16> beta_act;
  alignas(16) cute::ArrayEngine<float, 16> q_norm_inv;
  alignas(16) cute::ArrayEngine<float, 16> k_norm_inv;
  float a_log_exp;

  union {
    alignas(128) cute::ArrayEngine<
        BF16, cute::cosize_v<QKLayout>> g_bf16; // TMA load target
    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
  };
  union {
    alignas(128) cute::ArrayEngine<
        float, cute::cosize_v<GTotalLayout>> dt_bias; // TMA load target
    alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
  };
  alignas(16) cutlass::arch::ClusterTransactionBarrier tma_load_barrier;
};

// ==================== Kernel 1: Prepare ====================
template <
    class TmaLoadQ, class TmaLoadK, class TmaLoadBeta, class TmaLoadG,
    class TmaLoadDtBias, int CHUNK, int D, int NumThreads,
    bool IsVarlen = true>
CUTLASS_DEVICE void flash_kda_fwd_prepare_body(
    TmaLoadQ const &tma_load_q, TmaLoadK const &tma_load_k,
    TmaLoadBeta const &tma_load_beta, TmaLoadG const &tma_load_g,
    TmaLoadDtBias const &tma_load_dt_bias, float scale, int T_total, int H,
    int N, int64_t const *cu_seqlens, int total_tiles,
    float const *A_log_ptr,
    float gate_scale, SharedStorageK1<K1Layouts<D, CHUNK>> &shared_storage,
    int global_tile_idx, int head_idx,
    K1WorkspaceRawPointers ws_raw = {}) {
  // --- constants
  using BF16 = cutlass::bfloat16_t;
  using FP16 = cutlass::half_t;
  using Layouts = K1Layouts<D, CHUNK>;
  using MMALayout = typename Layouts::MMALayout;
  using QKLayout = typename Layouts::QKLayout;
  using GLayout = typename Layouts::GLayout;
  using BetaSmemLayout = typename Layouts::BetaSmemLayout;
  using GTotalLayout = typename Layouts::GTotalLayout;
  using LMLayout = typename Layouts::LMLayout;
  using TransposedLMLayout = typename Layouts::TransposedLMLayout;
  using TMAQKLayout = typename Layouts::TMAQKLayout;
  using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
  using TMAGTotalSmemLayout = typename Layouts::TMAGTotalSmemLayout;
  int const local_tid = int(threadIdx.x);
  static_assert(NumThreads == 128,
                "The production K1 configuration uses four warps.");
  constexpr uint32_t kTmaTransactionBytes =
      uint32_t(cute::cosize_v<QKLayout>) *
          uint32_t(3 * sizeof(BF16)) +        // q + k + g_bf16
      uint32_t(32) * uint32_t(sizeof(BF16)) + // beta (bf16, sigmoid fused)
      uint32_t(D) * uint32_t(sizeof(float));  // dt_bias

  // --- per-CTA tile info
  int seq_idx, tiles_before, local_t;
  int64_t bos, eos;
  int seq_len, t_tiles_this_seq;

  if constexpr (IsVarlen) {
    // Linear scan on cu_seqlens to find (seq_idx, local_t)
    seq_idx = 0;
    tiles_before = 0;
    for (int i = 0; i < N; i++) {
      int slen = int(cu_seqlens[i + 1] - cu_seqlens[i]);
      int n_tiles = (slen + CHUNK - 1) / CHUNK;
      if (tiles_before + n_tiles > global_tile_idx) {
        seq_idx = i;
        break;
      }
      tiles_before += n_tiles;
    }
    local_t = global_tile_idx - tiles_before;
    bos = cu_seqlens[seq_idx];
    eos = cu_seqlens[seq_idx + 1];
  } else {
    int T_seq = T_total / N;
    int tiles_per_seq = (T_seq + CHUNK - 1) / CHUNK;
    seq_idx = global_tile_idx / tiles_per_seq;
    tiles_before = seq_idx * tiles_per_seq;
    local_t = global_tile_idx - tiles_before;
    bos = seq_idx * T_seq;
    eos = bos + T_seq;
  }
  seq_len = int(eos - bos);
  t_tiles_this_seq = (seq_len + CHUNK - 1) / CHUNK;
  // Early exit for excess CTAs (total_tiles is an upper bound)
  if (local_t >= t_tiles_this_seq)
    return;

  // --- TMA load inputs (single-shot, no pipeline)
  // Only thread 0 issues TMA loads (not elect_one_sync which is per-warp)
  if (local_tid == 0) {
    using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
    shared_storage.tma_load_barrier.init(1);
    shared_storage.tma_load_barrier.arrive_and_expect_tx(kTmaTransactionBytes);

    //  Create the compile-time-known slice-0 TMA copy helper for q.
    //  Then that helper is used to partition the global q tile and
    //  shared-memory q tile for the actual TMA copy.
    Tensor g_q = tma_load_q.get_tma_tensor(make_shape(H, T_total, D));
    Tensor g_k = tma_load_k.get_tma_tensor(make_shape(H, T_total, D));
    Tensor g_beta = tma_load_beta.get_tma_tensor(make_shape(H * T_total));

    auto cta_tma_load_q = tma_load_q.get_slice(Int<0>{});
    auto cta_tma_load_k = tma_load_k.get_slice(Int<0>{});
    auto cta_tma_load_beta = tma_load_beta.get_slice(Int<0>{});

    auto qk_off = g_q.layout()(head_idx, int(bos) + local_t * CHUNK, 0); // basically since we want to load same head, token_start, & column, "qk_off" is used as it is in both g_q_tile, g_k_tile.
    auto tile_shape_3d = make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{});
    auto tile_stride_3d = stride(g_q.layout());
    Tensor g_q_tile = make_tensor(g_q.data() + qk_off,
                                  make_layout(tile_shape_3d, tile_stride_3d));
    Tensor g_k_tile = make_tensor(g_k.data() + qk_off,
                                  make_layout(tile_shape_3d, tile_stride_3d));

    int beta_linear = head_idx * T_total + (int(bos) + local_t * CHUNK);
    int beta_aligned = beta_linear & ~7; // rounding to the nearest multiple of 8. basically a ceil division with multiplication.
    auto beta_off = g_beta.layout()(beta_aligned);
    Tensor g_beta_tile =
        make_tensor(g_beta.data() + beta_off, BetaSmemLayout{});

    Tensor s_q_tile =
        make_tensor(make_smem_ptr(shared_storage.q.begin()), TMAQKLayout{});
    Tensor s_k_tile =
        make_tensor(make_smem_ptr(shared_storage.k.begin()), TMAQKLayout{});
    Tensor s_beta_tile = make_tensor(make_smem_ptr(shared_storage.beta.begin()),
                                     TMABetaSmemLayout{});

    // Issues the actual async TMA load from global to shared mem
    // with.(shared_storage.tma_load_barrier) adds the TMA completion barrier.
    // while reinterpret_cast<BarrierType&> makes sure that the barrier object
    // is the same exact barrier type that is needed by CuTe TMA API.
    cute::copy(tma_load_q.with(reinterpret_cast<BarrierType &>(
                   shared_storage.tma_load_barrier)),
               cta_tma_load_q.partition_S(g_q_tile),
               cta_tma_load_q.partition_D(s_q_tile));
    cute::copy(tma_load_k.with(reinterpret_cast<BarrierType &>(
                   shared_storage.tma_load_barrier)),
               cta_tma_load_k.partition_S(g_k_tile),
               cta_tma_load_k.partition_D(s_k_tile));
    cute::copy(tma_load_beta.with(reinterpret_cast<BarrierType &>(
                   shared_storage.tma_load_barrier)),
               cta_tma_load_beta.partition_S(g_beta_tile),
               cta_tma_load_beta.partition_D(s_beta_tile));

    // TMA load g_bf16 (same gmem layout as q/k)
    // From global b tile -> shared g_bf16 tile
    Tensor g_g = tma_load_g.get_tma_tensor(make_shape(H, T_total, D));
    auto cta_tma_load_g = tma_load_g.get_slice(Int<0>{});
    Tensor g_g_tile = make_tensor(g_g.data() + qk_off,
                                  make_layout(tile_shape_3d, tile_stride_3d));
    Tensor s_g_bf16_tile = make_tensor(
        make_smem_ptr(shared_storage.g_bf16.begin()), TMAQKLayout{});
    cute::copy(tma_load_g.with(reinterpret_cast<BarrierType &>(
                   shared_storage.tma_load_barrier)),
               cta_tma_load_g.partition_S(g_g_tile),
               cta_tma_load_g.partition_D(s_g_bf16_tile));

    // TMA load dt_bias [H, D] → [D] slice for current head
    Tensor g_dt = tma_load_dt_bias.get_tma_tensor(make_shape(H, D));
    auto cta_tma_load_dt = tma_load_dt_bias.get_slice(Int<0>{});
    auto dt_off = g_dt.layout()(head_idx, 0);
    Tensor g_dt_tile = make_tensor(
        g_dt.data() + dt_off,
        make_layout(make_shape(Int<1>{}, Int<D>{}), stride(g_dt.layout())));
    Tensor s_dt_tile = make_tensor(
        make_smem_ptr(shared_storage.dt_bias.begin()), TMAGTotalSmemLayout{});
    // dt_off starts at dt_bias[head_idx, 0]; the [1, D] tile copies the whole
    // row for this head.
    cute::copy(tma_load_dt_bias.with(reinterpret_cast<BarrierType &>(
                   shared_storage.tma_load_barrier)),
               cta_tma_load_dt.partition_S(g_dt_tile),
               cta_tma_load_dt.partition_D(s_dt_tile));
  }
  // This value is CTA-uniform. Computing it in every lane previously issued
  // 256 identical exponentials and one broadcast-like global load per warp.
  if (local_tid == 0) {
    shared_storage.a_log_exp = expf(A_log_ptr[head_idx]);
  }
  // --- Wait for TMA (q, k, beta, g_bf16, dt_bias)
  __syncthreads();
  shared_storage.tma_load_barrier.wait(0);
  cutlass::arch::fence_view_async_shared();
  __syncthreads();


  int compute_tid = local_tid;
  int beta_smem_offset = (head_idx * T_total + int(bos) + local_t * CHUNK) & 7;
  if (compute_tid < CHUNK) {
    shared_storage.beta_act.begin()[compute_tid] = sigmoid_tanh_approx_f32(
        float(shared_storage.beta.begin()[beta_smem_offset + compute_tid]));
  }
  int actual_len = min(CHUNK, seq_len - local_t * CHUNK);

  // --- QK L2 Normalization ---
  {
    constexpr int ELEMS_PER_THREAD = 8;
    constexpr int THREADS_PER_ROW = D / ELEMS_PER_THREAD; // 16
    constexpr int ROWS_PER_PASS = NumThreads / THREADS_PER_ROW;
    constexpr int ROW_PASSES = CHUNK / ROWS_PER_PASS;
    static_assert(CHUNK % ROWS_PER_PASS == 0);
    int my_col = (local_tid % THREADS_PER_ROW) * ELEMS_PER_THREAD;

    BF16 *q_smem = shared_storage.q.begin();
    BF16 *k_smem = shared_storage.k.begin();
    using BF16x8 = cutlass::AlignedArray<BF16, ELEMS_PER_THREAD, 16>;

#pragma unroll
    for (int pass = 0; pass < ROW_PASSES; ++pass) {
      int my_row = pass * ROWS_PER_PASS + local_tid / THREADS_PER_ROW;
      int row_offset = my_row * D + my_col;
      BF16x8 q_pack = *reinterpret_cast<BF16x8 const *>(q_smem + row_offset);
      BF16x8 k_pack = *reinterpret_cast<BF16x8 const *>(k_smem + row_offset);
      float q_sq = 0.0f, k_sq = 0.0f;

#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        float qv = bf16_to_f32(q_pack[i]);
        float kv = bf16_to_f32(k_pack[i]);
        q_sq += qv * qv;
        k_sq += kv * kv;
      }

#pragma unroll
      for (int delta = 8; delta >= 1; delta >>= 1) {
        q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, delta, THREADS_PER_ROW);
        k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, delta, THREADS_PER_ROW);
      }

      if ((local_tid % THREADS_PER_ROW) == 0) {
        shared_storage.q_norm_inv.begin()[my_row] = rsqrtf(q_sq + 1e-6f);
        shared_storage.k_norm_inv.begin()[my_row] = rsqrtf(k_sq + 1e-6f);
      }
    }
  }
  // --- Fused gate activation + cumsum ---
  // Q/K remain raw in shared memory. The decay pass consumes the row inverse
  // norms above and rounds normalized values to BF16 directly in registers.
  if (compute_tid < 128) {
    int col = compute_tid;
    BF16 const *g_bf16_smem = shared_storage.g_bf16.begin();
    float dt = shared_storage.dt_bias.begin()[col];
    float *g_smem = shared_storage.g.begin();
    float sum = 0.0f;
#pragma unroll
    for (int row = 0; row < CHUNK; ++row) {
      float g_val;
      if (row < actual_len) {
        g_val = bf16_to_f32(g_bf16_smem[row * D + col]) + dt;
        g_val = shared_storage.a_log_exp * g_val;
        g_val = gate_scale * sigmoid_tanh_approx_f32(g_val);
      } else {
        g_val = 0.0f;
      }
      sum += g_val;
      g_smem[row * D + col] = sum;
    }
    shared_storage.g_total.begin()[col] = sum;
  }
  __syncthreads();

  Tensor q_tile =
      make_tensor(make_smem_ptr(shared_storage.q.begin()), QKLayout{});
  Tensor k_tile =
      make_tensor(make_smem_ptr(shared_storage.k.begin()), QKLayout{});
  Tensor g_tile =
      make_tensor(make_smem_ptr(shared_storage.g.begin()), GLayout{});
  Tensor k_restored = make_tensor(
      make_smem_ptr(shared_storage.k_restored.begin()), MMALayout{});
  Tensor k_decayed =
      make_tensor(make_smem_ptr(shared_storage.k_decayed.begin()), MMALayout{});
  Tensor q_decayed =
      make_tensor(make_smem_ptr(shared_storage.q_decayed.begin()), MMALayout{});
  Tensor k_inv =
      make_tensor(make_smem_ptr(shared_storage.k_inv.begin()), MMALayout{});
  Tensor g_total = make_tensor(make_smem_ptr(shared_storage.g_total.begin()),
                               GTotalLayout{});

  // exp_g_total: compute exp(g_total) in smem before decay_apply
  if (compute_tid < 128) {
    float x = g_total(compute_tid);
    g_total(compute_tid) = ex2_approx_ftz_f32(x);
  }
  __syncthreads();

  // decay_apply
  if (compute_tid < 256) {
    static_assert(D % 64 == 0);
    static_assert(CHUNK % 8 == 0);

    // So g chooses column stripe, and warp_id + g chooses row. Then t chooses 2
    // columns inside that stripe.
    //   Each (warp_id, g) pair maps to one cell:
    //   row = (warp_id + g) % 8
    //   stripe = g
    //     in one warp:
    //     lanes 0..3    -> g=0, t=0..3
    //     lanes 4..7    -> g=1, t=0..3
    //     lanes 8..11   -> g=2, t=0..3
    //     ...
    //     lanes 28..31  -> g=7, t=0..3
    //     all g=0..7 exist at the same time across the 32 lanes.
    /*
      Warp 0
      row = (0 + g) % 8

      g=0 -> row0
      g=1 -> row1
      g=2 -> row2
      g=3 -> row3
      g=4 -> row4
      g=5 -> row5
      g=6 -> row6
      g=7 -> row7

      So warp 0 maps diagonally:
                stripe0 stripe1 stripe2 stripe3 stripe4 stripe5 stripe6 stripe7
      warp0      row0    row1    row2    row3    row4    row5    row6    row7
    */
    int lane = compute_tid % 32;
    int physical_warp_id = compute_tid / 32;
    int g = lane / 4;
    int t = lane % 4;

    // we can also use Int<1>{} in place of _1{}. same thing but diff alias
    auto vec8_2d = make_shape(_1{}, _8{}); // tile
    auto vec8_1d = make_shape(_8{});       // same. gets total of 8 cols values
    auto thr2_2d =
        make_shape(_1{}, _2{}); // tile used by lane to get it's 2 col values
    auto thr2_1d = make_shape(_2{}); // total sum for the lane-col values

    constexpr int N_M = CHUNK / 8;
    constexpr int N_N = D / 64;
    constexpr int N_TILES = N_M * N_N;
    constexpr int PHYSICAL_WARPS = NumThreads / 32;
    constexpr int WARP_PASSES = 8 / PHYSICAL_WARPS;
    static_assert(8 % PHYSICAL_WARPS == 0);

    float reg_g[WARP_PASSES][N_TILES][2];
    BF16 reg_q[WARP_PASSES][N_TILES][2];
    BF16 reg_k[WARP_PASSES][N_TILES][2];
    float reg_gt[WARP_PASSES][N_TILES][2];

#pragma unroll
    for (int pass = 0; pass < WARP_PASSES; ++pass) {
      int warp_id = physical_warp_id + pass * PHYSICAL_WARPS;
#pragma unroll
      for (int m_blk = 0; m_blk < CHUNK;
           m_blk += 8) { // looping over CHUNK-rows
#pragma unroll
        for (int n_blk = 0; n_blk < D; n_blk += 64) { // looping over D-dim cols
          int tile_idx = (m_blk / 8) * N_N + (n_blk / 64);
          // See docs/fwd-kernel1-decay-apply-thread-map.md for the warp/group
          // row-stripe mapping.
          int row =
              m_blk +
              ((warp_id + g) %
               8); //   row = start of 8-row block + offset within that
                   //   block. means the 8 groups in a warp use 8 diff rows.
          int col_base = n_blk + g * 8; // For n_blk = 0: g=0 -> col_base = 0  +
                                        // 0*8 = 0    -> cols 0..7
          int col_tile = col_base / 8;  // this is 8-column tile index as in
                                        // col_tile * 8 .. col_tile * 8 + 8

          // CuTe source says local_tile calls inner_partition, and
          // inner_partition first does zipped_divide, then slices the
          // second/Rest mode. NVIDIA docs describe the same concept here. See
          // docs/cute-local-tile-coordinate-example.md for why local_tile's
          // coord is a tile coordinate, not a raw element coordinate. This is
          // basically 8-column group-level tile
          Tensor tile_g =
              local_tile(g_tile, vec8_2d, make_coord(row, col_tile));
          Tensor tile_q =
              local_tile(q_tile, vec8_2d, make_coord(row, col_tile));
          Tensor tile_k =
              local_tile(k_tile, vec8_2d, make_coord(row, col_tile));
          Tensor tile_gt = local_tile(g_total, vec8_1d, make_coord(col_tile));

          // This is basically lane-level chunks with shared-mem view
          Tensor s_g = local_tile(tile_g, thr2_2d, make_coord(0, t));
          Tensor s_q = local_tile(tile_q, thr2_2d, make_coord(0, t));
          Tensor s_k = local_tile(tile_k, thr2_2d, make_coord(0, t));
          Tensor s_gt = local_tile(tile_gt, thr2_1d, make_coord(t));

          // creating register tensors with the same shape as slices.
          Tensor r_g = make_tensor_like<float>(s_g);
          Tensor r_q = make_tensor_like<BF16>(s_q);
          Tensor r_k = make_tensor_like<BF16>(s_k);
          Tensor r_gt = make_tensor_like<float>(s_gt);

          // basically a vectorized copy from smem to rmem when the layout is
          // aligned.
          cute::copy(AutoVectorizingCopy{}, s_g, r_g);
          cute::copy(AutoVectorizingCopy{}, s_q, r_q);
          cute::copy(AutoVectorizingCopy{}, s_k, r_k);
          cute::copy(AutoVectorizingCopy{}, s_gt, r_gt);

// See docs/fwd-kernel1-decay-apply-thread-map.md for why each thread stores 2
// values for each of 4 large tiles.
#pragma unroll
          for (int v = 0; v < 2; ++v) {
            reg_g[pass][tile_idx][v] = r_g(0, v);
            reg_q[pass][tile_idx][v] = r_q(0, v);
            reg_k[pass][tile_idx][v] = r_k(0, v);
            reg_gt[pass][tile_idx][v] = r_gt(v);
          }
        }
      }
    }

    // Sync before writing to union'd smem (q/k/g → k_decayed/q_decayed/k_inv)
    // All virtual-warp inputs must be resident before any union'd output
    // writes.
    __syncthreads();

#pragma unroll
    for (int pass = 0; pass < WARP_PASSES; ++pass) {
      int warp_id = physical_warp_id + pass * PHYSICAL_WARPS;
#pragma unroll
      for (int m_blk = 0; m_blk < CHUNK; m_blk += 8) {
#pragma unroll
        for (int n_blk = 0; n_blk < D; n_blk += 64) {
          int tile_idx = (m_blk / 8) * N_N + (n_blk / 64);
          int row = m_blk + ((warp_id + g) % 8);
          int col_base = n_blk + g * 8;
          int col_tile = col_base / 8;

          Tensor tile_qd =
              local_tile(q_decayed, vec8_2d, make_coord(row, col_tile));
          Tensor tile_kd =
              local_tile(k_decayed, vec8_2d, make_coord(row, col_tile));
          Tensor tile_kr =
              local_tile(k_restored, vec8_2d, make_coord(row, col_tile));
          Tensor tile_ki =
              local_tile(k_inv, vec8_2d, make_coord(row, col_tile));

          Tensor s_qd = local_tile(tile_qd, thr2_2d, make_coord(0, t));
          Tensor s_kd = local_tile(tile_kd, thr2_2d, make_coord(0, t));
          Tensor s_kr = local_tile(tile_kr, thr2_2d, make_coord(0, t));
          Tensor s_ki = local_tile(tile_ki, thr2_2d, make_coord(0, t));

          Tensor r_qd = make_tensor_like<BF16>(s_qd);
          Tensor r_kd = make_tensor_like<BF16>(s_kd);
          float q_inv = shared_storage.q_norm_inv.begin()[row];
          float k_inv_scale = shared_storage.k_norm_inv.begin()[row];
#pragma unroll
          for (int v = 0; v < 2; ++v) {
            float g_value = reg_g[pass][tile_idx][v];
            BF16 q = BF16(bf16_to_f32(reg_q[pass][tile_idx][v]) * q_inv);
            BF16 k =
                row < actual_len
                    ? BF16(bf16_to_f32(reg_k[pass][tile_idx][v]) * k_inv_scale)
                    : BF16(0);
            BF16 exp_cumsum = BF16(ex2_approx_ftz_f32(g_value));
            r_qd(0, v) = q * exp_cumsum * BF16(scale);
            r_kd(0, v) = k * exp_cumsum;
          }
          cute::copy(AutoVectorizingCopy{}, r_qd, s_qd);
          cute::copy(AutoVectorizingCopy{}, r_kd, s_kd);

          Tensor r_ki = make_tensor_like<BF16>(s_ki);
          Tensor r_kr = make_tensor_like<BF16>(s_kr);
#pragma unroll
          for (int v = 0; v < 2; ++v) {
            float g_value = reg_g[pass][tile_idx][v];
            BF16 k =
                row < actual_len
                    ? BF16(bf16_to_f32(reg_k[pass][tile_idx][v]) * k_inv_scale)
                    : BF16(0);
            BF16 inv_cumsum = BF16(ex2_approx_ftz_f32(-g_value));
            r_ki(0, v) = k * inv_cumsum;
            r_kr(0, v) = k * inv_cumsum * BF16(reg_gt[pass][tile_idx][v]);
          }
          cute::copy(AutoVectorizingCopy{}, r_ki, s_ki);
          cute::copy(AutoVectorizingCopy{}, r_kr, s_kr);
        }
      }
    }
  }
  __syncthreads();

  Tensor L = make_tensor(make_smem_ptr(shared_storage.L.begin()), LMLayout{});
  Tensor Mqk =
      make_tensor(make_smem_ptr(shared_storage.Mqk.begin()), LMLayout{});
  Tensor L_fp16 = make_tensor(
      make_smem_ptr(reinterpret_cast<FP16 *>(shared_storage.L.begin())),
      LMLayout{}); // this is because MMA compute L with fp16 output

  // L_Mqk
  if (compute_tid < 32) {
    mma_m16n16_bf16bf16fp16_1warp(k_decayed, k_inv, L_fp16, compute_tid);
  } else if (compute_tid >= 32 && compute_tid < 64) {
    mma_m16n16_bf16bf16bf16_1warp(q_decayed, k_inv, Mqk, compute_tid - 32);
  }
  __syncthreads();

  Tensor INV =
      make_tensor(make_smem_ptr(shared_storage.INV.begin()), LMLayout{});
  Tensor INV_fp16 = make_tensor(
      make_smem_ptr(reinterpret_cast<FP16 *>(shared_storage.INV.begin())),
      LMLayout{});

  // tril_IL + INV = I - L (merged, same thread same element)
  // dealing with a 16x16 matrix btw.
  for (int matrix_idx = compute_tid; matrix_idx < CHUNK * CHUNK;
       matrix_idx += NumThreads) {
    const int col_block_size = 8;
    int block_idx = matrix_idx / (CHUNK * col_block_size);
    int i = (matrix_idx / col_block_size) % CHUNK;
    int j = matrix_idx % col_block_size + block_idx * col_block_size;
    if (i <= j) { // causal mask w/o diagonal elements
      L_fp16(i, j) = FP16::bitcast(0);
    } else {
      L_fp16(i, j) = L_fp16(i, j) * FP16(shared_storage.beta_act.begin()[i]);
    }
    if (i < j) { // causal mask but with diagonal elements
      Mqk(i, j) = BF16::bitcast(0);
    }
    // INV = I - L (same thread reads L(i,j) it just wrote)
    FP16 x = L_fp16(i, j);
    INV_fp16(i, j) =
        (i == j ? FP16(1.0f) - x
                : -x); // lower-traiangular elements are -ve because we are
                       // subtracting with an Identity matrix
  }
  __syncthreads();

  // inv (Neumann series, fused in registers). See
  // docs/fwd-kernel1-neumann-inverse.md for the finite series/factorization.
  neumann_inv_fused_1warp(L_fp16, INV_fp16, INV, compute_tid);
  // Fence + sync combined: completion + TMA visibility
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  // The six global workspace stores and their H100 backpressure analysis are
  // summarized in docs/H100_RESULTS.md.
  if (local_tid == 0) {
    int ws_idx = head_idx * total_tiles + global_tile_idx;

    BF16 *k_decayed_dst =
        ws_raw.k_decayed + int64_t(ws_idx) * (CHUNK * D);
    cute::SM90_BULK_COPY_S2G::copy(shared_storage.k_decayed.begin(),
                                   k_decayed_dst,
                                   int32_t(CHUNK * D * sizeof(BF16)));
    tma_store_arrive();

    BF16 *q_decayed_dst =
        ws_raw.q_decayed + int64_t(ws_idx) * (CHUNK * D);
    cute::SM90_BULK_COPY_S2G::copy(shared_storage.q_decayed.begin(),
                                   q_decayed_dst,
                                   int32_t(CHUNK * D * sizeof(BF16)));
    tma_store_arrive();

    BF16 *k_restored_dst =
        ws_raw.k_restored + int64_t(ws_idx) * (CHUNK * D);
    cute::SM90_BULK_COPY_S2G::copy(shared_storage.k_restored.begin(),
                                   k_restored_dst,
                                   int32_t(CHUNK * D * sizeof(BF16)));
    tma_store_arrive();

    float *g_total_dst = ws_raw.g_total + int64_t(ws_idx) * D;
    cute::SM90_BULK_COPY_S2G::copy(shared_storage.g_total.begin(), g_total_dst,
                                   int32_t(D * sizeof(float)));
    tma_store_arrive();

    BF16 *inv_dst = ws_raw.inv + int64_t(ws_idx) * (CHUNK * CHUNK);
    cute::SM90_BULK_COPY_S2G::copy(shared_storage.INV.begin(), inv_dst,
                                   int32_t(CHUNK * CHUNK * sizeof(BF16)));
    tma_store_arrive();

    BF16 *mqk_dst = ws_raw.mqk + int64_t(ws_idx) * (CHUNK * CHUNK);
    cute::SM90_BULK_COPY_S2G::copy(shared_storage.Mqk.begin(), mqk_dst,
                                   int32_t(CHUNK * CHUNK * sizeof(BF16)));
    tma_store_arrive();
  }
  tma_store_wait<0>();
  __syncthreads();
}

template <class TmaLoadQ, class TmaLoadK, class TmaLoadBeta, class TmaLoadG,
          class TmaLoadDtBias, int CHUNK, int D, int NumThreads,
          bool IsVarlen = true>
__global__ void __launch_bounds__(NumThreads, 8) _flash_kda_fwd_prepare(
    CUTE_GRID_CONSTANT TmaLoadQ const tma_load_q,
    CUTE_GRID_CONSTANT TmaLoadK const tma_load_k,
    CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
    CUTE_GRID_CONSTANT TmaLoadG const tma_load_g,
    CUTE_GRID_CONSTANT TmaLoadDtBias const tma_load_dt_bias,
    float scale, int T_total, int H, int N, int64_t const *cu_seqlens,
    int total_tiles,
    float const *A_log_ptr, float gate_scale, K1WorkspaceRawPointers ws_raw) {
  using Layouts = K1Layouts<D, CHUNK>;
  using SharedStorageT = SharedStorageK1<Layouts>;
  extern __shared__ __align__(128) unsigned char shared_mem[];
  SharedStorageT &shared_storage =
      *reinterpret_cast<SharedStorageT *>(shared_mem);

  flash_kda_fwd_prepare_body<TmaLoadQ, TmaLoadK, TmaLoadBeta, TmaLoadG,
                             TmaLoadDtBias, CHUNK, D, NumThreads, IsVarlen>(
      tma_load_q, tma_load_k, tma_load_beta, tma_load_g, tma_load_dt_bias,
      scale, T_total, H, N, cu_seqlens, total_tiles, A_log_ptr, gate_scale,
      shared_storage, int(blockIdx.x), int(blockIdx.y), ws_raw);
}
