#pragma once

#include "utils.cuh"

// The production recurrence is full-width and keeps each compute warp's state
// columns resident in BF16 C fragments across the complete sequence.

template <int D, int CHUNK = 16, int CTA_N = D>
struct K2Layouts {
    static_assert(D % 16 == 0);
    static_assert(CTA_N % 32 == 0);
    static_assert(D % CTA_N == 0);

    // Full-width, read-only chunk operands. Their K/reduction dimension remains D
    // even when a CTA owns only CTA_N value/output columns.
    using MMALayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TransposedMMALayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<CHUNK>{}),
        LayoutRight{}
    ));

    // V, U, and output use the INTER atom. The tested SW128 alternatives made
    // uneven and fixed workloads slower despite improving tensor-map segment
    // size, so the MMA-friendly layout remains the production choice.
    using VOLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<CTA_N>{}),
        LayoutLeft{}
    ));
    using OutSmemLayout = VOLayout;
    using TMAOutLayout = decltype(composition(
        OutSmemLayout{}.layout_a(),
        OutSmemLayout{}.offset(),
        prepend(OutSmemLayout{}.layout_b())
    ));
    using BetaSmemLayout = Layout<Shape<Int<32>>, Stride<Int<1>>>;

    // The external state is [D_value, D_key]. StateSmemLayout preserves that
    // [N,K] view for TMA/MMA-B, while TransposedStateSmemLayout exposes the
    // arithmetic [K,N] view used by the state update.
    using StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CTA_N>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TransposedStateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<CTA_N>{}),
        LayoutRight{}
    ));
    using GTotalLayout = Layout<Shape<Int<D>>, Stride<Int<1>>>;
    using LMLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<CHUNK>{}),
        LayoutLeft{}
    ));

    using TMABetaSmemLayout = BetaSmemLayout;  // 1D TMA, no dummy dim
    using TMAVOLayout = decltype(composition(
        VOLayout{}.layout_a(),
        VOLayout{}.offset(),
        prepend(VOLayout{}.layout_b())
    ));
    using TMAStateSmemLayout = decltype(composition(
        StateSmemLayout{}.layout_a(),
        StateSmemLayout{}.offset(),
        prepend(StateSmemLayout{}.layout_b())
    ));
    // FP32 state layout (K_SW32 atom, same 8x8 atom structure as K_INTER bf16)
    using FP32StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW32_Atom<float>{},
        make_shape(Int<CTA_N>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TMAFP32StateSmemLayout = decltype(composition(
        FP32StateSmemLayout{}.layout_a(),
        FP32StateSmemLayout{}.offset(),
        prepend(FP32StateSmemLayout{}.layout_b())
    ));
};

template <class Layouts, int InputStages, int OutputStages>
struct SharedStorageK2 {
    using BF16 = cutlass::bfloat16_t;
    using VOLayout = typename Layouts::VOLayout;
    using OutSmemLayout = typename Layouts::OutSmemLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using MMALayout = typename Layouts::MMALayout;

    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_acc;

    struct InputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> v;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> Mqk;
    };

    struct OutputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<OutSmemLayout>> out;
    };

    // Anonymous union: pipeline buffers share space with fp32 state conversion buffer.
    // FP32 state load/store happens before/after the pipeline loop, so no overlap.
    union {
        struct {
            InputStorage input[InputStages];
            OutputStorage output[OutputStages];
        };
        alignas(128) char state_fp32_buf[cute::cosize_v<StateSmemLayout> * sizeof(float)];
    };

    typename cutlass::PipelineTmaAsync<InputStages>::SharedStorage load_pipeline;
    typename cutlass::PipelineAsync<OutputStages>::SharedStorage store_pipeline;
    alignas(16) cutlass::arch::ClusterTransactionBarrier state_acc_tma_barrier;
};

template <class CFragment, class BFragment>
CUTLASS_DEVICE void movm_transpose_c_to_b_16x16(
    CFragment const& source,
    BFragment& destination
) {
    static_assert(sizeof(CFragment) == 4 * sizeof(uint32_t));
    static_assert(sizeof(BFragment) == 4 * sizeof(uint32_t));

    auto const* source_regs = reinterpret_cast<uint32_t const*>(&source(0));
    auto* destination_regs = reinterpret_cast<uint32_t*>(&destination(0));
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        SM75_U32x1_MOVM_T::copy(source_regs[i], destination_regs[i]);
    }
}

// Map a scheduling rank to the original sequence index without materializing a
// permutation. One lane owns each candidate sequence; all lanes then count how
// many candidates are longer. Ties retain the original sequence order.
CUTLASS_DEVICE int sequence_for_descending_length_rank(
    int64_t const* cu_seqlens,
    int N,
    int schedule_rank
) {
    constexpr unsigned kFullWarpMask = 0xffffffffu;
    int lane = int(threadIdx.x) & 31;
    bool valid = lane < N;
    int64_t candidate_len = valid
        ? cu_seqlens[lane + 1] - cu_seqlens[lane]
        : int64_t{-1};
    int candidate_rank = 0;

    for (int other = 0; other < N; ++other) {
        int64_t other_len = cu_seqlens[other + 1] - cu_seqlens[other];
        if (valid &&
            (other_len > candidate_len ||
             (other_len == candidate_len && other < lane))) {
            ++candidate_rank;
        }
    }

    unsigned match = __ballot_sync(
        kFullWarpMask, valid && candidate_rank == schedule_rank);
    return __ffs(match) - 1;
}

// ==================== Kernel 2: Recurrence ====================
template <
    class TmaLoadV,
    class TmaLoadBeta,
    class TmaLoadState,
    class TmaStoreState,
    class TmaStoreOut,
    int CHUNK,
    int D,
    int CTA_N,
    int InputStages,
    int OutputStages,
    int NumThreads,
    bool HasStateIn = true,
    bool HasStateOut = true,
    bool StateFP32 = false,
    bool IsVarlen = true
>
CUTLASS_DEVICE void flash_kda_fwd_recurrence_body(
    TmaLoadV const& tma_load_v,
    TmaLoadBeta const& tma_load_beta,
    TmaLoadState const& tma_load_initial_state,
    TmaStoreState const& tma_store_final_state,
    TmaStoreOut const& tma_store_out,
    cutlass::bfloat16_t* out_raw_ptr,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens,
    int total_tiles,
    SharedStorageK2<K2Layouts<D, CHUNK, CTA_N>, InputStages, OutputStages>
        &shared_storage,
    int seq_idx,
    int head_idx,
    K1WorkspaceRawPointers ws_raw = {}
) {
    using BF16 = cutlass::bfloat16_t;
    using Layouts = K2Layouts<D, CHUNK, CTA_N>;
    using MMALayout = typename Layouts::MMALayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using VOLayout = typename Layouts::VOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using TMAVOLayout = typename Layouts::TMAVOLayout;
    using OutSmemLayout = typename Layouts::OutSmemLayout;
    using TMAOutLayout = typename Layouts::TMAOutLayout;
    using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
    using TMAStateSmemLayout = typename Layouts::TMAStateSmemLayout;
    int const local_tid = int(threadIdx.x);
    constexpr int kWarpSize = 32;
    constexpr int kComputeThreads = CTA_N;
    static_assert(CTA_N == D,
                  "The production K2 kernel owns all state/output columns.");
    static_assert(NumThreads >= kComputeThreads + 2 * kWarpSize);
    static_assert(NumThreads % kWarpSize == 0);

    // Transaction bytes: v + beta + k_decayed + q_decayed + k_restored + g_total + INV + Mqk
    constexpr uint32_t kTmaTransactionBytes =
        uint32_t(cute::cosize_v<VOLayout>) * uint32_t(sizeof(BF16)) +
        uint32_t(32) * uint32_t(sizeof(BF16)) +  // beta (bf16, sigmoid fused)
        uint32_t(cute::cosize_v<MMALayout>) * uint32_t(sizeof(BF16)) * 3 +
        uint32_t(cute::cosize_v<GTotalLayout>) * uint32_t(sizeof(float)) +
        uint32_t(cute::cosize_v<LMLayout>) * uint32_t(sizeof(BF16)) * 2;

    // --- warp specialization
    int warp_id = local_tid / kWarpSize;
    WarpRole warp_role = WarpRole::NonParticipant;
    if (warp_id < kComputeThreads / kWarpSize) {
        warp_role = WarpRole::MMA;
    } else if (warp_id < kComputeThreads / kWarpSize + 1) {
        warp_role = WarpRole::LOAD_QKG;
    } else if (warp_id < kComputeThreads / kWarpSize + 2) {
        warp_role = WarpRole::STORE;
    }

    using LoadPipelineState = cutlass::PipelineState<InputStages>;
    using LoadPipeline = cutlass::PipelineTmaAsync<InputStages>;
    LoadPipeline load_pipeline =
        make_load_pipeline<InputStages>(
        shared_storage.load_pipeline,
        kTmaTransactionBytes,
        warp_role, 1, kComputeThreads
    );
    using StorePipelineState = cutlass::PipelineState<OutputStages>;
    using StorePipeline = cutlass::PipelineAsync<OutputStages>;
    StorePipeline store_pipeline = make_store_pipeline<OutputStages>(
        shared_storage.store_pipeline,
        warp_role, kComputeThreads, 1
    );
    /*
      Unsharded launch:
                   seq0        seq1        seq2
      head0     CTA(0,0)    CTA(1,0)    CTA(2,0)
      head1     CTA(0,1)    CTA(1,1)    CTA(2,1)
      head2     CTA(0,2)    CTA(1,2)    CTA(2,2)
      head3     CTA(0,3)    CTA(1,3)    CTA(2,3)

      So this CTA:
      blockIdx.x = 1
      blockIdx.y = 2

      means:
      process sequence 1, head 2

    */
    constexpr int column_begin = 0;
    int64_t bos, eos;
    int tile_base;

    /*
       So inside the Kernel-2 CTA for seq_idx = 2:
          local t=0 needs Kernel 1 global tile 4
          local t=1 needs Kernel 1 global tile 5
          local t=2 needs Kernel 1 global tile 6
          local t=3 needs Kernel 1 global tile 7

          That is why we need:
          global_tile_idx = tile_base + t;

          For seq_idx = 2:
          tile_base = number of chunks before seq2
                    = seq0 chunks + seq1 chunks
                    = 3 + 1
                    = 4

          Then:
          t=0 -> tile_base + t = 4
          t=1 -> tile_base + t = 5
          t=2 -> tile_base + t = 6
          t=3 -> tile_base + t = 7

       t: is local chunk number in this seq
       tile_base: how many chunks before this seq
       tile_base + t: mapping to global_tile_idx from Kernel-1
    */
    if constexpr (IsVarlen) {
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
        // Compute tile_base via linear scan (no host-precomputed table)
        tile_base = 0;
        for (int i = 0; i < seq_idx; i++) {
            tile_base += (int(cu_seqlens[i + 1] - cu_seqlens[i]) + CHUNK - 1) / CHUNK;
        }
    } else {
        int T_seq = T_total / N;
        bos = seq_idx * T_seq;
        eos = bos + T_seq;
        tile_base = seq_idx * ((T_seq + CHUNK - 1) / CHUNK);
    }
    int seq_len  = int(eos - bos);
    int t_tiles  = (seq_len + CHUNK - 1) / CHUNK;
    bool lane_predicate = cute::elect_one_sync(); // one lane for issuing the TMA copy instruction

    // --- Load initial state
    if constexpr (HasStateIn && !StateFP32) {
        // BF16 state: TMA load directly into state_acc
        if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
            using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
            constexpr uint32_t kStateTransactionBytes = cute::cosize_v<StateSmemLayout> * sizeof(BF16);

            shared_storage.state_acc_tma_barrier.init(1);
            shared_storage.state_acc_tma_barrier.arrive_and_expect_tx(kStateTransactionBytes);

            Tensor g_init = tma_load_initial_state.get_tma_tensor(make_shape(N * H, D, D));
            auto init_off = g_init.layout()(seq_idx * H + head_idx, column_begin, 0);
            Tensor g_init_tile = make_tensor(g_init.data() + init_off,
                make_layout(make_shape(Int<1>{}, Int<CTA_N>{}, Int<D>{}), stride(g_init.layout())));
            Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TMAStateSmemLayout{});

            auto cta_tma_load_state = tma_load_initial_state.get_slice(Int<0>{});
            cute::copy(
                tma_load_initial_state.with(reinterpret_cast<BarrierType&>(shared_storage.state_acc_tma_barrier)),
                cta_tma_load_state.partition_S(g_init_tile),
                cta_tma_load_state.partition_D(s_state)
            );
        }
        __syncthreads();
        shared_storage.state_acc_tma_barrier.wait(0);
        cutlass::arch::fence_view_async_shared();
    } else if constexpr (HasStateIn && StateFP32) {
        // FP32 state: TMA load fp32 into pipeline buffer, then convert to bf16 in state_acc
        using FP32StateSmemLayout = typename Layouts::FP32StateSmemLayout;
        using TMAFP32StateSmemLayout = typename Layouts::TMAFP32StateSmemLayout;

        if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
            using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
            constexpr uint32_t kFP32StateTransactionBytes = cute::cosize_v<StateSmemLayout> * sizeof(float);

            shared_storage.state_acc_tma_barrier.init(1);
            shared_storage.state_acc_tma_barrier.arrive_and_expect_tx(kFP32StateTransactionBytes);

            Tensor g_init = tma_load_initial_state.get_tma_tensor(make_shape(N * H, D, D));
            auto init_off = g_init.layout()(seq_idx * H + head_idx, column_begin, 0);
            Tensor g_init_tile = make_tensor(g_init.data() + init_off,
                make_layout(make_shape(Int<1>{}, Int<CTA_N>{}, Int<D>{}), stride(g_init.layout())));
            Tensor s_fp32 = make_tensor(
                make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)),
                TMAFP32StateSmemLayout{});

            auto cta_tma_load_state = tma_load_initial_state.get_slice(Int<0>{});
            cute::copy(
                tma_load_initial_state.with(reinterpret_cast<BarrierType&>(shared_storage.state_acc_tma_barrier)),
                cta_tma_load_state.partition_S(g_init_tile),
                cta_tma_load_state.partition_D(s_fp32)
            );
        }
        __syncthreads();
        shared_storage.state_acc_tma_barrier.wait(0);
        cutlass::arch::fence_view_async_shared();

        // All threads: convert fp32 -> bf16 with layout transformation
        smem_cvt_fp32_to_bf16<FP32StateSmemLayout, StateSmemLayout, CTA_N, D, NumThreads>(
            reinterpret_cast<float*>(shared_storage.state_fp32_buf),
            shared_storage.state_acc.begin(),
            local_tid);
        __syncthreads();
    } else {
        // No state in: zero-initialize state_acc
        // kTotal is 128*128 = 16384 bf16 values & NumThreads = 192, every thread is strided with 192 like :
        //   thread 0 writes i = 0, 192, 384, ...
        //   thread 1 writes i = 1, 193, 385, ...
        {
            BF16* buf = shared_storage.state_acc.begin();
            constexpr int kTotal = cute::cosize_v<StateSmemLayout>;
            for (int i = local_tid; i < kTotal; i += NumThreads) {
                buf[i] = BF16(0);
            }
        }
        __syncthreads();
    }



    __syncthreads();

    // --- LOAD warp: issue TMA loads for v, beta, and workspace intermediates
    if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
        Tensor g_v = tma_load_v.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_beta =
            tma_load_beta.get_tma_tensor(make_shape(H * T_total));

        // init load pipeline producer with kInputStages=3, of load warp cycles.
        // the idea is to let the LOAD warp a small state object that tracks which shared-memory input stage, it should fill next.
        LoadPipelineState load_write = cutlass::make_producer_start_state<LoadPipeline>();
        // each tma_load_ object is a TMA descriptor with things like source & destination mem layout, copy op (TMA load).
        // get_slice(Int<0>{}) asks for a CTA copy helper for slice 0.
        auto cta_tma_load_v = tma_load_v.get_slice(Int<0>{});
        auto cta_tma_load_beta = tma_load_beta.get_slice(Int<0>{});

        for (int t = 0; t < t_tiles; ++t) { // local chunk index inside this sequence.
            load_pipeline.producer_acquire(load_write); // wait until the current input stage is free to write.
            using LoadBarrierType = typename LoadPipeline::ProducerBarrierType; // async barrier for the current stage. All TMA copies for this chunk are attached to this barrier. Compute warps waits until this barrier says ready.
            LoadBarrierType* tma_barrier = load_pipeline.producer_get_barrier(load_write);
            int stage = load_write.index(); // selects which stage to fill
            int ws_idx = head_idx * total_tiles + tile_base + t;


            // TMA load v
            auto v_off = g_v.layout()(head_idx, int(bos) + t * CHUNK, column_begin);
            Tensor g_v_tile = make_tensor(g_v.data() + v_off,
                make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<CTA_N>{}), stride(g_v.layout())));
            Tensor s_v_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].v.begin()), TMAVOLayout{});
            cute::copy(tma_load_v.with(*tma_barrier),
                cta_tma_load_v.partition_S(g_v_tile), cta_tma_load_v.partition_D(s_v_tile));

            // TMA load beta (1D)
            int beta_linear = head_idx * T_total + (int(bos) + t * CHUNK);
            int beta_aligned = beta_linear & ~7;
            auto beta_off = g_beta.layout()(beta_aligned);
            Tensor g_beta_tile = make_tensor(g_beta.data() + beta_off, BetaSmemLayout{});
            Tensor s_beta_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].beta.begin()), TMABetaSmemLayout{});
            cute::copy(
                tma_load_beta.with(*tma_barrier),
                cta_tma_load_beta.partition_S(g_beta_tile),
                cta_tma_load_beta.partition_D(s_beta_tile));

            // K1 stores byte images of its swizzled shared-memory tensors.
            // K2 uses the same layouts, so raw bulk copies restore them
            // directly without tensor-map coordinate work or repacking.
            cute::SM90_BULK_COPY_G2S::copy(
                ws_raw.k_decayed + int64_t(ws_idx) * (CHUNK * D),
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].k_decayed.begin(),
                int32_t(CHUNK * D * sizeof(BF16)));
            cute::SM90_BULK_COPY_G2S::copy(
                ws_raw.q_decayed + int64_t(ws_idx) * (CHUNK * D),
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].q_decayed.begin(),
                int32_t(CHUNK * D * sizeof(BF16)));
            cute::SM90_BULK_COPY_G2S::copy(
                ws_raw.k_restored + int64_t(ws_idx) * (CHUNK * D),
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].k_restored.begin(),
                int32_t(CHUNK * D * sizeof(BF16)));
            cute::SM90_BULK_COPY_G2S::copy(
                ws_raw.g_total + int64_t(ws_idx) * D,
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].g_total.begin(),
                int32_t(D * sizeof(float)));
            cute::SM90_BULK_COPY_G2S::copy(
                ws_raw.inv + int64_t(ws_idx) * (CHUNK * CHUNK),
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].INV.begin(),
                int32_t(CHUNK * CHUNK * sizeof(BF16)));
            cute::SM90_BULK_COPY_G2S::copy(
                ws_raw.mqk + int64_t(ws_idx) * (CHUNK * CHUNK),
                reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].Mqk.begin(),
                int32_t(CHUNK * CHUNK * sizeof(BF16)));

            ++load_write; // moving onto the next load producer pipeline stage.
        }
        load_pipeline.producer_tail(load_write);
    }

    /*
      LOAD warp loaded one chunk into shared input[stage]
      MMA warps consume that chunk
      MMA warps write output into shared output[out_stage]
      STORE warp later stores that output to global memory
    */

    // --- MMA warps
    // each warp works with two 16-column blocks of D=128 output/state columns
    // warp 0 = col 0... 31 -> block 0, 1 so on
    if (warp_role == WarpRole::MMA) {
        LoadPipelineState load_read; // load_read is the compute side handle for reading input[stage]
        StorePipelineState out_write = cutlass::make_producer_start_state<StorePipeline>(); // out_write is the compute side handle for producing output[stage]
        int compute_tid = local_tid;


        // Keep this warp's 32 value columns of the recurrent [K,V] state in
        // BF16 C fragments for the entire chunk loop. Phase 1 transposes each
        // fragment into an MMA-B operand with MOVM_T; Phase 6 updates the C
        // fragment in place. Shared memory is touched only at entry and, when
        // requested, once more before the final-state TMA store.
        Tensor resident_s_acc_T = make_tensor(
            make_smem_ptr(shared_storage.state_acc.begin()),
            TransposedStateSmemLayout{});
        auto resident_mma = make_tiled_mma(
            MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
            Layout<Shape<_1,_1>>{},
            Tile<_16,_16,_16>{});
        const int resident_warp_id = compute_tid / kWarpSize;
        const int resident_lane_id = compute_tid % kWarpSize;
        auto resident_thr_mma = resident_mma.get_slice(resident_lane_id);
        auto resident_load_c = make_tiled_copy_C(
            Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, resident_mma);
        auto resident_thr_load_c = resident_load_c.get_slice(resident_lane_id);
        Tensor resident_state_ref = local_tile(
            resident_s_acc_T,
            make_shape(Int<16>{}, Int<16>{}),
            make_coord(0, resident_warp_id * 2));
        auto resident_c_ref = resident_thr_mma.partition_C(resident_state_ref);
        using ResidentStateFragment = decltype(make_fragment_like<BF16>(
            resident_thr_mma.make_fragment_C(resident_c_ref)));
        constexpr int kResidentStateRowBlocks = D / 16;
        ResidentStateFragment resident_state[2][kResidentStateRowBlocks];

        #pragma unroll
        for (int m = 0; m < kResidentStateRowBlocks; ++m) {
            #pragma unroll
            for (int bi = 0; bi < 2; ++bi) {
                Tensor state_block = local_tile(
                    resident_s_acc_T,
                    make_shape(Int<16>{}, Int<16>{}),
                    make_coord(m, resident_warp_id * 2 + bi));
                copy(
                    resident_load_c,
                    resident_thr_load_c.partition_S(state_block),
                    resident_thr_load_c.retile_D(resident_state[bi][m]));
            }
        }

        for (int t = 0; t < t_tiles; ++t) {
            load_pipeline.consumer_wait(load_read); // consumer is the warp that loads.
            int load_stage = load_read.index();
            int out_stage = out_write.index();

            Tensor v_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].v.begin()), VOLayout{});
            Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].beta.begin()), BetaSmemLayout{});
            int beta_smem_offset = (head_idx * T_total + int(bos) + t * CHUNK) & 7;
            Tensor out_tile = make_tensor(make_smem_ptr(shared_storage.output[out_stage].out.begin()), OutSmemLayout{});

            Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()), MMALayout{});
            Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].q_decayed.begin()), MMALayout{});
            Tensor k_restored = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), MMALayout{});
            Tensor g_total = make_tensor(make_smem_ptr(shared_storage.input[load_stage].g_total.begin()), GTotalLayout{});
            Tensor INV = make_tensor(make_smem_ptr(shared_storage.input[load_stage].INV.begin()), LMLayout{});
            Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.input[load_stage].Mqk.begin()), LMLayout{});

            Tensor s_acc = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), StateSmemLayout{});
            Tensor s_acc_T = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayout{});

            // Fused MMA: v_sub, v_beta, U=INV@v, out=q@s, out+=Mqk@U, s_acc_update
            // Each warp handles TWO 16x16 column blocks (N=128 / 4 warps = 32 = 2 x 16)
            // U stays in registers via SM75_U32x1_MOVM_T (no smem round-trip)
            {
            Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), TransposedMMALayout{});

            auto mma = make_tiled_mma(
                MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                Layout<Shape<_1,_1>>{},
                Tile<_16,_16,_16>{}
            ); // since the atom is 16x8x16 but the tile shape is 16, 16, 16, CuTe covers for the rest of the N=16 logical tiles by doing the atom 2 times like 16x8 - 16x8, which makes it 16x16x16.

            const int warp_id = compute_tid / 32;
            const int lane_id = compute_tid % 32;
            const int group_id = (lane_id / 4) % 8;

            ThrMMA thr_mma = mma.get_slice(lane_id);

            // A copy: K_INTER → LDSM_N (for k_decayed, q_decayed, INV, Mqk)
            auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);

            // A copy: MN_INTER → LDSM_T (for k_restored_t in Phase 7)
            auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);

            // C load/store
            auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);
            auto smem_tiled_store_C = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
            auto smem_thr_store_C   = smem_tiled_store_C.get_slice(lane_id);

            // C load/store transposed (for Phase 6 state access via s_acc_T)
            auto smem_tiled_store_C_T = make_tiled_copy_C(Copy_Atom<SM90_U16x8_STSM_T, BF16>{}, mma);
            auto smem_thr_store_C_T   = smem_tiled_store_C_T.get_slice(lane_id);

            // so basically the idea is have reference tiles that will tell CuTe shape & layout for A, B, C and that tile is mostly 16x16 in size.
            // the reason we're doing this is because we want to create per-lane register fragment layouts. basically, for CuTe to know, for this lane_id, what part of the 16x16 tile, it has/works on?
            // thr_mma.partition_fragment_A does exactly that. 
            Tensor A_ref = local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor B_ref = local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor C_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

            /*
             the whole flow is:
              -> ldmatrix copy into tCrAi_k_view
              -> data lives in tCrAi_k
              -> transform into tCrA_k
              -> gemm uses tCrA_k
            */
            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref)); // can be read as creating a BF16 register fragment with the same shape/layout as this lane's A operand fragment. allocates per-lane register storage.
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k); // take the register fragment tCrAi_k and reinterpret it's layout as the destination layout expected by this ldmatrix copy. we need to do this because the ldmatrix instruction expects the destination registers in copy atom's destination layout. retile_D() does that.
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref); // make the actual MMA A operand fragment view used by gemm(thr_mma, tCrA_k)

            Tensor tCrAi_q = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_view = smem_thr_copy_A.retile_D(tCrAi_q);
            auto tCrA_q = thr_mma.partition_fragment_A(A_ref);

            auto tCrB = thr_mma.partition_fragment_B(B_ref);

            auto tCrC_ref = thr_mma.partition_C(C_ref);

            using AccFragT = decltype(thr_mma.make_fragment_C(tCrC_ref)); // type of accumulator fragment for this lane's C partition. float it is btw
            using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(tCrC_ref))); // BF16 type ofc
            using AFragT = decltype(thr_mma.partition_fragment_A(A_ref)); // type of an MMA A operand fragment
            using BFragT_u = decltype(thr_mma.partition_fragment_B(B_ref)); // type of an MMA B operand fragment

            AccFragT u_acc[2], out_acc[2]; // 2 here because each warp handles 2 16-column blocks
            #pragma unroll
            for (int i = 0; i < 2; ++i) { u_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(u_acc[i]); } // init each float acc fragment and clears it to zero.
            #pragma unroll
            for (int i = 0; i < 2; ++i) { out_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(out_acc[i]); }

            // ======== Phase 1: Dual GEMM k@s and q@s (k-loop, 2 blocks per warp) ========
            // so in this Copy atom 1 warp does 16x8 block twice to cover the 16x16 tile & then for all the 32 lanes of a warp, it cover the same M/CHUNK = 16 as row but 0...31 which is 16 cols two times. hope this makes sense.
            /*
                  Output C = [M,N] = [16,128]

                           N columns / D dim
                        0..15  16..31  32..47  48..63  64..79  80..95  96..111 112..127
              M 0..15   warp0  warp0   warp1   warp1   warp2   warp2   warp3   warp3
            */ 
            constexpr int K_BLOCKS = decltype(cute::size<1>(k_decayed))::value / 16; // size<1>(k_decayed) = D = 128 (it has original shape CHUNK, D), after this /16 is computing K-reduction dimension in 8 chunks

            {
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_k_view);
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_q_view);

            #pragma unroll
            for (int k = 0; k < K_BLOCKS; ++k) {
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                movm_transpose_c_to_b_16x16(resident_state[0][k], tCrB);

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[0]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[0]);

                movm_transpose_c_to_b_16x16(resident_state[1][k], tCrB);

                if (k + 1 < K_BLOCKS) {
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_k_view);
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_q_view);
                }

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[1]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[1]);
            }
            }

            // ======== Phase 2: Cast out (keep in regs), load v/INV/beta ========
            // so from Phase 1, we'd have filled up u_acc[2] & out_acc[2] for both 16-cols blocks owned by a warp.
            SFragT out_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); }); // so one of the reason we didn't used cute::identity{} here like before is because transform tries to do element-wise op match, which in this case is going to be Float = BF16. the error that shows up is operand types are 'cutlass::bfloat16_t' & 'const float' are no match.

            SFragT v_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor v_block = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * 2 + i));
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(v_block), smem_thr_load_C.retile_D(v_bf16[i])); // retile_D because destination needs to match the bf16 fragment.
            }

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(INV), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BF16 beta0 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id)))); // since group_id is 0..7 (+8), so these cover two token rows per-lane group as together both groups covers row 0..15
            BF16 beta1 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id + 8))));

            // ======== Phase 3: u = (v - u) * beta; u = INV @ u (per block) ========
            SFragT u_bf16[2]; // for the current u fragment in BF16
            uint32_t u_b_regs[4];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });

                // make_coord does something like this for:
                // make_coord(make_coord(a, 0), 0, d)
                //   frag[a][0][0][d]
                //
                //  make_coord(make_coord(a, 1), 0, d) is like:
                //   frag[a][1][0][d]
                //  so basically we're indexing over [2][2][0][2] which is written in CuTe like ((2,2),0,2) which are called "modes".
                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0;
                        u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1;
                    }
                }

                // we are doing this because here we need to do INV @ U but from before U has been C-type fragment [M, N] but now it's in B-type fragment [N,K]
                // so here SFragT u_bf16 has size: 8 & value_bytes: 2 meaning sfrag has 8 elements with each BF16 element being 2 bytes. it's layout is something like this: layout - ((_2,_2),_1,_2):((_1,_2),_0,_4). Now why 8 elements it's because the MMA fragment for C tile was Tile<_16,_16,_16> meaning 16x16 = 256 elements, since each warp has 32 lanes -> 256 elements / 32 lanes = 8 elements per lane.

                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0)); // like we discussed each lane has 8 BF16 elements ('u_bf16[i]' is the lane in question). Now each bf16 is 16-bits ofc (8 bf16 elements = 8 * 16-bits = 128-bits) & a uint32_t is 32-bits again ofc, 128 / 32 = 4 uint32_t registers of equivalent 1 lane u_bf16[i] space (u_c[0..3] are the registers in question). also each uint32_t has 2 BF16 values packed in it.

                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]); // okay, so the reason we convert u_bf16 to uint32_t is because SM75_U32x1_MOVM_T takes source & destination argument as uint32_t dtypes
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                auto tCrB_u_tmp = thr_mma.partition_fragment_B(B_ref); // manually filing it into b_dst because other we'd have to move it back to SMEM & then using smem_tiled_copy_B load it back to registers using ldmatrix. Saves us a round trip. we do this also because u_b_regs are simply register arrays, & to put the values from these to an actual MMA frament which is why we are creating this.
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_tmp(0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(u_acc[i]); // prev u_acc[i] has 'k_decayed @ s_acc' output. we need a new one for 'INV @ u'
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_tmp(_,_,Int<0>{}), u_acc[i]); // gemm does 'u_acc[i] = INV @ u'

                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); }); // BF16 again because Phase 4 & Phase 6 would need U again in BF16 B operand fragment.
            }

            // ======== Phase 4: Load Mqk, MOVM_T → tCrB_u_arr, Mqk@U + add out ========
            // we have two vars here: u_bf16 & out_bf16, where
            // u_bf16 = INV ((v - k_decayed @ s_acc) * beta)
            // out_bf16 = q_decayed @ s_acc
            // out = out + Mqk @ U
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(Mqk), tCrAi_k_view); // being copied as an A operand is because Mqk @ U makes it A
            cute::transform(tCrAi_k, tCrA_k, cute::identity{}); // views it as a ldmatrix tile & then moves it back to the actual A operand fragment.

            BFragT_u tCrB_u_arr[2];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                // making u_bf16 as a B operand fragment from being a C operand fragment.
                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                tCrB_u_arr[i] = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_arr[i](0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(out_acc[i]); // clear the Phase 1 value of out_acc
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]); // out_acc = Mqk @ U

                SFragT gemm_bf16;
                cute::transform(out_acc[i], gemm_bf16, [] __device__ (float x) { return BF16(x); });
                cute::transform(out_bf16[i], gemm_bf16, out_bf16[i], [] __device__ (BF16 c, BF16 a) { return c + a; });
            }

            // Late output-acquire (cycle trim): first out-stage write is in
            // phase 5, so ownership is needed only now; the wait typically
            // finds the stage already released.
            store_pipeline.producer_acquire(out_write);
            // ======== Phase 5: Store final out ========
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor out_block = local_tile(out_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * 2 + i));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(out_bf16[i]), smem_thr_store_C.partition_D(out_block)); // this uses stmatrix to store from registers to smem. from out_bf16 register fragment -> smem out_block. also we use retile_S, when we already know what lane register fragments are & for partition_D, when we already know a full tile and want to make it into per-lane register writes.
            }


            // ======== Phase 6: s_acc update ========
            // s_acc[D, D] = s_acc * g_total + k_restored_t[D, 16] @ U[16, D]
            // Each warp handles columns [warp_id*32, (warp_id+1)*32] = 2 x 16x16 blocks
            // U is already in tCrB_u_arr[0..1] as B operands (from Phase 4 MOVM_T)
            constexpr int PREFETCH = 1;
            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16; // k_restored_t shape is [128, 16]which means size<0> gives us 128 / 16 = 8 row blocks of state.

            Tensor tCrAi_kr = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_kr_view = smem_thr_copy_A_T.retile_D(tCrAi_kr);

            AFragT ring_A_kr[PREFETCH]; // PREFETCH = 1 is just one-slot buffering. also ring_A_kr loads 'k_restored_t' A fragment for current row block
            float ring_g0[PREFETCH], ring_g1[PREFETCH]; // g_total values for two row groups in this lane's fragment.

            #pragma unroll
            for (int i = 0; i < PREFETCH; ++i) {
                Tensor kr_block = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(i, 0));
                copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block), tCrAi_kr_view);
                cute::transform(tCrAi_kr, ring_A_kr[i], cute::identity{}); // this whole copy & transform b/w tCrAi_kr_view & tCrAi_kr & ring_A_kr can be skipped just like the s_block one below.


                ring_g0[i] = g_total(i * 16 + group_id); // so the reason this is 'i * 16', is only because state is in tiles of 16x16. the '+ 8' is mostly for group to handle two row positions 0..7 & 8..15
                ring_g1[i] = g_total(i * 16 + group_id + 8);
            }

            /*
              So one row-block iteration has this rhythm:

              Use current A and the register-resident state:
                u_acc[bi] = A_current @ U[bi]

              Overwrite A/g ring with next row block:
                ring_A_kr[slot] = A_next
                ring_g0/g1[slot] = g_next

              Update the current state fragment in place:
                resident_state[bi][m] =
                    resident_state[bi][m] * g_current + u_acc[bi]

              Only the final chunk exports resident_state to shared memory for
              the optional final-state TMA store.
            */
            #pragma unroll
            for (int m = 0; m < S_M_BLOCKS; ++m) {
                const int slot = m % PREFETCH;

                float g0 = ring_g0[slot];
                float g1 = ring_g1[slot];

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    clear(u_acc[bi]);
                    // See docs/fwd-kernel2-phase6-state-update-tiling.md for the slot-vs-bi tiling map.
                    gemm(thr_mma, ring_A_kr[slot](_,_,Int<0>{}), tCrB_u_arr[bi](_,_,Int<0>{}), u_acc[bi]); // computing u_acc[bi] = k_restored_t_block @ U_block
                }

                if (m + PREFETCH < S_M_BLOCKS) {
                    Tensor kr_next = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, 0));
                    copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_next), tCrAi_kr_view);
                    cute::transform(tCrAi_kr, ring_A_kr[slot], cute::identity{});

                    ring_g0[slot] = g_total((m + PREFETCH) * 16 + group_id); // pre-fetching the next g_total tiles is totally okay here because g0, g1 already have prev. values.
                    ring_g1[slot] = g_total((m + PREFETCH) * 16 + group_id + 8);
                }

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    auto& state_fragment = resident_state[bi][m];
                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        #pragma unroll
                        for (int d = 0; d < 2; ++d) {
                            // basically what is happening here is "new_state = old_state * g_total + update"
                            // c0 & c1 are two sets of C-fragment coordinates owned by this lane (mainly two row groups inside 16x16 tile).
                            // each lane updates 8 BF16 C-fragment elements 
                            auto c0 = make_coord(make_coord(a, 0), 0, d);
                            auto c1 = make_coord(make_coord(a, 1), 0, d);
                            state_fragment(c0) = BF16(bf16_to_f32(state_fragment(c0)) * g0 + u_acc[bi](c0));
                            state_fragment(c1) = BF16(bf16_to_f32(state_fragment(c1)) * g1 + u_acc[bi](c1));
                        }
                    }

                    // The store warp observes the last output-stage commit only
                    // after these writes and the following shared-memory fence.
                    if constexpr (HasStateOut) {
                        if (t + 1 == t_tiles) {
                            Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m, warp_id * 2 + bi));
                            copy(smem_tiled_store_C_T, smem_thr_store_C_T.retile_S(state_fragment), smem_thr_store_C_T.partition_D(s_block));
                        }
                    }
                }
            }
            }
            // See docs/fwd-kernel2-pipeline-handoff.md for the load/store pipeline ownership protocol.
            cutlass::arch::fence_view_async_shared();
            store_pipeline.producer_commit(out_write);
            load_pipeline.consumer_release(load_read);
            ++load_read;
            ++out_write;
        }
    }

    if (warp_role == WarpRole::STORE && lane_predicate) {
        Tensor g_out = tma_store_out.get_tma_tensor(make_shape(H, T_total, D));
        auto cta_tma_store = tma_store_out.get_slice(Int<0>{});
        StorePipelineState out_read;
        for (int t = 0; t < t_tiles; ++t) {
            store_pipeline.consumer_wait(out_read);
            int stage = out_read.index();
            int actual_len = min(CHUNK, seq_len - t * CHUNK);

            BF16* out_stage_ptr = shared_storage.output[stage].out.begin();


            if (actual_len < CHUNK) {
                Tensor s_out = make_tensor(make_smem_ptr(out_stage_ptr), OutSmemLayout{});
                for (int row = 0; row < actual_len; ++row) {
                    int64_t global_base =
                        (bos + t * CHUNK + row) * H * D +
                        head_idx * D + column_begin;
                    for (int col = 0; col < CTA_N; ++col) {
                        out_raw_ptr[global_base + col] = s_out(row, col);
                    }
                }
            } else {
                // TMA store for full tiles
                auto out_off = g_out.layout()(head_idx, int(bos) + t * CHUNK, column_begin);
                Tensor g_out_tile = make_tensor(g_out.data() + out_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<CTA_N>{}), stride(g_out.layout())));
                Tensor s_out_tile = make_tensor(make_smem_ptr(out_stage_ptr), TMAOutLayout{});
                cute::copy(
                    tma_store_out,
                    cta_tma_store.partition_S(s_out_tile),
                    cta_tma_store.partition_D(g_out_tile)
                );
                tma_store_arrive();
            }

            tma_store_wait<0>();
            store_pipeline.consumer_release(out_read);
            ++out_read;
        }

        if constexpr (HasStateOut && !StateFP32) {
            // BF16 state: TMA store directly from state_acc
            // See docs/fwd-kernel2-final-state-tma-shapes.md for the [N,H,D,D] -> [N*H,D,D] TMA tile mapping.
            Tensor g_final = tma_store_final_state.get_tma_tensor(make_shape(N * H, D, D));
            auto state_off = g_final.layout()(seq_idx * H + head_idx, column_begin, 0);
            Tensor g_final_tile = make_tensor(g_final.data() + state_off,
                make_layout(make_shape(Int<1>{}, Int<CTA_N>{}, Int<D>{}), stride(g_final.layout())));
            Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TMAStateSmemLayout{});

            auto cta_tma_store_state = tma_store_final_state.get_slice(Int<0>{});
            cute::copy(
                tma_store_final_state,
                cta_tma_store_state.partition_S(s_state),
                cta_tma_store_state.partition_D(g_final_tile)
            );
            tma_store_arrive();
        }
    }

    if constexpr (HasStateOut && StateFP32) {
        // FP32 state: all threads sync, convert bf16->fp32, then STORE warp does TMA
        using FP32StateSmemLayout = typename Layouts::FP32StateSmemLayout;
        using TMAFP32StateSmemLayout = typename Layouts::TMAFP32StateSmemLayout;

        __syncthreads();

        smem_cvt_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, CTA_N, D, NumThreads>(
            shared_storage.state_acc.begin(),
            reinterpret_cast<float*>(shared_storage.state_fp32_buf),
            local_tid);
        __syncthreads();

        if (warp_role == WarpRole::STORE && lane_predicate) {
            Tensor g_final = tma_store_final_state.get_tma_tensor(make_shape(N * H, D, D));
            auto state_off = g_final.layout()(seq_idx * H + head_idx, column_begin, 0);
            Tensor g_final_tile = make_tensor(g_final.data() + state_off,
                make_layout(make_shape(Int<1>{}, Int<CTA_N>{}, Int<D>{}), stride(g_final.layout())));
            Tensor s_fp32 = make_tensor(
                make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)),
                TMAFP32StateSmemLayout{});

            auto cta_tma_store_state = tma_store_final_state.get_slice(Int<0>{});
            cute::copy(
                tma_store_final_state,
                cta_tma_store_state.partition_S(s_fp32),
                cta_tma_store_state.partition_D(g_final_tile)
            );
            tma_store_arrive();
        }
    }

    __syncthreads();
}

template <
    class TmaLoadV,
    class TmaLoadBeta,
    class TmaLoadState,
    class TmaStoreState,
    class TmaStoreOut,
    int CHUNK,
    int D,
    int CTA_N,
    int InputStages,
    int OutputStages,
    int NumThreads,
    bool HasStateIn = true,
    bool HasStateOut = true,
    bool StateFP32 = false,
    bool IsVarlen = true
>
__global__ void __launch_bounds__(NumThreads, 2) _flash_kda_fwd_recurrence(
    CUTE_GRID_CONSTANT TmaLoadV const tma_load_v,
    CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
    CUTE_GRID_CONSTANT TmaLoadState const tma_load_initial_state,
    CUTE_GRID_CONSTANT TmaStoreState const tma_store_final_state,
    CUTE_GRID_CONSTANT TmaStoreOut const tma_store_out,
    cutlass::bfloat16_t* out_raw_ptr,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens,
    int total_tiles,
    K1WorkspaceRawPointers ws_raw
) {
    using Layouts = K2Layouts<D, CHUNK, CTA_N>;
    using SharedStorageT =
        SharedStorageK2<Layouts, InputStages, OutputStages>;
    extern __shared__ __align__(128) unsigned char shared_mem[];
    SharedStorageT& shared_storage =
        *reinterpret_cast<SharedStorageT*>(shared_mem);

    int seq_idx;
    int head_idx;
    if constexpr (IsVarlen) {
        if (N <= 32) {
            head_idx = int(blockIdx.x);
            seq_idx = sequence_for_descending_length_rank(
                cu_seqlens, N, int(blockIdx.y));
        } else {
            seq_idx = int(blockIdx.x);
            head_idx = int(blockIdx.y);
        }
    } else {
        seq_idx = int(blockIdx.x);
        head_idx = int(blockIdx.y);
    }

    flash_kda_fwd_recurrence_body<
        TmaLoadV, TmaLoadBeta,
        TmaLoadState, TmaStoreState, TmaStoreOut,
        CHUNK, D, CTA_N, InputStages, OutputStages, NumThreads,
        HasStateIn, HasStateOut, StateFP32, IsVarlen>(
        tma_load_v, tma_load_beta,
        tma_load_initial_state, tma_store_final_state, tma_store_out,
        out_raw_ptr, T_total, H, N, cu_seqlens, total_tiles,
        shared_storage, seq_idx, head_idx, ws_raw);
}
