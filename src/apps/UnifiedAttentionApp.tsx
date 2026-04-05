import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft } from 'lucide-react';
import { useUnifiedAttentionStore } from '../store/unifiedAttentionStore';
import ConfigDisplay from '../components/shared/ConfigDisplay';
import AttentionPipelineFlow from '../components/unified/AttentionPipelineFlow';
import AttentionStepByStep from '../components/unified/AttentionStepByStep';
import AttentionMemoryLayout from '../components/unified/AttentionMemoryLayout';
import Unified2DVisualization from '../components/unified/Unified2DVisualization';
import Unified3DVisualization from '../components/unified/Unified3DVisualization';
import GridLaunchVisualizer from '../components/shared/GridLaunchVisualizer';
import PseudoCode from '../components/shared/PseudoCode';
import ParameterGlossary from '../components/shared/ParameterGlossary';
import { createUnifiedGlossaryEntries } from '../data/glossaries/unifiedGlossary';

interface UnifiedAttentionAppProps {
  onBack: () => void;
  onNavigate?: (type: string) => void;
}

const UnifiedAttentionApp: React.FC<UnifiedAttentionAppProps> = ({ onBack, onNavigate }) => {
  const {
    config,
    gridLaunch,
    activeTab,
    highlightedCell,
    setActiveTab,
    setHighlightedCell,
    toggle3DKernel,
    initializeData
  } = useUnifiedAttentionStore();

  useEffect(() => {
    initializeData();
  }, [initializeData]);

  const tabs = [
    { id: 'pipeline', label: 'Pipeline' },
    { id: 'algorithm', label: 'Algorithm' },
    { id: 'pseudocode', label: 'Pseudocode' },
    { id: '2d-vs-3d', label: '2D vs 3D' },
    { id: 'memory', label: 'Memory' },
    { id: 'performance', label: 'Performance' }
  ] as const;

  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} />
        Back to Home
      </button>

      <header className="app-header">
        <h1>Unified Attention (2D/3D)</h1>
        <p>Adaptive kernel selection for optimal performance across sequence lengths</p>

        <div style={{
          marginTop: '15px',
          padding: '15px',
          background: '#f0f7ff',
          borderRadius: '8px',
          border: '2px solid #3498db',
          fontSize: '13px',
          lineHeight: '1.6',
          textAlign: 'left'
        }}>
          <strong>What is Unified Attention?</strong>
          <p style={{ marginTop: '8px', marginBottom: '8px' }}>
            Unified Attention is a Flash Attention variant from the aiter library that uses the same core algorithm
            (online softmax + tiling) but adds adaptive GPU kernel selection. It automatically chooses between
            two different grid launch strategies—2D or 3D—based on sequence length and workload characteristics.
            Both variants use Flash Attention's memory-efficient tiled computation, but differ in parallelization strategy.
          </p>
          <p style={{ marginTop: '8px', marginBottom: '0' }}>
            <strong>Core Algorithm:</strong> Flash Attention (online softmax, tiling, O(N²) → O(N) memory)<br/>
            <strong>Innovation:</strong> Adaptive grid selection - 2D kernel for short/decode (≤2048 tokens, single-pass),
            3D kernel for long/prefill (&gt;2048 tokens, segment-based parallelism with reduction phase).
            Same math, different GPU execution strategies.
          </p>
        </div>

        <div style={{
          marginTop: '15px',
          padding: '15px',
          background: '#f9f9f9',
          borderRadius: '8px',
          border: '2px solid #ddd',
          fontSize: '12px',
          fontFamily: 'monospace',
          textAlign: 'left'
        }}>
          <strong style={{ fontFamily: 'sans-serif' }}>Function Signature:</strong>
          <pre style={{ marginTop: '8px', marginBottom: '0', whiteSpace: 'pre-wrap' }}>
{`def unified_attention(
    q,                  # [total_tokens, num_heads, head_dim]
    k,                  # KV cache K pointer [num_pages, page_size, num_kv_heads, head_dim]
    v,                  # KV cache V pointer [num_pages, page_size, num_kv_heads, head_dim]
    out,                # [total_tokens, num_heads, head_dim] output
    cu_seqlens_q,       # [num_seqs + 1] cumulative Q token counts
    max_seqlen_q,       # max Q sequence length
    seqused_k,          # [num_seqs] KV cache length per sequence
    max_seqlen_k,       # max KV sequence length
    softmax_scale,      # 1 / sqrt(head_dim)
    causal,             # causal mask
    window_size,        # sliding window (-1, -1) = disabled
    block_table,        # [num_seqs, max_pages_per_seq] -> physical page IDs
    softcap,            # logit soft-capping (0 = disabled)
    q_descale, k_descale, v_descale,  # quantization scales
    alibi_slopes=None,  # ALiBi positional bias
    output_scale=None,  # FP8 output scale
    qq_bias=None,
    sinks=None,         # attention sinks
)`}</pre>
          <p style={{ marginTop: '8px', fontSize: '11px', fontFamily: 'sans-serif', color: '#888' }}>
            k and v are pointers to the paged KV cache (not per-sequence tensors).
            The kernel uses <code>block_table</code> to translate logical token positions to physical page addresses.
          </p>
        </div>
      </header>

      <div className="tab-navigation">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="overview-layout">
        <div className="left-panel">
          <ConfigDisplay
            title="Configuration"
            params={[
              { label: 'Batch Size', value: 1 },
              { label: 'Sequence Length', value: 512 },
              { label: 'Num Heads', value: 8 },
              { label: 'Head Dim', value: 64 },
              { label: 'Num KV Heads', value: 8 },
              { label: 'Block M', value: 64 },
              { label: 'Block N', value: 64 },
              { label: 'Num Segments', value: 4 },
              { label: 'Sliding Window', value: 'Disabled' },
              { label: 'Window Size', value: 256 },
              { label: 'Causal', value: 'Enabled' }
            ]}
          />

          <div className="toggle-container" style={{ marginTop: '15px' }}>
            <span>2D Kernel</span>
            <div
              className={`toggle-switch ${config.use3DKernel ? 'active' : ''}`}
              onClick={toggle3DKernel}
            >
              <div className="toggle-knob" />
            </div>
            <span>3D Kernel</span>
          </div>
        </div>

        <div className="center-panel">
          <div className={`tab-content ${activeTab === 'pipeline' ? 'active' : ''}`}>
            <AttentionPipelineFlow />
          </div>

          <div className={`tab-content ${activeTab === 'algorithm' ? 'active' : ''}`}>
            <AttentionStepByStep />
          </div>

          <div className={`tab-content ${activeTab === 'pseudocode' ? 'active' : ''}`}>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
              style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}
            >
              <h3>Kernel Pseudocode</h3>

              {onNavigate && (
                <div
                  onClick={() => onNavigate('triton2d-viz')}
                  style={{
                    padding: '12px 16px',
                    background: 'linear-gradient(135deg, #e94560 0%, #0f3460 100%)',
                    borderRadius: '8px',
                    color: 'white',
                    cursor: 'pointer',
                    fontWeight: 600,
                    fontSize: '14px',
                    textAlign: 'center',
                  }}
                >
                  Open Interactive Triton 2D Visualization →
                </div>
              )}

              <PseudoCode
                title="Triton 2D -- Single-Pass Unified Attention (Head-Merge)"
                subtitle="Single-pass paged-KV attention in Triton with GQA head-merging. Grid over KV heads, single pass."
                code={`# === CONFIG ===
# nhead_q = 8, nhead_kv = 2, hdim = 64
# num_queries_per_kv = nhead_q // nhead_kv = 8 // 2 = 4  (GQA-4)
#
BLOCK_M = 16         # Q tile height (M dimension of GEMM)
                     #   = BLOCK_Q tokens × num_queries_per_kv heads
BLOCK_Q = BLOCK_M // num_queries_per_kv  # = 16 // 4 = 4 token slots
BLOCK_N = 64         # KV tile width (N dimension of GEMM)
                     #   = how many KV tokens per loop iteration
#
# The core GEMM each iteration:
#   Q_tile    @ K_tile.T   = S
#   [BLOCK_M, hdim] @ [hdim, BLOCK_N] = [BLOCK_M, BLOCK_N]
#   [16, 64]        @ [64, 64]        = [16, 64]
#
# The 16 rows of Q_tile are laid out as:
#   row  0: token 0, head 0    ← 4 heads for token 0
#   row  1: token 0, head 1
#   row  2: token 0, head 2
#   row  3: token 0, head 3
#   row  4: token 1, head 0    ← 4 heads for token 1
#   row  5: token 1, head 1
#   row  6: token 1, head 2
#   row  7: token 1, head 3
#   row  8: token 2, head 0    ← 4 heads for token 2
#   ...
#   row 15: token 3, head 3
#
# All 4 heads in a row-group share the SAME K/V (same kv_head),
# but have DIFFERENT Q vectors, so they produce different outputs.

# === GRID ===
# (num_kv_heads, total_num_q_blocks) = (2, ceil(total_tokens / BLOCK_Q))
# For decode with 4 seqs: total_tokens=4, BLOCK_Q=4 → 1 block + padding
# Total WGs = 2 * (4 // 4 + 4) = 2 * 5 = 10
# Each workgroup: 2 warps, 128 threads

def workgroup_triton_2d(kv_head_idx, q_block_global_idx):
    # ── What is q_block_global_idx? ──
    #
    # Q tokens from all sequences are packed into a flat array.
    # Each sequence's tokens are chunked into blocks of BLOCK_Q=4 tokens.
    # q_block_global_idx numbers these blocks sequentially across all seqs.
    #
    # DECODE (1 token per seq, BLOCK_Q=4):
    #   Each seq has just 1 token, which takes 1 block (with 3 unused slots).
    #   Blocks map 1:1 to sequences -- no sharing, no multi-token blocks.
    #
    #   seq:         0         1         2         3
    #   tokens:     [t0]      [t1]      [t2]      [t3]
    #   blocks:    blk 0     blk 1     blk 2     blk 3
    #              (1 tok,   (1 tok,   (1 tok,   (1 tok,
    #               3 pad)   3 pad)    3 pad)    3 pad)
    #
    #   cum_blocks = [0, 1, 2, 3, 4]
    #   global_idx=2 → seq 2, the only block of seq 2
    #
    # PREFILL (multiple tokens per seq, BLOCK_Q=4):
    #   Now blocks actually contain multiple real tokens.
    #
    #   seq 0: 8 tokens  → ceil(8/4) = 2 blocks
    #   seq 1: 4 tokens  → ceil(4/4) = 1 block
    #   seq 2: 12 tokens → ceil(12/4) = 3 blocks
    #   seq 3: 4 tokens  → ceil(4/4) = 1 block
    #
    #   global idx:  0       1       2       3       4       5       6
    #              └──seq 0──┘    └seq 1┘  └─────seq 2──────┘    └seq 3┘
    #   tokens:    [0-3]   [4-7]   [0-3]   [0-3]   [4-7]  [8-11]  [0-3]
    #
    #   cum_blocks = [0, 2, 3, 6, 7]
    #   global_idx=4 → seq 2, local block 1 (tokens 4-7 of seq 2)

    # ── binary_search: global block idx → sequence index ──
    #
    # cu_seqlens_q stores cumulative TOKEN counts, but q_block_global_idx
    # is a BLOCK index. We need BLOCK_Q to bridge the two.
    #
    # Before launch, the host computes a cumulative BLOCK count array:
    #   seqlens_q = [cu_seqlens_q[i+1] - cu_seqlens_q[i] for each seq]
    #             = [8, 4, 12, 4]  (tokens per seq, prefill example)
    #   blocks_per_seq = [ceil(s / BLOCK_Q) for s in seqlens_q]
    #                  = [ceil(8/4), ceil(4/4), ceil(12/4), ceil(4/4)]
    #                  = [2, 1, 3, 1]
    #   cum_q_blocks = [0, 2, 3, 6, 7]  (cumulative sum with leading 0)
    #
    # This array is passed to the kernel. The binary search finds:
    #   cum_q_blocks[seq_idx] <= global_idx < cum_q_blocks[seq_idx+1]
    #
    # def binary_search(cum_q_blocks, global_idx):
    #     lo, hi = 0, num_seqs - 1
    #     while lo < hi:
    #         mid = (lo + hi) // 2
    #         if cum_q_blocks[mid + 1] <= global_idx:
    #             lo = mid + 1
    #         else:
    #             hi = mid
    #     return lo
    #
    # Prefill: global_idx=4, cum_q_blocks=[0, 2, 3, 6, 7]
    #   lo=0, hi=3 → mid=1: cum_q_blocks[2]=3 ≤ 4 → lo=2
    #   lo=2, hi=3 → mid=2: cum_q_blocks[3]=6 > 4  → hi=2
    #   lo=hi=2 → return 2  →  seq 2 ✓
    #     (seq 2 owns blocks [3,6), global 4 is block 1 within seq 2)
    #
    # Decode: cum_q_blocks=[0, 1, 2, 3, 4], global_idx=2
    #   lo=0, hi=3 → mid=1: cum_q_blocks[2]=2 ≤ 2 → lo=2
    #   lo=2, hi=3 → mid=2: cum_q_blocks[3]=3 > 2  → hi=2
    #   lo=hi=2 → return 2  →  seq 2 ✓  (block i = seq i)
    #
    # BLOCK_Q was needed to BUILD cum_q_blocks (tokens → blocks).
    # The kernel receives cum_q_blocks as a launch parameter.
    seq_idx = binary_search(cum_q_blocks, q_block_global_idx)
    q_start = cu_seqlens_q[seq_idx]           # token offset in flat Q array
    q_len   = cu_seqlens_q[seq_idx + 1] - q_start  # tokens in this seq
    kv_len  = seq_lens_k[seq_idx]             # KV cache length for this seq

    # Convert global block index → local block index within this sequence
    # Prefill: global=4, cum_q_blocks[seq=2]=3 → local = 4-3 = 1
    #   local block 1 × BLOCK_Q=4 → tokens 4-7 of seq 2
    # Decode: global=2, cum_q_blocks[seq=2]=2 → local = 2-2 = 0
    #   local block 0, token 0 (the only token), rest is padding
    q_block_local = q_block_global_idx - cum_q_blocks[seq_idx]
    if q_block_local * BLOCK_Q >= q_len:
        return  # padding block (beyond this seq's tokens), exit early

    # === HEAD-MERGE ===
    # Pack BLOCK_Q tokens × num_queries_per_kv heads into BLOCK_M rows
    # Formula: row = tok_index * num_queries_per_kv + head_within_group
    Q_tile = np.zeros([BLOCK_M, hdim])   # [16, 64]
    for row in range(BLOCK_M):           # row = 0..15
        tok  = q_block_local * BLOCK_Q + row // num_queries_per_kv
        #   row 0-3: tok = 0  (row // 4 = 0)
        #   row 4-7: tok = 1  (row // 4 = 1)
        head = kv_head_idx * num_queries_per_kv + row % num_queries_per_kv
        #   kv_head_idx=0: heads 0,1,2,3  (row % 4)
        #   kv_head_idx=1: heads 4,5,6,7
        if tok < q_len:
            Q_tile[row] = Q_flat[q_start + tok, head]
    # For decode (q_len=1): only rows 0-3 have data, rows 4-15 are zero padding

    acc = np.zeros([BLOCK_M, hdim])      # [16, 64] accumulator
    m = np.full([BLOCK_M], -np.inf)      # [16] running max per row
    l = np.zeros([BLOCK_M])              # [16] running exp-sum per row

    # Loop over ALL KV tokens in BLOCK_N=64 chunks
    num_tiles = ceil(kv_len / BLOCK_N)  # ceil(200 / 64) = 4
    for j in range(num_tiles):
        # Per-position page lookup within the tile
        K_tile = np.zeros([BLOCK_N, hdim])  # [64, 64]
        V_tile = np.zeros([BLOCK_N, hdim])  # [64, 64]
        for t in range(BLOCK_N):
            seq_offset = j * BLOCK_N + t    # absolute KV position
            if seq_offset < kv_len:
                # Paged KV: translate logical position → physical page + slot
                phys_page = block_table[seq_idx][seq_offset // page_size]
                slot      = seq_offset % page_size
                K_tile[t] = kv_cache[phys_page, slot, kv_head_idx, :]
                V_tile[t] = kv_cache[phys_page, slot, kv_head_idx, :]
        # K/V loaded ONCE, shared by all 4 heads in this workgroup

        # MFMA: scores for ALL heads at once
        S = Q_tile @ K_tile.T / sqrt(hdim)
        # [BLOCK_M, hdim] @ [hdim, BLOCK_N] = [BLOCK_M, BLOCK_N]
        # = [16, 64] @ [64, 64] = [16, 64]
        #   S[0,:] = head 0's scores against BLOCK_N=64 KV tokens
        #   S[1,:] = head 1's scores (same K, different Q)
        #   S[2,:] = head 2's scores
        #   S[3,:] = head 3's scores
        #   S[4,:] = head 0's scores for token 1 (if prefill)
        #   S[4..15,:] = zeros for decode (padding rows)

        # Causal mask + sliding window
        for row in range(BLOCK_M):
            tok = q_block_local * BLOCK_Q + row // num_queries_per_kv
            q_pos = q_start + tok + (kv_len - q_len)  # absolute position
            for col in range(BLOCK_N):
                kv_pos = j * BLOCK_N + col
                if kv_pos >= kv_len or kv_pos > q_pos:
                    S[row, col] = -np.inf

        # Online softmax update (each row independently)
        for row in range(BLOCK_M):
            m_new = max(m[row], np.max(S[row]))
            correction = np.exp(m[row] - m_new)   # rescale old accumulators
            P = np.exp(S[row] - m_new)             # softmax numerators
            l[row] = l[row] * correction + np.sum(P)
            acc[row] = acc[row] * correction + P @ V_tile
            m[row] = m_new

    # Store: un-merge heads back to their output positions
    for row in range(BLOCK_M):
        tok  = q_block_local * BLOCK_Q + row // num_queries_per_kv
        head = kv_head_idx * num_queries_per_kv + row % num_queries_per_kv
        if tok < q_len:
            out[q_start + tok, head] = acc[row] / l[row]`}
              />

              <PseudoCode
                title="Triton 3D -- Split-KV Unified Attention (Two Kernels)"
                subtitle="Splits KV into segments processed by parallel workgroups, then reduces. Head-merging included."
                code={`# === CONFIG ===
# Same head-merge as 2D, but KV sequence is split into parallel segments
BLOCK_M = 16         # Q tile height (M dimension) = 4 tokens × 4 heads
BLOCK_Q = BLOCK_M // num_queries_per_kv  # = 16 // 4 = 4 token slots
BLOCK_N = 64         # KV tile width (N dimension) = KV tokens per iteration
NUM_SEGMENTS = 16    # chosen by heuristic to fill GPU
#
# Why split? When batch is small and KV is long, the 2D kernel doesn't
# have enough workgroups to fill the GPU. 3D adds a segment dimension:
# each segment processes a SLICE of the KV range independently,
# then a reduce kernel merges the partial results.

# === GRID ===
# Attention: (total_q_blocks, num_kv_heads, NUM_SEGMENTS)
#   = (5, 2, 16) = 160 WGs  (vs 10 for 2D -- 16x more parallelism!)
# Reduce:    (total_q_tokens, num_query_heads)
#   = (4, 8) = 32 WGs

# === INTERMEDIATE BUFFERS (fp32, between the two kernels) ===
segm_output = np.zeros([total_q, nhead_q, NUM_SEGMENTS, hdim])
segm_max    = np.zeros([total_q, nhead_q, NUM_SEGMENTS])
segm_expsum = np.zeros([total_q, nhead_q, NUM_SEGMENTS])

# ============================================================
# KERNEL 1: Segment Attention (with head-merge)
# ============================================================
# Each WG processes the SAME Q tile but a DIFFERENT KV slice.

def workgroup_triton_3d(q_block_global_idx, kv_head_idx, segm_idx):
    # q_block_global_idx: same flat block index as 2D (see above)
    # segm_idx: which KV segment this WG processes (0..NUM_SEGMENTS-1)
    # All WGs with same (q_block_global_idx, kv_head_idx) but different
    # segm_idx process the SAME Q tile against DIFFERENT KV slices
    seq_idx = binary_search(cu_seqlens_q, q_block_global_idx, BLOCK_Q)
    kv_len = seq_lens_k[seq_idx]          # e.g. 2048

    # Divide KV range into NUM_SEGMENTS slices
    # With kv_len=2048, NUM_SEGMENTS=16, BLOCK_N=64:
    #   tiles_per_segment = ceil(2048 / (16 * 64)) = ceil(2) = 2
    #   segment 0: tiles 0-1  → KV positions 0-127
    #   segment 1: tiles 2-3  → KV positions 128-255
    #   ...
    #   segment 15: tiles 30-31 → KV positions 1920-2047
    tiles_per_segment = ceil(kv_len / (NUM_SEGMENTS * BLOCK_N))
    tile_start = segm_idx * tiles_per_segment
    tile_end   = min(tile_start + tiles_per_segment, ceil(kv_len / BLOCK_N))
    if tile_start * BLOCK_N >= kv_len:
        return  # this segment has no KV to process

    # Load Q tile with head-merge (same layout as Triton 2D)
    Q_tile = load_q_merged(seq_idx, kv_head_idx, q_block_global_idx)
    # [BLOCK_M, hdim] = [16, 64]

    acc = np.zeros([BLOCK_M, hdim])
    m = np.full([BLOCK_M], -np.inf)
    l = np.zeros([BLOCK_M])

    # Loop only over THIS segment's tile range (not all KV!)
    for j in range(tile_start, tile_end):
        K_tile, V_tile = page_lookup_tile(seq_idx, kv_head_idx, j, BLOCK_N)
        S = Q_tile @ K_tile.T / sqrt(hdim)
        # ... causal mask, online softmax update ...

    # Store PARTIAL results (not final -- reduce kernel will merge)
    for row in range(BLOCK_M):
        tok  = token_for_row(row)           # row // num_queries_per_kv
        head = head_for_row(row, kv_head_idx)  # kv_head * nqpkv + row % nqpkv
        segm_output[tok, head, segm_idx, :] = acc[row]   # unnormalized output
        segm_max[tok, head, segm_idx]        = m[row]     # local max
        segm_expsum[tok, head, segm_idx]     = l[row]     # local exp-sum

# ============================================================
# KERNEL 2: Reduce Segments (runs after ALL segment WGs finish)
# ============================================================
# Each WG merges all segments for one (token, head) pair.
# Same log-sum-exp trick as CK-SK's combine kernel.

def workgroup_reduce(query_token_idx, query_head_idx):
    seq_idx = token_to_seq(query_token_idx)
    kv_len = seq_lens_k[seq_idx]

    tiles_per_segment = ceil(kv_len / (NUM_SEGMENTS * BLOCK_N))
    act_num_segments = ceil(kv_len / (tiles_per_segment * BLOCK_N))

    # Step 1: find global max across all segments
    overall_max = -np.inf
    for s in range(act_num_segments):
        overall_max = max(overall_max, segm_max[query_token_idx, query_head_idx, s])

    # Step 2: rescale each segment's partial to the global max, then sum
    overall_expsum = 0.0
    output = np.zeros([hdim])
    for s in range(act_num_segments):
        # exp(local_max - global_max) corrects for different local scales
        correction = np.exp(segm_max[query_token_idx, query_head_idx, s] - overall_max)
        rescaled_expsum = segm_expsum[query_token_idx, query_head_idx, s] * correction
        overall_expsum += rescaled_expsum
        output += segm_output[query_token_idx, query_head_idx, s, :] * correction

    out[query_token_idx, query_head_idx] = output / overall_expsum`}
              />

              <PseudoCode
                title="Kernel Selection: 2D vs 3D"
                code={`def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, ...):
    # Step 1: Try CK-UA (fastest for certain decode shapes)
    if _try_ck_unified_attention(q, k, v, out, ...):
        return  # done

    # Step 2: Choose Triton 2D vs 3D
    num_2d_prgms = num_kv_heads * total_num_q_blocks
    target_prgms = get_num_sms() * 2

    if use_2d_kernel(head_size, sliding_window, all_decode,
                     max_seqlen_q, max_seqlen_k,
                     target_prgms, num_2d_prgms):
        kernel_unified_attention_2d[grid](...)
    else:
        kernel_unified_attention_3d[grid](...)
        reduce_segments[grid](...)

def use_2d_kernel(head_size, sliding_window, all_decode,
                  max_seqlen_q, max_seqlen_k,
                  target_num_prgms, num_2d_prgms):
    return (
        (sliding_window > 0)            # 2D handles sliding window
        or (max_seqlen_k <= 512)        # short KV doesn't need splitting
        or (num_2d_prgms > target_num_prgms)  # already enough WGs
    )`}
              />
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === '2d-vs-3d' ? 'active' : ''}`}>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              <h3 style={{ marginBottom: '20px' }}>2D vs 3D Kernel Comparison</h3>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h3>Current Mode: {config.use3DKernel ? '3D Kernel' : '2D Kernel'}</h3>
                <ul>
                  <li><strong>2D Kernel:</strong> Grid (num_kv_heads, total_q_blocks) - optimal for short sequences, decode, sliding window</li>
                  <li><strong>3D Kernel:</strong> Grid (q_blocks, kv_heads, segments) - optimal for long sequences with segment reduction</li>
                </ul>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
                <div>
                  <h4 style={{ marginBottom: '15px' }}>2D Kernel</h4>
                  <Unified2DVisualization />
                </div>
                <div>
                  <h4 style={{ marginBottom: '15px' }}>3D Kernel</h4>
                  <Unified3DVisualization />
                </div>
              </div>

              <GridLaunchVisualizer
                gridInfo={gridLaunch}
                is3D={config.use3DKernel}
                highlightedCell={highlightedCell}
                onCellHover={setHighlightedCell}
                title={config.use3DKernel ? '3D Grid Launch' : '2D Grid Launch'}
              />
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'memory' ? 'active' : ''}`}>
            <AttentionMemoryLayout />
          </div>

          <div className={`tab-content ${activeTab === 'performance' ? 'active' : ''}`}>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              <h3 style={{ marginBottom: '20px' }}>Performance Metrics</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '15px' }}>
                {[
                  { label: 'Q Blocks', value: Math.ceil(config.seqLen / config.blockM) },
                  { label: 'K Blocks', value: Math.ceil(config.seqLen / config.blockN) },
                  { label: 'Segments', value: config.use3DKernel ? config.numSegments : 1 },
                  { label: 'Total Thread Blocks', value: gridLaunch.gridX * gridLaunch.gridY * gridLaunch.gridZ }
                ].map(metric => (
                  <div key={metric.label} style={{
                    padding: '20px',
                    background: '#f8f9fa',
                    borderRadius: '8px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#333' }}>
                      {metric.value}
                    </div>
                    <div style={{ fontSize: '13px', color: '#666', marginTop: '4px' }}>
                      {metric.label}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      <ParameterGlossary
        entries={createUnifiedGlossaryEntries(config)}
        title="Unified Attention Parameter Reference"
        collapsible={true}
      />
    </div>
  );
};

export default UnifiedAttentionApp;
