import React from 'react';
import { ArrowLeft } from 'lucide-react';
import { ATTENTION_COLORS } from '../types/attention';
import PseudoCode from '../components/shared/PseudoCode';

interface CKSplitKVAppProps {
  onBack: () => void;
}

const CKSplitKVApp: React.FC<CKSplitKVAppProps> = ({ onBack }) => {
  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} />
        Back to Home
      </button>

      <header className="app-header">
        <h1 style={{
          background: ATTENTION_COLORS['ck-sk'].gradient,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          CK-SK (CK FMHA Split-KV)
        </h1>
        <p>Splits the KV sequence across multiple workgroups. Each computes a partial result
          over its KV slice. A separate combine kernel merges the partials using log-sum-exp.
          Needed when batch is too small to fill the GPU without splitting.</p>
      </header>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Key Concepts */}
        <div style={{
          padding: '20px',
          background: 'white',
          borderRadius: '12px',
          boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)'
        }}>
          <h3 style={{ marginBottom: '15px', color: '#333' }}>Key Concepts</h3>
          <ul style={{ lineHeight: '1.8', color: '#444', paddingLeft: '20px' }}>
            <li><strong>Two kernel launches:</strong> attention (many WGs) then combine (fewer WGs)</li>
            <li><strong>Split-KV:</strong> Divides KV sequence into num_splits parallel segments</li>
            <li><strong>Intermediate buffers:</strong> o_acc and lse_acc in fp32 between kernels</li>
            <li><strong>num_splits heuristic:</strong> target = multiProcessorCount × 4 total WGs</li>
            <li><strong>Head-merge support:</strong> In decode mode, grid uses nhead_kv instead of nhead_q</li>
            <li><strong>Features:</strong> Sliding window, sinks, softcap, paged KV</li>
          </ul>
        </div>

        {/* Config */}
        <PseudoCode
          title="Configuration (Example)"
          code={`nhead_q  = 8       # query heads
nhead_kv = 2       # key/value heads
hdim     = 64      # dimension per head
num_seqs = 4       # sequences in the batch
page_size = 64     # tokens per KV cache page

kM0 = 16           # Q tile height
num_splits = 8     # chosen by heuristic to fill GPU

# === GRID ===
# Attention kernel: dim3(q_tiles * num_splits, nhead_q, batch)
#   for decode: dim3(1 * 8, 8, 4) = 256 WGs
# Combine kernel:  dim3(q_tiles, nhead_q, batch)
#   for decode: dim3(1, 8, 4) = 32 WGs

# With head-merge (decode + GQA): nhead_q -> nhead_kv in grid,
# effective Q length = seqlen_q * num_queries_per_kv

# === INTERMEDIATE BUFFERS ===
o_acc   = np.zeros([nhead_q, num_splits, total_q, hdim])  # [8, 8, 4, 64] fp32
lse_acc = np.zeros([nhead_q, num_splits, total_q])        # [8, 8, 4]     fp32`}
        />

        {/* Kernel 1: Split-KV Attention */}
        <PseudoCode
          title="Kernel 1: Split-KV Attention"
          subtitle="Each WG processes ONE split of the KV range for one (head, batch, q_tile)"
          code={`def workgroup_splitkv_attention(q_head_idx, seq_idx, split_idx):
    kv_head_idx = q_head_idx // 4
    kv_len = seq_lens_k[seq_idx]

    # This split's KV range
    kv_per_split = ceil(kv_len / num_splits)
    kv_start = split_idx * kv_per_split
    kv_end   = min(kv_start + kv_per_split, kv_len)
    if kv_start >= kv_len:
        return  # nothing to do for this split

    # Load Q tile (same Q for all splits -- they differ only in KV range)
    Q_tile = load_q(seq_idx, q_head_idx)    # [kM0, hdim]

    acc = np.zeros([kM0, hdim])
    m = np.full([kM0], -np.inf)
    l = np.zeros([kM0])

    # Loop only over THIS split's KV range (not all KV)
    for kv_pos in range(kv_start, kv_end, bn0):
        # Page lookup for this KV chunk
        K_tile, V_tile = page_lookup(seq_idx, kv_head_idx, kv_pos, bn0)

        S = Q_tile @ K_tile.T / sqrt(hdim)
        # ... causal mask ...
        # ... online softmax update (same as before) ...

    # Store PARTIAL results (not final -- combine kernel will merge)
    for row in range(kM0):
        tok = token_for_row(row)
        o_acc[q_head_idx, split_idx, tok]   = acc[row]    # unnormalized
        lse_acc[q_head_idx, split_idx, tok] = m[row] + log(l[row])  # log-sum-exp`}
        />

        {/* Kernel 2: Combine */}
        <PseudoCode
          title="Kernel 2: Combine"
          subtitle="Runs after all split-KV WGs finish. Each WG merges all splits for one (head, batch, q_tile)."
          code={`def workgroup_combine(q_head_idx, seq_idx):
    tok = cu_seqlens_q[seq_idx]  # for decode: 1 token

    # Find global max across all splits
    global_max = -np.inf
    for s in range(num_splits):
        global_max = max(global_max, lse_acc[q_head_idx, s, tok])

    # Merge: rescale each split's partial to the global max, then sum
    total = 0.0
    output = np.zeros([hdim])
    for s in range(num_splits):
        # exp(local_lse - global_max) is the correction factor
        w = np.exp(lse_acc[q_head_idx, s, tok] - global_max)
        output += o_acc[q_head_idx, s, tok] * w
        total += w

    out[tok, q_head_idx] = output / total`}
        />

        {/* num_splits Heuristic */}
        <PseudoCode
          title="num_splits Heuristic"
          code={`def num_splits_heuristic(batch_nheads_mblocks, num_SMs):
    if batch_nheads_mblocks >= 0.8 * num_SMs:
        return 1   # enough work without splitting

    # Try each split count, pick smallest with good "wave efficiency"
    best_eff = 0
    for s in range(1, min(128, num_SMs) + 1):
        n_waves = (batch_nheads_mblocks * s) / num_SMs
        eff = n_waves / ceil(n_waves)
        best_eff = max(best_eff, eff)

    for s in range(1, min(128, num_SMs) + 1):
        eff = ...  # same calculation
        if eff >= 0.85 * best_eff:
            return s
    return 1`}
        />

        {/* Key Points */}
        <div style={{
          padding: '20px',
          background: 'white',
          borderRadius: '12px',
          boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)'
        }}>
          <h3 style={{ marginBottom: '15px', color: '#333' }}>Key Points</h3>
          <ul style={{ lineHeight: '1.8', color: '#444', paddingLeft: '20px' }}>
            <li>Two kernel launches: attention (many WGs) then combine (fewer WGs)</li>
            <li>Needs intermediate <code>o_acc</code> and <code>lse_acc</code> buffers in fp32</li>
            <li><code>num_splits</code> chosen by heuristic: target = <code>multiProcessorCount × 4</code> total WGs</li>
            <li>When batch is small and KV is long, splits provide parallelism the GPU needs</li>
            <li>Also supports head-merge for decode (<code>kMergeNumHeadGroupsSeqLenQ=true</code>):
              grid uses <code>nhead_kv</code> instead of <code>nhead_q</code>, effective Q length multiplied by
              <code>num_queries_per_kv</code></li>
            <li>Source: <code>3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp</code></li>
            <li>Python entry: <code>aiter/ops/mha.py</code> → <code>mha_varlen_fwd()</code> with <code>block_table</code></li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default CKSplitKVApp;
