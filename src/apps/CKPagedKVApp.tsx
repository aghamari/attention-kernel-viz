import React from 'react';
import { ArrowLeft } from 'lucide-react';
import { ATTENTION_COLORS } from '../types/attention';
import PseudoCode from '../components/shared/PseudoCode';

interface CKPagedKVAppProps {
  onBack: () => void;
}

const CKPagedKVApp: React.FC<CKPagedKVAppProps> = ({ onBack }) => {
  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} />
        Back to Home
      </button>

      <header className="app-header">
        <h1 style={{
          background: ATTENTION_COLORS['ck-pk'].gradient,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          CK-PK (CK FMHA PagedKV)
        </h1>
        <p>Single-kernel paged-KV attention from Composable Kernel. Each workgroup handles one Q head.
          Designed for <strong>prefill</strong> where Q tiles have many tokens.</p>
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
            <li><strong>Grid over Q heads:</strong> dim3(nhead_q, batch, q_tiles) — one WG per Q head, unlike CK-UA which uses KV heads</li>
            <li><strong>Per-token page lookup:</strong> Each KV token resolved individually via block_table</li>
            <li><strong>Single kernel:</strong> No split-KV, no combine step needed</li>
            <li><strong>Best for prefill:</strong> Many Q tokens fill bm0=128 tile, high MFMA utilization</li>
            <li><strong>No head-merge:</strong> Q heads sharing a KV head launch <strong>separate</strong> WGs that independently read the same K/V pages</li>
          </ul>
        </div>

        {/* Config */}
        <PseudoCode
          title="Configuration (Example)"
          code={`nhead_q  = 8       # query heads
nhead_kv = 2       # key/value heads
hdim     = 64      # dimension per head
num_queries_per_kv = nhead_q // nhead_kv  # = 4 (GQA-4)
num_seqs = 4       # sequences in the batch
page_size = 64     # tokens per KV cache page

bm0 = 128    # Q tile height (tokens per workgroup)
bn0 = 32     # KV tile width (KV tokens per loop iteration)

# === GRID ===
# dim3(nhead_q, batch, q_tiles)
# One workgroup per (Q head, batch, Q tile)
total_q_tiles = ceil(max_seqlen_q / bm0)  # e.g. ceil(1/128) = 1 for decode
# Total WGs = nhead_q * num_seqs * total_q_tiles = 8 * 4 * 1 = 32

# Each workgroup: 4 warps, 256 threads (on AMD, 1 warp = 64 threads)`}
        />

        {/* Pseudocode */}
        <PseudoCode
          title="Workgroup Pseudocode"
          subtitle="Example: q_head_idx=5, seq_idx=1, q_tile_idx=0"
          code={`def workgroup_ck_pk(q_head_idx, seq_idx, q_tile_idx):
    kv_head_idx = q_head_idx // 4         # = 1 (GQA-4 mapping)
    kv_len = seq_lens_k[seq_idx]          # = 200 for seq1

    # Load Q tile: bm0 rows of Q tokens for this head
    # For decode: only 1 token, rest is padding
    # For prefill: up to bm0 real tokens
    token_start = q_tile_idx * bm0
    Q_tile = np.zeros([bm0, hdim])                            # [128, 64]
    for row in range(bm0):
        tok = token_start + row
        if tok < seqlen_q[seq_idx]:
            Q_tile[row] = Q_flat[cu_seqlens_q[seq_idx] + tok, q_head_idx]

    # Accumulators (one per row)
    acc = np.zeros([bm0, hdim])    # [128, 64]
    m = np.full([bm0], -np.inf)    # [128]
    l = np.zeros([bm0])            # [128]

    # Loop over KV in chunks of bn0
    for kv_pos in range(0, kv_len, bn0):  # step by 32

        # Page lookup: resolve physical pages for this KV chunk
        K_tile = np.zeros([bn0, hdim])     # [32, 64]
        V_tile = np.zeros([bn0, hdim])     # [32, 64]
        for t in range(bn0):
            tok = kv_pos + t
            if tok < kv_len:
                page_col    = tok // page_size
                page_offset = tok % page_size
                phys_page = block_table[seq_idx][page_col]
                K_tile[t] = kv_cache[phys_page, page_offset, kv_head_idx, :]
                V_tile[t] = kv_cache[phys_page, page_offset, kv_head_idx, :]

        # Matrix multiply: all 256 threads cooperate on this MFMA
        S = Q_tile @ K_tile.T / sqrt(hdim)   # [128, 64] @ [64, 32] = [128, 32]

        # Mask out-of-range KV positions and causal violations
        for row in range(bm0):
            q_pos = token_start + row
            for col in range(bn0):
                kv_tok = kv_pos + col
                if kv_tok >= kv_len or kv_tok > q_pos:  # causal
                    S[row, col] = -np.inf

        # Online softmax update (each row independently)
        for row in range(bm0):
            m_new = max(m[row], np.max(S[row]))
            correction = np.exp(m[row] - m_new)
            P = np.exp(S[row] - m_new)               # [32]
            l[row] = l[row] * correction + np.sum(P)
            acc[row] = acc[row] * correction + P @ V_tile
            m[row] = m_new

    # Store results
    for row in range(bm0):
        tok = token_start + row
        if tok < seqlen_q[seq_idx]:
            out[cu_seqlens_q[seq_idx] + tok, q_head_idx] = acc[row] / l[row]`}
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
            <li>Grid iterates over <strong>Q heads</strong> (nhead_q=8), not KV heads</li>
            <li>Each WG handles 1 Q head — Q heads sharing a KV head launch <strong>separate</strong> WGs
              that independently read the same K/V pages</li>
            <li>No head-merging: for decode, only 1 row out of bm0=128 has real data</li>
            <li>Good for <strong>prefill</strong>: many Q tokens fill the tile, MFMA utilization is high</li>
            <li>Source: <code>3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_pagedkv_kernel.hpp</code></li>
            <li>Python entry: <code>aiter/ops/mha.py</code> → <code>mha_varlen_fwd_pagedkv()</code></li>
          </ul>
        </div>

        {/* MFMA Utilization Comparison */}
        <div style={{
          padding: '20px',
          background: 'white',
          borderRadius: '12px',
          boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)'
        }}>
          <h3 style={{ marginBottom: '15px', color: '#333' }}>MFMA Utilization for Decode (1 token, GQA-4)</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #ddd' }}>
                <th style={{ padding: '10px', textAlign: 'left' }}>Kernel</th>
                <th style={{ padding: '10px', textAlign: 'left' }}>Q Tile</th>
                <th style={{ padding: '10px', textAlign: 'left' }}>Useful Rows</th>
                <th style={{ padding: '10px', textAlign: 'left' }}>Utilization</th>
                <th style={{ padding: '10px', textAlign: 'left' }}>WGs per seq</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid #eee', background: '#fce4ec' }}>
                <td style={{ padding: '10px', fontWeight: 'bold' }}>CK-PK (bm0=128)</td>
                <td style={{ padding: '10px' }}>[128, 64]</td>
                <td style={{ padding: '10px' }}>1</td>
                <td style={{ padding: '10px', color: '#c62828' }}>0.8%</td>
                <td style={{ padding: '10px' }}>8 (one per Q head)</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '10px' }}>CK-UA tiny (kBlockM=16)</td>
                <td style={{ padding: '10px' }}>[16, 64]</td>
                <td style={{ padding: '10px' }}>4</td>
                <td style={{ padding: '10px', color: '#2e7d32' }}>25%</td>
                <td style={{ padding: '10px' }}>2 (one per KV head)</td>
              </tr>
              <tr style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '10px' }}>Triton 2D (BLOCK_M=16)</td>
                <td style={{ padding: '10px' }}>[16, 64]</td>
                <td style={{ padding: '10px' }}>4</td>
                <td style={{ padding: '10px', color: '#2e7d32' }}>25%</td>
                <td style={{ padding: '10px' }}>2 (one per KV head)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default CKPagedKVApp;
