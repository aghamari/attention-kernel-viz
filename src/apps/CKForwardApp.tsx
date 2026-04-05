import React from 'react';
import { ArrowLeft } from 'lucide-react';
import { ATTENTION_COLORS } from '../types/attention';
import PseudoCode from '../components/shared/PseudoCode';

interface CKForwardAppProps {
  onBack: () => void;
}

const CKForwardApp: React.FC<CKForwardAppProps> = ({ onBack }) => {
  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} />
        Back to Home
      </button>

      <header className="app-header">
        <h1 style={{
          background: ATTENTION_COLORS['ck-fwd'].gradient,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          CK-Fwd (CK FMHA Forward)
        </h1>
        <p>Standard flash-attention forward for contiguous (non-paged) KV tensors.
          Same as CK-PK but without page table indirection. Simplest path.</p>
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
            <li><strong>No page table:</strong> K/V accessed by direct offset, no block_table lookups</li>
            <li><strong>Contiguous KV:</strong> K/V stored as [total_kv_tokens, num_kv_heads, head_dim]</li>
            <li><strong>Grid over Q heads:</strong> dim3(nhead_q, q_tiles, batch) — same pattern as CK-PK</li>
            <li><strong>Features:</strong> Sliding window, causal mask, optional bias, softcap</li>
            <li><strong>Use case:</strong> Training or when KV is stored contiguously (rare in vLLM-style inference which always pages KV)</li>
          </ul>
        </div>

        {/* Memory Layout Comparison */}
        <div style={{
          padding: '20px',
          background: 'white',
          borderRadius: '12px',
          boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)'
        }}>
          <h3 style={{ marginBottom: '15px', color: '#333' }}>Memory Layout: Contiguous vs Paged</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
            <div style={{ padding: '15px', background: '#e8eaf6', borderRadius: '8px', border: '2px solid #3F51B5' }}>
              <strong>Contiguous (CK-Fwd):</strong>
              <pre style={{ fontSize: '12px', marginTop: '8px', whiteSpace: 'pre-wrap' }}>
{`k = [total_kv_tokens, num_kv_heads, head_size]
v = [total_kv_tokens, num_kv_heads, head_size]

# Direct access:
kv_start = cu_seqlens_k[batch_idx]
K_data = k[kv_start : kv_start + kv_len]`}
              </pre>
            </div>
            <div style={{ padding: '15px', background: '#fce4ec', borderRadius: '8px', border: '2px solid #E91E63' }}>
              <strong>Paged (CK-UA/SK/PK):</strong>
              <pre style={{ fontSize: '12px', marginTop: '8px', whiteSpace: 'pre-wrap' }}>
{`k_cache = [num_blks, page_size, num_kv_heads, head_size]
v_cache = [num_blks, page_size, num_kv_heads, head_size]

# Indirect access via block_table:
page = block_table[batch, token // page_size]
offset = token % page_size
K_data = k_cache[page, offset]`}
              </pre>
            </div>
          </div>
        </div>

        {/* Config */}
        <PseudoCode
          title="Configuration (Example)"
          code={`nhead_q  = 8       # query heads
nhead_kv = 2       # key/value heads
hdim     = 64      # dimension per head
num_seqs = 4       # sequences in the batch

# === GRID ===
# dim3(nhead_q, q_tiles, batch) = dim3(8, 1, 4) = 32 WGs`}
        />

        {/* Pseudocode */}
        <PseudoCode
          title="Workgroup Pseudocode"
          code={`def workgroup_ck_fwd(q_head_idx, q_tile_idx, seq_idx):
    kv_head_idx = q_head_idx // 4
    kv_len = seq_lens_k[seq_idx]

    Q_tile = load_q(seq_idx, q_head_idx, q_tile_idx)  # [kM0, hdim]

    acc = np.zeros([kM0, hdim])
    m = np.full([kM0], -np.inf)
    l = np.zeros([kM0])

    # KV is contiguous -- direct indexing, no page table
    kv_start = cu_seqlens_k[seq_idx]

    for kv_pos in range(0, kv_len, bn0):
        # Direct memory access -- K/V are contiguous per sequence
        K_tile = K_flat[kv_start + kv_pos : kv_start + kv_pos + bn0, kv_head_idx, :]
        V_tile = V_flat[kv_start + kv_pos : kv_start + kv_pos + bn0, kv_head_idx, :]

        S = Q_tile @ K_tile.T / sqrt(hdim)   # [kM0, bn0]

        # Mask out-of-range KV positions and causal violations
        for row in range(kM0):
            q_pos = q_tile_idx * kM0 + row
            for col in range(bn0):
                kv_tok = kv_pos + col
                if kv_tok >= kv_len or kv_tok > q_pos:  # causal
                    S[row, col] = -np.inf

        # Online softmax update
        for row in range(kM0):
            m_new = max(m[row], np.max(S[row]))
            correction = np.exp(m[row] - m_new)
            P = np.exp(S[row] - m_new)
            l[row] = l[row] * correction + np.sum(P)
            acc[row] = acc[row] * correction + P @ V_tile
            m[row] = m_new

    out[q_start + tok, q_head_idx] = acc / l`}
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
            <li>No page table, no block_table lookups — K/V accessed by direct offset</li>
            <li>Otherwise identical to CK-PK: same grid over Q heads, same tile sizes</li>
            <li>Used for prefill when KV is stored contiguously (not in paged cache)</li>
            <li>Rarely used in vLLM-style inference (which always pages KV)</li>
            <li>Source: <code>3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp</code></li>
            <li>Python entry: <code>aiter/ops/mha.py</code> → <code>mha_fwd()</code></li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default CKForwardApp;
