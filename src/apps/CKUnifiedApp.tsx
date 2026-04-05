import React, { useEffect } from 'react';
import { ArrowLeft } from 'lucide-react';
import { useCKUnifiedStore } from '../store/ckUnifiedStore';
import InputPanel from '../components/shared/InputPanel';
import GridLaunchVisualizer from '../components/shared/GridLaunchVisualizer';
import ParameterGlossary from '../components/shared/ParameterGlossary';
import PseudoCode from '../components/shared/PseudoCode';
import { createCKUnifiedGlossaryEntries } from '../data/glossaries/ckUnifiedGlossary';
import { ATTENTION_COLORS } from '../types/attention';

interface CKUnifiedAppProps {
  onBack: () => void;
}

const CKUnifiedApp: React.FC<CKUnifiedAppProps> = ({ onBack }) => {
  const {
    config,
    gridLaunch,
    avgQ,
    shouldUse,
    setConfig,
    recalculate
  } = useCKUnifiedStore();

  useEffect(() => {
    recalculate();
  }, [recalculate]);

  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} />
        Back to Home
      </button>

      <header className="app-header">
        <h1 style={{
          background: ATTENTION_COLORS['ck-ua'].gradient,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          CK-UA (CK Tile Unified Attention)
        </h1>
        <p>Single-kernel paged-KV attention with GQA head-merging. All Q heads in a GQA group
          are packed into the M dimension so one MFMA computes scores for all heads at once.</p>
      </header>

      <div className="overview-layout">
        <div className="left-panel">
          <InputPanel
            title="Configuration"
            sliders={[
              {
                label: 'Batch Size (num_seqs)',
                value: config.batchSize,
                min: 1,
                max: 512,
                onChange: (v) => setConfig({ batchSize: v })
              },
              {
                label: 'Sequence Length',
                value: config.seqLen,
                min: 1,
                max: 128,
                onChange: (v) => setConfig({ seqLen: v })
              },
              {
                label: 'Num Query Heads',
                value: config.numHeads,
                min: 8,
                max: 64,
                step: 8,
                onChange: (v) => setConfig({ numHeads: v })
              },
              {
                label: 'Num KV Heads',
                value: config.numKvHeads,
                min: 1,
                max: 16,
                onChange: (v) => setConfig({ numKvHeads: v })
              },
              {
                label: 'Head Dim',
                value: config.headDim,
                min: 64,
                max: 128,
                step: 64,
                onChange: (v) => setConfig({ headDim: v })
              }
            ]}
            selects={[
              {
                label: 'Page Block Size',
                value: config.pageBlockSize.toString(),
                options: [
                  { value: '32', label: '32' },
                  { value: '64', label: '64' }
                ],
                onChange: (v) => setConfig({ pageBlockSize: parseInt(String(v)) as 32 | 64 })
              },
              {
                label: 'Mask Type',
                value: config.maskType.toString(),
                options: [
                  { value: '0', label: 'No Mask' },
                  { value: '2', label: 'Causal' }
                ],
                onChange: (v) => setConfig({ maskType: parseInt(String(v)) as 0 | 2 })
              }
            ]}
          />
          <ParameterGlossary entries={createCKUnifiedGlossaryEntries(config)} />
        </div>

        <div className="center-panel">
          {/* Selector Logic */}
          <div>
            <PseudoCode
              title="Selector Logic"
              code={`def should_use_ck_ua():
    if max_seqlen_q != 1: return False          # decode only
    if window_size != (-1, -1): return False     # no sliding window
    if softcap != 0: return False                # no softcap
    if alibi_slopes is not None: return False    # no ALiBi
    if sinks is not None: return False           # no sinks

    # Only compiled for specific GQA configs
    if not ((head_size == 64 and num_queries_per_kv == 8) or
            (head_size == 128 and num_queries_per_kv == 1)):
        return False

    # CK-UA wins in "moderate occupancy" zone
    cu_count = get_num_sms()                      # 256 on MI300X
    triton_2d_wgs = num_kv_heads * num_seqs
    if not (cu_count * 4 <= triton_2d_wgs <= cu_count * 8):
        return False
    # On MI300X with 8 KV-heads: CK-UA activates for 128-256 seqs
    return True`}
            />
            <div style={{
              marginTop: '10px',
              padding: '10px 15px',
              background: shouldUse ? '#d4edda' : '#f8d7da',
              borderRadius: '6px',
              fontSize: '13px'
            }}>
              <strong>Current Config:</strong> {shouldUse ? '✓ CK-UA should be used' : '✗ CK-UA not optimal'}
              <br />
              avg_q = {avgQ.toFixed(2)}, tier = {config.tier}, kBlockM = {config.kBlockM},
              kBlockQ = {config.kBlockQ}, numQueriesPerKv = {config.numQueriesPerKv}
            </div>
          </div>

          {/* GQA Head-Merge */}
          <div style={{ padding: '20px', background: '#f8f9fa', borderRadius: '12px' }}>
            <h3 style={{ marginBottom: '15px' }}>GQA Head-Merge Optimization</h3>

            <div style={{ color: '#444', lineHeight: '1.7', fontSize: '14px', marginBottom: '15px' }}>
              <p style={{ marginBottom: '10px' }}>
                <strong><code>num_queries_per_kv</code> = {config.numQueriesPerKv}</strong> (GQA ratio):
                {config.numHeads} Q heads / {config.numKvHeads} KV heads = {config.numQueriesPerKv} Q heads share each KV head.
                Heads 0-{config.numQueriesPerKv - 1} all read from KV head 0,
                heads {config.numQueriesPerKv}-{config.numQueriesPerKv * 2 - 1} from KV head 1, etc.
              </p>
              <p style={{ marginBottom: '10px' }}>
                <strong><code>kBlockM</code> = {config.kBlockM}</strong> (Q tile height):
                total rows in the Q tile after merging. Computed as
                <code> kBlockQ × num_queries_per_kv = {config.kBlockQ} × {config.numQueriesPerKv} = {config.kBlockM}</code>.
              </p>
              <p>
                <strong><code>kBlockQ</code> = {config.kBlockQ}</strong> (token slots):
                how many Q <em>tokens</em> fit per tile. The remaining rows are filled by packing
                different Q heads for the same token, so one MFMA computes all heads at once.
              </p>
            </div>

            <div style={{
              padding: '15px',
              background: '#f0f7ff',
              borderRadius: '8px',
              border: '1px solid #b3d9ff',
              fontFamily: 'monospace',
              fontSize: '13px',
              lineHeight: '1.8',
              marginBottom: '15px'
            }}>
              Q tile layout (kBlockM={config.kBlockM} rows, decode with 1 token):<br/>
              {Array.from({ length: Math.min(config.numQueriesPerKv, 6) }, (_, h) =>
                <span key={h}>&nbsp;&nbsp;row {h}: token 0, head {h}<br/></span>
              )}
              {config.numQueriesPerKv > 6 && <span>&nbsp;&nbsp;...{config.numQueriesPerKv - 6} more heads...<br/></span>}
              &nbsp;&nbsp;rows {config.numQueriesPerKv}-{config.kBlockM - 1}: padding (unused for decode)<br/>
              <br/>
              S = Q_tile @ K_tile.T → [{config.kBlockM}, {config.headDim}] @ [{config.headDim}, {config.pageBlockSize}] = [{config.kBlockM}, {config.pageBlockSize}]<br/>
              &nbsp;&nbsp;One MFMA computes scores for ALL {config.numQueriesPerKv} heads simultaneously!
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
              <div style={{ padding: '15px', background: 'rgba(255, 87, 34, 0.08)', borderRadius: '8px' }}>
                <strong>Without head-merge (e.g. CK-PK):</strong>
                <pre style={{ fontSize: '12px', marginTop: '8px' }}>
{`Each Q-head gets its own workgroup:
for head in range(${config.numHeads}):
    S = Q[head] @ K[kv_head].T
    # [1, ${config.headDim}] @ [${config.headDim}, N] -- tiny GEMM!
# ${config.numHeads} WGs/seq × ${config.batchSize} seqs = ${config.numHeads * config.batchSize} WGs`}
                </pre>
              </div>
              <div style={{ padding: '15px', background: 'rgba(76, 175, 80, 0.08)', borderRadius: '8px' }}>
                <strong>With head-merge (CK-UA):</strong>
                <pre style={{ fontSize: '12px', marginTop: '8px' }}>
{`${config.numQueriesPerKv} Q-heads packed into M dimension:
Q_tile = [q_h0, q_h1, ..., q_h${config.numQueriesPerKv - 1}]  # ${config.numQueriesPerKv} rows
S = MFMA(Q_tile @ K.T)
# [${config.numQueriesPerKv}, ${config.headDim}] @ [${config.headDim}, N] -- ${config.numQueriesPerKv}× bigger!
# ${config.numKvHeads} WGs/seq × ${config.batchSize} seqs = ${config.numKvHeads * config.batchSize} WGs`}
                </pre>
              </div>
            </div>
          </div>

          {/* Tile Tiers */}
          <div style={{ padding: '20px', background: '#f8f9fa', borderRadius: '12px' }}>
            <h3 style={{ marginBottom: '15px' }}>Tile Tier Selection</h3>
            <p style={{ color: '#666', marginBottom: '15px' }}>
              Automatic tile size selection based on avg_q = total_q_tokens / num_seqs
            </p>
            <div style={{ display: 'grid', gap: '10px' }}>
              {[
                { tier: 'tiny', kBlockM: 16, warps: 1, threads: 64, mfma: '16x16x32', condition: 'avg_q ≤ 2' },
                { tier: 'bs32', kBlockM: 32, warps: 2, threads: 128, mfma: '16x16x32', condition: 'avg_q ≤ 4, block_size=32' },
                { tier: 'small', kBlockM: 64, warps: 2, threads: 128, mfma: '32x32x16', condition: 'avg_q ≤ 8' },
                { tier: 'medium', kBlockM: 128, warps: 4, threads: 256, mfma: '32x32x16', condition: 'max_seqlen_q ≤ 128' },
                { tier: 'large', kBlockM: 256, warps: 8, threads: 512, mfma: '32x32x16', condition: 'else' }
              ].map((t) => (
                <div
                  key={t.tier}
                  style={{
                    padding: '12px 16px',
                    background: config.tier === t.tier ? 'rgba(255, 87, 34, 0.12)' : 'white',
                    borderRadius: '8px',
                    border: config.tier === t.tier ? '2px solid #FF5722' : '1px solid #e0e0e0',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}
                >
                  <div>
                    <strong style={{ textTransform: 'uppercase' }}>{t.tier}</strong>
                    {config.tier === t.tier && <span style={{ color: '#FF5722', marginLeft: '8px' }}>● ACTIVE</span>}
                  </div>
                  <div style={{ fontSize: '13px', color: '#666' }}>
                    kBlockM={t.kBlockM} | {t.warps} warps/{t.threads} threads | MFMA {t.mfma} | {t.condition}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Grid Launch */}
          <GridLaunchVisualizer
            gridInfo={gridLaunch}
            is3D={false}
            title={config.isDecodeGrid
              ? `2D Decode Grid: dim3(${config.numKvHeads} kv_heads, ${config.batchSize} seqs)`
              : `1D Prefill Grid: dim3(${config.numKvHeads} kv_heads × total_q_blocks)`
            }
            colorScheme={ATTENTION_COLORS['ck-ua']}
          />

          {/* Pseudocode */}
          <PseudoCode
            title="Workgroup Pseudocode"
            subtitle={`One workgroup per (KV head, sequence). Config: kBlockM=${config.kBlockM}, kBlockQ=${config.kBlockQ}, kPageBlockSize=${config.pageBlockSize}`}
            code={`# === CONFIG (${config.tier} tier) ===
# nhead_q = ${config.numHeads}, nhead_kv = ${config.numKvHeads}, hdim = ${config.headDim}
# num_queries_per_kv = ${config.numHeads} // ${config.numKvHeads} = ${config.numQueriesPerKv}  (GQA-${config.numQueriesPerKv})
#
kBlockM = ${config.kBlockM}         # Q tile height (after merging heads into M)
kBlockQ = kBlockM // num_queries_per_kv  # = ${config.kBlockM} // ${config.numQueriesPerKv} = ${config.kBlockQ} token slots
kPageBlockSize = ${config.pageBlockSize}  # KV tile = one full page
#
# Q tile layout (kBlockM=${config.kBlockM} rows, decode with 1 token):
#   row  0: token 0, head 0    ← ${config.numQueriesPerKv} heads packed for token 0
#   row  1: token 0, head 1
${config.numQueriesPerKv > 2 ? `#   ...
#   row  ${config.numQueriesPerKv - 1}: token 0, head ${config.numQueriesPerKv - 1}` : `#   (only ${config.numQueriesPerKv} heads)`}
#   rows ${config.numQueriesPerKv}-${config.kBlockM - 1}: padding (zero rows, unused for decode)
#
# With prefill (e.g. 4 tokens), all ${config.kBlockM} rows would be used:
#   rows 0-${config.numQueriesPerKv - 1}: token 0, heads 0-${config.numQueriesPerKv - 1}
#   rows ${config.numQueriesPerKv}-${config.numQueriesPerKv * 2 - 1}: token 1, heads 0-${config.numQueriesPerKv - 1}
#   ...up to ${config.kBlockQ} tokens × ${config.numQueriesPerKv} heads = ${config.kBlockM} rows

# === GRID (decode) ===
# dim3(num_kv_heads, num_seqs) = dim3(${config.numKvHeads}, ${config.batchSize}) = ${config.numKvHeads * config.batchSize} workgroups
# One workgroup per (KV head, sequence)
# Compare: without head-merge would need ${config.numHeads} × ${config.batchSize} = ${config.numHeads * config.batchSize} WGs

# === ONE WORKGROUP: kv_head_idx=0, seq_idx=1 ===

def workgroup_ck_ua(kv_head_idx, seq_idx):
    kv_len = seq_lens_k[seq_idx]              # e.g. 200 tokens in KV cache
    q_start = cu_seqlens_q[seq_idx]           # where seq's Q tokens start in flat array
    q_len = cu_seqlens_q[seq_idx + 1] - q_start  # = 1 for decode

    # Q_flat is 3D: [total_tokens, nhead_q, hdim]
    # All seqs' tokens concatenated in dim 0, cu_seqlens_q marks boundaries.
    #
    # Decode example (4 seqs, 1 token each):
    #   Q_flat shape = [4, ${config.numHeads}, ${config.headDim}]
    #   cu_seqlens_q = [0, 1, 2, 3, 4]
    #   Q_flat[0, :, :] = seq 0's Q (${config.numHeads} heads × ${config.headDim}-dim)
    #   Q_flat[1, :, :] = seq 1's Q
    #   Q_flat[2, :, :] = seq 2's Q
    #   Q_flat[3, :, :] = seq 3's Q
    #
    # Q_flat[token_idx, head_idx, :] → one vector of [${config.headDim}]
    # Below we write Q_flat[token_idx, head_idx] as shorthand (the ":" is implicit)

    # === HEAD-MERGE: pack all Q heads for this KV group into M ===
    #
    # GQA-${config.numQueriesPerKv} mapping (${config.numHeads} Q heads, ${config.numKvHeads} KV heads):
${Array.from({ length: config.numKvHeads }, (_, i) => {
  const qStart = i * config.numQueriesPerKv;
  const qEnd = qStart + config.numQueriesPerKv - 1;
  return `    #   kv_head ${i} → Q heads ${qStart}-${qEnd}`;
}).join('\n')}
    #
    # This workgroup handles kv_head_idx, so it packs Q heads
    # ${`kv_head_idx * ${config.numQueriesPerKv}`} through ${`kv_head_idx * ${config.numQueriesPerKv} + ${config.numQueriesPerKv - 1}`} into the tile rows.
    Q_tile = np.zeros([kBlockM, hdim])          # [${config.kBlockM}, ${config.headDim}]
    for tok in range(q_len):                    # 0..0 for decode
        for h in range(num_queries_per_kv):     # 0..${config.numQueriesPerKv - 1}
            row = tok * num_queries_per_kv + h
            #   tok=0, h=0 → row=0    tok=0, h=1 → row=1    ...
            q_head = kv_head_idx * num_queries_per_kv + h
            #   e.g. kv_head_idx=3: q_heads ${3 * config.numQueriesPerKv},${3 * config.numQueriesPerKv + 1},..,${3 * config.numQueriesPerKv + config.numQueriesPerKv - 1}
            Q_tile[row] = Q_flat[q_start + tok, q_head]

    num_pages = ceil(kv_len / kPageBlockSize)   # ceil(200/${config.pageBlockSize}) = ${Math.ceil(200 / config.pageBlockSize)} pages

    # block_table maps (sequence, page_index) → physical page ID.
    # Pages are scattered in GPU memory (non-contiguous).
    #
    # kv_cache = [num_pages, page_size=${config.pageBlockSize}, nhead_kv, hdim]  -- physical pool
    # seq_lens_k = [100, 200, 80, 50]
    #
    # block_table = [
    #     [5, 12],       # seq 0: 100 tok → page 5 (tok 0-${config.pageBlockSize - 1}), page 12 (tok ${config.pageBlockSize}-99)
    #     [3, 7, 0, 9],  # seq 1: 200 tok → pages 3,7,0,9
    #     [10, 6],        # seq 2: 80 tok  → page 10 (tok 0-${config.pageBlockSize - 1}), page 6 (tok ${config.pageBlockSize}-79)
    #     [2],            # seq 3: 50 tok  → page 2 (tok 0-49)
    # ]
    #
    # Physical page IDs (5,12,3,7,0,9,...) are non-contiguous and
    # out of order -- allocated from a GPU memory free pool.
    #
    # To read token 130 of seq 1:
    #   page_i   = 130 // ${config.pageBlockSize} = ${Math.floor(130 / config.pageBlockSize)}  → block_table[1][${Math.floor(130 / config.pageBlockSize)}] = ${[3, 7, 0, 9][Math.floor(130 / config.pageBlockSize)]}
    #   slot     = 130 % ${config.pageBlockSize}  = ${130 % config.pageBlockSize}
    #   data     = kv_cache[${[3, 7, 0, 9][Math.floor(130 / config.pageBlockSize)]}, ${130 % config.pageBlockSize}, kv_head, :]

    acc = np.zeros([kBlockM, hdim])     # [${config.kBlockM}, ${config.headDim}] weighted output
    m = np.full([kBlockM], -np.inf)     # [${config.kBlockM}] running max score per row
    l = np.zeros([kBlockM])             # [${config.kBlockM}] running sum of exp(scores)

    # Loop over KV one page at a time
    for page_i in range(num_pages):       # 0, 1, 2, 3 for 200 tokens
        phys_page = block_table[seq_idx][page_i]
        # e.g. page_i=2 → block_table[1][2] = 0 → physical page 0

        # Load one full page of K and V (shared by ALL ${config.numQueriesPerKv} heads)
        # CK-UA loads the WHOLE page at once (vs CK-PK which does per-token lookup)
        K_tile = kv_cache[phys_page, :, kv_head_idx, :]  # [${config.pageBlockSize}, ${config.headDim}]
        V_tile = kv_cache[phys_page, :, kv_head_idx, :]  # [${config.pageBlockSize}, ${config.headDim}]

        # MFMA: scores for ALL ${config.numQueriesPerKv} heads at once
        S = Q_tile @ K_tile.T / sqrt(hdim)
        # [${config.kBlockM}, ${config.headDim}] @ [${config.headDim}, ${config.pageBlockSize}] = [${config.kBlockM}, ${config.pageBlockSize}]
        #   S[0,:] = head 0's scores against ${config.pageBlockSize} KV tokens
        #   S[1,:] = head 1's scores (different Q, same K)
        #   S[${config.numQueriesPerKv}-${config.kBlockM - 1},:] = padding rows (zeros × K = zeros)

        # Causal mask: mask out KV positions beyond sequence length
        for row in range(kBlockM):
            tok = row // num_queries_per_kv
            for col in range(kPageBlockSize):
                kv_tok = page_i * kPageBlockSize + col
                if kv_tok >= kv_len:
                    S[row, col] = -np.inf

        # Online softmax update (each row = one head independently)
        for row in range(kBlockM):
            m_new = max(m[row], np.max(S[row]))
            correction = np.exp(m[row] - m_new)   # rescale old accumulators
            P = np.exp(S[row] - m_new)             # softmax numerators for this tile
            l[row] = l[row] * correction + np.sum(P)
            acc[row] = acc[row] * correction + P @ V_tile
            m[row] = m_new

    # Un-merge: write each row back to its Q head's output position
    for tok in range(q_len):
        for h in range(num_queries_per_kv):
            row = tok * num_queries_per_kv + h
            q_head = kv_head_idx * num_queries_per_kv + h
            out[q_start + tok, q_head] = acc[row] / l[row]  # normalize by exp-sum`}
          />

          {/* Key Points */}
          <div style={{ padding: '20px', background: '#f8f9fa', borderRadius: '12px' }}>
            <h3 style={{ marginBottom: '15px' }}>Key Points</h3>
            <ul style={{ lineHeight: '1.8', color: '#444', paddingLeft: '20px' }}>
              <li>Grid iterates over <strong>KV heads</strong> (nhead_kv={config.numKvHeads}), not Q heads ({config.numHeads})</li>
              <li>Head-merge: {config.numQueriesPerKv} Q heads packed into M → one MFMA computes all {config.numQueriesPerKv} heads' scores</li>
              <li>Only {config.numKvHeads * config.batchSize} WGs for decode (vs {config.numHeads * config.batchSize} without head-merge)</li>
              <li>K/V loaded <strong>once</strong> per page, shared across all {config.numQueriesPerKv} heads in the group</li>
              <li>For decode with GQA-{config.numQueriesPerKv}: {config.numQueriesPerKv} useful rows out of {config.kBlockM}
                ({((config.numQueriesPerKv / config.kBlockM) * 100).toFixed(1)}% utilization)</li>
              <li>Source: <code>3rdparty/composable_kernel/example/ck_tile/42_unified_attention/</code></li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CKUnifiedApp;
