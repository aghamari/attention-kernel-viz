import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { usePagedAttentionStore } from '../store/pagedAttentionStore';
import ConfigDisplay from '../components/shared/ConfigDisplay';
import PseudoCode from '../components/shared/PseudoCode';
import ParameterGlossary from '../components/shared/ParameterGlossary';
import { createPagedGlossaryEntries } from '../data/glossaries/pagedGlossary';

interface PagedAttentionAppProps {
  onBack: () => void;
}

const PagedAttentionApp: React.FC<PagedAttentionAppProps> = ({ onBack }) => {
  const {
    config,
    blockTable,
    physicalBlocks,
    activeTab,
    activeSequence,
    setActiveTab,
    setActiveSequence,
    initializeBlocks
  } = usePagedAttentionStore();

  useEffect(() => {
    initializeBlocks();
  }, [initializeBlocks]);

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'pseudocode', label: 'Pseudocode' },
    { id: 'block-table', label: 'Block Table' },
    { id: 'kv-cache', label: 'KV Cache' },
    { id: 'v1-vs-v2', label: 'V1 vs V2' }
  ] as const;

  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} /> Back to Home
      </button>

      <header className="app-header">
        <h1>Paged Attention</h1>
        <p>Block-based KV cache management for efficient memory utilization</p>
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
            title="Example Configuration"
            params={[
              { label: 'Batch Size', value: config.batchSize },
              { label: 'Sequence Length', value: config.seqLen },
              { label: 'Num Heads', value: config.numHeads },
              { label: 'Head Dim', value: config.headDim },
              { label: 'Block Size', value: `${config.blockSize} tokens/block` },
              { label: 'Num Blocks', value: config.numBlocks },
              { label: 'Version', value: config.useV2 ? 'V2 (Multi Partition)' : 'V1 (Single Partition)' },
              { label: 'Num Partitions', value: config.useV2 ? config.numPartitions : 'N/A (V1)' },
            ]}
          />
          <ParameterGlossary
            entries={createPagedGlossaryEntries(config)}
            title="Paged Attention Parameter Reference"
            collapsible={true}
          />
        </div>

        <div className="center-panel">
          <div className={`tab-content ${activeTab === 'overview' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <h3 style={{ marginBottom: '20px' }}>Paged Attention Overview</h3>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h4>Key Concepts</h4>
                <ul>
                  <li><strong>Block Table:</strong> Maps logical KV blocks to physical memory locations</li>
                  <li><strong>Physical Blocks:</strong> Fixed-size memory chunks holding {config.blockSize} tokens each</li>
                  <li><strong>Non-Contiguous Storage:</strong> Sequences don't need contiguous memory</li>
                  <li><strong>Memory Sharing:</strong> Multiple sequences can reference same blocks (e.g., shared prefix)</li>
                </ul>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ marginBottom: '15px' }}>Memory Statistics</h4>
                <div className="metrics-dashboard">
                  <div className="metric-card">
                    <h4>Total Blocks</h4>
                    <span className="metric-value">{config.numBlocks}</span>
                  </div>
                  <div className="metric-card">
                    <h4>Used Blocks</h4>
                    <span className="metric-value">{physicalBlocks.filter(b => b.isUsed).length}</span>
                  </div>
                  <div className="metric-card">
                    <h4>Free Blocks</h4>
                    <span className="metric-value">{physicalBlocks.filter(b => !b.isUsed).length}</span>
                  </div>
                  <div className="metric-card">
                    <h4>Tokens/Block</h4>
                    <span className="metric-value">{config.blockSize}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'pseudocode' ? 'active' : ''}`}>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}
            >
              <h3>Kernel Pseudocode</h3>

              <PseudoCode
                title="PA Decode V1 (GQA path) -- Classic Paged Attention"
                subtitle="No head-merging. Separate code paths for MHA and GQA. Grid: (num_seqs, num_kv_heads, 1)"
                code={`# === GRID (V1, GQA path) ===
# (num_seqs, num_kv_heads, 1)  = (4, 2, 1) = 8 WGs
# Note: V1 MHA path uses (num_q_heads, num_seqs, 1) instead

def workgroup_pa_decode_v1_gqa(seq_idx, kv_head_idx):
    kv_len = seq_lens_k[seq_idx]

    # Load ALL Q heads for this GQA group
    # (different from head-merge: handled via dot product, not merged M)
    q_heads = []                                 # list of [hdim]
    for h in range(num_queries_per_kv):          # 0..3
        q_head = kv_head_idx * num_queries_per_kv + h
        q_heads.append(Q_flat[seq_idx, q_head])  # [64]

    # Accumulators per Q head
    acc = [np.zeros([hdim]) for _ in range(num_queries_per_kv)]
    m = [-np.inf] * num_queries_per_kv
    l = [0.0] * num_queries_per_kv

    for kv_pos in range(0, kv_len, BLOCK_KV):
        # Page lookup
        for t in range(BLOCK_KV):
            tok = kv_pos + t
            if tok < kv_len:
                phys_block = block_table[seq_idx][tok // block_size]
                offset = tok % block_size
                K[t] = k_cache[phys_block, kv_head_idx, offset, :]
                V[t] = v_cache[phys_block, kv_head_idx, offset, :]

        # Compute scores for EACH Q head separately
        for h in range(num_queries_per_kv):
            scores = q_heads[h] @ K.T / sqrt(hdim)  # [BLOCK_KV]
            # ... online softmax update for acc[h], m[h], l[h] ...

    for h in range(num_queries_per_kv):
        q_head = kv_head_idx * num_queries_per_kv + h
        out[seq_idx, q_head] = acc[h] / l[h]`}
              />

              <PseudoCode
                title="PA Decode V2 -- Partitioned (for long KV)"
                subtitle="Same split-KV pattern as CK-SK / Triton 3D. Each partition processes a KV slice, reduce kernel merges."
                code={`# === V2: PARTITIONED (for long KV) ===
# Attention grid: (num_seqs, num_kv_heads, max_num_partitions)
# Reduce grid:    (num_seqs, num_kv_heads, 1)
# Each partition processes a slice of KV, stores partial result,
# reduce kernel merges via log-sum-exp.

def workgroup_pa_v2_attention(seq_idx, kv_head_idx, partition_idx):
    kv_len = seq_lens_k[seq_idx]
    kv_per_part = ceil(kv_len / max_num_partitions)
    kv_start = partition_idx * kv_per_part
    kv_end   = min(kv_start + kv_per_part, kv_len)
    if kv_start >= kv_len:
        return

    # Same attention loop as V1, but only over this partition's KV range
    # Store partial: o_acc, lse_acc

def workgroup_pa_v2_reduce(seq_idx, kv_head_idx):
    # Merge all partitions via log-sum-exp (same as CK-SK combine)
    global_max = max(lse_acc[seq, head, :])
    for p in range(num_partitions):
        w = np.exp(lse_acc[seq, head, p] - global_max)
        output += o_acc[seq, head, p] * w
        total += w
    out[seq, head] = output / total`}
              />

              <PseudoCode
                title="PA Prefill -- Context Attention"
                subtitle="Prefill attention reading from paged KV cache + new tokens. Two-phase KV iteration."
                code={`# === GRID ===
# (batch, head, q_blocks)  where q_blocks = ceil(max_input_len / BLOCK)
# e.g. for prefill with 512 new tokens: (4, 8, 8) = 256 WGs
BLOCK = 64

def workgroup_pa_prefill(seq_idx, q_head_idx, q_block_idx):
    kv_head_idx = q_head_idx // num_queries_per_kv
    input_len = b_seq_len[seq_idx]       # new tokens in this prefill
    past_len  = b_ctx_len[seq_idx]       # existing cached tokens

    q_start = q_block_idx * BLOCK
    if q_start >= input_len:
        return

    # Load Q tile from NEW tokens (not from cache)
    Q_tile = Q_new[b_start_loc[seq_idx] + q_start : ..., q_head_idx]  # [BLOCK, hdim]

    acc = np.zeros([BLOCK, hdim])
    m = np.full([BLOCK], -np.inf)
    l = np.zeros([BLOCK])

    # Part 1: attend to CACHED KV tokens (paged)
    for kv_pos in range(0, past_len, BLOCK):
        K_tile = np.zeros([BLOCK, hdim])
        V_tile = np.zeros([BLOCK, hdim])
        for t in range(BLOCK):
            tok = kv_pos + t
            if tok < past_len:
                phys_block = block_table[seq_idx][tok // block_size]
                offset = tok % block_size
                K_tile[t] = k_cache[phys_block, offset, kv_head_idx, :]
                V_tile[t] = v_cache[phys_block, offset, kv_head_idx, :]
        S = Q_tile @ K_tile.T / sqrt(hdim)
        # ... mask, online softmax update ...

    # Part 2: attend to NEW K/V tokens (contiguous, being generated now)
    for kv_pos in range(0, input_len, BLOCK):
        K_tile = K_new[b_start_loc[seq_idx] + kv_pos : ..., kv_head_idx]
        V_tile = V_new[b_start_loc[seq_idx] + kv_pos : ..., kv_head_idx]
        S = Q_tile @ K_tile.T / sqrt(hdim)
        # Causal mask: new token at position past_len + q_pos can only
        # attend to tokens at positions <= past_len + q_pos
        # ... mask, online softmax update ...

    out[b_start_loc[seq_idx] + q_start : ..., q_head_idx] = acc / l`}
              />
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'block-table' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Block Table Visualization</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Mapping from logical sequence blocks to physical memory blocks
              </p>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ fontSize: '14px', color: '#666' }}>Select Sequence: </label>
                <select value={activeSequence} onChange={e => setActiveSequence(Number(e.target.value))} style={{ padding: '5px 10px', marginLeft: '10px' }}>
                  {Array.from({ length: config.batchSize }, (_, i) => (
                    <option key={i} value={i}>Sequence {i}</option>
                  ))}
                </select>
              </div>

              <div className="block-table">
                {Array.from({ length: config.batchSize }, (_, seq) => {
                  const blocks = blockTable.filter(b => b.seqIdx === seq);
                  return (
                    <div key={seq} className="block-table-row" style={{ opacity: seq === activeSequence ? 1 : 0.5 }}>
                      <div className="block-table-label">Seq {seq}:</div>
                      <div className="block-chain">
                        {blocks.map((block, idx) => (
                          <React.Fragment key={idx}>
                            <motion.div
                              className="block-item"
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              transition={{ delay: idx * 0.05 }}
                            >
                              B{block.physicalBlockIdx}
                            </motion.div>
                            {idx < blocks.length - 1 && <ArrowRight size={16} className="block-arrow" />}
                          </React.Fragment>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div style={{ marginTop: '30px' }}>
                <h4 style={{ marginBottom: '15px' }}>Physical Blocks</h4>
                <div className="physical-blocks">
                  {physicalBlocks.slice(0, 32).map((block, idx) => (
                    <motion.div
                      key={idx}
                      className={`physical-block ${block.isUsed ? 'used' : 'free'}`}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: idx * 0.02 }}
                    >
                      B{block.blockId}
                      {block.isUsed && <div style={{ fontSize: '9px' }}>S{block.seqIdx}</div>}
                    </motion.div>
                  ))}
                  {physicalBlocks.length > 32 && (
                    <div className="physical-block free">+{physicalBlocks.length - 32}</div>
                  )}
                </div>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'kv-cache' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>KV Cache Storage</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Each physical block stores K and V tensors for {config.blockSize} tokens
              </p>

              <div className="comparison-container">
                <div className="comparison-side left">
                  <h3>Block Structure</h3>
                  <div style={{ fontFamily: 'monospace', fontSize: '13px' }}>
                    <p>block_size = {config.blockSize} tokens</p>
                    <p>K: [{config.blockSize}, {config.numHeads}, {config.headDim}]</p>
                    <p>V: [{config.blockSize}, {config.numHeads}, {config.headDim}]</p>
                    <p>Total: {2 * config.blockSize * config.numHeads * config.headDim * 2} bytes/block</p>
                  </div>
                </div>
                <div className="comparison-side right">
                  <h3>Cache Layout</h3>
                  <div style={{ fontFamily: 'monospace', fontSize: '13px' }}>
                    <p>k_cache: [{config.numBlocks}, {config.blockSize}, {config.numHeads}, {config.headDim}]</p>
                    <p>v_cache: [{config.numBlocks}, {config.blockSize}, {config.numHeads}, {config.headDim}]</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'v1-vs-v2' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>V1 vs V2 Comparison</h3>

              <div className="comparison-container">
                <div className="comparison-side left" style={{ background: '#e3f2fd' }}>
                  <h3 style={{ borderColor: '#2196f3' }}>V1 - Single Partition</h3>
                  <ul style={{ listStyle: 'none', padding: 0 }}>
                    <li style={{ padding: '5px 0' }}>Single kernel processes all KV</li>
                    <li style={{ padding: '5px 0' }}>Best for short sequences (≤8K tokens)</li>
                    <li style={{ padding: '5px 0' }}>No reduction overhead</li>
                    <li style={{ padding: '5px 0' }}>Simpler implementation</li>
                  </ul>
                </div>
                <div className="comparison-side right" style={{ background: '#e8f5e9' }}>
                  <h3 style={{ borderColor: '#4caf50' }}>V2 - Multi Partition</h3>
                  <ul style={{ listStyle: 'none', padding: 0 }}>
                    <li style={{ padding: '5px 0' }}>KV split across {config.numPartitions} partitions</li>
                    <li style={{ padding: '5px 0' }}>Best for long sequences (&gt;8K tokens)</li>
                    <li style={{ padding: '5px 0' }}>Requires reduction kernel</li>
                    <li style={{ padding: '5px 0' }}>Higher parallelism</li>
                  </ul>
                </div>
              </div>

              <PseudoCode
                code={`# V2 Partition Reduction
for partition in range(num_partitions):
    partial_out[partition], partial_max[partition], partial_exp[partition] = \\
        attention_kernel(q, k_cache, v_cache, block_table, partition)

# Combine partitions
final_out = reduce_partitions(partial_out, partial_max, partial_exp)`}
              />
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PagedAttentionApp;
