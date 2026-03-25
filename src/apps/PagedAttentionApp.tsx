import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Play, ArrowRight } from 'lucide-react';
import { usePagedAttentionStore } from '../store/pagedAttentionStore';
import InputPanel from '../components/shared/InputPanel';
import MetricsDashboard from '../components/shared/MetricsDashboard';
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
    performance,
    currentPhase,
    activeTab,
    activeSequence,
    setConfig,
    setActiveTab,
    setActiveSequence,
    runSimulation,
    initializeBlocks
  } = usePagedAttentionStore();

  useEffect(() => {
    initializeBlocks();
  }, [initializeBlocks]);

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'block-table', label: 'Block Table' },
    { id: 'kv-cache', label: 'KV Cache' },
    { id: 'v1-vs-v2', label: 'V1 vs V2' }
  ] as const;

  const seqBlocks = blockTable.filter(b => b.seqIdx === activeSequence);

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
          <InputPanel
            title="Configuration"
            sliders={[
              { label: 'Batch Size', value: config.batchSize, min: 1, max: 16, onChange: v => setConfig({ batchSize: v }) },
              { label: 'Sequence Length', value: config.seqLen, min: 64, max: 2048, step: 64, onChange: v => setConfig({ seqLen: v }) },
              { label: 'Block Size', value: config.blockSize, min: 8, max: 64, step: 8, onChange: v => setConfig({ blockSize: v }) },
              { label: 'Num Blocks', value: config.numBlocks, min: 16, max: 128, step: 8, onChange: v => setConfig({ numBlocks: v }) },
              { label: 'Num Partitions', value: config.numPartitions, min: 1, max: 16, onChange: v => setConfig({ numPartitions: v }) }
            ]}
            selects={[
              { label: 'Version', value: config.useV2 ? 'v2' : 'v1', options: [{ value: 'v1', label: 'V1 (Single Partition)' }, { value: 'v2', label: 'V2 (Multi Partition)' }], onChange: v => setConfig({ useV2: v === 'v2' }) }
            ]}
          >
            <button className="run-simulation-btn" onClick={runSimulation}>
              <Play size={16} /> Run Simulation
            </button>
          </InputPanel>
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

              <MetricsDashboard metrics={performance} />
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
                <div className="comparison-side left" style={{ background: !config.useV2 ? '#e3f2fd' : '#f8f9fa' }}>
                  <h3 style={{ borderColor: !config.useV2 ? '#2196f3' : '#ccc' }}>V1 - Single Partition {!config.useV2 ? '(Active)' : ''}</h3>
                  <ul style={{ listStyle: 'none', padding: 0 }}>
                    <li style={{ padding: '5px 0' }}>Single kernel processes all KV</li>
                    <li style={{ padding: '5px 0' }}>Best for short sequences (≤8K tokens)</li>
                    <li style={{ padding: '5px 0' }}>No reduction overhead</li>
                    <li style={{ padding: '5px 0' }}>Simpler implementation</li>
                  </ul>
                </div>
                <div className="comparison-side right" style={{ background: config.useV2 ? '#e8f5e9' : '#f8f9fa' }}>
                  <h3 style={{ borderColor: config.useV2 ? '#4caf50' : '#ccc' }}>V2 - Multi Partition {config.useV2 ? '(Active)' : ''}</h3>
                  <ul style={{ listStyle: 'none', padding: 0 }}>
                    <li style={{ padding: '5px 0' }}>KV split across {config.numPartitions} partitions</li>
                    <li style={{ padding: '5px 0' }}>Best for long sequences (&gt;8K tokens)</li>
                    <li style={{ padding: '5px 0' }}>Requires reduction kernel</li>
                    <li style={{ padding: '5px 0' }}>Higher parallelism</li>
                  </ul>
                </div>
              </div>

              <div className="code-block" style={{ marginTop: '20px' }}>
                <pre>{`# V2 Partition Reduction
for partition in range(num_partitions):
    partial_out[partition], partial_max[partition], partial_exp[partition] = \\
        attention_kernel(q, k_cache, v_cache, block_table, partition)

# Combine partitions
final_out = reduce_partitions(partial_out, partial_max, partial_exp)`}</pre>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      <ParameterGlossary
        entries={createPagedGlossaryEntries(config)}
        title="Paged Attention Parameter Reference"
        collapsible={true}
      />
    </div>
  );
};

export default PagedAttentionApp;
