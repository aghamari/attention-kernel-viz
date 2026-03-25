import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Play } from 'lucide-react';
import { useUnifiedAttentionStore } from '../store/unifiedAttentionStore';
import InputPanel from '../components/shared/InputPanel';
import MetricsDashboard from '../components/shared/MetricsDashboard';
import AttentionPipelineFlow from '../components/unified/AttentionPipelineFlow';
import AttentionStepByStep from '../components/unified/AttentionStepByStep';
import AttentionMemoryLayout from '../components/unified/AttentionMemoryLayout';
import Unified2DVisualization from '../components/unified/Unified2DVisualization';
import Unified3DVisualization from '../components/unified/Unified3DVisualization';
import GridLaunchVisualizer from '../components/shared/GridLaunchVisualizer';
import ParameterGlossary from '../components/shared/ParameterGlossary';
import { createUnifiedGlossaryEntries } from '../data/glossaries/unifiedGlossary';

interface UnifiedAttentionAppProps {
  onBack: () => void;
}

const UnifiedAttentionApp: React.FC<UnifiedAttentionAppProps> = ({ onBack }) => {
  const {
    config,
    gridLaunch,
    performance,
    currentPhase,
    activeTab,
    highlightedCell,
    setConfig,
    setActiveTab,
    setHighlightedCell,
    toggle3DKernel,
    runSimulation,
    initializeData
  } = useUnifiedAttentionStore();

  useEffect(() => {
    initializeData();
  }, [initializeData]);

  const tabs = [
    { id: 'pipeline', label: 'Pipeline' },
    { id: 'algorithm', label: 'Algorithm' },
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
          <strong>Function Signature:</strong>
          <pre style={{ marginTop: '8px', marginBottom: '0', whiteSpace: 'pre-wrap' }}>
{`def unified_attention(
    q: Tensor,              # [batch, seq_len, num_heads, head_dim]
    k: Tensor,              # [batch, seq_len, num_kv_heads, head_dim]
    v: Tensor,              # [batch, seq_len, num_kv_heads, head_dim]
    causal: bool = True,
    sliding_window: Optional[int] = None,
    use_3d_kernel: bool = None,  # Auto-selected if None
    block_m: int = 64,
    block_n: int = 64,
    num_segments: int = 4
) -> Tensor:                # [batch, seq_len, num_heads, head_dim]
    """
    Adaptive 2D/3D attention with automatic kernel selection.
    Returns attended output combining information across sequence.
    """`}</pre>
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
          <InputPanel
            title="Configuration"
            sliders={[
              {
                label: 'Sequence Length',
                value: config.seqLen,
                min: 64,
                max: 8192,
                step: 64,
                onChange: (v) => setConfig({ seqLen: v })
              },
              {
                label: 'Num Heads',
                value: config.numHeads,
                min: 1,
                max: 32,
                onChange: (v) => setConfig({ numHeads: v })
              },
              {
                label: 'Head Dim',
                value: config.headDim,
                min: 32,
                max: 128,
                step: 16,
                onChange: (v) => setConfig({ headDim: v })
              },
              {
                label: 'KV Heads',
                value: config.numKvHeads,
                min: 1,
                max: config.numHeads,
                onChange: (v) => setConfig({ numKvHeads: v })
              },
              {
                label: 'Block M',
                value: config.blockM,
                min: 16,
                max: 128,
                step: 16,
                onChange: (v) => setConfig({ blockM: v })
              },
              {
                label: 'Num Segments',
                value: config.numSegments,
                min: 1,
                max: 16,
                onChange: (v) => setConfig({ numSegments: v })
              }
            ]}
            selects={[
              {
                label: 'Causal Mask',
                value: config.causal,
                options: [
                  { value: true, label: 'Enabled' },
                  { value: false, label: 'Disabled' }
                ],
                onChange: (v) => setConfig({ causal: v as boolean })
              },
              {
                label: 'Sliding Window',
                value: config.useSlidingWindow,
                options: [
                  { value: false, label: 'Disabled' },
                  { value: true, label: 'Enabled' }
                ],
                onChange: (v) => setConfig({ useSlidingWindow: v as boolean })
              }
            ]}
          >
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

            <button className="run-simulation-btn" onClick={() => {
              runSimulation();
              setActiveTab('algorithm');
            }}>
              <Play size={16} />
              Run Simulation
            </button>

            {currentPhase !== 'idle' && (
              <div style={{
                marginTop: '15px',
                padding: '15px',
                background: currentPhase === 'complete' ? '#e8f5e9' : '#fff3e0',
                borderRadius: '8px',
                border: `2px solid ${currentPhase === 'complete' ? '#4caf50' : '#ff9800'}`
              }}>
                <div style={{ fontSize: '14px', fontWeight: 'bold', color: currentPhase === 'complete' ? '#2e7d32' : '#e65100' }}>
                  Phase: {currentPhase.toUpperCase()}
                </div>
                <div style={{ marginTop: '10px', display: 'flex', gap: '5px' }}>
                  {(config.use3DKernel ? ['qk', 'softmax', 'av', 'reduction', 'complete'] : ['qk', 'softmax', 'av', 'complete']).map((phase, idx) => (
                    <div
                      key={phase}
                      style={{
                        flex: 1,
                        height: '8px',
                        borderRadius: '4px',
                        background: currentPhase === phase || (currentPhase === 'complete' && phase !== 'complete')
                          ? '#4caf50'
                          : (idx < (config.use3DKernel ? ['qk', 'softmax', 'av', 'reduction', 'complete'] : ['qk', 'softmax', 'av', 'complete']).indexOf(currentPhase) ? '#4caf50' : '#e0e0e0')
                      }}
                    />
                  ))}
                </div>
              </div>
            )}
          </InputPanel>
        </div>

        <div className="center-panel">
          <div className={`tab-content ${activeTab === 'pipeline' ? 'active' : ''}`}>
            <AttentionPipelineFlow />
          </div>

          <div className={`tab-content ${activeTab === 'algorithm' ? 'active' : ''}`}>
            <AttentionStepByStep />
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
              <MetricsDashboard
                metrics={performance}
                additionalMetrics={[
                  { label: 'Q Blocks', value: Math.ceil(config.seqLen / config.blockM) },
                  { label: 'K Blocks', value: Math.ceil(config.seqLen / config.blockN) },
                  { label: 'Segments', value: config.use3DKernel ? config.numSegments : 1 },
                  { label: 'Total Thread Blocks', value: gridLaunch.gridX * gridLaunch.gridY * gridLaunch.gridZ }
                ]}
              />
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
