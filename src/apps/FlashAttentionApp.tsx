import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Play } from 'lucide-react';
import { useFlashAttentionStore } from '../store/flashAttentionStore';
import InputPanel from '../components/shared/InputPanel';
import MetricsDashboard from '../components/shared/MetricsDashboard';
import ParameterGlossary from '../components/shared/ParameterGlossary';
import { createFlashGlossaryEntries } from '../data/glossaries/flashGlossary';

interface FlashAttentionAppProps {
  onBack: () => void;
}

const FlashAttentionApp: React.FC<FlashAttentionAppProps> = ({ onBack }) => {
  const {
    config,
    performance,
    currentPhase,
    activeTab,
    onlineSoftmax,
    setConfig,
    setActiveTab,
    runSimulation,
    stepOnlineSoftmax,
    resetOnlineSoftmax
  } = useFlashAttentionStore();

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'online-softmax', label: 'Online Softmax' },
    { id: 'causal-mask', label: 'Causal Mask' },
    { id: 'fp8', label: 'FP8 Support' }
  ] as const;

  const numBlocks = Math.ceil(config.seqLen / 64);

  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} />
        Back to Home
      </button>

      <header className="app-header">
        <h1>Flash Attention (MHA)</h1>
        <p>Memory-efficient attention with online softmax and tiled computation</p>
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
              { label: 'Sequence Length', value: config.seqLen, min: 64, max: 4096, step: 64, onChange: v => setConfig({ seqLen: v }) },
              { label: 'Num Heads', value: config.numHeads, min: 1, max: 32, onChange: v => setConfig({ numHeads: v }) },
              { label: 'Head Dim', value: config.headDim, min: 32, max: 128, step: 16, onChange: v => setConfig({ headDim: v }) },
              { label: 'Dropout', value: config.dropout, min: 0, max: 0.5, step: 0.05, onChange: v => setConfig({ dropout: v }) },
              { label: 'Softmax Scale', value: config.softmaxScale, min: 0.05, max: 0.5, step: 0.01, onChange: v => setConfig({ softmaxScale: v }) }
            ]}
            selects={[
              { label: 'Causal Mask', value: config.useCausalMask, options: [{ value: true, label: 'Enabled' }, { value: false, label: 'Disabled' }], onChange: v => setConfig({ useCausalMask: v as boolean }) },
              { label: 'ALiBi', value: config.useAlibi, options: [{ value: false, label: 'Disabled' }, { value: true, label: 'Enabled' }], onChange: v => setConfig({ useAlibi: v as boolean }) },
              { label: 'FP8', value: config.useFP8, options: [{ value: false, label: 'Disabled' }, { value: true, label: 'Enabled' }], onChange: v => setConfig({ useFP8: v as boolean }) }
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
              <h3 style={{ marginBottom: '20px' }}>Flash Attention Overview</h3>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h4>Key Features</h4>
                <ul>
                  <li><strong>Online Softmax:</strong> Compute softmax incrementally without materializing full attention matrix</li>
                  <li><strong>Tiled Computation:</strong> Process Q, K, V in blocks for memory efficiency</li>
                  <li><strong>IO-Aware:</strong> Minimize HBM reads/writes through register reuse</li>
                  <li><strong>Backward Pass:</strong> Recompute attention during backward for memory savings</li>
                </ul>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ marginBottom: '15px' }}>Computation Pipeline</h4>
                {['load-q', 'load-kv', 'qk-matmul', 'softmax', 'av-matmul', 'store'].map((phase, idx) => (
                  <div key={phase} className="flow-step">
                    <div className="flow-step-number" style={{
                      background: currentPhase === phase ? 'linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%)' : undefined
                    }}>
                      {idx + 1}
                    </div>
                    <div className="flow-step-content">
                      <h4 style={{ color: currentPhase === phase ? '#ff6b6b' : '#333' }}>
                        {phase.replace('-', ' ').toUpperCase()}
                      </h4>
                    </div>
                  </div>
                ))}
              </div>

              <MetricsDashboard metrics={performance} />
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'online-softmax' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Online Softmax Algorithm</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Compute softmax incrementally by tracking running max and sum
              </p>

              <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
                <button className="btn btn-primary" onClick={stepOnlineSoftmax}>
                  Step ({onlineSoftmax.currentBlock}/{numBlocks})
                </button>
                <button className="btn btn-secondary" onClick={resetOnlineSoftmax}>
                  Reset
                </button>
              </div>

              <div className="comparison-container">
                <div className="comparison-side left">
                  <h3>Running Statistics</h3>
                  <div style={{ fontFamily: 'monospace', fontSize: '14px' }}>
                    <p>Block: {onlineSoftmax.currentBlock}</p>
                    <p>Max So Far: {onlineSoftmax.maxSoFar === -Infinity ? '-∞' : onlineSoftmax.maxSoFar.toFixed(4)}</p>
                    <p>Exp Sum: {onlineSoftmax.expSumSoFar.toFixed(4)}</p>
                  </div>
                </div>
                <div className="comparison-side right">
                  <h3>Progress</h3>
                  <div style={{ height: '20px', background: '#f0f0f0', borderRadius: '10px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${(onlineSoftmax.currentBlock / numBlocks) * 100}%`,
                      height: '100%',
                      background: 'linear-gradient(90deg, #ff6b6b, #4ecdc4)',
                      transition: 'width 0.3s'
                    }} />
                  </div>
                </div>
              </div>

              <div className="code-block" style={{ marginTop: '20px' }}>
                <pre>{`# Online softmax update for each block
block_max = max(scores)
new_max = max(running_max, block_max)

# Correction factor for previous sum
correction = exp(running_max - new_max)

# Update running statistics
new_exp_sum = running_exp_sum * correction + sum(exp(scores - new_max))
output = output * correction + exp(scores - new_max) @ V

running_max, running_exp_sum = new_max, new_exp_sum`}</pre>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'causal-mask' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Causal Mask Visualization</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Lower triangular mask prevents attending to future tokens
              </p>

              <div style={{ display: 'flex', gap: '30px', flexWrap: 'wrap' }}>
                <div>
                  <h4 style={{ marginBottom: '10px' }}>Attention Mask</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: `repeat(8, 30px)`, gap: '2px' }}>
                    {Array.from({ length: 64 }, (_, i) => {
                      const row = Math.floor(i / 8);
                      const col = i % 8;
                      const isCausal = col <= row;
                      return (
                        <motion.div
                          key={i}
                          style={{
                            width: '30px',
                            height: '30px',
                            background: config.useCausalMask
                              ? (isCausal ? '#4ecdc4' : '#f0f0f0')
                              : '#4ecdc4',
                            borderRadius: '4px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '10px',
                            color: isCausal || !config.useCausalMask ? 'white' : '#999'
                          }}
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ delay: i * 0.01 }}
                        >
                          {config.useCausalMask && !isCausal ? '-∞' : ''}
                        </motion.div>
                      );
                    })}
                  </div>
                </div>

                {config.useAlibi && (
                  <div>
                    <h4 style={{ marginBottom: '10px' }}>ALiBi Slopes</h4>
                    <div style={{ fontFamily: 'monospace', fontSize: '13px' }}>
                      {Array.from({ length: Math.min(8, config.numHeads) }, (_, h) => (
                        <div key={h}>
                          Head {h}: slope = {Math.pow(2, -(8 / config.numHeads) * (h + 1)).toFixed(4)}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="code-block" style={{ marginTop: '20px' }}>
                <pre>{`# Causal mask application
if causal:
    mask = torch.triu(torch.full((seq_len, seq_len), -inf), diagonal=1)
    scores = scores + mask

# ALiBi (Attention with Linear Biases)
if alibi:
    slopes = 2 ** (-8 / num_heads * torch.arange(1, num_heads + 1))
    positions = torch.arange(seq_len)
    alibi_bias = slopes[:, None, None] * (positions[None, None, :] - positions[None, :, None])
    scores = scores + alibi_bias`}</pre>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'fp8' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>FP8 Quantization Support</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                8-bit floating point for 2x memory reduction and faster computation
              </p>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h4>FP8 Formats</h4>
                <ul>
                  <li><strong>E4M3:</strong> 4-bit exponent, 3-bit mantissa - for weights and activations</li>
                  <li><strong>E5M2:</strong> 5-bit exponent, 2-bit mantissa - for gradients</li>
                </ul>
              </div>

              <div className="comparison-container">
                <div className="comparison-side left" style={{ background: config.useFP8 ? '#e8f5e9' : '#f8f9fa' }}>
                  <h3 style={{ borderColor: config.useFP8 ? '#4caf50' : '#ccc' }}>FP8 Mode {config.useFP8 ? '(Active)' : ''}</h3>
                  <ul style={{ listStyle: 'none', padding: 0 }}>
                    <li>Memory: 8 bits per element</li>
                    <li>Range: ±448 (E4M3)</li>
                    <li>Precision: 3 mantissa bits</li>
                    <li>Speed: 2x faster on Tensor Cores</li>
                  </ul>
                </div>
                <div className="comparison-side right">
                  <h3>FP16/BF16 Mode</h3>
                  <ul style={{ listStyle: 'none', padding: 0 }}>
                    <li>Memory: 16 bits per element</li>
                    <li>Range: ±65504 (FP16)</li>
                    <li>Precision: 10 mantissa bits</li>
                    <li>Speed: Baseline</li>
                  </ul>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      <ParameterGlossary
        entries={createFlashGlossaryEntries(config)}
        title="Flash Attention Parameter Reference"
        collapsible={true}
      />
    </div>
  );
};

export default FlashAttentionApp;
