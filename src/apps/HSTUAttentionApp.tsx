import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Play } from 'lucide-react';
import { useHSTUAttentionStore } from '../store/hstuAttentionStore';
import InputPanel from '../components/shared/InputPanel';
import MetricsDashboard from '../components/shared/MetricsDashboard';
import ParameterGlossary from '../components/shared/ParameterGlossary';
import { createHSTUGlossaryEntries } from '../data/glossaries/hstuGlossary';

interface HSTUAttentionAppProps {
  onBack: () => void;
}

const HSTUAttentionApp: React.FC<HSTUAttentionAppProps> = ({ onBack }) => {
  const {
    config,
    performance,
    currentPhase,
    activeTab,
    activationComparison,
    jaggedSequences,
    setConfig,
    setActiveTab,
    runSimulation,
    updateActivationComparison,
    initializeJaggedSequences
  } = useHSTUAttentionStore();

  useEffect(() => {
    updateActivationComparison();
    initializeJaggedSequences();
  }, [updateActivationComparison, initializeJaggedSequences]);

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'silu-vs-softmax', label: 'SiLU vs Softmax' },
    { id: 'jagged-tensors', label: 'Jagged Tensors' },
    { id: 'alpha-scaling', label: 'Alpha Scaling' }
  ] as const;

  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} /> Back to Home
      </button>

      <header className="app-header">
        <h1>HSTU Attention</h1>
        <p>SiLU-gated attention with jagged tensor support</p>
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
              { label: 'Num Heads', value: config.numHeads, min: 1, max: 32, onChange: v => setConfig({ numHeads: v }) },
              { label: 'Alpha', value: config.alpha, min: 0.1, max: 2.0, step: 0.1, onChange: v => setConfig({ alpha: v }) }
            ]}
            selects={[
              { label: 'Jagged Tensors', value: config.useJaggedTensors, options: [{ value: true, label: 'Enabled' }, { value: false, label: 'Disabled' }], onChange: v => setConfig({ useJaggedTensors: v as boolean }) }
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
              <h3 style={{ marginBottom: '20px' }}>HSTU Attention Overview</h3>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h4>Key Differences from Standard Attention</h4>
                <ul>
                  <li><strong>SiLU Gating:</strong> Uses SiLU (x * sigmoid(x)) instead of softmax</li>
                  <li><strong>No Normalization:</strong> Scores are not normalized to sum to 1</li>
                  <li><strong>Alpha Scaling:</strong> Configurable gate parameter</li>
                  <li><strong>Jagged Support:</strong> Efficient handling of variable-length sequences</li>
                </ul>
              </div>

              <MetricsDashboard metrics={performance} />
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'silu-vs-softmax' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>SiLU vs Softmax Comparison</h3>

              <div className="activation-comparison">
                <div className="activation-box softmax">
                  <h4 style={{ color: '#1565c0', marginBottom: '10px' }}>Softmax</h4>
                  <div className="activation-formula">
                    softmax(x) = exp(x) / sum(exp(x))
                  </div>
                  <p style={{ fontSize: '13px', color: '#666', margin: '10px 0' }}>
                    Normalizes to probability distribution (sums to 1)
                  </p>
                  <div className="activation-values">
                    {activationComparison.softmaxOutput.slice(0, 4).map((v, i) => (
                      <span key={i} className="activation-value">{v.toFixed(3)}</span>
                    ))}
                  </div>
                </div>

                <div className="activation-box silu">
                  <h4 style={{ color: '#e65100', marginBottom: '10px' }}>SiLU</h4>
                  <div className="activation-formula">
                    silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                  </div>
                  <p style={{ fontSize: '13px', color: '#666', margin: '10px 0' }}>
                    Non-normalized gating (can sum to any value)
                  </p>
                  <div className="activation-values">
                    {activationComparison.siluOutput.slice(0, 4).map((v, i) => (
                      <span key={i} className="activation-value">{v.toFixed(3)}</span>
                    ))}
                  </div>
                </div>
              </div>

              <div style={{ marginTop: '30px' }}>
                <h4 style={{ marginBottom: '15px' }}>Input Values</h4>
                <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                  {activationComparison.input.slice(0, 8).map((v, i) => (
                    <span key={i} style={{ padding: '6px 12px', background: '#f0f0f0', borderRadius: '4px', fontFamily: 'monospace' }}>
                      {v.toFixed(3)}
                    </span>
                  ))}
                </div>
              </div>

              <div className="code-block" style={{ marginTop: '20px' }}>
                <pre>{`# HSTU Attention formula
# Instead of: attn = softmax(QK^T / sqrt(d)) @ V
# HSTU uses: attn = silu(alpha * QK^T) @ V

scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, Q, K]
gates = F.silu(alpha * scores)                  # SiLU gating
output = torch.matmul(gates, V)                 # Weighted sum`}</pre>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'jagged-tensors' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Jagged Tensor Support</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Efficient handling of variable-length sequences without padding
              </p>

              <div style={{ marginBottom: '30px' }}>
                <h4 style={{ marginBottom: '15px' }}>Sequence Lengths</h4>
                <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                  {jaggedSequences.map((seq, idx) => (
                    <motion.div
                      key={idx}
                      style={{
                        padding: '15px',
                        background: 'white',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        minWidth: '150px'
                      }}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.1 }}
                    >
                      <h5 style={{ color: '#666', marginBottom: '8px' }}>Sequence {seq.seqId}</h5>
                      <div style={{ fontFamily: 'monospace', fontSize: '13px' }}>
                        <p>Length: {seq.length}</p>
                        <p>Offset: {seq.startOffset}</p>
                      </div>
                      <div style={{
                        marginTop: '10px',
                        height: '8px',
                        background: '#f0f0f0',
                        borderRadius: '4px'
                      }}>
                        <div style={{
                          width: `${(seq.length / config.seqLen) * 100}%`,
                          height: '100%',
                          background: `hsl(${idx * 60}, 70%, 50%)`,
                          borderRadius: '4px'
                        }} />
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>

              <div className="comparison-container">
                <div className="comparison-side left">
                  <h3>Padded Batching</h3>
                  <ul style={{ listStyle: 'none', padding: 0, fontSize: '13px' }}>
                    <li style={{ padding: '5px 0' }}>Pad all sequences to max length</li>
                    <li style={{ padding: '5px 0' }}>Wasted computation on padding</li>
                    <li style={{ padding: '5px 0' }}>Simple implementation</li>
                    <li style={{ padding: '5px 0' }}>Memory inefficient</li>
                  </ul>
                </div>
                <div className="comparison-side right" style={{ background: config.useJaggedTensors ? '#e8f5e9' : '#f8f9fa' }}>
                  <h3 style={{ borderColor: config.useJaggedTensors ? '#4caf50' : '#ccc' }}>Jagged Tensors</h3>
                  <ul style={{ listStyle: 'none', padding: 0, fontSize: '13px' }}>
                    <li style={{ padding: '5px 0' }}>Store sequences contiguously</li>
                    <li style={{ padding: '5px 0' }}>No wasted computation</li>
                    <li style={{ padding: '5px 0' }}>Complex indexing</li>
                    <li style={{ padding: '5px 0' }}>Memory efficient</li>
                  </ul>
                </div>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'alpha-scaling' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Alpha Scaling Parameter</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Current alpha: {config.alpha.toFixed(2)}
              </p>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h4>Effect of Alpha</h4>
                <ul>
                  <li><strong>alpha &lt; 1:</strong> Smoother gating, more uniform attention</li>
                  <li><strong>alpha = 1:</strong> Standard SiLU behavior</li>
                  <li><strong>alpha &gt; 1:</strong> Sharper gating, more focused attention</li>
                </ul>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ marginBottom: '15px' }}>SiLU Response at Different Alpha</h4>
                <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                  {[0.5, 1.0, 1.5, 2.0].map(alpha => (
                    <div
                      key={alpha}
                      style={{
                        padding: '15px',
                        background: alpha === config.alpha ? '#e8f5e9' : 'white',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        border: alpha === config.alpha ? '2px solid #4caf50' : 'none'
                      }}
                    >
                      <h5 style={{ marginBottom: '10px' }}>alpha = {alpha}</h5>
                      <div style={{ fontFamily: 'monospace', fontSize: '12px' }}>
                        silu(1.0 * {alpha}) = {(1.0 * alpha / (1 + Math.exp(-1.0 * alpha))).toFixed(3)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="code-block">
                <pre>{`# Alpha-scaled SiLU gating
def silu_gated_attention(Q, K, V, alpha=${config.alpha}):
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Apply alpha scaling before SiLU
    scaled_scores = alpha * scores

    # SiLU gating (not softmax!)
    gates = F.silu(scaled_scores)

    # Compute output
    output = torch.matmul(gates, V)
    return output`}</pre>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      <ParameterGlossary
        entries={createHSTUGlossaryEntries(config)}
        title="HSTU Attention Parameter Reference"
        collapsible={true}
      />
    </div>
  );
};

export default HSTUAttentionApp;
