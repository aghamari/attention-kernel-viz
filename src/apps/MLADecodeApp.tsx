import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Play } from 'lucide-react';
import { useMLADecodeStore } from '../store/mlaDecodeStore';
import InputPanel from '../components/shared/InputPanel';
import MetricsDashboard from '../components/shared/MetricsDashboard';
import ParameterGlossary from '../components/shared/ParameterGlossary';
import { createMLAGlossaryEntries } from '../data/glossaries/mlaGlossary';

interface MLADecodeAppProps {
  onBack: () => void;
}

const MLADecodeApp: React.FC<MLADecodeAppProps> = ({ onBack }) => {
  const {
    config,
    performance,
    currentPhase,
    activeTab,
    ropeState,
    latentState,
    setConfig,
    setActiveTab,
    runSimulation,
    updateRoPEState,
    initializeLatentState
  } = useMLADecodeStore();

  useEffect(() => {
    initializeLatentState();
    updateRoPEState(0);
  }, [initializeLatentState, updateRoPEState]);

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'latent-compression', label: 'Latent Compression' },
    { id: 'rope-fusion', label: 'RoPE Fusion' },
    { id: 'two-stage', label: 'Two-Stage' }
  ] as const;

  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} /> Back to Home
      </button>

      <header className="app-header">
        <h1>MLA Decode + RoPE</h1>
        <p>Multi-Head Latent Attention with fused rotary embeddings</p>
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
              { label: 'Sequence Length', value: config.seqLen, min: 64, max: 2048, step: 64, onChange: v => setConfig({ seqLen: v }) },
              { label: 'Num Heads', value: config.numHeads, min: 1, max: 32, onChange: v => setConfig({ numHeads: v }) },
              { label: 'Head Dim', value: config.headDim, min: 32, max: 128, step: 16, onChange: v => setConfig({ headDim: v }) },
              { label: 'KV Latent Dim', value: config.kvLatentDim, min: 8, max: 64, step: 8, onChange: v => setConfig({ kvLatentDim: v }) },
              { label: 'RoPE Theta', value: config.ropeTheta, min: 1000, max: 100000, step: 1000, onChange: v => setConfig({ ropeTheta: v }) },
              { label: 'RoPE Scale', value: config.ropeScale, min: 0.5, max: 2.0, step: 0.1, onChange: v => setConfig({ ropeScale: v }) }
            ]}
            selects={[
              { label: 'Use RoPE', value: config.useRoPE, options: [{ value: true, label: 'Enabled' }, { value: false, label: 'Disabled' }], onChange: v => setConfig({ useRoPE: v as boolean }) }
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
              <h3 style={{ marginBottom: '20px' }}>MLA Decode Overview</h3>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h4>Multi-Head Latent Attention (MLA)</h4>
                <ul>
                  <li><strong>KV Compression:</strong> Reduce KV cache size via latent projection</li>
                  <li><strong>Fused RoPE:</strong> Apply rotary embeddings during attention</li>
                  <li><strong>Two-Stage:</strong> Separate QK and V computations</li>
                  <li><strong>Memory Efficient:</strong> {latentState.compressionRatio.toFixed(1)}x cache reduction</li>
                </ul>
              </div>

              <MetricsDashboard
                metrics={performance}
                additionalMetrics={[
                  { label: 'Compression Ratio', value: latentState.compressionRatio.toFixed(1), unit: 'x' },
                  { label: 'RoPE Theta', value: config.ropeTheta }
                ]}
              />
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'latent-compression' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>KV Latent Compression</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Compress KV cache from {config.headDim}D to {config.kvLatentDim}D ({latentState.compressionRatio.toFixed(1)}x reduction)
              </p>

              <div className="comparison-container">
                <div className="comparison-side left">
                  <h3>Original KV</h3>
                  <div style={{ fontFamily: 'monospace', fontSize: '13px' }}>
                    <p>Shape: [{config.seqLen}, {config.numKvHeads}, {config.headDim}]</p>
                    <p>Size: {(config.seqLen * config.numKvHeads * config.headDim * 2 / 1024).toFixed(1)} KB</p>
                  </div>
                  <div style={{ marginTop: '15px', display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: '2px' }}>
                    {Array.from({ length: Math.min(config.headDim, 32) }, (_, i) => (
                      <div
                        key={i}
                        style={{
                          width: '20px',
                          height: '20px',
                          background: `hsl(200, 70%, ${50 + Math.random() * 30}%)`,
                          borderRadius: '2px'
                        }}
                      />
                    ))}
                  </div>
                </div>
                <div className="comparison-side right" style={{ background: '#e8f5e9' }}>
                  <h3 style={{ borderColor: '#4caf50' }}>Compressed KV</h3>
                  <div style={{ fontFamily: 'monospace', fontSize: '13px' }}>
                    <p>Shape: [{config.seqLen}, {config.numKvHeads}, {config.kvLatentDim}]</p>
                    <p>Size: {(config.seqLen * config.numKvHeads * config.kvLatentDim * 2 / 1024).toFixed(1)} KB</p>
                  </div>
                  <div style={{ marginTop: '15px', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '2px' }}>
                    {Array.from({ length: Math.min(config.kvLatentDim, 16) }, (_, i) => (
                      <div
                        key={i}
                        style={{
                          width: '20px',
                          height: '20px',
                          background: `hsl(120, 70%, ${50 + Math.random() * 30}%)`,
                          borderRadius: '2px'
                        }}
                      />
                    ))}
                  </div>
                </div>
              </div>

              <div className="code-block" style={{ marginTop: '20px' }}>
                <pre>{`# KV Compression
# Original: K, V of shape [B, S, num_kv_heads, head_dim]
# Compressed: kv_latent of shape [B, S, num_kv_heads, kv_latent_dim]

kv_latent = compress_projection(torch.cat([K, V], dim=-1))
# kv_latent_dim << head_dim for memory savings

# During attention:
K_restored = restore_k_projection(kv_latent)
V_restored = restore_v_projection(kv_latent)`}</pre>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'rope-fusion' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Rotary Position Embedding (RoPE)</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Position-dependent rotation of query and key vectors
              </p>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ fontSize: '14px', color: '#666' }}>Position: </label>
                <input
                  type="range"
                  min={0}
                  max={config.seqLen - 1}
                  value={ropeState.position}
                  onChange={e => updateRoPEState(Number(e.target.value))}
                  style={{ width: '200px', marginLeft: '10px' }}
                />
                <span style={{ marginLeft: '10px' }}>{ropeState.position}</span>
              </div>

              <div className="comparison-container">
                <div className="comparison-side left">
                  <h3>Cosine Components</h3>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                    {ropeState.cosValues.slice(0, 8).map((v, i) => (
                      <div
                        key={i}
                        style={{
                          padding: '5px 10px',
                          background: `hsl(200, 70%, ${50 + v * 25}%)`,
                          borderRadius: '4px',
                          fontFamily: 'monospace',
                          fontSize: '11px',
                          color: 'white'
                        }}
                      >
                        {v.toFixed(3)}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="comparison-side right">
                  <h3>Sine Components</h3>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                    {ropeState.sinValues.slice(0, 8).map((v, i) => (
                      <div
                        key={i}
                        style={{
                          padding: '5px 10px',
                          background: `hsl(30, 70%, ${50 + v * 25}%)`,
                          borderRadius: '4px',
                          fontFamily: 'monospace',
                          fontSize: '11px',
                          color: 'white'
                        }}
                      >
                        {v.toFixed(3)}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="code-block" style={{ marginTop: '20px' }}>
                <pre>{`# RoPE Formula
# For dimension pair (2i, 2i+1):
freq = 1.0 / (theta ** (2i / dim))
angle = position * freq * scale

# Rotation:
x_rot[..., 2i] = x[..., 2i] * cos(angle) - x[..., 2i+1] * sin(angle)
x_rot[..., 2i+1] = x[..., 2i] * sin(angle) + x[..., 2i+1] * cos(angle)

# Current settings:
# theta = ${config.ropeTheta}, scale = ${config.ropeScale}`}</pre>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'two-stage' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Two-Stage Computation</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Separate QK dot product and V multiplication stages
              </p>

              <div style={{ marginBottom: '20px' }}>
                {['Stage 1: QK', 'Stage 2: V'].map((stage, idx) => (
                  <div key={idx} className="flow-step">
                    <div className="flow-step-number" style={{
                      background: currentPhase === (idx === 0 ? 'stage1-qk' : 'stage2-v')
                        ? 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
                        : undefined
                    }}>
                      {idx + 1}
                    </div>
                    <div className="flow-step-content">
                      <h4>{stage}</h4>
                      <p style={{ fontSize: '13px', color: '#666' }}>
                        {idx === 0
                          ? 'Compute Q @ K^T with RoPE applied'
                          : 'Apply softmax and multiply by V'}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="info-panel">
                <h4>Why Two-Stage?</h4>
                <ul>
                  <li>RoPE only applies to Q and K (not V)</li>
                  <li>Separate stages allow optimized memory access patterns</li>
                  <li>Stage 1 benefits from RoPE fusion</li>
                  <li>Stage 2 can use different parallelization strategy</li>
                </ul>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      <ParameterGlossary
        entries={createMLAGlossaryEntries(config)}
        title="MLA Decode + RoPE Parameter Reference"
        collapsible={true}
      />
    </div>
  );
};

export default MLADecodeApp;
