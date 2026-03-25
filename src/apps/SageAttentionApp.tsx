import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Play } from 'lucide-react';
import { useSageAttentionStore } from '../store/sageAttentionStore';
import InputPanel from '../components/shared/InputPanel';
import MetricsDashboard from '../components/shared/MetricsDashboard';
import ParameterGlossary from '../components/shared/ParameterGlossary';
import { createSageGlossaryEntries } from '../data/glossaries/sageGlossary';

interface SageAttentionAppProps {
  onBack: () => void;
}

const SageAttentionApp: React.FC<SageAttentionAppProps> = ({ onBack }) => {
  const {
    config,
    performance,
    currentPhase,
    activeTab,
    quantizationState,
    setConfig,
    setActiveTab,
    runSimulation,
    updateQuantizationState
  } = useSageAttentionStore();

  useEffect(() => {
    updateQuantizationState();
  }, [updateQuantizationState]);

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'fp8-quantization', label: 'FP8 Quantization' },
    { id: 'scale-factors', label: 'Scale Factors' },
    { id: 'mxfp4', label: 'MxFP4 Variant' }
  ] as const;

  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} /> Back to Home
      </button>

      <header className="app-header">
        <h1>Sage Attention</h1>
        <p>FP8 quantized attention with per-token scaling</p>
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
              { label: 'Head Dim', value: config.headDim, min: 32, max: 128, step: 16, onChange: v => setConfig({ headDim: v }) }
            ]}
            selects={[
              { label: 'FP8', value: config.useFP8, options: [{ value: true, label: 'Enabled' }, { value: false, label: 'Disabled' }], onChange: v => setConfig({ useFP8: v as boolean }) },
              { label: 'MxFP4', value: config.useMxFP4, options: [{ value: false, label: 'Disabled' }, { value: true, label: 'Enabled' }], onChange: v => setConfig({ useMxFP4: v as boolean }) },
              { label: 'Per-Token Scale', value: config.perTokenScale, options: [{ value: true, label: 'Enabled' }, { value: false, label: 'Per-Tensor' }], onChange: v => setConfig({ perTokenScale: v as boolean }) },
              { label: 'Smooth K', value: config.smoothK, options: [{ value: true, label: 'Enabled' }, { value: false, label: 'Disabled' }], onChange: v => setConfig({ smoothK: v as boolean }) }
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
              <h3 style={{ marginBottom: '20px' }}>Sage Attention Overview</h3>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h4>Key Optimizations</h4>
                <ul>
                  <li><strong>FP8 Quantization:</strong> 2x memory reduction, faster tensor core ops</li>
                  <li><strong>Per-Token Scaling:</strong> Better precision for varying magnitudes</li>
                  <li><strong>Smooth K:</strong> Apply smoothing to K for better quantization</li>
                  <li><strong>MxFP4:</strong> Ultra-low precision variant for extreme efficiency</li>
                </ul>
              </div>

              <MetricsDashboard
                metrics={performance}
                additionalMetrics={[
                  { label: 'Quantization Error', value: quantizationState.quantizationError.toFixed(4) }
                ]}
              />
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'fp8-quantization' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>FP8 Quantization Visualization</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Original values vs FP8 quantized values
              </p>

              <div style={{ marginBottom: '30px' }}>
                <h4 style={{ marginBottom: '15px' }}>Value Comparison</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
                  <div><strong>Original</strong></div>
                  <div><strong>FP8</strong></div>
                  <div><strong>Error</strong></div>
                  {quantizationState.originalValues.slice(0, 8).map((v, i) => (
                    <React.Fragment key={i}>
                      <div style={{
                        padding: '8px',
                        background: '#e3f2fd',
                        borderRadius: '4px',
                        fontFamily: 'monospace',
                        fontSize: '12px'
                      }}>
                        {v.toFixed(4)}
                      </div>
                      <div style={{
                        padding: '8px',
                        background: '#e8f5e9',
                        borderRadius: '4px',
                        fontFamily: 'monospace',
                        fontSize: '12px'
                      }}>
                        {quantizationState.fp8Values[i]?.toFixed(4)}
                      </div>
                      <div style={{
                        padding: '8px',
                        background: Math.abs(v - (quantizationState.fp8Values[i] || 0)) > 0.1 ? '#ffebee' : '#f0f0f0',
                        borderRadius: '4px',
                        fontFamily: 'monospace',
                        fontSize: '12px'
                      }}>
                        {Math.abs(v - (quantizationState.fp8Values[i] || 0)).toFixed(4)}
                      </div>
                    </React.Fragment>
                  ))}
                </div>
              </div>

              <div className="code-block">
                <pre>{`# FP8 E4M3 Quantization
# Range: ±448, 4-bit exponent, 3-bit mantissa

def quantize_fp8(x, scale):
    scaled = x / scale
    clamped = clamp(scaled, -448, 448)
    # Round to nearest representable FP8 value
    return round_fp8(clamped) * scale

# Scale factor calculation
scale = max(abs(x)) / 127.0  # Map to FP8 range`}</pre>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'scale-factors' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Scale Factor Strategies</h3>

              <div className="comparison-container">
                <div className="comparison-side left" style={{ background: !config.perTokenScale ? '#e8f5e9' : '#f8f9fa' }}>
                  <h3 style={{ borderColor: !config.perTokenScale ? '#4caf50' : '#ccc' }}>Per-Tensor Scale</h3>
                  <ul style={{ listStyle: 'none', padding: 0, fontSize: '13px' }}>
                    <li style={{ padding: '5px 0' }}>Single scale for entire tensor</li>
                    <li style={{ padding: '5px 0' }}>Lower overhead</li>
                    <li style={{ padding: '5px 0' }}>May lose precision for outliers</li>
                    <li style={{ padding: '5px 0' }}>Scale: {quantizationState.scaleFactors[0]?.toFixed(4) || 'N/A'}</li>
                  </ul>
                </div>
                <div className="comparison-side right" style={{ background: config.perTokenScale ? '#e8f5e9' : '#f8f9fa' }}>
                  <h3 style={{ borderColor: config.perTokenScale ? '#4caf50' : '#ccc' }}>Per-Token Scale</h3>
                  <ul style={{ listStyle: 'none', padding: 0, fontSize: '13px' }}>
                    <li style={{ padding: '5px 0' }}>Separate scale per token</li>
                    <li style={{ padding: '5px 0' }}>Higher overhead</li>
                    <li style={{ padding: '5px 0' }}>Better handles varying magnitudes</li>
                    <li style={{ padding: '5px 0' }}>
                      Scales: {quantizationState.scaleFactors.slice(0, 4).map(s => s.toFixed(3)).join(', ')}
                    </li>
                  </ul>
                </div>
              </div>

              {config.smoothK && (
                <div className="info-panel" style={{ marginTop: '20px' }}>
                  <h4>Smooth K Optimization</h4>
                  <p>Apply smoothing to K matrix to reduce outliers before quantization:</p>
                  <code style={{ display: 'block', marginTop: '10px', fontFamily: 'monospace' }}>
                    K_smooth = K - mean(K, dim=-1, keepdim=True)
                  </code>
                </div>
              )}
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'mxfp4' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>MxFP4 Ultra-Low Precision</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Mixed FP4: 2-bit exponent, 1-bit mantissa + sign
              </p>

              <div className="info-panel" style={{ marginBottom: '20px', background: config.useMxFP4 ? '#fff3e0' : '#f8f9fa' }}>
                <h4>MxFP4 Characteristics</h4>
                <ul>
                  <li><strong>Levels:</strong> Only ~16 representable values per sign</li>
                  <li><strong>Range:</strong> Approximately ±6</li>
                  <li><strong>Use Case:</strong> When memory is critical and some accuracy loss acceptable</li>
                  <li><strong>Speedup:</strong> ~4x vs FP16 on compatible hardware</li>
                </ul>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ marginBottom: '15px' }}>MxFP4 Quantization Levels</h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                  {[-6, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 6].map(level => (
                    <div
                      key={level}
                      style={{
                        padding: '8px 12px',
                        background: `hsl(${30 + level * 10}, 70%, 60%)`,
                        borderRadius: '4px',
                        fontFamily: 'monospace',
                        fontSize: '12px',
                        color: 'white'
                      }}
                    >
                      {level}
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ marginBottom: '15px' }}>MxFP4 vs FP8 Comparison</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
                  <div><strong>Original</strong></div>
                  <div><strong>FP8</strong></div>
                  <div><strong>MxFP4</strong></div>
                  {quantizationState.originalValues.slice(0, 6).map((v, i) => (
                    <React.Fragment key={i}>
                      <div style={{ padding: '6px', background: '#f0f0f0', borderRadius: '4px', fontFamily: 'monospace', fontSize: '11px' }}>
                        {v.toFixed(3)}
                      </div>
                      <div style={{ padding: '6px', background: '#e3f2fd', borderRadius: '4px', fontFamily: 'monospace', fontSize: '11px' }}>
                        {quantizationState.fp8Values[i]?.toFixed(3)}
                      </div>
                      <div style={{ padding: '6px', background: '#fff3e0', borderRadius: '4px', fontFamily: 'monospace', fontSize: '11px' }}>
                        {quantizationState.mxfp4Values[i]?.toFixed(3)}
                      </div>
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      <ParameterGlossary
        entries={createSageGlossaryEntries(config)}
        title="Sage Attention Parameter Reference"
        collapsible={true}
      />
    </div>
  );
};

export default SageAttentionApp;
