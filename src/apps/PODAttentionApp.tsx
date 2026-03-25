import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Play } from 'lucide-react';
import { usePODAttentionStore } from '../store/podAttentionStore';
import InputPanel from '../components/shared/InputPanel';
import MetricsDashboard from '../components/shared/MetricsDashboard';
import ParameterGlossary from '../components/shared/ParameterGlossary';
import { createPODGlossaryEntries } from '../data/glossaries/podGlossary';

interface PODAttentionAppProps {
  onBack: () => void;
}

const PODAttentionApp: React.FC<PODAttentionAppProps> = ({ onBack }) => {
  const {
    config,
    performance,
    currentPhase,
    activeTab,
    workloadDistribution,
    prefillProgress,
    decodeProgress,
    setConfig,
    setActiveTab,
    runSimulation,
    setPrefillRatio,
    updateWorkloadDistribution
  } = usePODAttentionStore();

  useEffect(() => {
    updateWorkloadDistribution();
  }, [updateWorkloadDistribution]);

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'dual-workload', label: 'Dual Workload' },
    { id: 'cu-allocation', label: 'CU Allocation' },
    { id: 'ratio-tuning', label: 'Ratio Tuning' }
  ] as const;

  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} /> Back to Home
      </button>

      <header className="app-header">
        <h1>POD Attention</h1>
        <p>Simultaneous prefill and decode on the same GPU</p>
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
              { label: 'Batch Size', value: config.batchSize, min: 1, max: 32, onChange: v => setConfig({ batchSize: v }) },
              { label: 'Prefill Seq Len', value: config.prefillSeqLen, min: 128, max: 4096, step: 128, onChange: v => setConfig({ prefillSeqLen: v }) },
              { label: 'Decode Tokens', value: config.decodeNumTokens, min: 8, max: 256, step: 8, onChange: v => setConfig({ decodeNumTokens: v }) },
              { label: 'Num Heads', value: config.numHeads, min: 1, max: 32, onChange: v => setConfig({ numHeads: v }) },
              { label: 'Prefill Ratio', value: config.prefillRatio, min: 0.1, max: 0.9, step: 0.05, onChange: v => setPrefillRatio(v) }
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
              <h3 style={{ marginBottom: '20px' }}>POD Attention Overview</h3>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h4>Prefill-Or-Decode (POD)</h4>
                <ul>
                  <li><strong>Dual Workload:</strong> Process prefill and decode simultaneously</li>
                  <li><strong>Resource Sharing:</strong> Split CUs between prefill and decode</li>
                  <li><strong>Workload Balance:</strong> Prefill is compute-bound, decode is memory-bound</li>
                  <li><strong>Efficiency:</strong> Better GPU utilization than sequential processing</li>
                </ul>
              </div>

              <MetricsDashboard
                metrics={performance}
                additionalMetrics={[
                  { label: 'Prefill CUs', value: workloadDistribution.prefillCUs.length },
                  { label: 'Decode CUs', value: workloadDistribution.decodeCUs.length }
                ]}
              />
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'dual-workload' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Dual Workload Execution</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Prefill and decode running concurrently
              </p>

              <div className="comparison-container">
                <div className="comparison-side left" style={{
                  background: currentPhase === 'prefill' ? '#fff3e0' : '#f8f9fa',
                  border: currentPhase === 'prefill' ? '2px solid #ff9800' : 'none'
                }}>
                  <h3 style={{ borderColor: '#ff9800', color: '#e65100' }}>Prefill Phase</h3>
                  <div style={{ marginBottom: '15px' }}>
                    <div style={{ fontSize: '13px', color: '#666', marginBottom: '5px' }}>
                      Processing {config.prefillSeqLen} tokens
                    </div>
                    <div style={{ height: '20px', background: '#f0f0f0', borderRadius: '10px', overflow: 'hidden' }}>
                      <motion.div
                        style={{
                          height: '100%',
                          background: 'linear-gradient(90deg, #ff9800, #ff5722)',
                          borderRadius: '10px'
                        }}
                        animate={{ width: `${prefillProgress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                    <div style={{ fontSize: '12px', color: '#888', marginTop: '5px' }}>
                      {prefillProgress}% complete
                    </div>
                  </div>
                  <ul style={{ listStyle: 'none', padding: 0, fontSize: '13px' }}>
                    <li>Compute-bound workload</li>
                    <li>High arithmetic intensity</li>
                    <li>Full attention computation</li>
                    <li>CUs allocated: {workloadDistribution.prefillCUs.length}</li>
                  </ul>
                </div>

                <div className="comparison-side right" style={{
                  background: currentPhase === 'decode' ? '#e3f2fd' : '#f8f9fa',
                  border: currentPhase === 'decode' ? '2px solid #2196f3' : 'none'
                }}>
                  <h3 style={{ borderColor: '#2196f3', color: '#1565c0' }}>Decode Phase</h3>
                  <div style={{ marginBottom: '15px' }}>
                    <div style={{ fontSize: '13px', color: '#666', marginBottom: '5px' }}>
                      Generating {config.decodeNumTokens} tokens
                    </div>
                    <div style={{ height: '20px', background: '#f0f0f0', borderRadius: '10px', overflow: 'hidden' }}>
                      <motion.div
                        style={{
                          height: '100%',
                          background: 'linear-gradient(90deg, #2196f3, #00bcd4)',
                          borderRadius: '10px'
                        }}
                        animate={{ width: `${decodeProgress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                    <div style={{ fontSize: '12px', color: '#888', marginTop: '5px' }}>
                      {decodeProgress}% complete
                    </div>
                  </div>
                  <ul style={{ listStyle: 'none', padding: 0, fontSize: '13px' }}>
                    <li>Memory-bound workload</li>
                    <li>Low arithmetic intensity</li>
                    <li>KV cache reads dominant</li>
                    <li>CUs allocated: {workloadDistribution.decodeCUs.length}</li>
                  </ul>
                </div>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'cu-allocation' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Compute Unit Allocation</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Distribution of 120 CUs between prefill ({(config.prefillRatio * 100).toFixed(0)}%) and decode ({(config.decodeRatio * 100).toFixed(0)}%)
              </p>

              <div style={{ marginBottom: '30px' }}>
                <h4 style={{ marginBottom: '15px' }}>CU Distribution</h4>
                <div style={{ display: 'flex', height: '60px', borderRadius: '8px', overflow: 'hidden' }}>
                  <motion.div
                    style={{
                      background: 'linear-gradient(180deg, #ff9800, #ff5722)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontWeight: 'bold'
                    }}
                    animate={{ width: `${config.prefillRatio * 100}%` }}
                    transition={{ duration: 0.5 }}
                  >
                    Prefill: {workloadDistribution.prefillCUs.length} CUs
                  </motion.div>
                  <motion.div
                    style={{
                      background: 'linear-gradient(180deg, #2196f3, #00bcd4)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontWeight: 'bold'
                    }}
                    animate={{ width: `${config.decodeRatio * 100}%` }}
                    transition={{ duration: 0.5 }}
                  >
                    Decode: {workloadDistribution.decodeCUs.length} CUs
                  </motion.div>
                </div>
              </div>

              <div className="comparison-container">
                <div className="comparison-side left">
                  <h3>Prefill Utilization</h3>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ff9800' }}>
                    {(workloadDistribution.prefillUtilization * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="comparison-side right">
                  <h3>Decode Utilization</h3>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#2196f3' }}>
                    {(workloadDistribution.decodeUtilization * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'ratio-tuning' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Prefill/Decode Ratio Tuning</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Adjust the balance based on your workload characteristics
              </p>

              <div style={{ marginBottom: '30px' }}>
                <h4 style={{ marginBottom: '15px' }}>Current Ratio: {(config.prefillRatio * 100).toFixed(0)}% Prefill / {(config.decodeRatio * 100).toFixed(0)}% Decode</h4>
                <input
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={config.prefillRatio}
                  onChange={e => setPrefillRatio(Number(e.target.value))}
                  style={{ width: '100%', height: '30px' }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#888' }}>
                  <span>More Decode</span>
                  <span>Balanced</span>
                  <span>More Prefill</span>
                </div>
              </div>

              <div className="info-panel">
                <h4>Tuning Guidelines</h4>
                <ul>
                  <li><strong>High Prefill Ratio:</strong> Best when processing many new requests</li>
                  <li><strong>High Decode Ratio:</strong> Best when generating long responses</li>
                  <li><strong>Balanced:</strong> Mixed workload with equal importance</li>
                  <li><strong>Dynamic:</strong> Some systems adjust ratio based on queue state</li>
                </ul>
              </div>

              <div className="code-block" style={{ marginTop: '20px' }}>
                <pre>{`# POD Attention Configuration
pod_config = {
    "prefill_ratio": ${config.prefillRatio.toFixed(2)},
    "decode_ratio": ${config.decodeRatio.toFixed(2)},
    "num_prefill_cus": ${workloadDistribution.prefillCUs.length},
    "num_decode_cus": ${workloadDistribution.decodeCUs.length},
}

# Launch concurrent kernels
prefill_kernel.launch(prefill_cus, prefill_batch)
decode_kernel.launch(decode_cus, decode_batch)

# Synchronize
torch.cuda.synchronize()`}</pre>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      <ParameterGlossary
        entries={createPODGlossaryEntries(config)}
        title="POD Attention Parameter Reference"
        collapsible={true}
      />
    </div>
  );
};

export default PODAttentionApp;
