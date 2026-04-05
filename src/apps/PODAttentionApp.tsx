import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft } from 'lucide-react';
import { usePODAttentionStore } from '../store/podAttentionStore';
import ConfigDisplay from '../components/shared/ConfigDisplay';
import ParameterGlossary from '../components/shared/ParameterGlossary';
import { createPODGlossaryEntries } from '../data/glossaries/podGlossary';

interface PODAttentionAppProps {
  onBack: () => void;
}

const PODAttentionApp: React.FC<PODAttentionAppProps> = ({ onBack }) => {
  const {
    config,
    activeTab,
    workloadDistribution,
    setActiveTab,
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
          <ConfigDisplay
            title="Configuration"
            params={[
              { label: 'Batch Size', value: 4 },
              { label: 'Sequence Length', value: 512 },
              { label: 'Num Heads', value: 8 },
              { label: 'Head Dim', value: 64 },
              { label: 'Num KV Heads', value: 8 },
              { label: 'Prefill Ratio', value: 0.7 },
              { label: 'Decode Ratio', value: 0.3 },
              { label: 'Prefill Seq Len', value: 512 },
              { label: 'Decode Num Tokens', value: 32 }
            ]}
          />
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

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                <div style={{ padding: '20px', background: '#fff3e0', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#ff9800' }}>
                    {workloadDistribution.prefillCUs.length}
                  </div>
                  <div style={{ fontSize: '13px', color: '#666', marginTop: '4px' }}>Prefill CUs</div>
                </div>
                <div style={{ padding: '20px', background: '#e3f2fd', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#2196f3' }}>
                    {workloadDistribution.decodeCUs.length}
                  </div>
                  <div style={{ fontSize: '13px', color: '#666', marginTop: '4px' }}>Decode CUs</div>
                </div>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'dual-workload' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Dual Workload Execution</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Prefill and decode running concurrently
              </p>

              <div className="comparison-container">
                <div className="comparison-side left">
                  <h3 style={{ borderColor: '#ff9800', color: '#e65100' }}>Prefill Phase</h3>
                  <div style={{ marginBottom: '15px' }}>
                    <div style={{ fontSize: '13px', color: '#666' }}>
                      Processing {config.prefillSeqLen} tokens
                    </div>
                  </div>
                  <ul style={{ listStyle: 'none', padding: 0, fontSize: '13px' }}>
                    <li>Compute-bound workload</li>
                    <li>High arithmetic intensity</li>
                    <li>Full attention computation</li>
                    <li>CUs allocated: {workloadDistribution.prefillCUs.length}</li>
                  </ul>
                </div>

                <div className="comparison-side right">
                  <h3 style={{ borderColor: '#2196f3', color: '#1565c0' }}>Decode Phase</h3>
                  <div style={{ marginBottom: '15px' }}>
                    <div style={{ fontSize: '13px', color: '#666' }}>
                      Generating {config.decodeNumTokens} tokens
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
