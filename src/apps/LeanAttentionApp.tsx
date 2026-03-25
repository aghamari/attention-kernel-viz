import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Play } from 'lucide-react';
import { useLeanAttentionStore } from '../store/leanAttentionStore';
import InputPanel from '../components/shared/InputPanel';
import MetricsDashboard from '../components/shared/MetricsDashboard';
import ParameterGlossary from '../components/shared/ParameterGlossary';
import { createLeanGlossaryEntries } from '../data/glossaries/leanGlossary';

interface LeanAttentionAppProps {
  onBack: () => void;
}

const LeanAttentionApp: React.FC<LeanAttentionAppProps> = ({ onBack }) => {
  const {
    config,
    workUnits,
    cuStates,
    performance,
    currentPhase,
    activeTab,
    currentTime,
    setConfig,
    setActiveTab,
    runSimulation,
    toggleWorkStealing,
    initializeWorkUnits
  } = useLeanAttentionStore();

  useEffect(() => {
    initializeWorkUnits();
  }, [initializeWorkUnits]);

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'streamk-schedule', label: 'StreamK Schedule' },
    { id: 'work-stealing', label: 'Work Stealing' },
    { id: 'persistent', label: 'Persistent Kernel' }
  ] as const;

  const getWorkColor = (workId: number) => {
    const colors = ['#e94560', '#4facfe', '#43e97b', '#fa709a', '#fee140', '#f093fb'];
    return colors[workId % colors.length];
  };

  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} /> Back to Home
      </button>

      <header className="app-header">
        <h1>Lean Attention (StreamK)</h1>
        <p>StreamK tiling with persistent kernels and work stealing</p>
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
              { label: 'Sequence Length', value: config.seqLen, min: 256, max: 8192, step: 256, onChange: v => setConfig({ seqLen: v }) },
              { label: 'Num Heads', value: config.numHeads, min: 1, max: 32, onChange: v => setConfig({ numHeads: v }) },
              { label: 'Num CUs', value: config.numCUs, min: 4, max: 120, step: 4, onChange: v => setConfig({ numCUs: v }) },
              { label: 'Tiles Per CU', value: config.tilesPerCU, min: 1, max: 8, onChange: v => setConfig({ tilesPerCU: v }) }
            ]}
            selects={[
              { label: 'Work Stealing', value: config.enableWorkStealing, options: [{ value: true, label: 'Enabled' }, { value: false, label: 'Disabled' }], onChange: v => setConfig({ enableWorkStealing: v as boolean }) },
              { label: 'Persistent Kernel', value: config.persistentKernel, options: [{ value: true, label: 'Enabled' }, { value: false, label: 'Disabled' }], onChange: v => setConfig({ persistentKernel: v as boolean }) }
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
              <h3 style={{ marginBottom: '20px' }}>Lean Attention Overview</h3>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h4>StreamK Design</h4>
                <ul>
                  <li><strong>Tile-based Decomposition:</strong> Attention split into fine-grained tiles</li>
                  <li><strong>Dynamic Scheduling:</strong> Tiles assigned to CUs at runtime</li>
                  <li><strong>Work Stealing:</strong> Idle CUs steal work from busy neighbors</li>
                  <li><strong>Persistent Kernels:</strong> Single kernel launch processes all tiles</li>
                </ul>
              </div>

              <MetricsDashboard
                metrics={performance}
                additionalMetrics={[
                  { label: 'Total Work Units', value: workUnits.length },
                  { label: 'CUs Active', value: cuStates.filter(c => !c.isIdle).length }
                ]}
              />
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'streamk-schedule' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>StreamK Gantt Chart</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Work distribution across {config.numCUs} compute units
              </p>

              <div className="streamk-gantt">
                {Array.from({ length: Math.min(config.numCUs, 8) }, (_, cu) => {
                  const cuWork = workUnits.filter(w => w.cuId === cu).slice(0, 12);
                  const totalTime = Math.max(...workUnits.map(w => w.endTime), 100);

                  return (
                    <div key={cu} className="gantt-row">
                      <div className="gantt-label">CU {cu}</div>
                      <div className="gantt-bar-container">
                        {cuWork.map((work, idx) => (
                          <motion.div
                            key={work.workId}
                            className="gantt-bar"
                            style={{
                              width: `${((work.endTime - work.startTime) / totalTime) * 100}%`,
                              marginLeft: idx === 0 ? `${(work.startTime / totalTime) * 100}%` : 0,
                              background: work.stolen
                                ? `repeating-linear-gradient(45deg, ${getWorkColor(work.workId)}, ${getWorkColor(work.workId)} 5px, transparent 5px, transparent 10px)`
                                : getWorkColor(work.workId)
                            }}
                            initial={{ width: 0 }}
                            animate={{ width: `${((work.endTime - work.startTime) / totalTime) * 100}%` }}
                            transition={{ delay: idx * 0.05 }}
                          >
                            W{work.workId}
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  );
                })}
                {config.numCUs > 8 && (
                  <div style={{ color: '#888', fontSize: '12px', marginLeft: '70px' }}>
                    +{config.numCUs - 8} more CUs
                  </div>
                )}
              </div>

              <div className="code-block" style={{ marginTop: '20px' }}>
                <pre>{`# StreamK Scheduling
total_tiles = num_q_blocks * num_kv_blocks * num_heads
tiles_per_cu = ceil(total_tiles / num_cus)

for cu_id in range(num_cus):
    start_tile = cu_id * tiles_per_cu
    end_tile = min((cu_id + 1) * tiles_per_cu, total_tiles)
    process_tiles(start_tile, end_tile)`}</pre>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'work-stealing' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Work Stealing Mechanism</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                {config.enableWorkStealing ? 'Work stealing is ENABLED' : 'Work stealing is DISABLED'}
              </p>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h4>How Work Stealing Works</h4>
                <ol>
                  <li>CU completes all assigned tiles</li>
                  <li>Checks neighbor CUs for remaining work</li>
                  <li>Atomically claims tiles from busy CU's queue</li>
                  <li>Processes stolen tiles</li>
                </ol>
              </div>

              <div className="comparison-container">
                <div className="comparison-side left" style={{ background: config.enableWorkStealing ? '#e8f5e9' : '#f8f9fa' }}>
                  <h3 style={{ borderColor: config.enableWorkStealing ? '#4caf50' : '#ccc' }}>With Work Stealing</h3>
                  <ul style={{ listStyle: 'none', padding: 0 }}>
                    <li>Better load balancing</li>
                    <li>Higher GPU utilization</li>
                    <li>Handles variable tile costs</li>
                    <li>Atomic overhead</li>
                  </ul>
                </div>
                <div className="comparison-side right" style={{ background: !config.enableWorkStealing ? '#e3f2fd' : '#f8f9fa' }}>
                  <h3 style={{ borderColor: !config.enableWorkStealing ? '#2196f3' : '#ccc' }}>Without Work Stealing</h3>
                  <ul style={{ listStyle: 'none', padding: 0 }}>
                    <li>Static assignment</li>
                    <li>Lower overhead</li>
                    <li>Potential load imbalance</li>
                    <li>Simpler implementation</li>
                  </ul>
                </div>
              </div>
            </motion.div>
          </div>

          <div className={`tab-content ${activeTab === 'persistent' ? 'active' : ''}`}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flow-container">
              <h3>Persistent Kernel Design</h3>
              <p style={{ color: '#666', marginBottom: '20px' }}>
                Single kernel launch processes all attention tiles
              </p>

              <div className="info-panel" style={{ marginBottom: '20px' }}>
                <h4>Benefits</h4>
                <ul>
                  <li><strong>Reduced Launch Overhead:</strong> Single kernel launch vs many small launches</li>
                  <li><strong>Better Scheduling:</strong> GPU hardware scheduler manages work distribution</li>
                  <li><strong>Memory Efficiency:</strong> Thread blocks persist and reuse shared memory</li>
                  <li><strong>Synchronization:</strong> Global memory used for inter-CU coordination</li>
                </ul>
              </div>

              <div className="code-block">
                <pre>{`# Persistent Kernel Pattern
@triton.jit
def attention_persistent(Q, K, V, Out, ...):
    # Each thread block processes multiple tiles
    while True:
        # Atomically get next tile index
        tile_idx = atomic_add(tile_counter, 1)
        if tile_idx >= total_tiles:
            break

        # Decode tile coordinates
        q_block, kv_block, head = decode_tile(tile_idx)

        # Process tile
        process_attention_tile(Q, K, V, Out, q_block, kv_block, head)`}</pre>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      <ParameterGlossary
        entries={createLeanGlossaryEntries(config)}
        title="Lean Attention (StreamK) Parameter Reference"
        collapsible={true}
      />
    </div>
  );
};

export default LeanAttentionApp;
