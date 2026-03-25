import React from 'react';
import { motion } from 'framer-motion';
import { useUnifiedAttentionStore } from '../../store/unifiedAttentionStore';
import InfoBox from '../shared/InfoBox';

const Unified2DVisualization: React.FC = () => {
  const { config, currentPhase } = useUnifiedAttentionStore();

  const numQBlocks = Math.ceil(config.seqLen / config.blockM);
  const displayBlocks = Math.min(numQBlocks, 8);
  const displayHeads = Math.min(config.numKvHeads, 6);

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'qk': return '#ff6b6b';
      case 'softmax': return '#4ecdc4';
      case 'av': return '#45b7d1';
      case 'complete': return '#98d85b';
      default: return '#e0e0e0';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flow-container"
    >
      <h3>2D Kernel Visualization</h3>
      <p style={{ color: '#666', marginBottom: '20px' }}>
        Grid: (num_kv_heads={config.numKvHeads}, total_q_blocks={numQBlocks}) = {config.numKvHeads * numQBlocks} thread blocks
      </p>

      <InfoBox type="concept">
        <strong>2D Kernel Characteristics:</strong>
        <ul style={{ marginTop: '8px', marginLeft: '20px' }}>
          <li><strong>Single-pass:</strong> No intermediate buffers or reduction needed</li>
          <li><strong>Online softmax:</strong> Computes softmax on-the-fly as K/V tiles are processed</li>
          <li><strong>Best for:</strong> Short sequences (≤2048), decode phase, sliding window attention</li>
          <li><strong>Work distribution:</strong> Each thread block processes {config.blockM} queries against all {config.seqLen} keys/values</li>
          <li><strong>Memory efficient:</strong> Only stores running max and sum for online softmax</li>
        </ul>
      </InfoBox>

      <div style={{ marginBottom: '30px' }}>
        <h4 style={{ marginBottom: '15px' }}>Grid Launch Pattern</h4>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {Array.from({ length: displayHeads }, (_, head) => (
            <div key={head} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ width: '80px', fontSize: '12px', color: '#666' }}>
                KV Head {head}:
              </div>
              <div style={{ display: 'flex', gap: '4px' }}>
                {Array.from({ length: displayBlocks }, (_, block) => (
                  <motion.div
                    key={block}
                    style={{
                      width: '50px',
                      height: '35px',
                      background: `linear-gradient(135deg, ${getPhaseColor(currentPhase)} 0%, #0f3460 100%)`,
                      borderRadius: '4px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '11px',
                      fontWeight: 500
                    }}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: (head * displayBlocks + block) * 0.03 }}
                  >
                    Q{block * config.blockM}-{Math.min((block + 1) * config.blockM - 1, config.seqLen - 1)}
                  </motion.div>
                ))}
                {numQBlocks > 8 && (
                  <div style={{
                    width: '40px',
                    height: '35px',
                    background: '#ddd',
                    borderRadius: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#666',
                    fontSize: '11px'
                  }}>
                    +{numQBlocks - 8}
                  </div>
                )}
              </div>
            </div>
          ))}
          {config.numKvHeads > 6 && (
            <div style={{ color: '#888', fontSize: '12px', marginLeft: '80px' }}>
              ... and {config.numKvHeads - 6} more KV heads
            </div>
          )}
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h4 style={{ marginBottom: '15px' }}>Computation Flow</h4>
        <div className="flow-step">
          <div className="flow-step-number">1</div>
          <div className="flow-step-content">
            <h4>Load Q Block</h4>
            <p>Load BLOCK_M rows of Q matrix into shared memory</p>
          </div>
        </div>
        <div className="flow-step">
          <div className="flow-step-number">2</div>
          <div className="flow-step-content">
            <h4>Tile Over KV</h4>
            <p>Process K,V in tiles of BLOCK_N with online softmax</p>
          </div>
        </div>
        <div className="flow-step">
          <div className="flow-step-number">3</div>
          <div className="flow-step-content">
            <h4>Output</h4>
            <p>Write final attention output directly (no reduction needed)</p>
          </div>
        </div>
      </div>

      <div className="code-block">
        <pre>{`# 2D Grid Launch
grid = (num_kv_heads, total_num_q_blocks)
block = (BLOCK_SIZE,)

# Each block processes:
# - Q[q_block_start:q_block_end, :]  # BLOCK_M queries
# - K, V for single KV head          # all KV tokens`}</pre>
      </div>
    </motion.div>
  );
};

export default Unified2DVisualization;
