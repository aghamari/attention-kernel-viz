import React from 'react';
import { motion } from 'framer-motion';
import { useUnifiedAttentionStore } from '../../store/unifiedAttentionStore';
import InfoBox from '../shared/InfoBox';

const Unified3DVisualization: React.FC = () => {
  const { config, segments, currentPhase } = useUnifiedAttentionStore();

  const numQBlocks = Math.ceil(config.seqLen / config.blockM);
  const displaySegments = Math.min(config.numSegments, 6);

  const getSegmentColor = (segIdx: number) => {
    const colors = ['#e94560', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140'];
    return colors[segIdx % colors.length];
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flow-container"
    >
      <h3>3D Kernel Visualization</h3>
      <p style={{ color: '#666', marginBottom: '20px' }}>
        Grid: (q_blocks={numQBlocks}, kv_heads={config.numKvHeads}, segments={config.numSegments}) = {numQBlocks * config.numKvHeads * config.numSegments} thread blocks
      </p>

      <InfoBox type="concept">
        <strong>3D Kernel Characteristics:</strong>
        <ul style={{ marginTop: '8px', marginLeft: '20px' }}>
          <li><strong>Segment parallelism:</strong> Divides KV sequence into {config.numSegments} segments for parallel processing</li>
          <li><strong>Each segment:</strong> Computes partial attention for KV range [{Math.floor(config.seqLen / config.numSegments * 0)} - {Math.floor(config.seqLen / config.numSegments * 1)}), etc.</li>
          <li><strong>Reduction required:</strong> Separate kernel combines segment results using global max/exp-sum</li>
          <li><strong>Best for:</strong> Long sequences (&gt;2048), maximum GPU utilization with {numQBlocks * config.numKvHeads * config.numSegments} parallel blocks</li>
          <li><strong>Trade-off:</strong> Higher parallelism but requires extra reduction step and intermediate buffers</li>
        </ul>
      </InfoBox>

      <div style={{ marginBottom: '30px' }}>
        <h4 style={{ marginBottom: '15px' }}>Segment Distribution</h4>
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          {Array.from({ length: displaySegments }, (_, segIdx) => (
            <motion.div
              key={segIdx}
              style={{
                flex: '1 1 200px',
                background: 'white',
                borderRadius: '8px',
                padding: '15px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                borderTop: `4px solid ${getSegmentColor(segIdx)}`
              }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: segIdx * 0.1 }}
            >
              <h4 style={{ color: getSegmentColor(segIdx), marginBottom: '10px' }}>
                Segment {segIdx}
              </h4>
              <div style={{ fontSize: '13px', color: '#666' }}>
                <p>KV Range: [{segments[segIdx]?.startIdx || 0} - {segments[segIdx]?.endIdx || 0})</p>
                <p>Partial Max: tracking</p>
                <p>Partial ExpSum: tracking</p>
              </div>
              <div style={{
                marginTop: '10px',
                height: '8px',
                background: '#f0f0f0',
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <motion.div
                  style={{
                    height: '100%',
                    background: getSegmentColor(segIdx),
                    borderRadius: '4px'
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: currentPhase === 'complete' ? '100%' : '0%' }}
                  transition={{ duration: 1, delay: segIdx * 0.2 }}
                />
              </div>
            </motion.div>
          ))}
        </div>
        {config.numSegments > 6 && (
          <div style={{ color: '#888', fontSize: '12px', marginTop: '10px' }}>
            +{config.numSegments - 6} more segments
          </div>
        )}
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h4 style={{ marginBottom: '15px' }}>3D Computation Flow</h4>
        <div className="flow-step">
          <div className="flow-step-number">1</div>
          <div className="flow-step-content">
            <h4>Parallel Segment Processing</h4>
            <p>Each segment computes partial attention for its KV range</p>
          </div>
        </div>
        <div className="flow-step">
          <div className="flow-step-number">2</div>
          <div className="flow-step-content">
            <h4>Track Partial Statistics</h4>
            <p>Each segment maintains partial max and exp-sum for softmax</p>
          </div>
        </div>
        <div className="flow-step">
          <div className="flow-step-number">3</div>
          <div className="flow-step-content">
            <h4>Segment Reduction</h4>
            <p>reduce_segments() combines partial results with proper rescaling</p>
          </div>
        </div>
      </div>

      <div className="comparison-container">
        <div className="comparison-side left">
          <h3>Per-Segment Outputs</h3>
          <div style={{ fontSize: '13px', fontFamily: 'monospace' }}>
            <p>partial_out[seg] = shape (q_blocks, heads, head_dim)</p>
            <p>partial_max[seg] = shape (q_blocks, heads)</p>
            <p>partial_exp_sum[seg] = shape (q_blocks, heads)</p>
          </div>
        </div>
        <div className="comparison-side right">
          <h3>After Reduction</h3>
          <div style={{ fontSize: '13px', fontFamily: 'monospace' }}>
            <p>final_out = combine(partial_out, partial_max, partial_exp_sum)</p>
            <p>global_max = max(partial_max)</p>
            <p>global_exp_sum = sum(partial_exp_sum * correction)</p>
          </div>
        </div>
      </div>

      <div className="code-block" style={{ marginTop: '20px' }}>
        <pre>{`# 3D Grid Launch
grid = (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)
block = (BLOCK_SIZE,)

# Each block processes:
# - Q[q_block_start:q_block_end, :]  # BLOCK_M queries
# - K, V[segment_start:segment_end]  # segment of KV

# After all segments complete:
reduce_segments(partial_out, partial_max, partial_exp_sum) -> final_out`}</pre>
      </div>
    </motion.div>
  );
};

export default Unified3DVisualization;
