import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, XCircle } from 'lucide-react';
import { useUnifiedAttentionStore } from '../../store/unifiedAttentionStore';
import { shouldUse2DKernel } from '../../types/attention';

const KernelSelectionLogic: React.FC = () => {
  const { config } = useUnifiedAttentionStore();

  const isShortSequence = config.seqLen <= 2048;
  const isSlidingWindow = config.useSlidingWindow;
  const isDecodeOnly = config.seqLen === 1;
  const recommendUse2D = shouldUse2DKernel(config);

  const conditions = [
    {
      label: 'Short Sequence (seqLen <= 2048)',
      met: isShortSequence,
      value: `seqLen = ${config.seqLen}`
    },
    {
      label: 'Sliding Window Enabled',
      met: isSlidingWindow,
      value: isSlidingWindow ? `window = ${config.windowSize}` : 'Disabled'
    },
    {
      label: 'Decode-Only (seqLen = 1)',
      met: isDecodeOnly,
      value: isDecodeOnly ? 'Yes' : 'No'
    }
  ];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="selection-logic"
    >
      <h3 style={{ marginBottom: '20px' }}>Kernel Selection Heuristic</h3>

      <div className="info-panel" style={{ marginBottom: '20px' }}>
        <h4>use_2d_kernel() Decision Function</h4>
        <p style={{ marginTop: '10px', fontSize: '14px' }}>
          The kernel automatically selects between 2D and 3D based on workload characteristics.
          Use 2D when: short sequences OR sliding window OR decode-only batch.
        </p>
      </div>

      <div className="decision-tree">
        {conditions.map((cond, idx) => (
          <motion.div
            key={idx}
            className={`decision-node condition`}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: idx * 0.1 }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              {cond.met ? (
                <CheckCircle size={20} color="#4caf50" />
              ) : (
                <XCircle size={20} color="#f44336" />
              )}
              <div>
                <div style={{ fontWeight: 500 }}>{cond.label}</div>
                <div style={{ fontSize: '13px', color: '#666' }}>{cond.value}</div>
              </div>
            </div>
          </motion.div>
        ))}

        <motion.div
          style={{
            padding: '15px',
            textAlign: 'center',
            fontSize: '20px',
            color: '#666'
          }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          {conditions.some(c => c.met) ? 'OR condition met' : 'No conditions met'}
        </motion.div>

        <motion.div
          className={`decision-node ${recommendUse2D ? 'result-2d' : 'result-3d'}`}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            <div style={{
              width: '60px',
              height: '60px',
              borderRadius: '50%',
              background: recommendUse2D ? '#2196f3' : '#4caf50',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: '24px',
              fontWeight: 'bold'
            }}>
              {recommendUse2D ? '2D' : '3D'}
            </div>
            <div>
              <div style={{ fontWeight: 600, fontSize: '18px' }}>
                Recommended: {recommendUse2D ? '2D Kernel' : '3D Kernel'}
              </div>
              <div style={{ fontSize: '14px', color: '#666', marginTop: '5px' }}>
                {recommendUse2D
                  ? 'Single-pass, no segment reduction overhead'
                  : 'Segment-parallel for maximum GPU utilization'}
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      <div className="comparison-container" style={{ marginTop: '30px' }}>
        <div className="comparison-side left">
          <h3>2D Kernel Benefits</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li style={{ padding: '8px 0' }}>Lower memory overhead</li>
            <li style={{ padding: '8px 0' }}>No reduction kernel needed</li>
            <li style={{ padding: '8px 0' }}>Better for short sequences</li>
            <li style={{ padding: '8px 0' }}>Simpler synchronization</li>
          </ul>
        </div>
        <div className="comparison-side right">
          <h3>3D Kernel Benefits</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li style={{ padding: '8px 0' }}>Higher parallelism</li>
            <li style={{ padding: '8px 0' }}>Better CU utilization</li>
            <li style={{ padding: '8px 0' }}>Scales with sequence length</li>
            <li style={{ padding: '8px 0' }}>Optimal for long prefill</li>
          </ul>
        </div>
      </div>

      <div className="code-block" style={{ marginTop: '20px' }}>
        <pre>{`def use_2d_kernel(config):
    """Heuristic for 2D vs 3D kernel selection"""
    is_short_sequence = config.seq_len <= 2048
    is_sliding_window = config.use_sliding_window
    is_decode_only = config.seq_len == 1

    return is_short_sequence or is_sliding_window or is_decode_only`}</pre>
      </div>
    </motion.div>
  );
};

export default KernelSelectionLogic;
