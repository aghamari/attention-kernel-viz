import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowDown, Play, RotateCcw } from 'lucide-react';
import { SegmentInfo } from '../../types/attention';

interface SegmentReductionProps {
  segments: SegmentInfo[];
}

const SegmentReduction: React.FC<SegmentReductionProps> = ({ segments }) => {
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const displaySegments = segments.slice(0, 4);

  const runAnimation = () => {
    setIsAnimating(true);
    setCurrentStep(0);

    let step = 0;
    const interval = setInterval(() => {
      step++;
      setCurrentStep(step);
      if (step >= 4) {
        clearInterval(interval);
        setIsAnimating(false);
      }
    }, 1000);
  };

  const reset = () => {
    setIsAnimating(false);
    setCurrentStep(0);
  };

  const getSegmentColor = (idx: number) => {
    const colors = ['#e94560', '#4facfe', '#43e97b', '#fa709a'];
    return colors[idx % colors.length];
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flow-container"
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h3>Segment Reduction</h3>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn btn-primary" onClick={runAnimation} disabled={isAnimating}>
            <Play size={14} /> Animate
          </button>
          <button className="btn btn-secondary" onClick={reset}>
            <RotateCcw size={14} /> Reset
          </button>
        </div>
      </div>

      <div className="info-panel" style={{ marginBottom: '20px' }}>
        <h4>reduce_segments() Algorithm</h4>
        <ol>
          <li>Find global maximum across all segment partial maxes</li>
          <li>Compute correction factors: exp(partial_max - global_max)</li>
          <li>Rescale partial outputs and exp-sums</li>
          <li>Sum rescaled outputs and divide by global exp-sum</li>
        </ol>
      </div>

      {/* Segment partial results */}
      <div style={{ marginBottom: '30px' }}>
        <h4 style={{ marginBottom: '15px' }}>Partial Results from Each Segment</h4>
        <div className="segment-container">
          {displaySegments.map((seg, idx) => (
            <motion.div
              key={idx}
              className="segment-block"
              style={{ borderTop: `4px solid ${getSegmentColor(idx)}` }}
              initial={{ opacity: 0, y: 20 }}
              animate={{
                opacity: currentStep >= 1 ? 0.5 : 1,
                y: 0,
                scale: currentStep >= 1 ? 0.95 : 1
              }}
              transition={{ delay: idx * 0.1 }}
            >
              <h4>Segment {idx}</h4>
              <div className="segment-values">
                <div className="segment-value">max: {(Math.random() * 2 - 1).toFixed(3)}</div>
                <div className="segment-value">exp_sum: {(Math.random() * 10).toFixed(3)}</div>
                <div className="segment-value">out: [...]</div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Reduction arrows */}
      <AnimatePresence>
        {currentStep >= 1 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{ textAlign: 'center', marginBottom: '20px' }}
          >
            <div style={{ display: 'flex', justifyContent: 'center', gap: '40px' }}>
              {displaySegments.map((_, idx) => (
                <motion.div
                  key={idx}
                  initial={{ y: -20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: idx * 0.1 }}
                >
                  <ArrowDown size={24} color={getSegmentColor(idx)} />
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Reduction steps */}
      <AnimatePresence>
        {currentStep >= 2 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{
              background: '#fff3e0',
              padding: '20px',
              borderRadius: '8px',
              marginBottom: '20px',
              border: '2px solid #ff9800'
            }}
          >
            <h4 style={{ color: '#e65100', marginBottom: '10px' }}>Step 1: Find Global Max</h4>
            <div className="code-block" style={{ background: '#1e1e1e', padding: '10px' }}>
              <code>global_max = max(seg0.max, seg1.max, seg2.max, seg3.max) = 1.234</code>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {currentStep >= 3 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{
              background: '#e3f2fd',
              padding: '20px',
              borderRadius: '8px',
              marginBottom: '20px',
              border: '2px solid #2196f3'
            }}
          >
            <h4 style={{ color: '#1565c0', marginBottom: '10px' }}>Step 2: Compute Corrections</h4>
            <div className="code-block" style={{ background: '#1e1e1e', padding: '10px' }}>
              <code>{`correction[i] = exp(partial_max[i] - global_max)
rescaled_out[i] = partial_out[i] * correction[i]
rescaled_exp_sum[i] = partial_exp_sum[i] * correction[i]`}</code>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Final output */}
      <AnimatePresence>
        {currentStep >= 4 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              background: 'linear-gradient(135deg, #e94560 0%, #0f3460 100%)',
              padding: '20px',
              borderRadius: '8px',
              color: 'white',
              textAlign: 'center'
            }}
          >
            <h4 style={{ marginBottom: '10px' }}>Final Output</h4>
            <div style={{ fontFamily: 'monospace', fontSize: '14px' }}>
              output = sum(rescaled_out) / sum(rescaled_exp_sum)
            </div>
            <div style={{ marginTop: '10px', fontSize: '13px', opacity: 0.9 }}>
              Numerically stable combination of all segment contributions
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="code-block" style={{ marginTop: '20px' }}>
        <pre>{`def reduce_segments(partial_out, partial_max, partial_exp_sum):
    # Find global maximum for numerical stability
    global_max = partial_max.max(dim=0)

    # Compute correction factors
    correction = exp(partial_max - global_max)

    # Rescale and sum
    rescaled_out = (partial_out * correction.unsqueeze(-1)).sum(dim=0)
    rescaled_exp_sum = (partial_exp_sum * correction).sum(dim=0)

    # Final output
    return rescaled_out / rescaled_exp_sum.unsqueeze(-1)`}</pre>
      </div>
    </motion.div>
  );
};

export default SegmentReduction;
