import React from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft } from 'lucide-react';
import FeatureMatrix from '../components/shared/FeatureMatrix';
import { AttentionType } from '../types/attention';

interface FeatureMatrixAppProps {
  onBack: () => void;
  onNavigate?: (type: AttentionType) => void;
}

const FeatureMatrixApp: React.FC<FeatureMatrixAppProps> = ({ onBack, onNavigate }) => {
  const handleNavigate = (kernelId: string) => {
    const kernelMap: { [key: string]: AttentionType } = {
      'unified-2d': 'unified',
      'unified-3d': 'unified',
      'ck-ua': 'ck-ua',
      'ck-sk': 'ck-sk',
      'ck-pk': 'ck-pk',
      'ck-fwd': 'ck-fwd'
    };

    const type = kernelMap[kernelId];
    if (type && onNavigate) {
      onNavigate(type);
    }
  };

  return (
    <div className="app-container" style={{ minHeight: '100vh', padding: '20px' }}>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        style={{ marginBottom: '20px' }}
      >
        <button
          onClick={onBack}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '10px 20px',
            background: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '8px',
            color: 'white',
            fontSize: '14px',
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
          }}
        >
          <ArrowLeft size={16} />
          Back to Kernels
        </button>
      </motion.div>

      <FeatureMatrix onNavigate={handleNavigate} />

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        style={{
          marginTop: '40px',
          padding: '24px',
          background: 'rgba(255, 255, 255, 0.03)',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }}
      >
        <h3 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '16px' }}>About This Comparison</h3>
        <div style={{ display: 'grid', gap: '16px', fontSize: '14px', lineHeight: '1.6', color: '#ccc' }}>
          <p>
            <strong>Triton Kernels:</strong> Implemented in Python using Triton, with automatic code generation and tuning.
            Support a wide range of features including ALiBi, FP8 output, and comprehensive masking options.
          </p>
          <p>
            <strong>CK-Tile Kernels:</strong> Implemented in C++ using AMD's Composable Kernel library, optimized for RDNA/CDNA architectures.
            Feature MFMA (Matrix Fused Multiply-Add) instructions and advanced GPU-specific optimizations.
          </p>
          <p>
            <strong>CK-UA:</strong> Unique head-merging optimization packs multiple query heads into the M dimension of GEMM tiles,
            improving efficiency for GQA workloads. Uses tier-based tile sizing (Tiny/BS32/Small/Medium/Large) selected by avg_q heuristic.
          </p>
          <p>
            <strong>CK-SK:</strong> Split-KV variant divides KV sequence across workgroups with a separate combine kernel.
            Optimal for long sequences where num_splits heuristic targets multiProcessorCount × 4 workgroups.
          </p>
          <p>
            <strong>Data from:</strong> attention_pipelines.md documentation from AIter codebase, covering Triton unified attention
            and CK-Tile FMHA variants (UA/SK/PK/Fwd).
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default FeatureMatrixApp;
