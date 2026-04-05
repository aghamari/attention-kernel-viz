import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Check, X, Info } from 'lucide-react';

interface FeatureMatrixProps {
  onNavigate?: (kernel: string) => void;
}

interface FeatureSupport {
  supported: boolean;
  note?: string;
}

const FeatureMatrix: React.FC<FeatureMatrixProps> = ({ onNavigate }) => {
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  const kernels = [
    { id: 'unified-2d', name: 'Triton 2D', color: '#e94560' },
    { id: 'unified-3d', name: 'Triton 3D', color: '#0f3460' },
    { id: 'ck-ua', name: 'CK-UA', color: '#FF5722' },
    { id: 'ck-sk', name: 'CK-SK', color: '#00BCD4' },
    { id: 'ck-pk', name: 'CK-PK', color: '#E91E63' },
    { id: 'ck-fwd', name: 'CK-Fwd', color: '#3F51B5' }
  ];

  const features: { name: string; description: string; matrix: { [key: string]: FeatureSupport } }[] = [
    {
      name: 'Head-merge',
      description: 'Pack multiple Q heads into M dimension for single MFMA',
      matrix: {
        'unified-2d': { supported: true },
        'unified-3d': { supported: true },
        'ck-ua': { supported: true, note: 'Core optimization' },
        'ck-sk': { supported: true, note: 'Decode only' },
        'ck-pk': { supported: false },
        'ck-fwd': { supported: false }
      }
    },
    {
      name: 'Paged KV Cache',
      description: 'Block-based KV cache with indirection table',
      matrix: {
        'unified-2d': { supported: true },
        'unified-3d': { supported: true },
        'ck-ua': { supported: true },
        'ck-sk': { supported: true },
        'ck-pk': { supported: true },
        'ck-fwd': { supported: false }
      }
    },
    {
      name: 'KV Splitting',
      description: 'Splits KV sequence across multiple workgroups for parallelism',
      matrix: {
        'unified-2d': { supported: false },
        'unified-3d': { supported: true, note: 'Via segments' },
        'ck-ua': { supported: false },
        'ck-sk': { supported: true, note: 'Via num_splits' },
        'ck-pk': { supported: false },
        'ck-fwd': { supported: false }
      }
    },
    {
      name: 'Combine Kernel',
      description: 'Separate reduction kernel to merge split-KV partial results',
      matrix: {
        'unified-2d': { supported: false },
        'unified-3d': { supported: true, note: 'reduce_segments' },
        'ck-ua': { supported: false },
        'ck-sk': { supported: true, note: 'Log-sum-exp merge' },
        'ck-pk': { supported: false },
        'ck-fwd': { supported: false }
      }
    },
    {
      name: 'Sliding Window',
      description: 'Attend only to recent K tokens within window',
      matrix: {
        'unified-2d': { supported: true },
        'unified-3d': { supported: true },
        'ck-ua': { supported: false },
        'ck-sk': { supported: true },
        'ck-pk': { supported: true },
        'ck-fwd': { supported: true }
      }
    },
    {
      name: 'Causal Mask',
      description: 'Autoregressive masking (Q[i] attends to K[≤i])',
      matrix: {
        'unified-2d': { supported: true },
        'unified-3d': { supported: true },
        'ck-ua': { supported: true },
        'ck-sk': { supported: true },
        'ck-pk': { supported: true },
        'ck-fwd': { supported: true }
      }
    },
    {
      name: 'Softcap',
      description: 'Soft-clamp logits before softmax (tanh-based)',
      matrix: {
        'unified-2d': { supported: true },
        'unified-3d': { supported: true },
        'ck-ua': { supported: false },
        'ck-sk': { supported: true },
        'ck-pk': { supported: true },
        'ck-fwd': { supported: true }
      }
    },
    {
      name: 'ALiBi',
      description: 'Attention with Linear Biases positional encoding',
      matrix: {
        'unified-2d': { supported: true },
        'unified-3d': { supported: true },
        'ck-ua': { supported: false },
        'ck-sk': { supported: false },
        'ck-pk': { supported: false },
        'ck-fwd': { supported: false }
      }
    },
    {
      name: 'Sinks',
      description: 'Preserve attention to initial tokens in sliding window',
      matrix: {
        'unified-2d': { supported: true },
        'unified-3d': { supported: true },
        'ck-ua': { supported: false },
        'ck-sk': { supported: true },
        'ck-pk': { supported: true },
        'ck-fwd': { supported: true }
      }
    },
    {
      name: 'FP8 Output',
      description: 'Quantize output to FP8 for memory efficiency',
      matrix: {
        'unified-2d': { supported: true },
        'unified-3d': { supported: false },
        'ck-ua': { supported: false },
        'ck-sk': { supported: false },
        'ck-pk': { supported: false },
        'ck-fwd': { supported: false }
      }
    }
  ];

  const dataTypes: { [key: string]: string } = {
    'unified-2d': 'fp16/bf16',
    'unified-3d': 'fp16/bf16',
    'ck-ua': 'fp16/bf16',
    'ck-sk': 'fp16/bf16/fp8',
    'ck-pk': 'fp16/bf16',
    'ck-fwd': 'fp16/bf16/fp8'
  };

  const hdimSupport: { [key: string]: string } = {
    'unified-2d': 'Any',
    'unified-3d': 'Any',
    'ck-ua': '64, 128',
    'ck-sk': '32-256',
    'ck-pk': '32-256',
    'ck-fwd': '32-256'
  };

  const utilizationData = [
    { kernel: 'CK-PK (bm0=128)', tile: '[128, 64]', useful: '1', util: '0.8%', wgs: '8 (one per Q head)', highlight: false },
    { kernel: 'CK-UA tiny (kBlockM=16)', tile: '[16, 64]', useful: '4', util: '25%', wgs: '2 (one per KV head)', highlight: true },
    { kernel: 'Triton 2D (BLOCK_M=16)', tile: '[16, 64]', useful: '4', util: '25%', wgs: '2 (one per KV head)', highlight: true },
    { kernel: 'PA Decode V1', tile: 'per-head dot', useful: 'N/A', util: 'N/A', wgs: '2 (GQA) or 8 (MHA)', highlight: false }
  ];

  return (
    <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      <motion.h2
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        style={{
          fontSize: '32px',
          fontWeight: 700,
          marginBottom: '10px',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}
      >
        Attention Kernel Feature Matrix
      </motion.h2>
      <p style={{ color: '#888', marginBottom: '30px' }}>
        Compare capabilities across Triton and CK-Tile attention kernels
      </p>

      <div style={{ overflowX: 'auto' }}>
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            background: 'rgba(255, 255, 255, 0.03)',
            borderRadius: '12px',
            overflow: 'hidden'
          }}
        >
          <thead>
            <tr style={{ background: 'rgba(255, 255, 255, 0.05)' }}>
              <th
                style={{
                  padding: '16px',
                  textAlign: 'left',
                  fontWeight: 600,
                  borderBottom: '2px solid rgba(255, 255, 255, 0.1)'
                }}
              >
                Feature
              </th>
              {kernels.map((kernel) => (
                <th
                  key={kernel.id}
                  style={{
                    padding: '16px',
                    textAlign: 'center',
                    fontWeight: 600,
                    borderBottom: '2px solid rgba(255, 255, 255, 0.1)',
                    cursor: onNavigate ? 'pointer' : 'default',
                    transition: 'background 0.2s'
                  }}
                  onClick={() => onNavigate?.(kernel.id)}
                >
                  <div style={{ color: kernel.color, fontWeight: 700 }}>{kernel.name}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {features.map((feature, idx) => (
              <motion.tr
                key={feature.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05 }}
                style={{
                  borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                  background:
                    selectedFeature === feature.name ? 'rgba(102, 126, 234, 0.1)' : 'transparent',
                  transition: 'background 0.2s'
                }}
                onMouseEnter={() => setSelectedFeature(feature.name)}
                onMouseLeave={() => setSelectedFeature(null)}
              >
                <td style={{ padding: '16px' }}>
                  <div style={{ fontWeight: 500 }}>{feature.name}</div>
                  <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
                    {feature.description}
                  </div>
                </td>
                {kernels.map((kernel) => {
                  const support = feature.matrix[kernel.id];
                  return (
                    <td
                      key={kernel.id}
                      style={{
                        padding: '16px',
                        textAlign: 'center'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                        {support.supported ? (
                          <Check size={20} color="#4ade80" strokeWidth={3} />
                        ) : (
                          <X size={20} color="#f87171" strokeWidth={2} />
                        )}
                        {support.note && (
                          <div
                            style={{
                              position: 'relative',
                              display: 'inline-block'
                            }}
                            title={support.note}
                          >
                            <Info size={14} color="#888" />
                          </div>
                        )}
                      </div>
                      {support.note && (
                        <div style={{ fontSize: '11px', color: '#888', marginTop: '4px' }}>
                          {support.note}
                        </div>
                      )}
                    </td>
                  );
                })}
              </motion.tr>
            ))}

            {/* Data Types Row */}
            <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)', background: 'rgba(255, 255, 255, 0.02)' }}>
              <td style={{ padding: '16px' }}>
                <div style={{ fontWeight: 500 }}>Data Types</div>
                <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
                  Supported precision formats
                </div>
              </td>
              {kernels.map((kernel) => (
                <td key={kernel.id} style={{ padding: '16px', textAlign: 'center' }}>
                  <code style={{ fontSize: '12px', color: '#4ade80' }}>{dataTypes[kernel.id]}</code>
                </td>
              ))}
            </tr>

            {/* Head Dimension Row */}
            <tr style={{ background: 'rgba(255, 255, 255, 0.02)' }}>
              <td style={{ padding: '16px' }}>
                <div style={{ fontWeight: 500 }}>Compiled Head Dims</div>
                <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
                  Supported head dimension sizes
                </div>
              </td>
              {kernels.map((kernel) => (
                <td key={kernel.id} style={{ padding: '16px', textAlign: 'center' }}>
                  <code style={{ fontSize: '12px', color: '#60a5fa' }}>{hdimSupport[kernel.id]}</code>
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      {/* MFMA Utilization Section */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        style={{
          marginTop: '30px',
          padding: '20px',
          background: 'rgba(255, 255, 255, 0.03)',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }}
      >
        <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '16px' }}>
          MFMA Utilization for Decode (1 token, GQA-4)
        </h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid rgba(255, 255, 255, 0.1)' }}>
                <th style={{ padding: '10px', textAlign: 'left', fontSize: '14px' }}>Kernel</th>
                <th style={{ padding: '10px', textAlign: 'left', fontSize: '14px' }}>Q Tile</th>
                <th style={{ padding: '10px', textAlign: 'left', fontSize: '14px' }}>Useful Rows</th>
                <th style={{ padding: '10px', textAlign: 'left', fontSize: '14px' }}>Utilization</th>
                <th style={{ padding: '10px', textAlign: 'left', fontSize: '14px' }}>WGs per seq</th>
              </tr>
            </thead>
            <tbody>
              {utilizationData.map((row) => (
                <tr
                  key={row.kernel}
                  style={{
                    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                    background: row.highlight ? 'rgba(76, 175, 80, 0.08)' : 'transparent'
                  }}
                >
                  <td style={{ padding: '10px', fontWeight: 500, fontSize: '14px' }}>{row.kernel}</td>
                  <td style={{ padding: '10px', fontSize: '14px', fontFamily: 'monospace' }}>{row.tile}</td>
                  <td style={{ padding: '10px', fontSize: '14px' }}>{row.useful}</td>
                  <td style={{
                    padding: '10px',
                    fontSize: '14px',
                    fontWeight: 600,
                    color: row.highlight ? '#4ade80' : (row.util === 'N/A' ? '#888' : '#f87171')
                  }}>
                    {row.util}
                  </td>
                  <td style={{ padding: '10px', fontSize: '14px' }}>{row.wgs}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Legend */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        style={{
          marginTop: '20px',
          padding: '20px',
          background: 'rgba(102, 126, 234, 0.1)',
          borderRadius: '12px',
          border: '1px solid rgba(102, 126, 234, 0.2)'
        }}
      >
        <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '12px' }}>Legend</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Check size={18} color="#4ade80" strokeWidth={3} />
            <span style={{ fontSize: '14px' }}>Feature supported</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <X size={18} color="#f87171" strokeWidth={2} />
            <span style={{ fontSize: '14px' }}>Feature not supported</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Info size={16} color="#888" />
            <span style={{ fontSize: '14px' }}>Additional notes available (hover)</span>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default FeatureMatrix;
