import React from 'react';
import { motion } from 'framer-motion';
import { ArrowDown } from 'lucide-react';
import { useUnifiedAttentionStore } from '../../store/unifiedAttentionStore';
import InfoBox from '../shared/InfoBox';

const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes.toFixed(0)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
};

const AttentionMemoryLayout: React.FC = () => {
  const { config, memoryLayout } = useUnifiedAttentionStore();

  const totalGlobalMemory = memoryLayout.globalMemory.reduce((sum, item) => sum + item.bytes, 0);
  const totalSharedMemory = memoryLayout.sharedMemory.reduce((sum, item) => sum + item.bytes, 0);

  return (
    <div style={{ padding: '20px' }}>
      <h3 style={{ marginBottom: '20px' }}>Memory Layout & Data Movement</h3>

      <InfoBox type="concept">
        <strong>GPU Memory Hierarchy:</strong>
        <ul style={{ marginTop: '8px', marginLeft: '20px' }}>
          <li><strong>Global Memory:</strong> Large but slow (~1 TB/s). Stores all input/output tensors.</li>
          <li><strong>Shared Memory:</strong> Small but fast (~19 TB/s). Shared within a thread block. Used for tiling.</li>
          <li><strong>Registers:</strong> Tiny but fastest (~100 TB/s). Private to each thread. Used for computations.</li>
        </ul>
      </InfoBox>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '30px',
        marginTop: '30px'
      }}>
        {/* Global Memory */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          style={{
            padding: '20px',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: '12px',
            color: '#fff'
          }}
        >
          <h4 style={{ marginBottom: '15px', display: 'flex', justifyContent: 'space-between' }}>
            <span>Global Memory (HBM)</span>
            <span style={{ fontSize: '14px', fontWeight: 'normal' }}>
              Total: {formatBytes(totalGlobalMemory)}
            </span>
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
            {memoryLayout.globalMemory.map((item, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: idx * 0.05 }}
                style={{
                  padding: '15px',
                  background: 'rgba(255, 255, 255, 0.15)',
                  borderRadius: '8px',
                  backdropFilter: 'blur(10px)'
                }}
              >
                <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{item.name}</div>
                <div style={{ fontSize: '12px', opacity: 0.9 }}>{item.shape}</div>
                <div style={{ fontSize: '14px', marginTop: '5px', fontWeight: 'bold' }}>
                  {formatBytes(item.bytes)}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <ArrowDown size={32} color="#666" />
        </div>

        {/* Shared Memory */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          style={{
            padding: '20px',
            background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
            borderRadius: '12px',
            color: '#fff'
          }}
        >
          <h4 style={{ marginBottom: '15px', display: 'flex', justifyContent: 'space-between' }}>
            <span>Shared Memory (Per Thread Block)</span>
            <span style={{ fontSize: '14px', fontWeight: 'normal' }}>
              Total: {formatBytes(totalSharedMemory)} / 48 KB limit
            </span>
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px' }}>
            {memoryLayout.sharedMemory.map((item, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 + idx * 0.05 }}
                style={{
                  padding: '15px',
                  background: 'rgba(255, 255, 255, 0.15)',
                  borderRadius: '8px',
                  backdropFilter: 'blur(10px)'
                }}
              >
                <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{item.name}</div>
                <div style={{ fontSize: '14px', marginTop: '5px', fontWeight: 'bold' }}>
                  {formatBytes(item.bytes)}
                </div>
              </motion.div>
            ))}
          </div>
          <InfoBox type="note" color="#fff">
            <span style={{ color: '#333' }}>
              Shared memory usage: {((totalSharedMemory / (48 * 1024)) * 100).toFixed(1)}% of 48 KB limit.
              {totalSharedMemory > 48 * 1024 && (
                <strong style={{ color: '#f44336' }}> ⚠️ EXCEEDS LIMIT - kernel may not launch!</strong>
              )}
            </span>
          </InfoBox>
        </motion.div>

        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <ArrowDown size={32} color="#666" />
        </div>

        {/* Registers */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.4 }}
          style={{
            padding: '20px',
            background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            borderRadius: '12px',
            color: '#fff'
          }}
        >
          <h4 style={{ marginBottom: '15px' }}>Registers (Per Thread)</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px' }}>
            {memoryLayout.registers.map((item, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.4 + idx * 0.05 }}
                style={{
                  padding: '15px',
                  background: 'rgba(255, 255, 255, 0.15)',
                  borderRadius: '8px',
                  backdropFilter: 'blur(10px)'
                }}
              >
                <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{item.name}</div>
                <div style={{ fontSize: '14px', marginTop: '5px', fontWeight: 'bold' }}>
                  {item.count} registers
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Data Movement Pipeline */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        style={{
          marginTop: '40px',
          padding: '20px',
          background: '#f9f9f9',
          borderRadius: '8px',
          border: '2px solid #ddd'
        }}
      >
        <h4 style={{ marginBottom: '15px' }}>Data Movement Pipeline</h4>
        <div style={{ fontSize: '13px', lineHeight: '2', fontFamily: 'monospace' }}>
          <div style={{ padding: '10px', background: '#fff', borderRadius: '6px', marginBottom: '10px' }}>
            <strong>1. Load Q/K tiles from Global → Shared:</strong>
            <div style={{ marginLeft: '20px', color: '#666' }}>
              global_mem[q_offset] → shared_mem[Q_tile] ({formatBytes(config.blockM * config.headDim * 4)})
              <br />
              global_mem[k_offset] → shared_mem[K_tile] ({formatBytes(config.blockN * config.headDim * 4)})
            </div>
          </div>

          <div style={{ padding: '10px', background: '#fff', borderRadius: '6px', marginBottom: '10px' }}>
            <strong>2. Load fragments from Shared → Registers:</strong>
            <div style={{ marginLeft: '20px', color: '#666' }}>
              Each thread loads a small fragment (e.g., 8×8) for computation
            </div>
          </div>

          <div style={{ padding: '10px', background: '#fff', borderRadius: '6px', marginBottom: '10px' }}>
            <strong>3. Compute in Registers:</strong>
            <div style={{ marginLeft: '20px', color: '#666' }}>
              Matmul (Q @ K^T), Softmax, Matmul (Scores @ V)
              <br />
              All intermediate values stay in registers (fastest!)
            </div>
          </div>

          <div style={{ padding: '10px', background: '#fff', borderRadius: '6px', marginBottom: '10px' }}>
            <strong>4. Store partial results to Shared:</strong>
            <div style={{ marginLeft: '20px', color: '#666' }}>
              Accumulate results in shared memory for reduction
            </div>
          </div>

          <div style={{ padding: '10px', background: '#fff', borderRadius: '6px' }}>
            <strong>5. Write final output to Global:</strong>
            <div style={{ marginLeft: '20px', color: '#666' }}>
              shared_mem[output_tile] → global_mem[output_offset]
            </div>
          </div>
        </div>
      </motion.div>

      {/* Tiling Benefits */}
      <InfoBox type="important">
        <strong>Why Tiling Matters:</strong>
        <ul style={{ marginTop: '8px', marginLeft: '20px' }}>
          <li>Global memory: ~1 TB/s bandwidth</li>
          <li>Shared memory: ~19 TB/s bandwidth (19× faster!)</li>
          <li>Registers: ~100 TB/s effective (100× faster!)</li>
          <li>By loading tiles into shared memory and reusing them, we minimize slow global memory accesses</li>
          <li>Block size {config.blockM}×{config.blockN} means each tile is reused {config.blockM} times for Q and {config.blockN} times for K</li>
        </ul>
      </InfoBox>
    </div>
  );
};

export default AttentionMemoryLayout;
