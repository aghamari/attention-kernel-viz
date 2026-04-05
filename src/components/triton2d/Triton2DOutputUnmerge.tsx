import React from 'react';
import { motion } from 'framer-motion';
import { useTriton2DVizStore } from '../../store/triton2dVizStore';

const HEAD_COLORS = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#16a085'];

const Triton2DOutputUnmerge: React.FC = () => {
  const { qTile, selectedWG, config, currentStage, kvTileSteps } = useTriton2DVizStore();
  if (!selectedWG || qTile.length === 0) return null;
  if (currentStage !== 'output' && currentStage !== 'complete') return null;

  const lastStep = kvTileSteps[kvTileSteps.length - 1];
  if (!lastStep) return null;

  return (
    <div style={{ padding: '20px', background: 'white', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }}>
      <h3 style={{ margin: '0 0 4px', color: '#333' }}>Output: Un-merge Heads</h3>
      <p style={{ fontSize: '12px', color: '#666', margin: '0 0 12px' }}>
        Each row's accumulator is normalized (acc[row] / l[row]) and written
        back to out[q_start + tok, q_head].
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
        {qTile.filter(r => r.isReal).map((row, i) => (
          <motion.div
            key={row.row}
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.08 }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 10px',
              background: '#f8f9fa',
              borderRadius: '6px',
              borderLeft: `4px solid ${HEAD_COLORS[row.head % HEAD_COLORS.length]}`,
            }}
          >
            <span style={{ fontSize: '11px', fontFamily: 'monospace', color: '#555', width: '40px' }}>
              row {row.row}
            </span>

            <span style={{ fontSize: '11px', color: '#999', width: '24px' }}>→</span>

            <span style={{
              fontSize: '11px',
              fontFamily: 'monospace',
              fontWeight: 600,
              color: '#333',
            }}>
              out[tok {row.tok}, Q head {row.qHeadGlobal}]
            </span>

            <span style={{ fontSize: '10px', color: '#999', marginLeft: 'auto' }}>
              acc[{row.row}] / l[{row.row}] = normalized output
            </span>

            {lastStep && (
              <span style={{
                fontSize: '10px',
                fontFamily: 'monospace',
                color: '#666',
                background: '#e8e8e8',
                padding: '2px 6px',
                borderRadius: '4px',
              }}>
                l={lastStep.lAfter[row.row]?.toFixed(2) ?? '?'}
              </span>
            )}
          </motion.div>
        ))}
      </div>

      {currentStage === 'complete' && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          style={{
            marginTop: '16px',
            padding: '12px 16px',
            background: '#e8f5e9',
            borderRadius: '8px',
            border: '2px solid #4caf50',
            textAlign: 'center',
            fontSize: '14px',
            fontWeight: 600,
            color: '#2e7d32',
          }}
        >
          Attention complete for Seq {selectedWG.seqIdx}, KV Head {selectedWG.kvHeadIdx}
          — {qTile.filter(r => r.isReal).length} outputs written
        </motion.div>
      )}
    </div>
  );
};

export default Triton2DOutputUnmerge;
