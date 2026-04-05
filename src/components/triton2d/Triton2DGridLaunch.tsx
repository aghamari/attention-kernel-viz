import React from 'react';
import { motion } from 'framer-motion';
import { useTriton2DVizStore } from '../../store/triton2dVizStore';

const SEQ_COLORS = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'];

const Triton2DGridLaunch: React.FC = () => {
  const { simData, selectedWG, selectWorkgroup } = useTriton2DVizStore();
  if (!simData) return null;
  const { config, totalQBlocks, workgroups, cumQBlocks } = simData;

  return (
    <div style={{ padding: '20px', background: 'white', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }}>
      <h3 style={{ margin: '0 0 8px', color: '#333' }}>Grid Launch</h3>
      <p style={{ fontSize: '12px', color: '#666', margin: '0 0 16px' }}>
        dim3({config.nheadKV} kv_heads, {totalQBlocks} q_blocks) = {config.nheadKV * totalQBlocks} workgroups.
        Click a cell to select.
      </p>

      <div style={{ overflowX: 'auto' }}>
        {/* Column headers: kv_head */}
        <div style={{ display: 'flex', gap: '3px', marginLeft: '80px', marginBottom: '4px' }}>
          {Array.from({ length: config.nheadKV }, (_, kv) => (
            <div key={kv} style={{
              width: '70px',
              textAlign: 'center',
              fontSize: '11px',
              fontWeight: 600,
              color: '#555',
            }}>
              KV Head {kv}
            </div>
          ))}
        </div>

        {/* Grid rows: q_block_global */}
        {Array.from({ length: totalQBlocks }, (_, gIdx) => {
          const seqIdx = cumQBlocks.findIndex((_, i) => i < cumQBlocks.length - 1 && gIdx >= cumQBlocks[i] && gIdx < cumQBlocks[i + 1]);
          const localBlock = gIdx - cumQBlocks[seqIdx];
          return (
            <div key={gIdx} style={{ display: 'flex', gap: '3px', marginBottom: '3px' }}>
              <div style={{
                width: '76px',
                fontSize: '10px',
                color: '#666',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                paddingRight: '4px',
              }}>
                blk {gIdx} (S{seqIdx}.{localBlock})
              </div>
              {Array.from({ length: config.nheadKV }, (_, kvHead) => {
                const isSelected = selectedWG?.kvHeadIdx === kvHead && selectedWG?.qBlockGlobal === gIdx;
                return (
                  <motion.div
                    key={kvHead}
                    whileHover={{ scale: 1.08 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => selectWorkgroup(kvHead, gIdx)}
                    style={{
                      width: '70px',
                      height: '34px',
                      background: isSelected
                        ? 'linear-gradient(135deg, #e94560 0%, #0f3460 100%)'
                        : SEQ_COLORS[seqIdx % SEQ_COLORS.length],
                      borderRadius: '6px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '10px',
                      fontWeight: isSelected ? 700 : 500,
                      cursor: 'pointer',
                      border: isSelected ? '2px solid #fff' : '1px solid rgba(255,255,255,0.3)',
                      boxShadow: isSelected ? '0 0 8px rgba(233,69,96,0.5)' : 'none',
                    }}
                  >
                    S{seqIdx} H{kvHead}
                  </motion.div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Triton2DGridLaunch;
