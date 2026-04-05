import React from 'react';
import { motion } from 'framer-motion';
import { useTriton2DVizStore } from '../../store/triton2dVizStore';

const SEQ_COLORS = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'];

const Triton2DInputData: React.FC = () => {
  const { simData, currentStage, currentTileIdx, kvTileSteps, selectedWG } = useTriton2DVizStore();
  if (!simData) return null;
  const { config, cuSeqlensQ, cumQBlocks, blockTable, kvCache } = simData;

  const isKVStage = currentStage === 'kv_tile' && kvTileSteps.length > 0 && selectedWG;
  const activeStep = isKVStage ? kvTileSteps[Math.min(currentTileIdx, kvTileSteps.length - 1)] : null;
  const activePhysPage = activeStep?.physPage ?? -1;
  const activePageCol = activeStep?.pageCol ?? -1;
  const activeSeqIdx = selectedWG?.seqIdx ?? -1;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* seq_lens_k + cu_seqlens_q */}
      <div style={{ padding: '12px', background: '#f8f9fa', borderRadius: '8px' }}>
        <h4 style={{ margin: '0 0 8px', fontSize: '13px', color: '#555' }}>Sequence Lengths</h4>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {config.seqLensK.map((kl, i) => (
            <div key={i} style={{
              padding: '4px 10px',
              background: SEQ_COLORS[i % SEQ_COLORS.length],
              borderRadius: '6px',
              color: 'white',
              fontSize: '12px',
              fontWeight: 600,
            }}>
              Seq {i}: q={config.seqLensQ[i]}, kv={kl}
            </div>
          ))}
        </div>
        <div style={{ marginTop: '8px', fontFamily: 'monospace', fontSize: '11px', color: '#666' }}>
          cu_seqlens_q = [{cuSeqlensQ.join(', ')}]
          <br />
          cum_q_blocks = [{cumQBlocks.join(', ')}]
        </div>
      </div>

      {/* Q_flat */}
      <div style={{ padding: '12px', background: '#f8f9fa', borderRadius: '8px' }}>
        <h4 style={{ margin: '0 0 8px', fontSize: '13px', color: '#555' }}>
          Q_flat [{simData.qFlat.length}, {config.nheadQ}, {config.hdim}]
        </h4>
        <div style={{ display: 'flex', gap: '2px' }}>
          {simData.qFlat.map((_, tIdx) => {
            const seqIdx = cuSeqlensQ.findIndex((cs, i) => i < cuSeqlensQ.length - 1 && tIdx >= cs && tIdx < cuSeqlensQ[i + 1]);
            return (
              <motion.div
                key={tIdx}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: tIdx * 0.05 }}
                style={{
                  width: '32px',
                  height: '28px',
                  background: SEQ_COLORS[seqIdx % SEQ_COLORS.length],
                  borderRadius: '4px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '10px',
                  fontWeight: 600,
                }}
              >
                T{tIdx}
              </motion.div>
            );
          })}
        </div>
        <div style={{ fontSize: '10px', color: '#999', marginTop: '4px' }}>
          Each token has {config.nheadQ} Q heads × {config.hdim} dims
        </div>
      </div>

      {/* block_table */}
      <div style={{ padding: '12px', background: '#f8f9fa', borderRadius: '8px' }}>
        <h4 style={{ margin: '0 0 8px', fontSize: '13px', color: '#555' }}>
          block_table [seq → physical pages]
        </h4>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {blockTable.table.map((pages, seq) => (
            <div key={seq} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span style={{
                width: '40px',
                fontSize: '11px',
                color: SEQ_COLORS[seq % SEQ_COLORS.length],
                fontWeight: 600,
              }}>
                Seq {seq}:
              </span>
              {pages.map((pageId, col) => {
                const isActive = isKVStage && seq === activeSeqIdx && col === activePageCol;
                return (
                  <motion.div
                    key={col}
                    animate={{
                      scale: isActive ? 1.15 : 1,
                      boxShadow: isActive ? '0 0 8px 2px rgba(233, 69, 96, 0.6)' : 'none',
                    }}
                    transition={{ duration: 0.3 }}
                    style={{
                      width: '32px',
                      height: '24px',
                      background: SEQ_COLORS[seq % SEQ_COLORS.length],
                      borderRadius: '4px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '10px',
                      fontWeight: 600,
                      border: isActive ? '2px solid #e94560' : '2px solid transparent',
                    }}
                    title={`Seq ${seq}, page_col ${col} → phys page ${pageId} (tokens ${col * config.pageSize}-${Math.min((col + 1) * config.pageSize - 1, config.seqLensK[seq] - 1)})`}
                  >
                    P{pageId}
                  </motion.div>
                );
              })}
              <span style={{ fontSize: '10px', color: '#999' }}>
                ({config.seqLensK[seq]} tok)
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* kv_cache pages */}
      <div style={{ padding: '12px', background: '#f8f9fa', borderRadius: '8px' }}>
        <h4 style={{ margin: '0 0 8px', fontSize: '13px', color: '#555' }}>
          kv_cache [{kvCache.numPages} pages, {config.pageSize} slots, {config.nheadKV} heads, {config.hdim} dim]
        </h4>
        <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
          {Array.from({ length: Math.min(kvCache.numPages, 16) }, (_, pageId) => {
            const ownerSeq = blockTable.table.findIndex(pages => pages.includes(pageId));
            const isUsed = ownerSeq >= 0;
            const isActive = isKVStage && pageId === activePhysPage;
            return (
              <motion.div
                key={pageId}
                animate={{
                  scale: isActive ? 1.2 : 1,
                  boxShadow: isActive ? '0 0 8px 2px rgba(233, 69, 96, 0.6)' : 'none',
                }}
                transition={{ duration: 0.3 }}
                style={{
                  width: '36px',
                  height: '28px',
                  background: isUsed ? SEQ_COLORS[ownerSeq % SEQ_COLORS.length] : '#e0e0e0',
                  borderRadius: '4px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: isUsed ? 'white' : '#999',
                  fontSize: '10px',
                  fontWeight: 600,
                  opacity: isActive ? 1 : (isUsed ? 0.7 : 0.3),
                  border: isActive ? '2px solid #e94560' : '2px solid transparent',
                }}
                title={isUsed ? `Page ${pageId} → Seq ${ownerSeq}` : `Page ${pageId} (free)`}
              >
                P{pageId}
              </motion.div>
            );
          })}
        </div>
        <div style={{ fontSize: '10px', color: '#999', marginTop: '4px' }}>
          {isKVStage
            ? <span style={{ color: '#e94560', fontWeight: 600 }}>Reading page {activePhysPage} (highlighted)</span>
            : 'Pages colored by owning sequence. Gray = free.'
          }
        </div>
      </div>
    </div>
  );
};

export default Triton2DInputData;
