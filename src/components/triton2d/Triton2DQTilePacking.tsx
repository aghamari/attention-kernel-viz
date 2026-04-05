import React from 'react';
import { motion } from 'framer-motion';
import { useTriton2DVizStore } from '../../store/triton2dVizStore';

const HEAD_COLORS = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#16a085'];

const Triton2DQTilePacking: React.FC = () => {
  const { qTile, config, selectedWG, currentStage, qTilePackingRow } = useTriton2DVizStore();
  if (!selectedWG || qTile.length === 0) return null;

  const packingToken = currentStage === 'q_tile_packing'
    ? Math.floor((qTilePackingRow + 1) / config.numQueriesPerKV)
    : config.BLOCK_Q;
  const showAll = currentStage !== 'q_tile_packing';

  return (
    <div style={{ padding: '20px', background: 'white', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }}>
      <h3 style={{ margin: '0 0 4px', color: '#333' }}>Step 1: Pack Q Tile (Head-Merge)</h3>

      {/* Explanation */}
      <div style={{
        padding: '12px 14px',
        background: '#e3f2fd',
        borderRadius: '8px',
        border: '1px solid #90caf9',
        marginBottom: '16px',
        fontSize: '13px',
        color: '#333',
        lineHeight: '1.6',
      }}>
        <strong>What's happening:</strong> Q_flat is 3D: <code>[total_tokens, nhead_q={config.nheadQ}, hdim={config.hdim}]</code>.
        Each (token, head) entry is a <strong>vector of {config.hdim} numbers</strong>, not a scalar.
        <br /><br />
        This workgroup picks the Q vectors for its tokens and heads, then <strong>stacks them
        row-by-row into a 2D tile</strong> of shape [BLOCK_M={config.BLOCK_M}, hdim={config.hdim}].
        KV head {selectedWG.kvHeadIdx} serves {config.numQueriesPerKV} Q heads
        (Q{selectedWG.kvHeadIdx * config.numQueriesPerKV}–Q{selectedWG.kvHeadIdx * config.numQueriesPerKV + config.numQueriesPerKV - 1}),
        so {config.BLOCK_Q} token slots × {config.numQueriesPerKV} heads = <strong>{config.BLOCK_M} rows</strong>.
        {selectedWG.qLen < config.BLOCK_Q &&
          ` This sequence has only ${selectedWG.qLen} token${selectedWG.qLen > 1 ? 's' : ''} — rows ${selectedWG.qLen * config.numQueriesPerKV}–${config.BLOCK_M - 1} are zero padding.`
        }
      </div>

      <div style={{ display: 'flex', gap: '24px', alignItems: 'flex-start' }}>
        {/* LEFT: Source -- Q_flat showing ALL heads, highlighting this WG's heads */}
        <div>
          <h4 style={{ margin: '0 0 4px', fontSize: '12px', color: '#555' }}>
            Q_flat[token, head, :] — all {config.nheadQ} heads
          </h4>
          <p style={{ margin: '0 0 8px', fontSize: '10px', color: '#999' }}>
            Each cell = [{config.hdim}]-dim vector. This WG gathers
            heads <strong>{selectedWG.kvHeadIdx * config.numQueriesPerKV}–{selectedWG.kvHeadIdx * config.numQueriesPerKV + config.numQueriesPerKV - 1}</strong> (KV head {selectedWG.kvHeadIdx}'s group).
          </p>
          {/* KV head group labels */}
          <div style={{ display: 'flex', marginLeft: '44px', marginBottom: '2px' }}>
            {Array.from({ length: config.nheadKV }, (_, kvh) => (
              <div key={kvh} style={{
                width: `${config.numQueriesPerKV * 42}px`,
                textAlign: 'center',
                fontSize: '8px',
                fontWeight: 600,
                color: kvh === selectedWG.kvHeadIdx ? '#333' : '#ccc',
                borderBottom: kvh === selectedWG.kvHeadIdx ? '2px solid #e94560' : '1px solid #eee',
                paddingBottom: '2px',
                marginRight: '4px',
              }}>
                KV head {kvh}{kvh === selectedWG.kvHeadIdx ? ' ← this WG' : ''}
              </div>
            ))}
          </div>
          {/* Column headers: ALL Q heads */}
          <div style={{ display: 'flex', marginLeft: '44px', marginBottom: '4px' }}>
            {Array.from({ length: config.nheadQ }, (_, qh) => {
              const ownerKVHead = Math.floor(qh / config.numQueriesPerKV);
              const isMyHead = ownerKVHead === selectedWG.kvHeadIdx;
              const hWithin = qh % config.numQueriesPerKV;
              return (
                <div key={qh} style={{
                  width: '40px',
                  textAlign: 'center',
                  fontSize: '8px',
                  color: isMyHead ? HEAD_COLORS[hWithin % HEAD_COLORS.length] : '#ccc',
                  fontWeight: isMyHead ? 600 : 400,
                }}>
                  Q{qh}
                </div>
              );
            })}
          </div>
          {/* Grid rows: tokens × ALL heads */}
          {Array.from({ length: config.BLOCK_Q }, (_, tok) => {
            const isRealToken = tok < selectedWG.qLen;
            const isActive = showAll ? isRealToken : (tok < packingToken);
            return (
              <div key={tok} style={{ display: 'flex', alignItems: 'center', marginBottom: '3px' }}>
                <span style={{
                  width: '40px',
                  fontSize: '10px',
                  fontFamily: 'monospace',
                  color: isRealToken ? '#333' : '#bbb',
                  textAlign: 'right',
                  paddingRight: '4px',
                }}>
                  {isRealToken ? `tok ${tok}` : 'empty'}
                </span>
                {Array.from({ length: config.nheadQ }, (_, qh) => {
                  const ownerKVHead = Math.floor(qh / config.numQueriesPerKV);
                  const isMyHead = ownerKVHead === selectedWG.kvHeadIdx;
                  const hWithin = qh % config.numQueriesPerKV;
                  const isGathered = isMyHead && isRealToken;
                  return (
                    <motion.div
                      key={qh}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{
                        opacity: isGathered && isActive ? 1 : (isRealToken && isMyHead ? 0.5 : 0.15),
                        scale: isGathered && isActive ? 1 : 0.9,
                      }}
                      transition={{ duration: 0.3 }}
                      style={{
                        width: '38px',
                        height: '28px',
                        margin: '0 1px',
                        background: isRealToken
                          ? (isMyHead ? HEAD_COLORS[hWithin % HEAD_COLORS.length] : '#d0d0d0')
                          : '#e8e8e8',
                        borderRadius: '4px',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '7px',
                        color: isMyHead && isRealToken ? 'white' : '#aaa',
                        fontWeight: 600,
                        border: isGathered && isActive ? '2px solid rgba(0,0,0,0.3)' : 'none',
                      }}
                    >
                      <span>Q{qh}</span>
                      {isRealToken && <span style={{ fontSize: '6px', opacity: 0.7 }}>[{config.hdim}d]</span>}
                    </motion.div>
                  );
                })}
              </div>
            );
          })}
        </div>

        {/* ARROW */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          fontSize: '20px',
          color: '#999',
          paddingTop: '24px',
        }}>
          →
        </div>

        {/* RIGHT: Packed tile grouped by token */}
        <div>
          <h4 style={{ margin: '0 0 8px', fontSize: '12px', color: '#555' }}>
            Packed: Q_tile [BLOCK_M={config.BLOCK_M}, hdim={config.hdim}]
          </h4>
          {Array.from({ length: config.BLOCK_Q }, (_, tok) => {
            const isReal = tok < selectedWG.qLen;
            const isActive = showAll ? true : (tok < packingToken);
            const startRow = tok * config.numQueriesPerKV;
            return (
              <div key={tok} style={{
                marginBottom: '6px',
                padding: '4px 6px',
                borderRadius: '6px',
                border: isReal
                  ? '2px solid rgba(0,0,0,0.15)'
                  : '1px dashed #ddd',
                background: isReal ? 'rgba(255,255,255,0.5)' : '#fafafa',
                opacity: isActive ? 1 : 0.2,
              }}>
                {/* Group label */}
                <div style={{
                  fontSize: '9px',
                  fontWeight: 600,
                  color: isReal ? '#555' : '#bbb',
                  marginBottom: '2px',
                }}>
                  {isReal ? `Token ${tok}` : 'Padding'} — rows {startRow}–{startRow + config.numQueriesPerKV - 1}
                </div>
                {/* Rows in this group */}
                {Array.from({ length: config.numQueriesPerKV }, (_, h) => {
                  const row = startRow + h;
                  const qRow = qTile[row];
                  if (!qRow) return null;
                  return (
                    <motion.div
                      key={h}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: isActive ? 1 : 0.15, x: 0 }}
                      transition={{ duration: 0.2, delay: showAll ? row * 0.01 : 0 }}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '3px',
                        height: '18px',
                        marginBottom: '1px',
                      }}
                    >
                      <span style={{
                        width: '20px',
                        fontSize: '8px',
                        color: '#999',
                        fontFamily: 'monospace',
                        textAlign: 'right',
                      }}>
                        r{row}
                      </span>
                      <span style={{
                        width: '16px',
                        fontSize: '8px',
                        color: HEAD_COLORS[h % HEAD_COLORS.length],
                        fontWeight: 600,
                      }}>
                        h{h}
                      </span>
                      {/* hdim values */}
                      <div style={{ display: 'flex', gap: '1px' }}>
                        {qRow.values.map((v, d) => (
                          <div key={d} style={{
                            width: '12px',
                            height: '14px',
                            background: qRow.isReal
                              ? HEAD_COLORS[h % HEAD_COLORS.length]
                              : '#e0e0e0',
                            borderRadius: '1px',
                            opacity: qRow.isReal ? (0.3 + Math.abs(v) * 0.7) : 0.2,
                            fontSize: '6px',
                            color: 'white',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                          }}>
                            {qRow.isReal ? v.toFixed(0) : ''}
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap', fontSize: '10px' }}>
        {Array.from({ length: config.numQueriesPerKV }, (_, h) => (
          <div key={h} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{
              width: '10px', height: '10px', borderRadius: '2px',
              background: HEAD_COLORS[h % HEAD_COLORS.length],
            }} />
            Head {h} (Q{selectedWG.kvHeadIdx * config.numQueriesPerKV + h})
          </div>
        ))}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#999' }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#e0e0e0' }} />
          Padding (zeros)
        </div>
      </div>
    </div>
  );
};

export default Triton2DQTilePacking;
