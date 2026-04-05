import React from 'react';
import { motion } from 'framer-motion';
import { useTriton2DVizStore } from '../../store/triton2dVizStore';

const Triton2DMainLoop: React.FC = () => {
  const { kvTileSteps, currentTileIdx, selectedWG, simData, config, currentStage } = useTriton2DVizStore();
  if (!selectedWG || !simData || kvTileSteps.length === 0) return null;
  if (currentStage !== 'kv_tile' && currentStage !== 'output' && currentStage !== 'complete') return null;

  const step = kvTileSteps[Math.min(currentTileIdx, kvTileSteps.length - 1)];
  const numTiles = Math.ceil(selectedWG.kvLen / config.BLOCK_N);

  const scoreColorScale = (v: number, masked: boolean) => {
    if (masked || v === -Infinity) return '#f0f0f0';
    const norm = Math.min(Math.max((v + 2) / 4, 0), 1);
    const r = Math.round(255 * (1 - norm));
    const g = Math.round(100 + 155 * norm);
    const b = Math.round(255 * norm);
    return `rgb(${r},${g},${b})`;
  };

  return (
    <div style={{ padding: '20px', background: 'white', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }}>
      <h3 style={{ margin: '0 0 8px', color: '#333' }}>
        Step 2: Compute Attention Scores
      </h3>

      {/* Tile timeline: shows what happens at each tile */}
      <div style={{
        display: 'flex',
        gap: '6px',
        marginBottom: '16px',
        alignItems: 'stretch',
      }}>
        {Array.from({ length: numTiles }, (_, t) => {
          const tileStart = t * config.BLOCK_N;
          const tileEnd = Math.min(tileStart + config.BLOCK_N, selectedWG.kvLen);
          const realCount = tileEnd - tileStart;
          const isDone = t < currentTileIdx;
          const isCurrent = t === currentTileIdx;
          return (
            <div key={t} style={{
              flex: 1,
              padding: '6px 8px',
              background: isCurrent ? '#e94560' : isDone ? '#4caf50' : '#f0f0f0',
              borderRadius: '6px',
              color: isCurrent || isDone ? 'white' : '#888',
              fontSize: '9px',
              lineHeight: '1.5',
              fontWeight: isCurrent ? 600 : 400,
            }}>
              <div style={{ fontWeight: 600, fontSize: '10px' }}>
                {isDone ? '✓' : isCurrent ? '→' : ''} Tile {t}
              </div>
              <div>S{t} [{config.BLOCK_M}×{config.BLOCK_N}]</div>
              <div>KV {tileStart}–{tileEnd - 1}{realCount < config.BLOCK_N ? ` (${realCount} real)` : ''}</div>
              <div style={{ fontSize: '8px', opacity: 0.8 }}>
                {isDone ? 'm,l,acc updated → S discarded'
                  : isCurrent ? 'computing now...'
                  : 'pending'}
              </div>
            </div>
          );
        })}
        {/* Final output */}
        <div style={{
          flex: 1,
          padding: '6px 8px',
          background: currentStage === 'output' || currentStage === 'complete' ? '#0f3460' : '#f0f0f0',
          borderRadius: '6px',
          color: currentStage === 'output' || currentStage === 'complete' ? 'white' : '#bbb',
          fontSize: '9px',
          lineHeight: '1.5',
        }}>
          <div style={{ fontWeight: 600, fontSize: '10px' }}>Output</div>
          <div>acc / l</div>
          <div>= softmax(S_full) @ V</div>
          <div style={{ fontSize: '8px', opacity: 0.8 }}>
            [{config.BLOCK_M}×{config.hdim}]
          </div>
        </div>
      </div>

      {/* Page lookup: block_table + kv_cache side by side */}
      <div style={{
        padding: '14px',
        background: '#fff3e0',
        borderRadius: '8px',
        border: '1px solid #ffcc80',
        marginBottom: '12px',
      }}>
        <div style={{ fontSize: '12px', fontWeight: 600, color: '#555', marginBottom: '10px' }}>
          Page Lookup: need KV tokens {step.kvStart}–{step.kvEnd - 1}
        </div>

        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
          {/* LEFT: block_table row */}
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '10px', color: '#888', marginBottom: '4px', fontWeight: 600 }}>
              block_table[seq {selectedWG.seqIdx}]
            </div>
            <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap', marginBottom: '8px' }}>
              {simData.blockTable.table[selectedWG.seqIdx].map((pageId, col) => {
                const isActive = col === step.pageCol;
                return (
                  <motion.div
                    key={col}
                    animate={{ scale: isActive ? 1.1 : 1 }}
                    style={{
                      width: '38px',
                      height: '30px',
                      background: isActive ? '#e94560' : '#ffe0b2',
                      borderRadius: '4px',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '8px',
                      fontWeight: 600,
                      color: isActive ? 'white' : '#666',
                      border: isActive ? '2px solid #c62828' : '1px solid #ffcc80',
                    }}
                  >
                    <span style={{ fontSize: '9px' }}>P{pageId}</span>
                    <span style={{ fontSize: '7px', opacity: 0.7 }}>
                      col {col}
                    </span>
                  </motion.div>
                );
              })}
            </div>
            {/* Read result with slot math */}
            <div style={{
              padding: '8px 10px',
              background: '#e94560',
              borderRadius: '6px',
              color: 'white',
              fontSize: '10px',
              fontWeight: 600,
              lineHeight: '1.6',
            }}>
              page_col = {step.kvStart} ÷ {config.pageSize} = <strong>{step.pageCol}</strong>
              <br />
              block_table[{selectedWG.seqIdx}][{step.pageCol}] → page <strong>{step.physPage}</strong>
            </div>
          </div>

          {/* Arrow */}
          <div style={{ display: 'flex', alignItems: 'center', paddingTop: '20px', fontSize: '18px', color: '#e94560' }}>
            →
          </div>

          {/* RIGHT: kv_cache pages */}
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '10px', color: '#888', marginBottom: '4px', fontWeight: 600 }}>
              kv_cache (physical pages)
            </div>
            <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap', marginBottom: '8px' }}>
              {Array.from({ length: Math.min(simData.kvCache.numPages, 12) }, (_, pageId) => {
                const ownerSeq = simData.blockTable.table.findIndex(pages => pages.includes(pageId));
                const isActive = pageId === step.physPage;
                const isUsed = ownerSeq >= 0;
                return (
                  <motion.div
                    key={pageId}
                    animate={{ scale: isActive ? 1.15 : 1 }}
                    style={{
                      width: '30px',
                      height: '24px',
                      background: isActive ? '#e94560'
                        : isUsed ? 'rgba(0,0,0,0.12)' : '#f0f0f0',
                      borderRadius: '3px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '8px',
                      fontWeight: isActive ? 700 : 500,
                      color: isActive ? 'white' : (isUsed ? '#555' : '#bbb'),
                      border: isActive ? '2px solid #c62828' : '1px solid #e0e0e0',
                    }}
                  >
                    P{pageId}
                  </motion.div>
                );
              })}
            </div>
            {/* What's being read with slot math */}
            <div style={{
              padding: '8px 10px',
              background: '#e94560',
              borderRadius: '6px',
              color: 'white',
              fontSize: '10px',
              fontWeight: 600,
              lineHeight: '1.6',
            }}>
              {(() => {
                const firstSlot = step.kvStart % config.pageSize;
                const numReal = step.kvEnd - step.kvStart;
                const lastSlot = firstSlot + numReal - 1;
                return (
                  <>
                    slot = token_pos % page_size
                    <br />
                    first: {step.kvStart} % {config.pageSize} = <strong>{firstSlot}</strong>,
                    {' '}last: {step.kvEnd - 1} % {config.pageSize} = <strong>{lastSlot}</strong>
                    <br />
                    kv_cache[page {step.physPage}, slots {firstSlot}–{lastSlot}, <strong>head {selectedWG.kvHeadIdx}</strong>]
                    <br />
                    <span style={{ fontSize: '9px', opacity: 0.7 }}>
                      (page has all {config.nheadKV} heads, but this WG only reads head {selectedWG.kvHeadIdx}'s slice)
                    </span>
                    {numReal < config.BLOCK_N && <span style={{ opacity: 0.7 }}> — {numReal} real, {config.BLOCK_N - numReal} zeros</span>}
                  </>
                );
              })()}
            </div>
          </div>
        </div>
      </div>

      {/* K tile: how it's filled from kv_cache */}
      <div style={{
        padding: '12px',
        background: '#e3f2fd',
        borderRadius: '8px',
        border: '1px solid #90caf9',
        marginBottom: '12px',
      }}>
        <div style={{ fontSize: '12px', fontWeight: 600, color: '#1565c0', marginBottom: '4px' }}>
          K_tile [{config.BLOCK_N} KV positions × {config.hdim} head dims] — head {selectedWG.kvHeadIdx} only
        </div>
        <div style={{ fontSize: '10px', color: '#666', marginBottom: '6px' }}>
          Read from kv_cache[page {step.physPage}].
          The page stores all {config.nheadKV} KV heads, but we only extract head {selectedWG.kvHeadIdx}'s key vectors.
          <br />
          {config.BLOCK_N} rows = BLOCK_N (KV token positions).
          {' '}{config.hdim} cols = hdim (dimension per key vector).
          {config.BLOCK_N === config.hdim && <span style={{ color: '#e94560' }}> Both happen to be {config.hdim} here — they're unrelated!</span>}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          {step.kTile.map((row, t) => {
            const kvPos = step.kvStart + t;
            const isReal = kvPos < selectedWG.kvLen;
            return (
              <div key={t} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{
                  width: '60px',
                  fontSize: '9px',
                  fontFamily: 'monospace',
                  color: isReal ? '#1565c0' : '#bbb',
                  textAlign: 'right',
                }}>
                  {isReal ? `kv pos ${kvPos}` : 'beyond kv_len'}
                </span>
                <span style={{ fontSize: '9px', color: '#999', width: '40px' }}>
                  slot {kvPos % config.pageSize}
                </span>
                <div style={{ display: 'flex', gap: '1px' }}>
                  {row.map((v, d) => (
                    <div key={d} style={{
                      width: '14px',
                      height: '14px',
                      background: isReal ? '#1565c0' : '#e0e0e0',
                      borderRadius: '2px',
                      opacity: isReal ? (0.3 + Math.abs(v) * 0.7) : 0.2,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '6px',
                      color: 'white',
                    }}>
                      {isReal ? v.toFixed(0) : ''}
                    </div>
                  ))}
                </div>
                <span style={{ fontSize: '8px', color: '#999' }}>
                  [{config.hdim}d]
                </span>
              </div>
            );
          })}
        </div>
        <div style={{ fontSize: '10px', color: '#888', marginTop: '6px' }}>
          Each row = one KV token's key vector [{config.hdim} dims], read from page {step.physPage}.
          {step.kvEnd < step.kvStart + config.BLOCK_N &&
            ` Last ${config.BLOCK_N - (step.kvEnd - step.kvStart)} rows are zeros (past kv_len=${selectedWG.kvLen}).`
          }
        </div>
      </div>

      {/* Matrix multiply visualization */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '8px',
        padding: '10px',
        background: '#f3e5f5',
        borderRadius: '8px',
        border: '1px solid #ce93d8',
        marginBottom: '12px',
        flexWrap: 'wrap',
      }}>
        <div style={{
          padding: '6px 10px',
          background: '#7b1fa2',
          borderRadius: '6px',
          color: 'white',
          fontSize: '11px',
          fontWeight: 600,
          textAlign: 'center',
        }}>
          Q_tile<br />
          <span style={{ fontSize: '9px', opacity: 0.8 }}>[{config.BLOCK_M} tok×heads × {config.hdim} hdim]</span>
        </div>
        <span style={{ fontSize: '16px', color: '#7b1fa2', fontWeight: 700 }}>@</span>
        <div style={{
          padding: '6px 10px',
          background: '#1565c0',
          borderRadius: '6px',
          color: 'white',
          fontSize: '11px',
          fontWeight: 600,
          textAlign: 'center',
        }}>
          K_tile.T<br />
          <span style={{ fontSize: '9px', opacity: 0.8 }}>[{config.hdim} hdim × {config.BLOCK_N} kv_pos]</span>
        </div>
        <span style={{ fontSize: '16px', color: '#555', fontWeight: 700 }}>/ √{config.hdim}</span>
        <span style={{ fontSize: '16px', color: '#555', fontWeight: 700 }}>=</span>
        <div style={{
          padding: '6px 10px',
          background: '#e94560',
          borderRadius: '6px',
          color: 'white',
          fontSize: '11px',
          fontWeight: 600,
          textAlign: 'center',
        }}>
          S<br />
          <span style={{ fontSize: '9px', opacity: 0.8 }}>[{config.BLOCK_M} tok×heads × {config.BLOCK_N} kv_pos]</span>
        </div>
      </div>

      {/* Score matrix S */}
      <div style={{ marginBottom: '12px' }}>
        <h4 style={{ margin: '0 0 6px', fontSize: '13px', color: '#555' }}>
          Result: S [{config.BLOCK_M} × {config.BLOCK_N}]
        </h4>
        <p style={{ fontSize: '11px', color: '#888', margin: '0 0 6px' }}>
          Each cell S[row, col] = dot(Q_tile[row], K[col]) / √{config.hdim}.
          Gray "—" = masked (-inf). Always {config.BLOCK_M}×{config.BLOCK_N}, same size every tile.
        </p>
        <div style={{ overflowX: 'auto' }}>
          {/* Column header: KV cache positions */}
          <div style={{ marginLeft: '80px', marginBottom: '2px' }}>
            <div style={{
              fontSize: '10px',
              color: '#e94560',
              fontWeight: 600,
              marginBottom: '2px',
              maxWidth: `${config.BLOCK_N * 32}px`,
              textAlign: 'center',
              borderBottom: '2px solid #e94560',
              paddingBottom: '3px',
            }}>
              ← BLOCK_N = {config.BLOCK_N} KV positions (of {selectedWG.kvLen} total) →
            </div>
            <div style={{ display: 'flex' }}>
              {Array.from({ length: config.BLOCK_N }, (_, c) => (
                <div key={c} style={{
                  width: '32px',
                  fontSize: '9px',
                  color: step.kvStart + c < selectedWG.kvLen ? '#555' : '#ccc',
                  textAlign: 'center',
                  fontFamily: 'monospace',
                }}>
                  {step.kvStart + c}
                </div>
              ))}
            </div>
          </div>
          <div style={{ display: 'flex' }}>
            {/* BLOCK_M label */}
            <div style={{
              width: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <span style={{
                writingMode: 'vertical-rl',
                transform: 'rotate(180deg)',
                fontSize: '9px',
                fontWeight: 600,
                color: '#0f3460',
                whiteSpace: 'nowrap',
              }}>
                BLOCK_M = {config.BLOCK_M}
              </span>
            </div>
            <div>
              {/* Score grid with token group brackets */}
              {Array.from({ length: config.BLOCK_Q }, (_, tok) => {
                const startRow = tok * config.numQueriesPerKV;
                const isReal = tok < selectedWG.qLen;
                return (
                  <div key={tok} style={{
                    display: 'flex',
                    alignItems: 'stretch',
                    marginBottom: tok < config.BLOCK_Q - 1 ? '3px' : '0',
                  }}>
                    {/* Token group label */}
                    <div style={{
                      width: '22px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderLeft: `3px solid ${isReal ? '#0f3460' : '#ddd'}`,
                      borderTop: `1px solid ${isReal ? '#0f3460' : '#ddd'}`,
                      borderBottom: `1px solid ${isReal ? '#0f3460' : '#ddd'}`,
                      borderRadius: '3px 0 0 3px',
                    }}>
                      <span style={{
                        writingMode: 'vertical-rl',
                        transform: 'rotate(180deg)',
                        fontSize: '7px',
                        fontWeight: 600,
                        color: isReal ? '#0f3460' : '#ccc',
                        whiteSpace: 'nowrap',
                      }}>
                        {isReal ? `t${tok}` : 'pad'}
                      </span>
                    </div>
                    {/* Rows for this token */}
                    <div>
                      {Array.from({ length: config.numQueriesPerKV }, (_, h) => {
                        const r = startRow + h;
                        const scoreRow = step.scores[r];
                        if (!scoreRow) return null;
                        return (
                          <div key={h} style={{ display: 'flex', alignItems: 'center', marginBottom: '1px' }}>
                            <span style={{
                              width: '38px',
                              fontSize: '8px',
                              color: '#999',
                              textAlign: 'right',
                              paddingRight: '4px',
                              fontFamily: 'monospace',
                            }}>
                              r{r} h{h}
                            </span>
                            {scoreRow.map((score, c) => (
                              <motion.div
                                key={c}
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ delay: (r * config.BLOCK_N + c) * 0.002 }}
                                style={{
                                  width: '30px',
                                  height: '16px',
                                  margin: '0 1px',
                                  background: scoreColorScale(score, step.masked[r][c]),
                                  borderRadius: '2px',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  fontSize: '7px',
                                  color: step.masked[r][c] ? '#ccc' : '#fff',
                                  fontWeight: 500,
                                }}
                                title={step.masked[r][c] ? 'masked (-inf)' : `score: ${score}`}
                              >
                                {step.masked[r][c] ? '—' : score.toFixed(1)}
                              </motion.div>
                            ))}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Online softmax state */}
      <div style={{
        padding: '10px 14px',
        background: '#f3e5f5',
        borderRadius: '8px',
        border: '1px solid #ce93d8',
        marginBottom: '12px',
        fontSize: '12px',
        color: '#555',
      }}>
        <strong>Online Softmax:</strong> Instead of storing all scores and computing softmax at the end,
        we maintain a running max (m) and exp-sum (l) per row. After each tile, these get updated.
        When all tiles are done, the final output = acc / l.
      </div>
      <div style={{ display: 'flex', gap: '16px' }}>
        <div style={{ flex: 1 }}>
          <h4 style={{ margin: '0 0 6px', fontSize: '12px', color: '#555' }}>
            Running Max (m) — tracks highest score seen so far
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
            {step.mAfter.slice(0, config.numQueriesPerKV * Math.min(config.BLOCK_Q, selectedWG.qLen)).map((m, r) => (
              <div key={r} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '20px', fontSize: '8px', color: '#999' }}>r{r}</span>
                <div style={{
                  height: '10px',
                  width: `${Math.max(0, Math.min((m + 3) / 6 * 100, 100))}%`,
                  maxWidth: '120px',
                  background: '#e94560',
                  borderRadius: '3px',
                }} />
                <span style={{ fontSize: '9px', color: '#666', fontFamily: 'monospace' }}>
                  {m === -Infinity ? '-∞' : m.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <h4 style={{ margin: '0 0 6px', fontSize: '12px', color: '#555' }}>
            Running Exp-Sum (l) — denominator for softmax normalization
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
            {step.lAfter.slice(0, config.numQueriesPerKV * Math.min(config.BLOCK_Q, selectedWG.qLen)).map((l, r) => (
              <div key={r} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '20px', fontSize: '8px', color: '#999' }}>r{r}</span>
                <div style={{
                  height: '10px',
                  width: `${Math.min(l / 10 * 100, 100)}%`,
                  maxWidth: '120px',
                  background: '#0f3460',
                  borderRadius: '3px',
                }} />
                <span style={{ fontSize: '9px', color: '#666', fontFamily: 'monospace' }}>
                  {l.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Triton2DMainLoop;
