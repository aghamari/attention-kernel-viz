import React, { useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, SkipForward, SkipBack, RotateCcw } from 'lucide-react';
import { useTriton2DVizStore, VizStage } from '../store/triton2dVizStore';
import Triton2DInputData from '../components/triton2d/Triton2DInputData';
import Triton2DGridLaunch from '../components/triton2d/Triton2DGridLaunch';
import Triton2DQTilePacking from '../components/triton2d/Triton2DQTilePacking';
import Triton2DMainLoop from '../components/triton2d/Triton2DMainLoop';
import Triton2DOutputUnmerge from '../components/triton2d/Triton2DOutputUnmerge';

interface Triton2DVizAppProps {
  onBack: () => void;
}

const STAGE_LABELS: Record<VizStage, string> = {
  idle: 'Ready',
  setup: 'Data Structures',
  q_tile_packing: 'Q Tile Head-Merge',
  kv_tile: 'KV Tile Loop',
  output: 'Output Un-merge',
  complete: 'Complete',
};

const STAGE_ORDER: VizStage[] = ['idle', 'setup', 'q_tile_packing', 'kv_tile', 'output', 'complete'];

const Triton2DVizApp: React.FC<Triton2DVizAppProps> = ({ onBack }) => {
  const {
    currentStage,
    selectedWG,
    config,
    initialize,
    nextStep,
    prevStep,
    reset,
    currentTileIdx,
    kvTileSteps,
    qTilePackingRow,
  } = useTriton2DVizStore();

  useEffect(() => {
    initialize();
  }, [initialize]);

  const handleReset = useCallback(() => {
    reset();
    initialize();
  }, [reset, initialize]);

  const handleNext = useCallback(() => {
    if (currentStage === 'idle') initialize();
    else nextStep();
  }, [currentStage, initialize, nextStep]);

  const stageIdx = STAGE_ORDER.indexOf(currentStage);

  return (
    <div className="app-container">
      <button className="back-btn" onClick={onBack}>
        <ArrowLeft size={16} />
        Back to Home
      </button>

      <header className="app-header">
        <h1 style={{
          background: 'linear-gradient(135deg, #e94560 0%, #0f3460 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}>
          Triton 2D Unified Attention — Interactive
        </h1>
        <p>Step through a single-pass head-merged attention workgroup with concrete data</p>

        {/* Inputs & kernel parameters */}
        <div style={{
          marginTop: '12px',
          display: 'flex',
          gap: '12px',
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}>
          {/* Inputs */}
          <div style={{
            padding: '10px 14px',
            background: '#f0f7ff',
            borderRadius: '8px',
            border: '1px solid #b3d9ff',
            fontSize: '12px',
            lineHeight: '1.6',
            textAlign: 'left',
          }}>
            <strong style={{ color: '#1565c0' }}>Inputs:</strong><br />
            q (Q_flat): [{config.numSeqs} tokens, {config.nheadQ} heads, {config.hdim} hdim]<br />
            kv_cache: [{config.pageSize} slots/page, {config.nheadKV} KV heads, {config.hdim} hdim]<br />
            block_table: [{config.numSeqs} seqs, variable pages] → physical page IDs<br />
            seq_lens_k: [{config.seqLensK.join(', ')}] (KV tokens per seq)
          </div>

          {/* Model config */}
          <div style={{
            padding: '10px 14px',
            background: '#f3e5f5',
            borderRadius: '8px',
            border: '1px solid #ce93d8',
            fontSize: '12px',
            lineHeight: '1.6',
            textAlign: 'left',
          }}>
            <strong style={{ color: '#7b1fa2' }}>Model:</strong><br />
            nhead_q = {config.nheadQ}, nhead_kv = {config.nheadKV}<br />
            num_queries_per_kv = {config.numQueriesPerKV} (GQA-{config.numQueriesPerKV})<br />
            hdim = {config.hdim} (dimension per head)
          </div>

          {/* Kernel tile parameters */}
          <div style={{
            padding: '10px 14px',
            background: '#fff3e0',
            borderRadius: '8px',
            border: '1px solid #ffcc80',
            fontSize: '12px',
            lineHeight: '1.6',
            textAlign: 'left',
          }}>
            <strong style={{ color: '#e65100' }}>Kernel tiles:</strong><br />
            BLOCK_M = {config.BLOCK_M} (Q tile rows = {config.BLOCK_Q} tokens × {config.numQueriesPerKV} heads)<br />
            BLOCK_Q = {config.BLOCK_Q} (token slots per tile)<br />
            BLOCK_N = {config.BLOCK_N} (KV positions per iteration)<br />
            page_size = {config.pageSize} (tokens per physical page)
          </div>
        </div>
      </header>

      {/* Stage progress + controls -- STICKY */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        padding: '12px 20px',
        background: 'white',
        borderRadius: '12px',
        boxShadow: '0 4px 15px rgba(0,0,0,0.15)',
        marginBottom: '20px',
        position: 'sticky',
        top: '0',
        zIndex: 100,
      }}>
        {/* Stage indicator */}
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', gap: '4px', marginBottom: '8px' }}>
            {STAGE_ORDER.map((s, i) => (
              <motion.div
                key={s}
                animate={{
                  background: i < stageIdx ? '#4caf50'
                    : i === stageIdx ? 'linear-gradient(135deg, #e94560 0%, #0f3460 100%)'
                    : '#e0e0e0',
                }}
                style={{
                  flex: 1,
                  height: '6px',
                  borderRadius: '3px',
                  background: i < stageIdx ? '#4caf50' : i === stageIdx ? '#e94560' : '#e0e0e0',
                }}
              />
            ))}
          </div>
          <div style={{ fontSize: '13px', fontWeight: 600, color: '#333' }}>
            Stage: {STAGE_LABELS[currentStage]}
            {currentStage === 'kv_tile' && ` (tile ${currentTileIdx + 1}/${kvTileSteps.length})`}
            {currentStage === 'q_tile_packing' && ` (token ${Math.floor(qTilePackingRow / config.numQueriesPerKV) + 1}/${config.BLOCK_Q})`}
          </div>
          {selectedWG && (
            <div style={{ fontSize: '11px', color: '#888' }}>
              Workgroup: KV head {selectedWG.kvHeadIdx}, Seq {selectedWG.seqIdx}
              (q_len={selectedWG.qLen}, kv_len={selectedWG.kvLen})
            </div>
          )}
        </div>

        {/* Playback controls */}
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
            onClick={handleReset}
            style={{
              padding: '8px',
              background: '#f0f0f0',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
            }}
            title="Reset"
          >
            <RotateCcw size={16} />
          </button>
          <button
            onClick={prevStep}
            disabled={currentStage === 'idle'}
            style={{
              padding: '8px',
              background: currentStage === 'idle' ? '#e0e0e0' : '#f0f0f0',
              border: 'none',
              borderRadius: '8px',
              cursor: currentStage === 'idle' ? 'default' : 'pointer',
              display: 'flex',
              alignItems: 'center',
            }}
            title="Previous step"
          >
            <SkipBack size={16} />
          </button>
          <button
            onClick={handleNext}
            disabled={currentStage === 'complete'}
            style={{
              padding: '8px 16px',
              background: currentStage === 'complete' ? '#e0e0e0' : 'linear-gradient(135deg, #e94560 0%, #0f3460 100%)',
              border: 'none',
              borderRadius: '8px',
              cursor: currentStage === 'complete' ? 'default' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              color: currentStage === 'complete' ? '#999' : 'white',
              fontWeight: 600,
              fontSize: '13px',
            }}
            title="Next step"
          >
            Step <SkipForward size={14} />
          </button>
        </div>
      </div>

      <div className="overview-layout">
        {/* Left panel: input data + grid launch */}
        <div className="left-panel" style={{ maxHeight: '80vh', overflowY: 'auto' }}>
          <Triton2DInputData />
          <div style={{ marginTop: '16px' }}>
            <Triton2DGridLaunch />
          </div>
        </div>

        {/* Center panel: step-by-step visualization */}
        <div className="center-panel" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {currentStage === 'idle' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{
                padding: '40px',
                textAlign: 'center',
                color: '#888',
              }}
            >
              <p style={{ fontSize: '16px', marginBottom: '12px' }}>
                Press <strong>Step →</strong> to begin
              </p>
              <p style={{ fontSize: '13px' }}>
                Select a workgroup from the grid on the left to follow a specific (kv_head, seq) pair
              </p>
            </motion.div>
          )}

          {currentStage === 'setup' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{
                padding: '20px',
                background: '#e3f2fd',
                borderRadius: '12px',
                border: '1px solid #90caf9',
              }}
            >
              <h3 style={{ margin: '0 0 8px', color: '#1565c0' }}>Setup: Data Structures Ready</h3>
              <p style={{ fontSize: '13px', color: '#555', margin: 0 }}>
                Q_flat, kv_cache, and block_table are initialized. The grid launches
                {' '}{useTriton2DVizStore.getState().simData?.workgroups.length ?? 0} workgroups.
                Press Step → to start packing the Q tile with head-merge.
              </p>
            </motion.div>
          )}

          {currentStage === 'q_tile_packing' && <Triton2DQTilePacking />}
          {currentStage === 'kv_tile' && <Triton2DMainLoop />}
          {(currentStage === 'output' || currentStage === 'complete') && <Triton2DOutputUnmerge />}
        </div>
      </div>
    </div>
  );
};

export default Triton2DVizApp;
