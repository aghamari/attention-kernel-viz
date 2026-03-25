import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { useUnifiedAttentionStore } from '../../store/unifiedAttentionStore';
import MatrixBox from '../shared/MatrixBox';
import InfoBox from '../shared/InfoBox';

const AttentionPipelineFlow: React.FC = () => {
  const { config, tensors, currentStep } = useUnifiedAttentionStore();

  const hiddenDim = config.numHeads * config.headDim;

  return (
    <div style={{ padding: '20px' }}>
      <h3 style={{ marginBottom: '20px' }}>Attention Pipeline Overview</h3>

      <InfoBox type="concept">
        <strong>Q, K, V Explained:</strong>
        <ul style={{ marginTop: '8px', marginLeft: '20px' }}>
          <li><strong>Q (Query):</strong> "What am I looking for?" - represents the current token</li>
          <li><strong>K (Key):</strong> "What do I contain?" - represents all tokens in sequence</li>
          <li><strong>V (Value):</strong> "What information do I carry?" - the actual content to aggregate</li>
          <li>Attention computes: which Keys match each Query, then sums corresponding Values</li>
        </ul>
      </InfoBox>

      <InfoBox type="important">
        <strong>Unified = Flash Attention + Adaptive Grid Selection</strong>
        <ul style={{ marginTop: '8px', marginLeft: '20px' }}>
          <li><strong>Core algorithm:</strong> Flash Attention (online softmax, tiled computation) - same for both 2D and 3D</li>
          <li><strong>2D kernel (≤2048 tokens):</strong> Single-pass, grid=(kv_heads, q_blocks), optimal for decode/short sequences</li>
          <li><strong>3D kernel (&gt;2048 tokens):</strong> Multi-segment, grid=(q_blocks, kv_heads, segments), requires reduction phase</li>
          <li><strong>Current selection:</strong> seqLen={config.seqLen} → {config.use3DKernel ? '3D' : '2D'} kernel (automatic)</li>
        </ul>
      </InfoBox>

      {/* Pipeline Visualization */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '30px',
        marginTop: '30px'
      }}>
        {/* Step 1: Input */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <MatrixBox
            label="Input"
            dims={`[B×S×D]`}
            shape={`[B×S×D]`}
            actualDims={tensors.input ? tensors.input.actualDims : `${config.batchSize}×${config.seqLen}×${hiddenDim}`}
            memorySize={tensors.input?.memoryStr || '0 B'}
            description="Input token embeddings. Each token is represented as a D-dimensional vector."
            color="#3498db"
            isActive={currentStep === 'idle'}
            sampleValues={tensors.input?.sampleValues}
          />
          <ArrowRight size={24} color="#666" />
          <div style={{
            padding: '15px 20px',
            background: '#f5f5f5',
            borderRadius: '8px',
            border: '2px solid #9e9e9e',
            fontFamily: 'monospace',
            fontSize: '12px'
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Q/K/V Projection</div>
            <div>Q = Input @ W_Q</div>
            <div>K = Input @ W_K</div>
            <div>V = Input @ W_V</div>
          </div>
        </div>

        {/* Step 2: Q, K, V */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginLeft: '40px' }}>
          <div style={{ display: 'flex', gap: '15px' }}>
            <MatrixBox
              label="Q (Queries)"
              dims={`[B×S×H×D]`}
              shape={`[B×S×H×D]`}
              actualDims={tensors.q?.actualDims || `${config.batchSize}×${config.seqLen}×${config.numHeads}×${config.headDim}`}
              memorySize={tensors.q?.memoryStr || '0 B'}
              description="Query vectors: What each token is looking for"
              color="#3498db"
              isActive={currentStep === 'proj'}
              sampleValues={tensors.q?.sampleValues}
            />
            <MatrixBox
              label="K (Keys)"
              dims={`[B×S×H×D]`}
              shape={`[B×S×H×D]`}
              actualDims={tensors.k?.actualDims || `${config.batchSize}×${config.seqLen}×${config.numHeads}×${config.headDim}`}
              memorySize={tensors.k?.memoryStr || '0 B'}
              description="Key vectors: What each token contains"
              color="#e74c3c"
              isActive={currentStep === 'proj'}
              sampleValues={tensors.k?.sampleValues}
            />
            <MatrixBox
              label="V (Values)"
              dims={`[B×S×H×D]`}
              shape={`[B×S×H×D]`}
              actualDims={tensors.v?.actualDims || `${config.batchSize}×${config.seqLen}×${config.numHeads}×${config.headDim}`}
              memorySize={tensors.v?.memoryStr || '0 B'}
              description="Value vectors: The information to aggregate"
              color="#2ecc71"
              isActive={currentStep === 'proj'}
              sampleValues={tensors.v?.sampleValues}
            />
          </div>
          <ArrowRight size={24} color="#666" />
          <div style={{
            padding: '15px 20px',
            background: '#f5f5f5',
            borderRadius: '8px',
            border: '2px solid #9e9e9e',
            fontFamily: 'monospace',
            fontSize: '12px'
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Attention Scores</div>
            <div>Scores = Q @ K^T / √d</div>
            {config.causal && <div style={{ color: '#e74c3c' }}>+ Causal Mask</div>}
          </div>
        </div>

        {/* Step 3: Scores */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginLeft: '80px' }}>
          <MatrixBox
            label="Attention Scores"
            dims={`[B×H×S×S]`}
            shape={`[B×H×S×S]`}
            actualDims={tensors.scores?.actualDims || `${config.batchSize}×${config.numHeads}×${config.seqLen}×${config.seqLen}`}
            memorySize={tensors.scores?.memoryStr || '0 B'}
            description="Raw attention scores showing how much each token attends to others"
            color="#f39c12"
            isActive={currentStep === 'qk-matmul'}
            sampleValues={tensors.scores?.sampleValues}
          />
          <ArrowRight size={24} color="#666" />
          <div style={{
            padding: '15px 20px',
            background: '#f5f5f5',
            borderRadius: '8px',
            border: '2px solid #9e9e9e',
            fontFamily: 'monospace',
            fontSize: '12px'
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Softmax</div>
            <div>softmax(Scores)</div>
          </div>
          <ArrowRight size={24} color="#666" />
          <MatrixBox
            label="Attention Weights"
            dims={`[B×H×S×S]`}
            shape={`[B×H×S×S]`}
            actualDims={tensors.softmaxScores?.actualDims || `${config.batchSize}×${config.numHeads}×${config.seqLen}×${config.seqLen}`}
            memorySize={tensors.softmaxScores?.memoryStr || '0 B'}
            description="Normalized attention probabilities (sum to 1)"
            color="#9b59b6"
            isActive={currentStep === 'softmax'}
            sampleValues={tensors.softmaxScores?.sampleValues}
          />
        </div>

        {/* Step 4: Output */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginLeft: '40px' }}>
          <div style={{
            padding: '15px 20px',
            background: '#f5f5f5',
            borderRadius: '8px',
            border: '2px solid #9e9e9e',
            fontFamily: 'monospace',
            fontSize: '12px'
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Weighted Sum</div>
            <div>Output = Weights @ V</div>
          </div>
          <ArrowRight size={24} color="#666" />
          <MatrixBox
            label="Output"
            dims={`[B×S×D]`}
            shape={`[B×S×D]`}
            actualDims={tensors.output?.actualDims || `${config.batchSize}×${config.seqLen}×${hiddenDim}`}
            memorySize={tensors.output?.memoryStr || '0 B'}
            description="Final attended output combining information from all relevant tokens"
            color="#2ecc71"
            isActive={currentStep === 'av-matmul' || currentStep === 'output'}
            sampleValues={tensors.output?.sampleValues}
          />
        </div>
      </div>

      {/* Formula Summary */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        style={{
          marginTop: '40px',
          padding: '20px',
          background: '#f9f9f9',
          borderRadius: '8px',
          border: '2px solid #ddd'
        }}
      >
        <h4 style={{ marginBottom: '15px' }}>Complete Attention Formula</h4>
        <div style={{ fontFamily: 'monospace', fontSize: '14px', lineHeight: '2' }}>
          <div>Attention(Q, K, V) = softmax(Q @ K^T / √{config.headDim}) @ V</div>
          <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
            Where:
            <ul style={{ marginTop: '5px', marginLeft: '20px' }}>
              <li>Q: Queries [{config.batchSize}×{config.seqLen}×{config.numHeads}×{config.headDim}]</li>
              <li>K: Keys [{config.batchSize}×{config.seqLen}×{config.numHeads}×{config.headDim}]</li>
              <li>V: Values [{config.batchSize}×{config.seqLen}×{config.numHeads}×{config.headDim}]</li>
              <li>√{config.headDim} = {Math.sqrt(config.headDim).toFixed(2)} (scaling factor)</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default AttentionPipelineFlow;
