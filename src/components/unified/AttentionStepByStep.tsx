import React from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react';
import { useUnifiedAttentionStore } from '../../store/unifiedAttentionStore';
import InfoBox from '../shared/InfoBox';

const AttentionStepByStep: React.FC = () => {
  const { config, stepIndex, nextStep, prevStep, resetSteps, tensors } = useUnifiedAttentionStore();

  const steps = [
    {
      id: 'idle',
      title: 'Step 0: Initial State',
      description: 'We start with input token embeddings ready for processing.',
      formula: 'Input ∈ ℝ^{B×S×D}',
      code: `# Input dimensions
batch_size = ${config.batchSize}
seq_len = ${config.seqLen}
hidden_dim = ${config.numHeads * config.headDim}
input_shape = [${config.batchSize}, ${config.seqLen}, ${config.numHeads * config.headDim}]`,
      explanation: 'Each token in the sequence is represented as a dense vector in D-dimensional space.'
    },
    {
      id: 'proj',
      title: 'Step 1: Q/K/V Projection',
      description: 'Project input to Query, Key, and Value spaces using learned weight matrices.',
      formula: `Q = Input @ W_Q ∈ ℝ^{B×S×H×D}
K = Input @ W_K ∈ ℝ^{B×S×H×D}
V = Input @ W_V ∈ ℝ^{B×S×H×D}`,
      code: `# Projection
q = input @ w_q.reshape(${config.numHeads * config.headDim}, ${config.numHeads}, ${config.headDim})
k = input @ w_k.reshape(${config.numHeads * config.headDim}, ${config.numHeads}, ${config.headDim})
v = input @ w_v.reshape(${config.numHeads * config.headDim}, ${config.numHeads}, ${config.headDim})

# Output shapes
q.shape = [${config.batchSize}, ${config.seqLen}, ${config.numHeads}, ${config.headDim}]
k.shape = [${config.batchSize}, ${config.seqLen}, ${config.numHeads}, ${config.headDim}]
v.shape = [${config.batchSize}, ${config.seqLen}, ${config.numHeads}, ${config.headDim}]`,
      explanation: `Each head learns different aspects of attention. With ${config.numHeads} heads, we can attend to ${config.numHeads} different patterns simultaneously.`
    },
    {
      id: 'qk-matmul',
      title: 'Step 2: Compute Attention Scores',
      description: 'Calculate similarity between queries and keys using matrix multiplication.',
      formula: `Scores = (Q @ K^T) / √${config.headDim} ∈ ℝ^{B×H×S×S}`,
      code: `# Compute scores
scores = torch.matmul(q, k.transpose(-2, -1))
scores = scores / math.sqrt(${config.headDim})  # Scale by √d = ${Math.sqrt(config.headDim).toFixed(2)}

# Output shape
scores.shape = [${config.batchSize}, ${config.numHeads}, ${config.seqLen}, ${config.seqLen}]

${config.causal ? `# Apply causal mask (prevent looking ahead)
mask = torch.triu(torch.ones(${config.seqLen}, ${config.seqLen}), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))` : '# No causal masking'}`,
      explanation: `Scaling by √${config.headDim} prevents the dot products from growing too large. ${config.causal ? 'Causal masking ensures each position only attends to previous positions.' : 'Without causal masking, tokens can attend to the entire sequence.'}`
    },
    {
      id: 'softmax',
      title: 'Step 3: Apply Softmax',
      description: 'Normalize scores into probability distribution for each query.',
      formula: 'Attention Weights = softmax(Scores) ∈ ℝ^{B×H×S×S}',
      code: `# Apply softmax
attention_weights = torch.softmax(scores, dim=-1)

# Properties:
# - Each row sums to 1.0
# - All values are in [0, 1]
# - Represents probability distribution over keys

# Example at position [0,0,0,:]:
# Before softmax: ${tensors.scores?.sampleValues[0]?.map(v => v.toFixed(2)).join(', ') || '[...]'}
# After softmax:  ${tensors.softmaxScores?.sampleValues[0]?.map(v => v.toFixed(2)).join(', ') || '[...]'} (sum ≈ 1.0)`,
      explanation: 'Softmax converts arbitrary scores into normalized probabilities. High scores become large probabilities, low scores become small probabilities.'
    },
    {
      id: 'av-matmul',
      title: 'Step 4: Weighted Sum of Values',
      description: 'Aggregate value vectors weighted by attention probabilities.',
      formula: 'Output = Attention Weights @ V ∈ ℝ^{B×S×H×D}',
      code: `# Weighted sum
output = torch.matmul(attention_weights, v)

# Output shape
output.shape = [${config.batchSize}, ${config.seqLen}, ${config.numHeads}, ${config.headDim}]

# Interpretation:
# For each query position, we compute a weighted average of all value vectors.
# The weights are determined by the attention probabilities.`,
      explanation: 'This is where information actually flows. Tokens with high attention weights contribute more to the output.'
    },
    {
      id: 'output',
      title: 'Step 5: Final Output',
      description: 'Concatenate heads and project back to original dimension.',
      formula: 'Final = Concat(heads) @ W_O ∈ ℝ^{B×S×D}',
      code: `# Concatenate all heads
output = output.reshape(${config.batchSize}, ${config.seqLen}, ${config.numHeads * config.headDim})

# Final projection (optional, often included)
output = output @ w_o

# Final shape
output.shape = [${config.batchSize}, ${config.seqLen}, ${config.numHeads * config.headDim}]`,
      explanation: 'We combine information from all attention heads into a single output representation.'
    }
  ];

  const currentStepData = steps[stepIndex];

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h3>Step-by-Step Algorithm</h3>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={resetSteps}
            style={{
              padding: '8px 15px',
              background: '#f5f5f5',
              border: '2px solid #ddd',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <RotateCcw size={16} />
            Reset
          </button>
          <button
            onClick={prevStep}
            disabled={stepIndex === 0}
            style={{
              padding: '8px 15px',
              background: stepIndex === 0 ? '#f5f5f5' : '#3498db',
              color: stepIndex === 0 ? '#999' : '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: stepIndex === 0 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <ChevronLeft size={16} />
            Previous
          </button>
          <button
            onClick={nextStep}
            disabled={stepIndex === steps.length - 1}
            style={{
              padding: '8px 15px',
              background: stepIndex === steps.length - 1 ? '#f5f5f5' : '#3498db',
              color: stepIndex === steps.length - 1 ? '#999' : '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: stepIndex === steps.length - 1 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            Next
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      {/* Progress indicator */}
      <div style={{ marginBottom: '30px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
          {steps.map((step, idx) => (
            <div
              key={step.id}
              style={{
                flex: 1,
                textAlign: 'center',
                fontSize: '11px',
                color: idx === stepIndex ? '#3498db' : idx < stepIndex ? '#4caf50' : '#999',
                fontWeight: idx === stepIndex ? 'bold' : 'normal'
              }}
            >
              {step.id === 'idle' ? 'Start' : step.id}
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: '5px' }}>
          {steps.map((step, idx) => (
            <div
              key={step.id}
              style={{
                flex: 1,
                height: '8px',
                borderRadius: '4px',
                background: idx === stepIndex ? '#3498db' : idx < stepIndex ? '#4caf50' : '#e0e0e0',
                transition: 'background 0.3s'
              }}
            />
          ))}
        </div>
      </div>

      {/* Current step content */}
      <motion.div
        key={currentStepData.id}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
      >
        <InfoBox type="concept">
          <h4 style={{ marginBottom: '10px', fontSize: '16px' }}>{currentStepData.title}</h4>
          <p style={{ marginBottom: '10px' }}>{currentStepData.description}</p>
        </InfoBox>

        {/* Formula */}
        <div style={{
          padding: '20px',
          background: '#fff',
          border: '2px solid #3498db',
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <h5 style={{ marginBottom: '10px', color: '#3498db' }}>Formula</h5>
          <pre style={{
            fontFamily: 'monospace',
            fontSize: '14px',
            whiteSpace: 'pre-wrap',
            margin: 0
          }}>
            {currentStepData.formula}
          </pre>
        </div>

        {/* Code */}
        <div style={{
          padding: '20px',
          background: '#1e1e1e',
          color: '#d4d4d4',
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <h5 style={{ marginBottom: '10px', color: '#4ec9b0' }}>Code (PyTorch)</h5>
          <pre style={{
            fontFamily: 'monospace',
            fontSize: '12px',
            whiteSpace: 'pre-wrap',
            margin: 0,
            lineHeight: '1.6'
          }}>
            {currentStepData.code}
          </pre>
        </div>

        {/* Explanation */}
        <InfoBox type="note">
          <strong>Explanation:</strong>
          <p style={{ marginTop: '8px' }}>{currentStepData.explanation}</p>
        </InfoBox>
      </motion.div>
    </div>
  );
};

export default AttentionStepByStep;
