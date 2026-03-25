import { GlossaryEntry } from '../../components/shared/ParameterGlossary';
import { createCommonGlossaryEntries } from './commonGlossary';

export const createHSTUGlossaryEntries = (config: any): GlossaryEntry[] => {
  const commonEntries = createCommonGlossaryEntries(config);

  const specificEntries: GlossaryEntry[] = [
    {
      name: 'alpha',
      type: 'float',
      category: 'specific',
      range: '0.1-2.0',
      currentValue: config.alpha || 1.0,
      defaultValue: 1.0,
      description: 'SiLU gate scaling parameter. Controls sharpness of gating function.',
      sampleValues: [
        { value: 0.5, explanation: 'Smooth gating (gentle activation)' },
        { value: 1.0, explanation: 'Standard SiLU activation' },
        { value: 2.0, explanation: 'Sharp gating (aggressive activation)' }
      ],
      formula: `SiLU with scaling:
  SiLU(x) = x × σ(α × x)
  where σ(x) = 1 / (1 + e^(-x))

Effect of α:
  α < 1: Smoother, more gradual gating
  α = 1: Standard SiLU
  α > 1: Sharper, more binary-like gating

Gradient:
  d/dx SiLU(x) = σ(αx) + αx × σ(αx) × (1 - σ(αx))`
    },
    {
      name: 'maxSeqLenK',
      type: 'int',
      category: 'specific',
      range: '64-2048',
      currentValue: config.maxSeqLenK || 512,
      defaultValue: 512,
      description: 'Maximum key sequence length. For variable-length sequences in the key/value.',
      sampleValues: [
        { value: 256, explanation: 'Short key sequences' },
        { value: 512, explanation: 'Standard context' },
        { value: 2048, explanation: 'Extended key context' }
      ]
    },
    {
      name: 'maxSeqLenQ',
      type: 'int',
      category: 'specific',
      range: '64-2048',
      currentValue: config.maxSeqLenQ || 256,
      defaultValue: 256,
      description: 'Maximum query sequence length. Can differ from maxSeqLenK for cross-attention.',
      sampleValues: [
        { value: 128, explanation: 'Short query sequences' },
        { value: 256, explanation: 'Standard query length' },
        { value: 1024, explanation: 'Long query sequences' }
      ],
      formula: `Typical Scenarios:

Self-Attention:
  maxSeqLenQ = maxSeqLenK
  Q, K, V all from same sequence

Cross-Attention:
  maxSeqLenQ ≠ maxSeqLenK
  Example (encoder-decoder):
    Q from decoder (shorter)
    K, V from encoder (longer)

Memory:
  Attention scores: [B × H × maxSeqLenQ × maxSeqLenK]`
    },
    {
      name: 'useJaggedTensors',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.useJaggedTensors ?? true,
      defaultValue: true,
      description: 'Store sequences contiguously without padding. More memory-efficient for variable-length batches.',
      sampleValues: [
        { value: false, explanation: 'Padded tensors (simple, wastes memory)' },
        { value: true, explanation: 'Jagged tensors (efficient, complex indexing)' }
      ],
      formula: `Padded Tensors:
  Sequences: [100, 200, 50, 150]
  Padded to: [200, 200, 200, 200]
  Storage: 800 tokens
  Waste: 350 tokens (44%)

Jagged Tensors:
  Store: [seq1|seq2|seq3|seq4] contiguously
  Metadata: offsets = [0, 100, 300, 350, 500]
  Storage: 500 tokens
  Waste: 0 tokens

  Access seq[i]:
    start = offsets[i]
    end = offsets[i+1]
    data = buffer[start:end]

Benefits:
  - No wasted memory on padding
  - Better cache locality
  - Harder to implement (complex indexing)`
    },
    {
      name: 'SiLU (Swish)',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Sigmoid Linear Unit - smooth, non-monotonic activation function.',
      sampleValues: [],
      formula: `SiLU(x) = x × σ(x) = x / (1 + e^(-x))

Properties:
  - Smooth (differentiable everywhere)
  - Non-monotonic (dips below zero for x < 0)
  - Unbounded above, bounded below
  - Self-gating (x gates itself via sigmoid)

vs ReLU:
  ReLU(x) = max(0, x)  # Sharp, monotonic
  SiLU(x) = x × σ(x)   # Smooth, non-monotonic

Used in: GPT, Transformer FFN, HSTU attention gating`
    },
    {
      name: 'Gating vs Normalization',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Distinction between multiplicative gating (SiLU) and normalizing functions (Softmax).',
      sampleValues: [],
      formula: `Softmax (Normalization):
  softmax(x)[i] = exp(x[i]) / Σ exp(x[j])
  Properties:
    - Outputs sum to 1 (probability distribution)
    - All outputs in [0, 1]
    - Used for: Attention weights

SiLU/Gating (Activation):
  SiLU(x) = x × σ(x)
  Properties:
    - Unbounded output
    - NOT normalized (no sum constraint)
    - Used for: Feature activation/gating

Key Difference:
  Softmax → Probability (normalized, interpretable)
  SiLU    → Activation (unbounded, expressive)`
    },
    {
      name: 'Jagged Tensors',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: config.useJaggedTensors ? 'Enabled' : 'Disabled',
      defaultValue: 'Enabled',
      description: 'Variable-length tensors stored contiguously without padding.',
      sampleValues: [],
      formula: `Standard Batching (Padded):
  Batch = [
    [tok1, tok2, tok3, PAD,  PAD],
    [tok1, tok2, PAD,  PAD,  PAD],
    [tok1, tok2, tok3, tok4, tok5]
  ]
  Shape: [3, 5]  # Rectangular, includes padding

Jagged Tensors:
  Values: [tok1, tok2, tok3, tok1, tok2, tok1, tok2, tok3, tok4, tok5]
  Offsets: [0, 3, 5, 10]

  To access sequence i:
    seq[i] = values[offsets[i]:offsets[i+1]]

Advantages:
  - No memory waste
  - Better memory bandwidth (no padding transfers)
  - More complex indexing

Libraries: PyTorch NestedTensor, JAX, custom CUDA kernels`
    },
    {
      name: 'HSTU Attention',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Attention variant using SiLU gating instead of softmax normalization.',
      sampleValues: [],
      formula: `Standard Attention:
  scores = Q @ K^T
  weights = softmax(scores)  ← Normalization
  output = weights @ V

HSTU Attention:
  scores = Q @ K^T
  gates = SiLU(α × scores)   ← Gating (no normalization!)
  output = gates @ V

Key Differences:
  1. No softmax normalization
  2. Gates can be > 1 or < 0
  3. Output NOT weighted average
  4. α controls gate sharpness

Trade-offs:
  + More expressive (unbounded gates)
  + Faster (no exp/sum for softmax)
  - Less interpretable (not probabilities)
  - May need careful tuning (α parameter)`
    }
  ];

  return [...commonEntries, ...specificEntries];
};
