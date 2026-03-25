import { GlossaryEntry } from '../../components/shared/ParameterGlossary';
import { createCommonGlossaryEntries } from './commonGlossary';

export const createFlashGlossaryEntries = (config: any): GlossaryEntry[] => {
  const commonEntries = createCommonGlossaryEntries(config);

  const specificEntries: GlossaryEntry[] = [
    {
      name: 'dropout',
      type: 'float',
      category: 'specific',
      range: '0-0.5',
      currentValue: config.dropout || 0.0,
      defaultValue: 0.0,
      description: 'Attention dropout rate. Randomly zeros out attention weights during training for regularization.',
      sampleValues: [
        { value: 0.0, explanation: 'No dropout (inference mode)' },
        { value: 0.1, explanation: 'Light regularization (training)' },
        { value: 0.3, explanation: 'Strong regularization (prevents overfitting)' }
      ]
    },
    {
      name: 'useCausalMask',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.useCausalMask ?? true,
      defaultValue: true,
      description: 'Apply lower-triangular causal mask. Essential for autoregressive language modeling.',
      sampleValues: [
        { value: true, explanation: 'Causal/autoregressive (GPT, LLaMA)' },
        { value: false, explanation: 'Bidirectional (BERT, encoding tasks)' }
      ]
    },
    {
      name: 'useAlibi',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.useAlibi ?? false,
      defaultValue: false,
      description: 'Use Attention with Linear Biases (ALiBi) for position encoding. Alternative to learned positional embeddings.',
      sampleValues: [
        { value: false, explanation: 'Standard positional encoding (absolute or RoPE)' },
        { value: true, explanation: 'ALiBi position bias (better length extrapolation)' }
      ],
      formula: `ALiBi Bias: -slope × |i - j|

Applied to attention scores before softmax:
  scores[i,j] = Q[i] @ K[j]^T - slope × |i - j|

Benefits:
  - No learned position embeddings needed
  - Better extrapolation to longer sequences
  - Linear position bias favors nearby tokens`
    },
    {
      name: 'alibiSlope',
      type: 'float',
      category: 'specific',
      range: '-1.0 to 1.0',
      currentValue: config.alibiSlope || 0.0,
      defaultValue: 0.0,
      description: 'ALiBi bias slope parameter. Controls how quickly attention decays with distance.',
      sampleValues: [
        { value: -0.5, explanation: 'Negative bias (penalize distant tokens)' },
        { value: 0.0, explanation: 'Disabled (no position bias)' },
        { value: 0.5, explanation: 'Positive bias (favor distant tokens - rare)' }
      ]
    },
    {
      name: 'useFP8',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.useFP8 ?? false,
      defaultValue: false,
      description: 'Use FP8 E4M3 quantization for Q/K/V tensors. Reduces memory and increases throughput at slight accuracy cost.',
      sampleValues: [
        { value: false, explanation: 'FP16/BF16 (standard precision)' },
        { value: true, explanation: 'FP8 (2× memory reduction, ~1.5× speedup)' }
      ],
      formula: `FP8 E4M3 Format:
  1 sign bit
  4 exponent bits
  3 mantissa bits

Range: [-448, 448] with subnormals
Precision: ~0.5% relative error

Memory Savings:
  FP32 → FP16: 2× reduction
  FP16 → FP8:  2× reduction (4× total from FP32)`
    },
    {
      name: 'softmaxScale',
      type: 'float',
      category: 'specific',
      range: '0.05-0.5',
      currentValue: config.softmaxScale || 0.125,
      defaultValue: 0.125,
      description: 'Scaling factor for attention scores before softmax. Typically 1/√(head_dim) to prevent saturation.',
      sampleValues: [
        { value: 0.125, explanation: '1/√64 for head_dim=64' },
        { value: 0.0884, explanation: '1/√128 for head_dim=128' },
        { value: 0.0625, explanation: '1/√256 for head_dim=256' }
      ],
      formula: `Softmax Scale: 1 / √(head_dim)

Why scaling?
  - Dot products grow with dimension
  - Large logits → softmax saturation
  - Small gradients → poor training

Scaled Attention:
  scores = (Q @ K^T) / √d
  weights = softmax(scores)
  output = weights @ V`
    },
    {
      name: 'Flash Attention Algorithm',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Memory-efficient attention via online softmax and tiling. Reduces O(N²) → O(N) memory usage.',
      sampleValues: [],
      formula: `Standard Attention:
  S = Q @ K^T          # [N×N] - memory bottleneck!
  P = softmax(S)       # [N×N] - requires materializing S
  O = P @ V            # [N×d]

Flash Attention:
  - Tiles Q, K, V into blocks
  - Computes attention incrementally
  - Uses online softmax (streaming algorithm)
  - Never materializes full [N×N] matrix

Memory: O(N²) → O(N)
Speed: Faster via better memory access patterns`
    },
    {
      name: 'Online Softmax',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Streaming softmax computation that processes data in chunks without storing full attention matrix.',
      sampleValues: [],
      formula: `Standard Softmax (requires full data):
  softmax(x)[i] = exp(x[i]) / Σ exp(x[j])

Online Softmax (incremental):
  Track running: max_val, sum_exp
  For each new block:
    1. Update max_val
    2. Rescale previous sum_exp
    3. Add new exp values
    4. Compute normalized output

Enables: Tiled attention without storing [N×N] scores`
    }
  ];

  return [...commonEntries, ...specificEntries];
};
