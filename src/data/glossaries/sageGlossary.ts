import { GlossaryEntry } from '../../components/shared/ParameterGlossary';
import { createCommonGlossaryEntries } from './commonGlossary';

export const createSageGlossaryEntries = (config: any): GlossaryEntry[] => {
  const commonEntries = createCommonGlossaryEntries(config);

  const specificEntries: GlossaryEntry[] = [
    {
      name: 'useFP8',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.useFP8 ?? true,
      defaultValue: true,
      description: 'Use FP8 E4M3 quantization for attention tensors. 2× memory reduction, ~1.5× speedup.',
      sampleValues: [
        { value: false, explanation: 'FP16/BF16 (standard precision)' },
        { value: true, explanation: 'FP8 (2× memory savings, slight accuracy loss)' }
      ],
      formula: `FP8 E4M3 Format:
  1 sign bit + 4 exponent bits + 3 mantissa bits

Range: [-448, 448] with subnormals
Precision: ~0.5-1% relative error

Memory Savings:
  FP16: 2 bytes per value
  FP8:  1 byte per value
  Reduction: 2×

Attention Memory:
  Q, K, V: [B×S×H×D] × 3 tensors
  FP16: 6BSD bytes
  FP8:  3BSD bytes (50% savings)`
    },
    {
      name: 'useMxFP4',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.useMxFP4 ?? false,
      defaultValue: false,
      description: 'Use ultra-low MxFP4 precision (4-bit mantissa). Requires useFP8 = true. Experimental: 4× memory reduction.',
      sampleValues: [
        { value: false, explanation: 'Standard (FP8 or FP16)' },
        { value: true, explanation: 'Aggressive compression (4× savings, more accuracy loss)' }
      ],
      formula: `MxFP4 (Microscaling FP4):
  Block-wise 4-bit floating point
  Each block (e.g., 32 values):
    - Shared exponent (8 bits)
    - 32× 4-bit mantissas

Total: 8 + 32×4 = 136 bits for 32 values
  = 4.25 bits/value (vs 16 bits for FP16)

Compression: ~4× vs FP16

Trade-off:
  + Massive memory savings
  + Faster memory transfers
  - Lower precision (acceptable for many tasks)
  - Requires careful tuning`
    },
    {
      name: 'perTokenScale',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.perTokenScale ?? true,
      defaultValue: true,
      description: 'Per-token scaling vs per-tensor scaling. Better precision for varying token magnitudes.',
      sampleValues: [
        { value: false, explanation: 'Per-tensor (single scale for all tokens)' },
        { value: true, explanation: 'Per-token (individual scale per token, better quality)' }
      ],
      formula: `Per-Tensor Scaling:
  scale = max(abs(X)) / FP8_max
  X_quant = round(X / scale)

  Problem: One outlier token → poor quantization for others

Per-Token Scaling:
  For each token i:
    scale[i] = max(abs(X[i])) / FP8_max
    X_quant[i] = round(X[i] / scale[i])

  Benefit: Each token optimally scaled
  Cost: Store scale[i] for each token (small overhead)`
    },
    {
      name: 'smoothK',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.smoothK ?? true,
      defaultValue: true,
      description: 'Smooth K matrix before quantization. Reduces quantization error by normalizing outliers.',
      sampleValues: [
        { value: false, explanation: 'Direct quantization (may have outliers)' },
        { value: true, explanation: 'Smooth first (better quantization quality)' }
      ],
      formula: `Problem: Outliers in K hurt quantization
  If max(K) = 1000, but most values ~ 1
  → Poor precision for typical values

SmoothQuant/Smooth-K:
  1. Compute smoothing factor per channel:
       s = sqrt(max(abs(K)) × max(abs(Q)))

  2. Transfer scale from K to Q:
       K_smooth = K / s
       Q_smooth = Q × s

  3. Quantize smoothed tensors:
       K_quant = quantize(K_smooth)
       Q_quant = quantize(Q_smooth)

Result: K has reduced outliers → better quantization
Math unchanged: (Q×s) @ (K/s)^T = Q @ K^T`
    },
    {
      name: 'FP8 E4M3',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: '8-bit floating point format optimized for deep learning.',
      sampleValues: [],
      formula: `Format: 1 sign + 4 exponent + 3 mantissa

E4M3 Characteristics:
  Max value: 448
  Min positive: ~0.002 (subnormal)
  Precision: 3 mantissa bits → ~0.5-1% error

vs FP16:
  FP16: 1 + 5 exp + 10 mantissa
  Range: [-65504, 65504]
  Precision: ~0.01% error

Trade-off:
  FP8: Narrower range, lower precision
  But: 2× memory, 2× bandwidth, faster compute`
    },
    {
      name: 'MxFP4',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Microscaling FP4 - block-wise 4-bit floating point with shared exponent.',
      sampleValues: [],
      formula: `Block Structure (block_size=32):
  1× Shared exponent: 8 bits
  32× Mantissas: 4 bits each = 128 bits
  Total: 136 bits for 32 values

Encoding:
  shared_exp = max_exponent(block)
  For each value:
    mantissa[i] = round(value[i] / 2^shared_exp)

Decoding:
  value[i] = mantissa[i] × 2^shared_exp

Advantages:
  - 4× compression vs FP16
  - Preserves relative magnitudes within block
  - Hardware-friendly (simple ops)

Limitations:
  - Poor for values spanning many orders of magnitude
  - Block size trade-off (larger = more compression risk)`
    },
    {
      name: 'SmoothQuant',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Technique to migrate quantization difficulty from activations to weights via smoothing.',
      sampleValues: [],
      formula: `Observation:
  Activations (Q,K,V) have outliers
  Weights are easier to quantize

SmoothQuant:
  1. Compute per-channel smoothing factor:
       s_j = (max|X_j|)^α / (max|W_j|)^(1-α)
       where α ∈ [0,1] (typically 0.5)

  2. Migrate scale:
       X' = X / s  (smoother activations)
       W' = W × s  (absorb scale into weights)

  3. Quantize both:
       X_q = quantize(X')
       W_q = quantize(W')

Result: X' @ W' = X @ W (mathematically equivalent)
But: X' easier to quantize (fewer outliers)`
    }
  ];

  return [...commonEntries, ...specificEntries];
};
