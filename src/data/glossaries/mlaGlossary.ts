import { GlossaryEntry } from '../../components/shared/ParameterGlossary';
import { createCommonGlossaryEntries } from './commonGlossary';

export const createMLAGlossaryEntries = (config: any): GlossaryEntry[] => {
  const commonEntries = createCommonGlossaryEntries(config);

  const specificEntries: GlossaryEntry[] = [
    {
      name: 'kvLatentDim',
      type: 'int',
      category: 'specific',
      range: '8-64',
      currentValue: config.kvLatentDim || 32,
      defaultValue: 32,
      description: 'Compressed KV cache dimension. Compression ratio = headDim / kvLatentDim.',
      sampleValues: [
        { value: 16, explanation: '4× compression (headDim=64)' },
        { value: 32, explanation: '2× compression (headDim=64)' },
        { value: 64, explanation: 'No compression (headDim=64)' }
      ],
      formula: `Compression Ratio: headDim / kvLatentDim

Example (headDim=64):
  kvLatentDim=16 → 4× compression (75% memory saved)
  kvLatentDim=32 → 2× compression (50% memory saved)
  kvLatentDim=64 → 1× (no compression)

KV Cache Size:
  Original: [B×S×H×headDim] × 2
  Compressed: [B×S×H×kvLatentDim] × 2

Memory Saved: (1 - kvLatentDim/headDim) × 100%`
    },
    {
      name: 'ropeTheta',
      type: 'float',
      category: 'specific',
      range: '1000-100000',
      currentValue: config.ropeTheta || 10000,
      defaultValue: 10000,
      description: 'RoPE base frequency parameter. Higher = slower position decay, better for long contexts.',
      sampleValues: [
        { value: 10000, explanation: 'Standard (original RoPE)' },
        { value: 50000, explanation: 'Extended context (5× base)' },
        { value: 100000, explanation: 'Very long context (10× base)' }
      ],
      formula: `RoPE Frequencies:
  θ_i = ropeTheta^(-2i/d)
  where i = 0, 1, ..., d/2-1

Example (d=64, ropeTheta=10000):
  θ_0 = 10000^0     = 1
  θ_1 = 10000^(-1/32) ≈ 0.851
  θ_31 = 10000^(-31/32) ≈ 0.0001

Higher ropeTheta:
  → Lower frequencies
  → Slower position decay
  → Better long-range modeling`
    },
    {
      name: 'ropeScale',
      type: 'float',
      category: 'specific',
      range: '0.5-2.0',
      currentValue: config.ropeScale || 1.0,
      defaultValue: 1.0,
      description: 'RoPE scaling factor for extended contexts. Scales position indices before rotation.',
      sampleValues: [
        { value: 1.0, explanation: 'Standard (no scaling)' },
        { value: 2.0, explanation: 'Doubled context length' },
        { value: 4.0, explanation: 'Quadrupled context length' }
      ],
      formula: `Position Scaling:
  scaled_pos = position / ropeScale

Effect:
  ropeScale=2.0 → positions appear 2× closer
  → Can handle 2× longer sequences
  → Interpolates between trained positions

Example:
  Trained on seqLen=2048, ropeTheta=10000
  With ropeScale=2.0:
    → Can handle seqLen=4096
    → Position 4000 treated as position 2000

Trade-off:
  + Extends context without retraining
  - Slight quality degradation on very long sequences`
    },
    {
      name: 'useRoPE',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.useRoPE ?? true,
      defaultValue: true,
      description: 'Fuse RoPE (Rotary Position Embeddings) into attention. Alternative to absolute position embeddings.',
      sampleValues: [
        { value: false, explanation: 'No position encoding (or use absolute)' },
        { value: true, explanation: 'RoPE (relative positions, better extrapolation)' }
      ]
    },
    {
      name: 'nope_size',
      type: 'int',
      category: 'specific',
      range: '8-64',
      currentValue: config.nope_size || 32,
      defaultValue: 32,
      description: 'Non-rotary position embedding dimension. Part of KV not subject to rotation.',
      sampleValues: [
        { value: 16, explanation: 'Small non-rotary component' },
        { value: 32, explanation: 'Balanced split' },
        { value: 48, explanation: 'Large non-rotary component' }
      ],
      formula: `Head Dimension Split:
  headDim = nope_size + rope_size

Example (headDim=64):
  nope_size=32, rope_size=32
  → Half rotated, half not rotated

Non-Rotary Component:
  - Not affected by RoPE rotation
  - Can capture position-independent features
  - Useful for content-based matching

Rotary Component:
  - Affected by RoPE rotation
  - Captures position-dependent features
  - Relative position encoding`
    },
    {
      name: 'rope_size',
      type: 'int',
      category: 'specific',
      range: '8-64',
      currentValue: config.rope_size || 32,
      defaultValue: 32,
      description: 'Rotary position embedding dimension. Must satisfy: nope_size + rope_size = headDim.',
      sampleValues: [
        { value: 16, explanation: 'Small rotary component' },
        { value: 32, explanation: 'Balanced split' },
        { value: 48, explanation: 'Large rotary component' }
      ]
    },
    {
      name: 'MLA (Multi-Head Latent Attention)',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Attention variant that compresses KV cache using latent compression.',
      sampleValues: [],
      formula: `Standard Attention KV Cache:
  K: [B×S×H×D]
  V: [B×S×H×D]
  Memory: 2 × B × S × H × D

MLA Compressed KV Cache:
  K_latent: [B×S×H×L]  where L = kvLatentDim
  V_latent: [B×S×H×L]
  Memory: 2 × B × S × H × L

Compression:
  K_latent = compress(K)  # e.g., via learned projection
  During attention:
    K = decompress(K_latent)
    Standard attention with decompressed K, V

Memory Saved: (1 - L/D) × 100%
  Example: L=32, D=64 → 50% savings`
    },
    {
      name: 'RoPE (Rotary Position Embeddings)',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Relative position encoding via rotation matrices applied to Q and K.',
      sampleValues: [],
      formula: `RoPE Rotation:
  For position m, rotate query/key:
    [x_0]     [cos(mθ_i)  -sin(mθ_i)] [x_0]
    [x_1]  =  [sin(mθ_i)   cos(mθ_i)] [x_1]

  Applied to each dimension pair (x_2i, x_2i+1)

Frequencies:
  θ_i = ropeTheta^(-2i/d)

Relative Position:
  Q_m @ K_n^T encodes relative distance |m-n|

Advantages:
  - No learned position embeddings
  - Naturally handles relative positions
  - Better length extrapolation
  - Rotation preserves magnitude

Used in: LLaMA, GPT-NeoX, PaLM`
    },
    {
      name: 'Two-Stage Attention',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'MLA decoding splits attention into two stages: scoring and application.',
      sampleValues: [],
      formula: `Single-Stage (Standard):
  scores = Q @ K^T
  weights = softmax(scores / √d)
  output = weights @ V

Two-Stage (MLA Decode):
  Stage 1: Compute attention weights
    K = decompress(K_latent)
    scores = Q @ K^T
    weights = softmax(scores / √d)

  Stage 2: Apply to values
    V = decompress(V_latent)
    output = weights @ V

Benefits:
  - Can pipeline stages
  - Decompress K, V only when needed
  - Better memory locality

Cost:
  - More complex implementation
  - Potential for redundant decompression`
    },
    {
      name: 'RoPE + Non-RoPE Split',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: `nope=${config.nope_size || 32}, rope=${config.rope_size || 32}`,
      defaultValue: 'nope=32, rope=32',
      description: 'Splitting head dimension into rotary and non-rotary components.',
      sampleValues: [],
      formula: `Head Dimension: D = nope_size + rope_size

Query/Key Structure:
  Q = [Q_nope, Q_rope]  # Concatenated
    Q_nope: [B×S×H×nope_size]  ← No rotation
    Q_rope: [B×S×H×rope_size]  ← Apply RoPE

Attention Computation:
  scores = (Q_nope @ K_nope^T) + (Q_rope @ K_rope^T)
           -------------------   ---------------------
           Position-independent   Position-dependent

Rationale:
  - Some features don't need position info
  - Separating allows model flexibility
  - nope captures content similarity
  - rope captures positional relationships`
    }
  ];

  return [...commonEntries, ...specificEntries];
};
