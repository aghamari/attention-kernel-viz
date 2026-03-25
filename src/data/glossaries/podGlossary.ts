import { GlossaryEntry } from '../../components/shared/ParameterGlossary';
import { createCommonGlossaryEntries } from './commonGlossary';

export const createPODGlossaryEntries = (config: any): GlossaryEntry[] => {
  const commonEntries = createCommonGlossaryEntries(config);

  const specificEntries: GlossaryEntry[] = [
    {
      name: 'prefillRatio',
      type: 'float',
      category: 'specific',
      range: '0.1-0.9',
      currentValue: config.prefillRatio || 0.6,
      defaultValue: 0.6,
      description: 'Fraction of GPU CUs allocated to prefill workload. Must sum with decodeRatio to 1.0.',
      sampleValues: [
        { value: 0.3, explanation: 'Decode-heavy (more generation, less input)' },
        { value: 0.6, explanation: 'Balanced (equal priority)' },
        { value: 0.8, explanation: 'Prefill-heavy (large context processing)' }
      ],
      formula: `CU Allocation (MI300X with 120 CUs):
  Prefill CUs = ⌊120 × prefillRatio⌋
  Decode CUs = ⌊120 × decodeRatio⌋

Example (prefillRatio=0.6):
  Prefill: 72 CUs (compute-bound)
  Decode:  48 CUs (memory-bound)

Constraint: prefillRatio + decodeRatio = 1.0`
    },
    {
      name: 'decodeRatio',
      type: 'float',
      category: 'specific',
      range: '0.1-0.9',
      currentValue: config.decodeRatio || 0.4,
      defaultValue: 0.4,
      description: 'Fraction of GPU CUs allocated to decode workload. Must sum with prefillRatio to 1.0.',
      sampleValues: [
        { value: 0.2, explanation: 'Prefill-heavy allocation' },
        { value: 0.4, explanation: 'Balanced (standard)' },
        { value: 0.7, explanation: 'Decode-heavy (many concurrent generations)' }
      ]
    },
    {
      name: 'prefillSeqLen',
      type: 'int',
      category: 'specific',
      range: '128-4096',
      currentValue: config.prefillSeqLen || 1024,
      defaultValue: 1024,
      description: 'Sequence length for prefill (context processing). Compute-bound workload with high arithmetic intensity.',
      sampleValues: [
        { value: 256, explanation: 'Short prompt' },
        { value: 1024, explanation: 'Standard context (article, code)' },
        { value: 4096, explanation: 'Long context (document, book chapter)' }
      ],
      formula: `Prefill Characteristics:
  Workload: Process full context at once
  Compute: O(N²) attention (N = prefillSeqLen)
  Arithmetic Intensity: ~100 ops/byte (compute-bound)
  Bottleneck: GPU compute (FLOPS)
  KV Cache: Build initial cache

Memory: B × N × H × D × 2 (K+V)`
    },
    {
      name: 'decodeNumTokens',
      type: 'int',
      category: 'specific',
      range: '8-256',
      currentValue: config.decodeNumTokens || 64,
      defaultValue: 64,
      description: 'Number of tokens to generate in decode phase. Memory-bound workload with low arithmetic intensity.',
      sampleValues: [
        { value: 16, explanation: 'Short generation (Q&A, completion)' },
        { value: 64, explanation: 'Medium generation (paragraph)' },
        { value: 256, explanation: 'Long generation (essay, code)' }
      ],
      formula: `Decode Characteristics:
  Workload: Generate tokens one-by-one
  Compute: O(N) per token (read cached KV)
  Arithmetic Intensity: ~10 ops/byte (memory-bound)
  Bottleneck: Memory bandwidth (HBM)
  KV Cache: Read existing + append new

Per Token:
  Q: [B×1×H×D]
  K,V: Read [B×(context+i)×H×D] from cache`
    },
    {
      name: 'Prefill vs Decode',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Two distinct phases of LLM inference with different computational characteristics.',
      sampleValues: [],
      formula: `PREFILL (Context Encoding):
  Input: Full prompt [B×N]
  Output: Initial KV cache + first token
  Compute: O(N²) attention
  Characteristic: Compute-bound
  Arithmetic Intensity: ~100 ops/byte
  Parallelism: High (process all tokens together)

DECODE (Token Generation):
  Input: Single new token [B×1]
  Output: Next token prediction
  Compute: O(N) attention (read cached KV)
  Characteristic: Memory-bound
  Arithmetic Intensity: ~10 ops/byte
  Parallelism: Low (sequential generation)`
    },
    {
      name: 'POD (Prefill-Optimized-Decode)',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Concurrent execution of prefill and decode on the same GPU by partitioning compute resources.',
      sampleValues: [],
      formula: `Traditional Approach:
  Run prefill → Wait → Run decode
  Problem: GPU underutilized during decode (memory-bound)

POD Approach:
  Partition GPU: prefillRatio + decodeRatio = 1.0
  Run prefill AND decode concurrently
  Example:
    - 72 CUs: Prefill (compute-intensive)
    - 48 CUs: Decode (memory-intensive)

Benefits:
  - Better GPU utilization
  - Higher throughput (more requests/sec)
  - Balanced compute + memory bandwidth usage

Trade-off:
  - Slightly slower individual requests
  - Much higher aggregate throughput`
    },
    {
      name: 'CU Allocation',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Partitioning GPU compute units between concurrent workloads.',
      sampleValues: [],
      formula: `GPU Resources (MI300X):
  Total CUs: 120
  Total HBM Bandwidth: ~5 TB/s
  Total Compute: ~1.3 PFLOPS (FP16)

Allocation:
  Prefill gets: 120 × prefillRatio CUs
    - Uses compute heavily
    - Moderate memory bandwidth

  Decode gets: 120 × decodeRatio CUs
    - Uses memory bandwidth heavily
    - Light compute

Optimal Split:
  Balance compute vs memory bottlenecks
  Typical: 60-70% prefill, 30-40% decode`
    }
  ];

  return [...commonEntries, ...specificEntries];
};
