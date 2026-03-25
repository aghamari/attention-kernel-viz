import { GlossaryEntry } from '../../components/shared/ParameterGlossary';

export const createCommonGlossaryEntries = (config: any): GlossaryEntry[] => [
  {
    name: 'batchSize',
    type: 'int',
    category: 'common',
    range: '1-64',
    currentValue: config.batchSize || 1,
    defaultValue: 1,
    description: 'Number of sequences processed in parallel. Higher batch sizes improve GPU utilization but require more memory.',
    sampleValues: [
      { value: 1, explanation: 'Single sequence (minimal memory)' },
      { value: 8, explanation: 'Typical serving batch' },
      { value: 32, explanation: 'High-throughput training' }
    ]
  },
  {
    name: 'seqLen',
    type: 'int',
    category: 'common',
    range: '64-8192',
    currentValue: config.seqLen || 512,
    defaultValue: 512,
    description: 'Number of tokens in the input sequence. Determines context window size and memory usage (O(N²) for attention scores).',
    sampleValues: [
      { value: 512, explanation: 'Short context (e.g., chat messages)' },
      { value: 2048, explanation: 'Medium context (e.g., articles)' },
      { value: 4096, explanation: 'Long context (e.g., documents)' }
    ]
  },
  {
    name: 'numHeads',
    type: 'int',
    category: 'common',
    range: '1-32',
    currentValue: config.numHeads || 8,
    defaultValue: 8,
    description: 'Number of attention heads for multi-head attention. Allows the model to attend to different representation subspaces.',
    sampleValues: [
      { value: 8, explanation: 'Standard for smaller models' },
      { value: 16, explanation: 'Common for medium models' },
      { value: 32, explanation: 'Large models (e.g., GPT-3)' }
    ]
  },
  {
    name: 'headDim',
    type: 'int',
    category: 'common',
    range: '32-128',
    currentValue: config.headDim || 64,
    defaultValue: 64,
    description: 'Dimension of each attention head. Total model dimension = numHeads × headDim.',
    sampleValues: [
      { value: 64, explanation: 'Standard dimension (512 total with 8 heads)' },
      { value: 80, explanation: 'Common in some architectures' },
      { value: 128, explanation: 'Larger heads (1024 total with 8 heads)' }
    ]
  },
  {
    name: 'numKvHeads',
    type: 'int',
    category: 'common',
    range: '1-numHeads',
    currentValue: config.numKvHeads || config.numHeads || 8,
    defaultValue: 8,
    description: 'Number of key/value heads (can be < query heads for Grouped Query Attention). Reduces KV cache memory usage.',
    sampleValues: [
      { value: 1, explanation: 'Multi-Query Attention (MQA) - minimal memory' },
      { value: 4, explanation: 'Grouped Query Attention (GQA) - balanced' },
      { value: 8, explanation: 'Standard Multi-Head Attention (MHA)' }
    ]
  },
  {
    name: 'causal',
    type: 'bool',
    category: 'common',
    range: 'true/false',
    currentValue: config.causal ?? true,
    defaultValue: true,
    description: 'Apply causal masking (prevent attention to future tokens). Essential for autoregressive language modeling.',
    sampleValues: [
      { value: true, explanation: 'Language modeling, text generation' },
      { value: false, explanation: 'Bidirectional tasks (e.g., BERT, encoding)' }
    ]
  },
  {
    name: 'Query (Q)',
    type: 'string',
    category: 'technical',
    range: 'N/A',
    currentValue: '[B×S×H×D]',
    defaultValue: 'N/A',
    description: '"What am I looking for?" - Represents the current token asking questions about the sequence.',
    sampleValues: []
  },
  {
    name: 'Key (K)',
    type: 'string',
    category: 'technical',
    range: 'N/A',
    currentValue: '[B×S×H×D]',
    defaultValue: 'N/A',
    description: '"What do I contain?" - Represents all tokens in the sequence being queried.',
    sampleValues: []
  },
  {
    name: 'Value (V)',
    type: 'string',
    category: 'technical',
    range: 'N/A',
    currentValue: '[B×S×H×D]',
    defaultValue: 'N/A',
    description: '"What information do I carry?" - The actual content that gets aggregated based on attention weights.',
    sampleValues: []
  },
  {
    name: 'Attention Mechanism',
    type: 'string',
    category: 'formula',
    range: 'N/A',
    currentValue: 'N/A',
    defaultValue: 'N/A',
    description: 'Complete attention computation with scaled dot-product and softmax normalization.',
    sampleValues: [],
    formula: `Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V

Where:
  Q @ K^T = Attention Scores (dot product similarity)
  / √d = Scaling factor (d = head_dim)
  softmax(...) = Attention Weights (normalized probabilities)
  @ V = Weighted sum of values (output)

Memory Complexity: O(N²) for standard attention
Flash Attention: O(N) via online softmax`
  },
  {
    name: 'Tiling',
    type: 'string',
    category: 'technical',
    range: 'N/A',
    currentValue: 'N/A',
    defaultValue: 'N/A',
    description: 'Break computation into blocks that fit in fast shared memory (~48 KB per block). Reduces global memory access.',
    sampleValues: []
  },
  {
    name: 'Online Softmax',
    type: 'string',
    category: 'technical',
    range: 'N/A',
    currentValue: 'N/A',
    defaultValue: 'N/A',
    description: 'Compute softmax incrementally without materializing the full attention matrix. Core technique of Flash Attention for memory efficiency.',
    sampleValues: []
  },
  {
    name: 'GPU Memory Hierarchy',
    type: 'string',
    category: 'technical',
    range: 'N/A',
    currentValue: 'N/A',
    defaultValue: 'N/A',
    description: 'GPU memory levels ordered by speed and size. Optimized kernels maximize use of faster memory.',
    sampleValues: [],
    formula: `Registers: Fastest, ~65K per SM, private per thread
Shared Memory: Fast, ~48 KB per block, ~19 TB/s
L1/L2 Cache: Medium, automatic caching
Global Memory (HBM): Slow but large, ~40 GB, ~1.5 TB/s

Optimization: Keep frequently accessed data in shared memory`
  }
];
