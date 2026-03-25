import { GlossaryEntry } from '../../components/shared/ParameterGlossary';
import { createCommonGlossaryEntries } from './commonGlossary';

export const createPagedGlossaryEntries = (config: any): GlossaryEntry[] => {
  const commonEntries = createCommonGlossaryEntries(config);

  const specificEntries: GlossaryEntry[] = [
    {
      name: 'blockSize',
      type: 'int',
      category: 'specific',
      range: '8-64',
      currentValue: config.blockSize || 16,
      defaultValue: 16,
      description: 'Number of tokens per physical memory block. Determines granularity of KV cache management.',
      sampleValues: [
        { value: 8, explanation: 'Fine-grained allocation (less fragmentation)' },
        { value: 16, explanation: 'Balanced (standard vLLM default)' },
        { value: 32, explanation: 'Coarse blocks (fewer lookups, more waste)' }
      ],
      formula: `Block Memory:
  Per block = blockSize × 2 (K+V) × num_kv_heads × head_dim

Example (blockSize=16, 8 heads, dim=64):
  16 × 2 × 8 × 64 = 16,384 values per block
  In FP16: 32 KB per block`
    },
    {
      name: 'numBlocks',
      type: 'int',
      category: 'specific',
      range: '16-128',
      currentValue: config.numBlocks || 32,
      defaultValue: 32,
      description: 'Total number of physical KV cache blocks. Determines maximum concurrent sequences.',
      sampleValues: [
        { value: 16, explanation: 'Limited capacity (few concurrent seqs)' },
        { value: 32, explanation: 'Moderate capacity' },
        { value: 128, explanation: 'High capacity (many concurrent seqs)' }
      ],
      formula: `Max Concurrent Sequences:
  Depends on sequence lengths and block allocation

Example:
  numBlocks = 64, blockSize = 16
  Total capacity = 64 × 16 = 1024 tokens

  - 4 sequences × 256 tokens each = 4 × 16 blocks = 64 blocks ✓
  - 8 sequences × 128 tokens each = 8 × 8 blocks  = 64 blocks ✓`
    },
    {
      name: 'maxContextLen',
      type: 'int',
      category: 'specific',
      range: '512-8192',
      currentValue: config.maxContextLen || 2048,
      defaultValue: 2048,
      description: 'Maximum sequence length supported. Must satisfy: maxContextLen ≤ numBlocks × blockSize.',
      sampleValues: [
        { value: 512, explanation: 'Short context window' },
        { value: 2048, explanation: 'Standard context (GPT-3.5)' },
        { value: 8192, explanation: 'Extended context (GPT-4)' }
      ]
    },
    {
      name: 'useV2',
      type: 'bool',
      category: 'specific',
      range: 'true/false/auto',
      currentValue: config.useV2 ?? 'auto',
      defaultValue: 'auto',
      description: 'Use V2 (multi-partition) vs V1 (single-partition) kernel. Auto selects based on context length.',
      sampleValues: [
        { value: false, explanation: 'V1 kernel - single partition (context ≤ 8K)' },
        { value: true, explanation: 'V2 kernel - multi-partition reduction (context > 8K)' },
        { value: 'auto', explanation: 'Automatic selection' }
      ],
      formula: `V1 (Single Partition):
  - Direct attention computation
  - No reduction overhead
  - Best for: context_len ≤ 8192

V2 (Multi Partition):
  - Split KV cache across partitions
  - Parallel processing + reduction
  - Best for: context_len > 8192

Selection: Use V2 if maxContextLen > 8192`
    },
    {
      name: 'numPartitions',
      type: 'int',
      category: 'specific',
      range: '1-16',
      currentValue: config.numPartitions || 4,
      defaultValue: 4,
      description: 'Number of partitions for V2 reduction. Only used when useV2 = true.',
      sampleValues: [
        { value: 1, explanation: 'No partitioning (V1 mode)' },
        { value: 4, explanation: 'Balanced parallelism' },
        { value: 8, explanation: 'Maximum parallelism for very long contexts' }
      ]
    },
    {
      name: 'Block Table',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Maps logical KV blocks to physical memory locations. Enables non-contiguous memory allocation.',
      sampleValues: [],
      formula: `Block Table: [num_sequences][max_blocks_per_seq]

Lookup:
  logical_block_id = token_position / blockSize
  physical_block_id = block_table[seq_id][logical_block_id]
  offset = token_position % blockSize

Access KV:
  K_ptr = blocks[physical_block_id] + offset
  V_ptr = blocks[physical_block_id] + blockSize + offset

Benefits:
  - No memory fragmentation
  - Efficient sharing (e.g., beam search)
  - Dynamic allocation/deallocation`
    },
    {
      name: 'KV Cache',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Cached key/value tensors to avoid recomputation during autoregressive generation.',
      sampleValues: [],
      formula: `Generation without KV Cache:
  Token 1: Compute K[0:1], V[0:1]
  Token 2: Compute K[0:2], V[0:2]  ← recomputes token 1!
  Token 3: Compute K[0:3], V[0:3]  ← recomputes tokens 1,2!
  Cost: O(N²) for N tokens

Generation with KV Cache:
  Token 1: Compute K[0], V[0], store in cache
  Token 2: Compute K[1], V[1], append to cache
  Token 3: Compute K[2], V[2], append to cache
  Cost: O(N) for N tokens

Memory: [batch, max_seq_len, num_kv_heads, head_dim] × 2 (K+V)`
    },
    {
      name: 'Partition Reduction',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'V2 kernel phase that combines attention outputs from multiple KV partitions.',
      sampleValues: [],
      formula: `V2 Algorithm:
1. Split KV cache into P partitions
2. Compute partial attention for each partition:
     O_p = softmax(Q @ K_p^T) @ V_p
     Also track: max_logit_p, sum_exp_p

3. Reduction phase:
     global_max = max(max_logit_1, ..., max_logit_P)
     For each partition p:
       rescale_p = exp(max_logit_p - global_max)
       O_final += O_p × rescale_p × sum_exp_p
     O_final /= Σ(rescale_p × sum_exp_p)

Enables: Very long contexts (>8K tokens) with bounded memory`
    }
  ];

  return [...commonEntries, ...specificEntries];
};
