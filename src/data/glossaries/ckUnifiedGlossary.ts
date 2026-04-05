import { GlossaryEntry } from '../../components/shared/ParameterGlossary';
import { createCommonGlossaryEntries } from './commonGlossary';

export const createCKUnifiedGlossaryEntries = (config: any): GlossaryEntry[] => {
  const commonEntries = createCommonGlossaryEntries(config);

  const specificEntries: GlossaryEntry[] = [
    {
      name: 'kBlockM',
      type: 'int',
      category: 'specific',
      range: '16-256',
      currentValue: config.kBlockM || 64,
      defaultValue: 64,
      description: 'Q tile size (rows in M dimension). Automatically selected based on tier. Packs GQA heads into rows.',
      sampleValues: [
        { value: 16, explanation: 'Tiny tier - pure decode (avg_q ≤ 2)' },
        { value: 32, explanation: 'BS32 tier - block_size=32 decode (avg_q ≤ 4)' },
        { value: 64, explanation: 'Small tier - short decode (avg_q ≤ 8)' },
        { value: 128, explanation: 'Medium tier - prefill (max_seqlen_q ≤ 128)' },
        { value: 256, explanation: 'Large tier - long prefill' }
      ],
      formula: `Tier Selection:
avg_q = num_tokens / num_seqs
if avg_q ≤ 2: Tiny (kBlockM=16, 1 warp, MFMA 16x16x32)
if avg_q ≤ 4 && block_size=32: BS32 (kBlockM=32, 2 warps)
if avg_q ≤ 8: Small (kBlockM=64, 2 warps, MFMA 32x32x16)
if max_seqlen_q ≤ 128: Medium (kBlockM=128, 4 warps)
else: Large (kBlockM=256, 8 warps)`
    },
    {
      name: 'kBlockQ',
      type: 'int',
      category: 'specific',
      range: 'kBlockM / numQueriesPerKv',
      currentValue: config.kBlockQ || 8,
      defaultValue: 8,
      description: 'Number of Q tokens per tile after GQA head-merging. kBlockQ = kBlockM / num_queries_per_kv',
      sampleValues: [
        { value: 2, explanation: 'kBlockM=16, GQA-8' },
        { value: 8, explanation: 'kBlockM=64, GQA-8' },
        { value: 16, explanation: 'kBlockM=128, GQA-8' }
      ],
      formula: 'kBlockQ = kBlockM / numQueriesPerKv'
    },
    {
      name: 'pageBlockSize',
      type: 'int',
      category: 'specific',
      range: '32, 64',
      currentValue: config.pageBlockSize || 64,
      defaultValue: 64,
      description: 'Paged KV cache block size (tokens per physical page). 32 or 64 supported.',
      sampleValues: [
        { value: 32, explanation: 'Smaller pages, finer granularity' },
        { value: 64, explanation: 'Larger pages, fewer indirections' }
      ]
    },
    {
      name: 'numQueriesPerKv',
      type: 'int',
      category: 'specific',
      range: '1, 8',
      currentValue: config.numQueriesPerKv || 8,
      defaultValue: 8,
      description: 'GQA ratio: number of Q-heads sharing each KV-head. Enables head-merge optimization.',
      sampleValues: [
        { value: 1, explanation: 'MHA (Multi-Head Attention) - each Q-head has own KV-head' },
        { value: 8, explanation: 'GQA-8 - 8 Q-heads share 1 KV-head (head-merge optimization)' }
      ],
      formula: `Head Merge:
With GQA-8, kBlockM=64:
64 rows = 8 Q-tokens × 8 Q-heads
All 8 Q-heads attend to same KV (1 KV-head)
Single MFMA computes scores for all 8 heads simultaneously`
    },
    {
      name: 'tier',
      type: 'enum',
      category: 'specific',
      range: 'tiny, bs32, small, medium, large',
      currentValue: config.tier || 'small',
      defaultValue: 'small',
      description: 'Tile tier automatically selected by avg_q and max_seqlen_q heuristics',
      sampleValues: [
        { value: 'tiny', explanation: 'Pure decode (avg_q ≤ 2): kBlockM=16, 1 warp' },
        { value: 'bs32', explanation: 'BS32 decode: kBlockM=32, 2 warps' },
        { value: 'small', explanation: 'Short decode (avg_q ≤ 8): kBlockM=64, 2 warps' },
        { value: 'medium', explanation: 'Prefill: kBlockM=128, 4 warps' },
        { value: 'large', explanation: 'Long prefill: kBlockM=256, 8 warps' }
      ]
    },
    {
      name: 'isDecodeGrid',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.isDecodeGrid ?? true,
      defaultValue: true,
      description: '2D decode grid vs 1D prefill grid selection',
      sampleValues: [
        { value: true, explanation: '2D decode grid: dim3(num_kv_heads, num_seqs)' },
        { value: false, explanation: '1D prefill grid: dim3(num_kv_heads * total_num_q_blocks)' }
      ],
      formula: `Grid Launch:
Decode (seqLen=1): grid = dim3(num_kv_heads, num_seqs)
Prefill (seqLen>1): grid = dim3(num_kv_heads * total_num_q_blocks)
total_num_q_blocks = ⌈num_tokens / kBlockQ⌉`
    },
    {
      name: 'maskType',
      type: 'enum',
      category: 'specific',
      range: '0, 2',
      currentValue: config.maskType || 2,
      defaultValue: 2,
      description: 'Masking mode: 0=no_mask, 2=causal',
      sampleValues: [
        { value: 0, explanation: 'No mask - full bidirectional attention' },
        { value: 2, explanation: 'Causal mask - autoregressive (Q[i] attends to K[≤i])' }
      ]
    },
    {
      name: 'scaleS',
      type: 'float',
      category: 'specific',
      range: '0.0-1.0',
      currentValue: config.scaleS || (1.0 / Math.sqrt(config.headDim)),
      defaultValue: 1.0 / Math.sqrt(64),
      description: 'Softmax scale factor (typically 1/sqrt(head_dim))',
      sampleValues: [
        { value: 0.125, explanation: 'head_dim=64: 1/sqrt(64) = 1/8' },
        { value: 0.088, explanation: 'head_dim=128: 1/sqrt(128)' }
      ],
      formula: 'scaleS = 1 / sqrt(head_dim)'
    }
  ];

  return [...commonEntries, ...specificEntries];
};
