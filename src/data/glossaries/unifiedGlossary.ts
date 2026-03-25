import { GlossaryEntry } from '../../components/shared/ParameterGlossary';
import { createCommonGlossaryEntries } from './commonGlossary';

export const createUnifiedGlossaryEntries = (config: any): GlossaryEntry[] => {
  const commonEntries = createCommonGlossaryEntries(config);

  const specificEntries: GlossaryEntry[] = [
    {
      name: 'blockM',
      type: 'int',
      category: 'specific',
      range: '16-128',
      currentValue: config.blockM || 64,
      defaultValue: 64,
      description: 'Query block size for tiling. Determines how many query rows are processed together in shared memory.',
      sampleValues: [
        { value: 16, explanation: 'Low memory usage, more kernel launches' },
        { value: 64, explanation: 'Balanced memory and performance' },
        { value: 128, explanation: 'High throughput, requires more shared memory' }
      ]
    },
    {
      name: 'blockN',
      type: 'int',
      category: 'specific',
      range: '16-128',
      currentValue: config.blockN || 64,
      defaultValue: 64,
      description: 'Key/Value block size for tiling. Determines how many key/value columns are processed together.',
      sampleValues: [
        { value: 16, explanation: 'Fine-grained tiling' },
        { value: 64, explanation: 'Square tiles (balanced, same as blockM)' },
        { value: 128, explanation: 'Rectangular tiles (more parallelism)' }
      ]
    },
    {
      name: 'numSegments',
      type: 'int',
      category: 'specific',
      range: '1-16',
      currentValue: config.numSegments || 4,
      defaultValue: 4,
      description: 'Number of KV segments for 3D kernel. More segments = more parallelism but requires reduction phase.',
      sampleValues: [
        { value: 1, explanation: '2D mode (no segmentation)' },
        { value: 4, explanation: 'Balanced parallelism for 3D kernel' },
        { value: 8, explanation: 'Maximum GPU utilization for long sequences' }
      ]
    },
    {
      name: 'use3DKernel',
      type: 'bool',
      category: 'specific',
      range: 'true/false/auto',
      currentValue: config.use3DKernel ?? 'auto',
      defaultValue: 'auto',
      description: 'Adaptive selection between 2D and 3D grid launch. Auto selects based on sequence length.',
      sampleValues: [
        { value: false, explanation: '2D kernel - single-pass, simple (seqLen ≤ 2048)' },
        { value: true, explanation: '3D kernel - segment reduction, parallel (seqLen > 2048)' },
        { value: 'auto', explanation: 'Automatic selection based on workload' }
      ],
      formula: `Selection Heuristic:
Use 2D if: seqLen ≤ 2048 OR useSlidingWindow = true
Use 3D if: seqLen > 2048 AND useSlidingWindow = false

2D Grid: (num_kv_heads, ⌈seqLen / blockM⌉)
3D Grid: (⌈seqLen / blockM⌉, num_kv_heads, num_segments)`
    },
    {
      name: 'useSlidingWindow',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.useSlidingWindow ?? false,
      defaultValue: false,
      description: 'Enable sliding window attention (local attention). Forces 2D kernel selection for efficient sparse patterns.',
      sampleValues: [
        { value: false, explanation: 'Full attention (all tokens attend to all)' },
        { value: true, explanation: 'Local attention (window-based, saves memory)' }
      ]
    },
    {
      name: 'windowSize',
      type: 'int',
      category: 'specific',
      range: '64-2048',
      currentValue: config.windowSize || 256,
      defaultValue: 256,
      description: 'Window size for sliding window attention. Only used when useSlidingWindow = true.',
      sampleValues: [
        { value: 128, explanation: 'Small local window' },
        { value: 256, explanation: 'Standard window size' },
        { value: 512, explanation: 'Large local context' }
      ],
      formula: `Each token can attend to:
  - windowSize tokens before it
  - Itself
  - (if not causal) windowSize tokens after it

Memory Savings: O(N × W) instead of O(N²)
Where W = windowSize, N = seqLen`
    },
    {
      name: 'Grid Launch Strategy',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: config.use3DKernel ? '3D' : '2D',
      defaultValue: '2D',
      description: 'Configuration of GPU thread blocks for parallel execution.',
      sampleValues: [],
      formula: `2D Grid Launch:
  Grid: (num_kv_heads, ⌈seqLen / blockM⌉)
  - Simple, single-pass
  - Good for decode, short sequences
  - No reduction overhead

3D Grid Launch:
  Grid: (⌈seqLen / blockM⌉, num_kv_heads, num_segments)
  - Parallel segment processing
  - Good for prefill, long sequences
  - Requires final reduction to combine segments`
    },
    {
      name: 'Segment Reduction',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Final phase in 3D kernel that combines partial attention outputs from multiple segments.',
      sampleValues: [],
      formula: `Each segment computes partial attention for its KV range:
  Segment i: attends to K[i×S/n : (i+1)×S/n]

Final reduction combines all segments:
  O_final[q] = Σ(O_segment[q,i] × exp_sum[q,i]) / Σ(exp_sum[q,i])

Where exp_sum = sum of exponentials for softmax normalization`
    }
  ];

  return [...commonEntries, ...specificEntries];
};
