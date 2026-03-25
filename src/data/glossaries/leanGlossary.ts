import { GlossaryEntry } from '../../components/shared/ParameterGlossary';
import { createCommonGlossaryEntries } from './commonGlossary';

export const createLeanGlossaryEntries = (config: any): GlossaryEntry[] => {
  const commonEntries = createCommonGlossaryEntries(config);

  const specificEntries: GlossaryEntry[] = [
    {
      name: 'numCUs',
      type: 'int',
      category: 'specific',
      range: '4-120',
      currentValue: config.numCUs || 8,
      defaultValue: 8,
      description: 'Number of compute units (GPU streaming multiprocessors). MI300X has 120 CUs, A100 has 108 SMs.',
      sampleValues: [
        { value: 8, explanation: 'Limited CUs (testing, small GPU)' },
        { value: 60, explanation: 'Half MI300X capacity' },
        { value: 120, explanation: 'Full MI300X (maximum parallelism)' }
      ],
      formula: `Common GPU CU/SM counts:
  AMD MI300X: 120 CUs
  AMD MI250X: 110 CUs
  NVIDIA A100: 108 SMs
  NVIDIA H100: 132 SMs

Higher numCUs = more parallel work tiles`
    },
    {
      name: 'tilesPerCU',
      type: 'int',
      category: 'specific',
      range: '1-8',
      currentValue: config.tilesPerCU || 4,
      defaultValue: 4,
      description: 'Number of work tiles assigned to each CU. More tiles = better load balancing but higher overhead.',
      sampleValues: [
        { value: 1, explanation: 'Minimal tiles (less balancing, lower overhead)' },
        { value: 4, explanation: 'Balanced (good load distribution)' },
        { value: 8, explanation: 'Fine-grained (best balance, more overhead)' }
      ],
      formula: `Total Work Tiles = ⌈seqLen / tileM⌉ × ⌈seqLen / tileN⌉

Tile Assignment:
  Static: Each CU gets tilesPerCU tiles
  Dynamic (with stealing): CUs take from queue

Trade-off:
  More tiles → Better load balance, more scheduling overhead
  Fewer tiles → Less overhead, worse load balance`
    },
    {
      name: 'enableWorkStealing',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.enableWorkStealing ?? true,
      defaultValue: true,
      description: 'Allow idle CUs to steal work from busy neighbors. Improves load balance for irregular workloads.',
      sampleValues: [
        { value: false, explanation: 'Static assignment (predictable, may have idle CUs)' },
        { value: true, explanation: 'Dynamic stealing (better utilization, slight overhead)' }
      ],
      formula: `Without Work Stealing:
  CU[i] processes tiles[i×T : (i+1)×T]
  Problem: Some tiles finish early → idle CUs

With Work Stealing:
  Each CU has local queue
  When idle: CU steals from neighbor's queue
  Result: All CUs stay busy until work complete

Efficiency = actual_work / (numCUs × max_CU_time)
  Work stealing increases efficiency for irregular tiles`
    },
    {
      name: 'persistentKernel',
      type: 'bool',
      category: 'specific',
      range: 'true/false',
      currentValue: config.persistentKernel ?? true,
      defaultValue: true,
      description: 'Use single persistent kernel vs multiple kernel launches. Reduces launch overhead for multi-tile workloads.',
      sampleValues: [
        { value: false, explanation: 'Multiple launches (simple, higher overhead)' },
        { value: true, explanation: 'Single persistent kernel (lower overhead, complex)' }
      ],
      formula: `Non-Persistent:
  for each tile:
    launch_kernel(tile)  ← overhead per launch
  Total overhead: N × launch_cost

Persistent:
  launch_kernel_once():
    while tiles_remaining:
      tile = get_next_tile()
      process(tile)
  Total overhead: 1 × launch_cost

Best for: Multiple tiles per GPU (tilesPerCU > 1)`
    },
    {
      name: 'StreamK Decomposition',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: 'N/A',
      defaultValue: 'N/A',
      description: 'Stream-K is a decomposition strategy for load-balanced GEMM/attention. Splits work into fine-grained tiles.',
      sampleValues: [],
      formula: `Standard Tiling:
  Split Q into M/blockM row blocks
  Split K,V into N/blockN col blocks
  Problem: Uneven load if some tiles finish faster

StreamK Tiling:
  Smaller tiles (more granular)
  Dynamic scheduling across CUs
  Work stealing for idle units

Benefits:
  - Better load balance
  - Higher GPU utilization
  - Especially for irregular shapes/sparsity`
    },
    {
      name: 'Work Stealing',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: config.enableWorkStealing ? 'Enabled' : 'Disabled',
      defaultValue: 'Enabled',
      description: 'Dynamic task redistribution where idle workers take work from busy workers.',
      sampleValues: [],
      formula: `Algorithm:
1. Each CU has local work queue
2. CU processes tiles from its queue
3. When queue empty:
     - Try to steal from neighbor
     - Neighbor = (CU_id + offset) % numCUs
4. Repeat until all work done

Stealing Strategy:
  - Take half of neighbor's remaining work
  - Reduces contention
  - Maintains locality

Overhead: ~5-10% from synchronization
Benefit: 20-40% better utilization for irregular work`
    },
    {
      name: 'Persistent Kernel',
      type: 'string',
      category: 'technical',
      range: 'N/A',
      currentValue: config.persistentKernel ? 'Enabled' : 'Disabled',
      defaultValue: 'Enabled',
      description: 'Kernel stays active and processes multiple work items instead of relaunching.',
      sampleValues: [],
      formula: `Traditional Multi-Launch:
  for tile in tiles:
    kernel<<<grid, block>>>(tile)
    wait()

  Overhead: Launch latency × num_tiles (~5-10μs per launch)

Persistent Kernel:
  kernel<<<numCUs, block>>>():
    while (tile = get_next_tile()):
      process(tile)

  Overhead: Single launch (~5-10μs total)

When beneficial:
  - Many tiles (>100)
  - Small tiles (quick processing)
  - Work stealing (dynamic scheduling)`
    }
  ];

  return [...commonEntries, ...specificEntries];
};
