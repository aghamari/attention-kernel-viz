// Attention Kernel Types for Visualization

// Base configuration shared across attention types
export interface BaseAttentionConfig {
  batchSize: number;
  seqLen: number;
  numHeads: number;
  headDim: number;
  numKvHeads: number;
}

// Unified Attention Config (2D & 3D)
export interface UnifiedAttentionConfig extends BaseAttentionConfig {
  blockM: number;         // Query block size
  blockN: number;         // Key/Value block size
  numSegments: number;    // For 3D kernel
  useSlidingWindow: boolean;
  windowSize: number;
  use3DKernel: boolean;   // Toggle between 2D and 3D
  causal: boolean;
}

// Flash Attention (MHA) Config
export interface FlashAttentionConfig extends BaseAttentionConfig {
  dropout: number;
  useCausalMask: boolean;
  useAlibi: boolean;
  alibiSlope: number;
  useFP8: boolean;
  softmaxScale: number;
}

// Paged Attention Config
export interface PagedAttentionConfig extends BaseAttentionConfig {
  blockSize: number;       // Tokens per physical block
  numBlocks: number;       // Total physical blocks
  maxContextLen: number;
  useV2: boolean;          // Use partitioned version
  numPartitions: number;
}

// Lean Attention (StreamK) Config
export interface LeanAttentionConfig extends BaseAttentionConfig {
  numCUs: number;          // Compute units
  tilesPerCU: number;
  enableWorkStealing: boolean;
  persistentKernel: boolean;
}

// HSTU Attention Config
export interface HSTUAttentionConfig extends BaseAttentionConfig {
  alpha: number;           // SiLU gate parameter
  maxSeqLenK: number;
  maxSeqLenQ: number;
  useJaggedTensors: boolean;
}

// MLA Decode + RoPE Config
export interface MLADecodeConfig extends BaseAttentionConfig {
  kvLatentDim: number;     // Compressed KV dimension
  ropeTheta: number;       // RoPE base
  ropeScale: number;
  useRoPE: boolean;
  nope_size: number;
  rope_size: number;
}

// Sage Attention Config
export interface SageAttentionConfig extends BaseAttentionConfig {
  useFP8: boolean;
  useMxFP4: boolean;
  perTokenScale: boolean;
  smoothK: boolean;
}

// POD Attention Config
export interface PODAttentionConfig extends BaseAttentionConfig {
  prefillRatio: number;    // 0-1 portion for prefill
  decodeRatio: number;     // 0-1 portion for decode
  prefillSeqLen: number;
  decodeNumTokens: number;
}

// CK-UA (CK Tile Unified Attention) Config
export interface CKUnifiedConfig extends BaseAttentionConfig {
  kBlockM: 16 | 32 | 64 | 128 | 256;  // Q tile size per tier
  kBlockQ: number;                     // = kBlockM / numQueriesPerKv
  pageBlockSize: 32 | 64;              // Page block size
  maskType: 0 | 2;                     // 0=no_mask, 2=causal
  tier: 'tiny' | 'bs32' | 'small' | 'medium' | 'large';
  scaleS: number;                      // Softmax scale
  scaleK: number;                      // K quantization scale
  scaleV: number;                      // V quantization scale
  isDecodeGrid: boolean;               // 2D decode vs 1D prefill
  numQueriesPerKv: number;             // GQA ratio
}

// CK-SK (CK FMHA Split-KV) Config
export interface CKSplitKVConfig extends BaseAttentionConfig {
  numSplits: number;           // Chosen by heuristic (target = CU_count * 4)
  maxSeqLenQ: number;
  maxSeqLenK: number;
  softmaxScale: number;
  logitsSoftCap: number;       // 0 = disabled
  windowSizeLeft: number;      // -1 = no window
  windowSizeRight: number;
  sinkSize: number;
  isCausal: boolean;
}

// CK-PK (CK FMHA PagedKV) Config
export interface CKPagedKVConfig extends BaseAttentionConfig {
  bm0: 16 | 32 | 128;          // Q tile size
  bn0: number;                  // KV tile size (typically 32)
  pageSize: 32 | 64 | 128 | 256;
  maxSeqLenQ: number;
  maxSeqLenK: number;
  windowSizeLeft: number;
  windowSizeRight: number;
  sinkSize: number;
  isCausal: boolean;
}

// CK-Fwd (CK FMHA Forward Non-Paged) Config
export interface CKForwardConfig extends BaseAttentionConfig {
  bm0: number;                  // Q tile size
  maxSeqLenQ: number;
  maxSeqLenK: number;
  softmaxScale: number;
  logitsSoftCap: number;        // 0 = disabled
  dropoutP: number;
  windowSizeLeft: number;
  windowSizeRight: number;
  isCausal: boolean;
  useBias: boolean;
}

// Grid launch info for visualization
export interface GridLaunchInfo {
  gridX: number;
  gridY: number;
  gridZ: number;
  blockX: number;
  blockY: number;
  blockZ: number;
  totalThreads: number;
}

// Segment info for 3D unified attention
export interface SegmentInfo {
  segmentId: number;
  startIdx: number;
  endIdx: number;
  partialMax: number;
  partialExpSum: number;
  partialOutput: number[];
}

// Block table entry for paged attention
export interface BlockTableEntry {
  logicalBlockIdx: number;
  physicalBlockIdx: number;
  seqIdx: number;
}

// Performance metrics
export interface PerformanceMetrics {
  flops: number;
  memoryBandwidth: number;
  computeUtilization: number;
  occupancy: number;
  arithmeticIntensity: number;
}

// Sample tensor data for visualization
export interface SampleTensor {
  shape: number[];          // Actual numeric shape
  shapeStr: string;         // e.g., "[B×S×D]"
  actualDims: string;       // e.g., "2×1024×768"
  memoryBytes: number;
  memoryStr: string;        // e.g., "1.57 MB"
  sampleValues: number[][]; // Small 4×4 sample for display
}

// Memory layout for visualization
export interface MemoryLayoutInfo {
  globalMemory: { name: string; bytes: number; shape: string }[];
  sharedMemory: { name: string; bytes: number }[];
  registers: { name: string; count: number }[];
}

// Algorithm step for step-by-step visualization
export type AlgorithmStep =
  | 'idle'
  | 'proj'        // Q/K/V projection
  | 'qk-matmul'   // Q @ K^T
  | 'softmax'     // Softmax normalization
  | 'av-matmul'   // Attention @ V
  | 'output';     // Final output

// Attention type enum
export type AttentionType =
  | 'unified'
  | 'flash'
  | 'paged'
  | 'lean'
  | 'hstu'
  | 'mla'
  | 'sage'
  | 'pod'
  | 'ck-ua'
  | 'ck-sk'
  | 'ck-pk'
  | 'ck-fwd'
  | 'triton2d-viz'
  | 'comparison';

// Color schemes for different attention types
export const ATTENTION_COLORS: Record<AttentionType, { primary: string; secondary: string; gradient: string }> = {
  unified: {
    primary: '#e94560',
    secondary: '#0f3460',
    gradient: 'linear-gradient(135deg, #e94560 0%, #0f3460 100%)'
  },
  flash: {
    primary: '#ff6b6b',
    secondary: '#4ecdc4',
    gradient: 'linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%)'
  },
  paged: {
    primary: '#667eea',
    secondary: '#764ba2',
    gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
  },
  lean: {
    primary: '#f093fb',
    secondary: '#f5576c',
    gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
  },
  hstu: {
    primary: '#4facfe',
    secondary: '#00f2fe',
    gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
  },
  mla: {
    primary: '#43e97b',
    secondary: '#38f9d7',
    gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
  },
  sage: {
    primary: '#fa709a',
    secondary: '#fee140',
    gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
  },
  pod: {
    primary: '#a18cd1',
    secondary: '#fbc2eb',
    gradient: 'linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)'
  },
  'ck-ua': {
    primary: '#FF5722',
    secondary: '#FF9800',
    gradient: 'linear-gradient(135deg, #FF5722 0%, #FF9800 100%)'
  },
  'ck-sk': {
    primary: '#00BCD4',
    secondary: '#009688',
    gradient: 'linear-gradient(135deg, #00BCD4 0%, #009688 100%)'
  },
  'ck-pk': {
    primary: '#E91E63',
    secondary: '#9C27B0',
    gradient: 'linear-gradient(135deg, #E91E63 0%, #9C27B0 100%)'
  },
  'ck-fwd': {
    primary: '#3F51B5',
    secondary: '#2196F3',
    gradient: 'linear-gradient(135deg, #3F51B5 0%, #2196F3 100%)'
  },
  'triton2d-viz': {
    primary: '#e94560',
    secondary: '#0f3460',
    gradient: 'linear-gradient(135deg, #e94560 0%, #0f3460 100%)'
  },
  comparison: {
    primary: '#607D8B',
    secondary: '#455A64',
    gradient: 'linear-gradient(135deg, #607D8B 0%, #455A64 100%)'
  }
};

// Helper functions
export function calculateGridLaunch2D(
  config: UnifiedAttentionConfig
): GridLaunchInfo {
  const numQBlocks = Math.ceil(config.seqLen / config.blockM);
  return {
    gridX: config.numKvHeads,
    gridY: numQBlocks,
    gridZ: 1,
    blockX: 128,
    blockY: 1,
    blockZ: 1,
    totalThreads: config.numKvHeads * numQBlocks * 128
  };
}

export function calculateGridLaunch3D(
  config: UnifiedAttentionConfig
): GridLaunchInfo {
  const numQBlocks = Math.ceil(config.seqLen / config.blockM);
  return {
    gridX: numQBlocks,
    gridY: config.numKvHeads,
    gridZ: config.numSegments,
    blockX: 128,
    blockY: 1,
    blockZ: 1,
    totalThreads: numQBlocks * config.numKvHeads * config.numSegments * 128
  };
}

export function shouldUse2DKernel(config: UnifiedAttentionConfig): boolean {
  // Heuristic from unified_attention.py
  const isShortSequence = config.seqLen <= 2048;
  const isSlidingWindow = config.useSlidingWindow;
  const isDecodeOnly = config.seqLen === 1;

  return isShortSequence || isSlidingWindow || isDecodeOnly;
}

export function calculateFLOPs(config: BaseAttentionConfig): number {
  // Attention FLOPs: 4 * batch * heads * seq^2 * head_dim
  return 4 * config.batchSize * config.numHeads *
         config.seqLen * config.seqLen * config.headDim;
}

export function calculateMemoryBandwidth(config: BaseAttentionConfig): number {
  // Simplified memory bandwidth calculation (bytes)
  const qkvSize = config.batchSize * config.seqLen * config.numHeads * config.headDim * 4 * 3;
  const outputSize = config.batchSize * config.seqLen * config.numHeads * config.headDim * 4;
  return (qkvSize + outputSize) / 1e9; // GB
}

// CK-UA tier selection based on avg_q
export function selectCKUATier(avgQ: number, maxSeqLenQ: number, blockSize: number): {
  tier: 'tiny' | 'bs32' | 'small' | 'medium' | 'large';
  kBlockM: 16 | 32 | 64 | 128 | 256;
  warps: number;
  mfma: string;
} {
  if (avgQ <= 2) {
    return { tier: 'tiny', kBlockM: 16, warps: 1, mfma: '16x16x32' };
  }
  if (blockSize === 32 && avgQ <= 4) {
    return { tier: 'bs32', kBlockM: 32, warps: 2, mfma: '16x16x32' };
  }
  if (avgQ <= 8) {
    return { tier: 'small', kBlockM: 64, warps: 2, mfma: '32x32x16' };
  }
  if (maxSeqLenQ <= 128) {
    return { tier: 'medium', kBlockM: 128, warps: 4, mfma: '32x32x16' };
  }
  return { tier: 'large', kBlockM: 256, warps: 8, mfma: '32x32x16' };
}

// CK-UA selector logic (from attention_pipelines.md)
export function shouldUseCKUA(
  maxSeqLenQ: number,
  numSeqs: number,
  numKvHeads: number,
  windowSize: [number, number],
  blockSize: number,
  maxSeqLenK: number,
  headSize: number,
  numQueriesPerKv: number,
  cuCount: number = 256
): boolean {
  if (maxSeqLenQ !== 1) return false;           // decode only
  if (windowSize[0] !== -1 || windowSize[1] !== -1) return false;  // no sliding window
  if (blockSize < 32) return false;
  if (blockSize < 64 && maxSeqLenK < 256) return false;

  // Only compiled for these configs
  const validConfig =
    (headSize === 64 && numQueriesPerKv === 8) ||
    (headSize === 128 && numQueriesPerKv === 1);
  if (!validConfig) return false;

  const triton2dWgs = numKvHeads * numSeqs;
  return cuCount * 4 <= triton2dWgs && triton2dWgs <= cuCount * 8;
}

// Calculate grid launch for CK-UA
export function calculateCKUAGrid(
  numKvHeads: number,
  numSeqs: number,
  totalNumQBlocks: number,
  isDecodeGrid: boolean
): GridLaunchInfo {
  if (isDecodeGrid) {
    // 2D decode grid: dim3(num_kv_heads, num_seqs)
    return {
      gridX: numKvHeads,
      gridY: numSeqs,
      gridZ: 1,
      blockX: 256,
      blockY: 1,
      blockZ: 1,
      totalThreads: numKvHeads * numSeqs * 256
    };
  } else {
    // 1D prefill grid: dim3(num_kv_heads * total_num_q_blocks)
    return {
      gridX: numKvHeads * totalNumQBlocks,
      gridY: 1,
      gridZ: 1,
      blockX: 256,
      blockY: 1,
      blockZ: 1,
      totalThreads: numKvHeads * totalNumQBlocks * 256
    };
  }
}

// Calculate grid launch for CK-SK
export function calculateCKSKGrid(
  batchSize: number,
  numHeadsQ: number,
  numSplits: number,
  maxSeqLenQ: number
): { attention: GridLaunchInfo; combine: GridLaunchInfo } {
  // Attention kernel: dim3(batch * nhead_q * num_splits)
  const attentionTotal = batchSize * numHeadsQ * numSplits;
  // Combine kernel: dim3(batch * nhead_q * max_seqlen_q)
  const combineTotal = batchSize * numHeadsQ * maxSeqLenQ;

  return {
    attention: {
      gridX: attentionTotal,
      gridY: 1,
      gridZ: 1,
      blockX: 256,
      blockY: 1,
      blockZ: 1,
      totalThreads: attentionTotal * 256
    },
    combine: {
      gridX: combineTotal,
      gridY: 1,
      gridZ: 1,
      blockX: 256,
      blockY: 1,
      blockZ: 1,
      totalThreads: combineTotal * 256
    }
  };
}

// Calculate num_splits heuristic for CK-SK
export function calculateNumSplits(
  batchSize: number,
  numHeadsK: number,
  maxSeqLenQ: number,
  cuCount: number = 256
): number {
  // Target = multiProcessorCount * 4
  const target = cuCount * 4;
  const baseWgs = batchSize * numHeadsK * maxSeqLenQ;
  if (baseWgs >= target) return 1;
  return Math.min(32, Math.ceil(target / baseWgs));
}
