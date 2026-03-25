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

// Animation state
export interface AnimationState {
  isPlaying: boolean;
  currentStep: number;
  totalSteps: number;
  speed: number;
}

// Visualization mode
export type VisualizationMode =
  | 'overview'
  | 'grid-launch'
  | 'memory-access'
  | 'computation'
  | 'reduction';

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
  | 'pod';

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
