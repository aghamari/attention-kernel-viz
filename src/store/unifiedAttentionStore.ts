// Unified Attention Store - State management for 2D/3D attention visualization

import { create } from 'zustand';
import {
  UnifiedAttentionConfig,
  GridLaunchInfo,
  SegmentInfo,
  PerformanceMetrics,
  SampleTensor,
  MemoryLayoutInfo,
  AlgorithmStep,
  calculateGridLaunch2D,
  calculateGridLaunch3D,
  shouldUse2DKernel,
  calculateFLOPs,
  calculateMemoryBandwidth
} from '../types/attention';

interface UnifiedAttentionState {
  config: UnifiedAttentionConfig;
  gridLaunch: GridLaunchInfo;
  segments: SegmentInfo[];
  performance: PerformanceMetrics;
  currentPhase: 'idle' | 'qk' | 'softmax' | 'av' | 'reduction' | 'complete';
  activeTab: 'pipeline' | 'algorithm' | '2d-vs-3d' | 'memory' | 'performance';
  highlightedCell: { x: number; y: number; z: number } | null;

  // New: Step-by-step state
  currentStep: AlgorithmStep;
  stepIndex: number;

  // New: Computed tensors
  tensors: {
    input: SampleTensor | null;
    q: SampleTensor | null;
    k: SampleTensor | null;
    v: SampleTensor | null;
    scores: SampleTensor | null;
    softmaxScores: SampleTensor | null;
    output: SampleTensor | null;
  };

  // New: Memory layout
  memoryLayout: MemoryLayoutInfo;

  // Actions
  setConfig: (config: Partial<UnifiedAttentionConfig>) => void;
  setActiveTab: (tab: UnifiedAttentionState['activeTab']) => void;
  setHighlightedCell: (cell: UnifiedAttentionState['highlightedCell']) => void;
  runSimulation: () => void;
  setPhase: (phase: UnifiedAttentionState['currentPhase']) => void;
  toggle3DKernel: () => void;
  initializeData: () => void;
  nextStep: () => void;
  prevStep: () => void;
  resetSteps: () => void;
  computeTensors: () => void;
  calculateMemory: () => void;
}

const defaultConfig: UnifiedAttentionConfig = {
  batchSize: 1,
  seqLen: 512,
  numHeads: 8,
  headDim: 64,
  numKvHeads: 8,
  blockM: 64,
  blockN: 64,
  numSegments: 4,
  useSlidingWindow: false,
  windowSize: 256,
  use3DKernel: false,
  causal: true
};

// Helper: Generate sample tensor values
function generateSampleValues(rows: number, cols: number): number[][] {
  const values: number[][] = [];
  for (let i = 0; i < Math.min(rows, 4); i++) {
    const row: number[] = [];
    for (let j = 0; j < Math.min(cols, 4); j++) {
      row.push(Math.random() * 2 - 1); // Random values between -1 and 1
    }
    values.push(row);
  }
  return values;
}

// Helper: Format bytes to human-readable string
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes.toFixed(0)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// Helper: Create sample tensor
function createSampleTensor(
  shape: number[],
  shapeStr: string,
  config: UnifiedAttentionConfig
): SampleTensor {
  const totalElements = shape.reduce((a, b) => a * b, 1);
  const memoryBytes = totalElements * 4; // Assuming FP32

  return {
    shape,
    shapeStr,
    actualDims: shape.join('×'),
    memoryBytes,
    memoryStr: formatBytes(memoryBytes),
    sampleValues: generateSampleValues(shape[shape.length - 2] || 4, shape[shape.length - 1] || 4)
  };
}

export const useUnifiedAttentionStore = create<UnifiedAttentionState>((set, get) => ({
  config: defaultConfig,
  gridLaunch: calculateGridLaunch2D(defaultConfig),
  segments: [],
  performance: {
    flops: 0,
    memoryBandwidth: 0,
    computeUtilization: 0,
    occupancy: 0,
    arithmeticIntensity: 0
  },
  currentPhase: 'idle',
  activeTab: 'pipeline',
  highlightedCell: null,

  // New state
  currentStep: 'idle',
  stepIndex: 0,
  tensors: {
    input: null,
    q: null,
    k: null,
    v: null,
    scores: null,
    softmaxScores: null,
    output: null
  },
  memoryLayout: {
    globalMemory: [],
    sharedMemory: [],
    registers: []
  },

  setConfig: (newConfig) => {
    set((state) => {
      const config = { ...state.config, ...newConfig };

      // Automatically determine 2D vs 3D based on heuristic
      if (!('use3DKernel' in newConfig)) {
        config.use3DKernel = !shouldUse2DKernel(config);
      }

      const gridLaunch = config.use3DKernel
        ? calculateGridLaunch3D(config)
        : calculateGridLaunch2D(config);

      return { config, gridLaunch };
    });
    get().initializeData();
  },

  setActiveTab: (tab) => set({ activeTab: tab }),

  setHighlightedCell: (cell) => set({ highlightedCell: cell }),

  toggle3DKernel: () => {
    set((state) => {
      const use3DKernel = !state.config.use3DKernel;
      const config = { ...state.config, use3DKernel };
      const gridLaunch = use3DKernel
        ? calculateGridLaunch3D(config)
        : calculateGridLaunch2D(config);
      return { config, gridLaunch };
    });
    get().initializeData();
  },

  initializeData: () => {
    const { config } = get();

    // Generate segment data for 3D visualization
    const segments: SegmentInfo[] = [];
    const kvLen = config.seqLen;
    const segmentSize = Math.ceil(kvLen / config.numSegments);

    for (let i = 0; i < config.numSegments; i++) {
      segments.push({
        segmentId: i,
        startIdx: i * segmentSize,
        endIdx: Math.min((i + 1) * segmentSize, kvLen),
        partialMax: -Infinity,
        partialExpSum: 0,
        partialOutput: []
      });
    }

    // Calculate performance metrics
    const flops = calculateFLOPs(config);
    const memoryBandwidth = calculateMemoryBandwidth(config);
    const arithmeticIntensity = flops / (memoryBandwidth * 1e9);

    set({
      segments,
      performance: {
        flops,
        memoryBandwidth,
        computeUtilization: 0.85,
        occupancy: Math.min(1.0, get().gridLaunch.gridX * get().gridLaunch.gridY / 108),
        arithmeticIntensity
      }
    });

    // Compute tensors and memory layout
    get().computeTensors();
    get().calculateMemory();
  },

  runSimulation: () => {
    const { config } = get();
    const phases: UnifiedAttentionState['currentPhase'][] = config.use3DKernel
      ? ['qk', 'softmax', 'av', 'reduction', 'complete']
      : ['qk', 'softmax', 'av', 'complete'];

    // Reset to start
    set({ currentPhase: 'idle', stepIndex: 0, currentStep: 'idle' });

    let phaseIndex = 0;
    let stepIdx = 0;
    const steps: AlgorithmStep[] = ['idle', 'proj', 'qk-matmul', 'softmax', 'av-matmul', 'output'];

    const runNextStep = () => {
      if (stepIdx < steps.length) {
        set({
          currentStep: steps[stepIdx],
          stepIndex: stepIdx,
          currentPhase: phaseIndex < phases.length ? phases[phaseIndex] : 'complete'
        });
        stepIdx++;
        if (stepIdx % 2 === 0) phaseIndex++;
        setTimeout(runNextStep, 1500);
      }
    };

    setTimeout(runNextStep, 500);
  },

  setPhase: (phase) => set({ currentPhase: phase }),

  // New step-by-step methods
  nextStep: () => {
    const steps: AlgorithmStep[] = ['idle', 'proj', 'qk-matmul', 'softmax', 'av-matmul', 'output'];
    const { stepIndex } = get();
    const nextIndex = Math.min(stepIndex + 1, steps.length - 1);
    set({ stepIndex: nextIndex, currentStep: steps[nextIndex] });
  },

  prevStep: () => {
    const steps: AlgorithmStep[] = ['idle', 'proj', 'qk-matmul', 'softmax', 'av-matmul', 'output'];
    const { stepIndex } = get();
    const prevIndex = Math.max(stepIndex - 1, 0);
    set({ stepIndex: prevIndex, currentStep: steps[prevIndex] });
  },

  resetSteps: () => {
    set({ stepIndex: 0, currentStep: 'idle' });
  },

  computeTensors: () => {
    const { config } = get();
    const B = config.batchSize;
    const S = config.seqLen;
    const H = config.numHeads;
    const D = config.headDim;
    const hiddenDim = H * D;

    const tensors = {
      input: createSampleTensor([B, S, hiddenDim], '[B×S×D]', config),
      q: createSampleTensor([B, S, H, D], '[B×S×H×D]', config),
      k: createSampleTensor([B, S, H, D], '[B×S×H×D]', config),
      v: createSampleTensor([B, S, H, D], '[B×S×H×D]', config),
      scores: createSampleTensor([B, H, S, S], '[B×H×S×S]', config),
      softmaxScores: createSampleTensor([B, H, S, S], '[B×H×S×S]', config),
      output: createSampleTensor([B, S, hiddenDim], '[B×S×D]', config)
    };

    set({ tensors });
  },

  calculateMemory: () => {
    const { config } = get();
    const B = config.batchSize;
    const S = config.seqLen;
    const H = config.numHeads;
    const D = config.headDim;
    const hiddenDim = H * D;

    const globalMemory = [
      { name: 'Input', bytes: B * S * hiddenDim * 4, shape: `[${B}×${S}×${hiddenDim}]` },
      { name: 'W_Q', bytes: hiddenDim * hiddenDim * 4, shape: `[${hiddenDim}×${hiddenDim}]` },
      { name: 'W_K', bytes: hiddenDim * hiddenDim * 4, shape: `[${hiddenDim}×${hiddenDim}]` },
      { name: 'W_V', bytes: hiddenDim * hiddenDim * 4, shape: `[${hiddenDim}×${hiddenDim}]` },
      { name: 'Q', bytes: B * S * H * D * 4, shape: `[${B}×${S}×${H}×${D}]` },
      { name: 'K', bytes: B * S * H * D * 4, shape: `[${B}×${S}×${H}×${D}]` },
      { name: 'V', bytes: B * S * H * D * 4, shape: `[${B}×${S}×${H}×${D}]` },
      { name: 'Output', bytes: B * S * hiddenDim * 4, shape: `[${B}×${S}×${hiddenDim}]` }
    ];

    const sharedMemory = [
      { name: 'Q tile', bytes: config.blockM * D * 4 },
      { name: 'K tile', bytes: config.blockN * D * 4 },
      { name: 'V tile', bytes: config.blockN * D * 4 },
      { name: 'Scores tile', bytes: config.blockM * config.blockN * 4 }
    ];

    const registers = [
      { name: 'Accumulators', count: config.blockM * D },
      { name: 'Max values', count: config.blockM },
      { name: 'Sum values', count: config.blockM }
    ];

    set({ memoryLayout: { globalMemory, sharedMemory, registers } });
  }
}));
