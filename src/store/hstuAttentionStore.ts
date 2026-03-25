// HSTU Attention Store - State management for SiLU gating visualization

import { create } from 'zustand';
import {
  HSTUAttentionConfig,
  PerformanceMetrics,
  calculateFLOPs,
  calculateMemoryBandwidth
} from '../types/attention';

interface ActivationComparison {
  input: number[];
  softmaxOutput: number[];
  siluOutput: number[];
}

interface JaggedSequence {
  seqId: number;
  length: number;
  startOffset: number;
}

interface HSTUAttentionState {
  config: HSTUAttentionConfig;
  performance: PerformanceMetrics;
  currentPhase: 'idle' | 'qk' | 'silu-gate' | 'av' | 'complete';
  activeTab: 'overview' | 'silu-vs-softmax' | 'jagged-tensors' | 'alpha-scaling';
  activationComparison: ActivationComparison;
  jaggedSequences: JaggedSequence[];

  // Actions
  setConfig: (config: Partial<HSTUAttentionConfig>) => void;
  setActiveTab: (tab: HSTUAttentionState['activeTab']) => void;
  runSimulation: () => void;
  setPhase: (phase: HSTUAttentionState['currentPhase']) => void;
  updateActivationComparison: () => void;
  initializeJaggedSequences: () => void;
}

const defaultConfig: HSTUAttentionConfig = {
  batchSize: 4,
  seqLen: 256,
  numHeads: 8,
  headDim: 64,
  numKvHeads: 8,
  alpha: 1.0,
  maxSeqLenK: 512,
  maxSeqLenQ: 256,
  useJaggedTensors: true
};

// SiLU activation: x * sigmoid(x)
function silu(x: number): number {
  return x * (1 / (1 + Math.exp(-x)));
}

// Softmax for an array
function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

export const useHSTUAttentionStore = create<HSTUAttentionState>((set, get) => ({
  config: defaultConfig,
  performance: {
    flops: calculateFLOPs(defaultConfig),
    memoryBandwidth: calculateMemoryBandwidth(defaultConfig),
    computeUtilization: 0.88,
    occupancy: 0.9,
    arithmeticIntensity: 55
  },
  currentPhase: 'idle',
  activeTab: 'overview',
  activationComparison: {
    input: [],
    softmaxOutput: [],
    siluOutput: []
  },
  jaggedSequences: [],

  setConfig: (newConfig) => {
    set((state) => {
      const config = { ...state.config, ...newConfig };
      return { config };
    });
    get().updateActivationComparison();
    get().initializeJaggedSequences();
  },

  setActiveTab: (tab) => set({ activeTab: tab }),

  updateActivationComparison: () => {
    const { config } = get();

    // Generate sample input values
    const input = Array.from({ length: 8 }, () => Math.random() * 4 - 2);

    // Apply softmax
    const softmaxOutput = softmax(input);

    // Apply SiLU with alpha scaling
    const siluOutput = input.map(x => silu(x * config.alpha));

    set({
      activationComparison: {
        input,
        softmaxOutput,
        siluOutput
      }
    });
  },

  initializeJaggedSequences: () => {
    const { config } = get();

    const jaggedSequences: JaggedSequence[] = [];
    let offset = 0;

    for (let i = 0; i < config.batchSize; i++) {
      // Variable length sequences
      const length = Math.floor(config.seqLen * (0.5 + Math.random() * 0.5));
      jaggedSequences.push({
        seqId: i,
        length,
        startOffset: offset
      });
      offset += length;
    }

    set({ jaggedSequences });
  },

  runSimulation: () => {
    const phases: HSTUAttentionState['currentPhase'][] = [
      'qk', 'silu-gate', 'av', 'complete'
    ];

    let phaseIndex = 0;

    const runNextPhase = () => {
      if (phaseIndex < phases.length) {
        set({ currentPhase: phases[phaseIndex] });
        phaseIndex++;
        setTimeout(runNextPhase, 1000);
      }
    };

    get().updateActivationComparison();
    get().initializeJaggedSequences();
    set({ currentPhase: 'idle' });
    setTimeout(runNextPhase, 500);
  },

  setPhase: (phase) => set({ currentPhase: phase })
}));
