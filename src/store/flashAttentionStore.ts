// Flash Attention Store - State management for MHA visualization

import { create } from 'zustand';
import {
  FlashAttentionConfig,
  PerformanceMetrics,
  calculateFLOPs,
  calculateMemoryBandwidth
} from '../types/attention';

interface OnlineSoftmaxState {
  currentBlock: number;
  maxSoFar: number;
  expSumSoFar: number;
  outputAccum: number[];
}

interface FlashAttentionState {
  config: FlashAttentionConfig;
  performance: PerformanceMetrics;
  currentPhase: 'idle' | 'load-q' | 'load-kv' | 'qk-matmul' | 'softmax' | 'av-matmul' | 'store' | 'complete';
  activeTab: 'overview' | 'online-softmax' | 'causal-mask' | 'fp8';
  onlineSoftmax: OnlineSoftmaxState;
  causalMaskActive: boolean;

  // Actions
  setConfig: (config: Partial<FlashAttentionConfig>) => void;
  setActiveTab: (tab: FlashAttentionState['activeTab']) => void;
  runSimulation: () => void;
  setPhase: (phase: FlashAttentionState['currentPhase']) => void;
  stepOnlineSoftmax: () => void;
  resetOnlineSoftmax: () => void;
}

const defaultConfig: FlashAttentionConfig = {
  batchSize: 1,
  seqLen: 512,
  numHeads: 8,
  headDim: 64,
  numKvHeads: 8,
  dropout: 0.0,
  useCausalMask: true,
  useAlibi: false,
  alibiSlope: 0.0,
  useFP8: false,
  softmaxScale: 0.125
};

export const useFlashAttentionStore = create<FlashAttentionState>((set, get) => ({
  config: defaultConfig,
  performance: {
    flops: calculateFLOPs(defaultConfig),
    memoryBandwidth: calculateMemoryBandwidth(defaultConfig),
    computeUtilization: 0.85,
    occupancy: 0.9,
    arithmeticIntensity: 50
  },
  currentPhase: 'idle',
  activeTab: 'overview',
  onlineSoftmax: {
    currentBlock: 0,
    maxSoFar: -Infinity,
    expSumSoFar: 0,
    outputAccum: []
  },
  causalMaskActive: true,

  setConfig: (newConfig) => {
    set((state) => {
      const config = { ...state.config, ...newConfig };
      return {
        config,
        performance: {
          flops: calculateFLOPs(config),
          memoryBandwidth: calculateMemoryBandwidth(config),
          computeUtilization: 0.85,
          occupancy: 0.9,
          arithmeticIntensity: calculateFLOPs(config) / (calculateMemoryBandwidth(config) * 1e9)
        },
        causalMaskActive: config.useCausalMask
      };
    });
  },

  setActiveTab: (tab) => set({ activeTab: tab }),

  runSimulation: () => {
    const phases: FlashAttentionState['currentPhase'][] = [
      'load-q', 'load-kv', 'qk-matmul', 'softmax', 'av-matmul', 'store', 'complete'
    ];

    let phaseIndex = 0;

    const runNextPhase = () => {
      if (phaseIndex < phases.length) {
        set({ currentPhase: phases[phaseIndex] });
        phaseIndex++;
        setTimeout(runNextPhase, 800);
      }
    };

    set({ currentPhase: 'idle' });
    setTimeout(runNextPhase, 300);
  },

  setPhase: (phase) => set({ currentPhase: phase }),

  stepOnlineSoftmax: () => {
    set((state) => {
      const { config, onlineSoftmax } = state;
      const numBlocks = Math.ceil(config.seqLen / 64);

      if (onlineSoftmax.currentBlock >= numBlocks) {
        return state;
      }

      // Simulate online softmax step
      const blockMax = Math.random() * 2 - 1;
      const newMax = Math.max(onlineSoftmax.maxSoFar, blockMax);
      const correction = Math.exp(onlineSoftmax.maxSoFar - newMax);
      const newExpSum = onlineSoftmax.expSumSoFar * correction + Math.exp(blockMax - newMax);

      return {
        onlineSoftmax: {
          currentBlock: onlineSoftmax.currentBlock + 1,
          maxSoFar: newMax,
          expSumSoFar: newExpSum,
          outputAccum: [...onlineSoftmax.outputAccum, newExpSum]
        }
      };
    });
  },

  resetOnlineSoftmax: () => {
    set({
      onlineSoftmax: {
        currentBlock: 0,
        maxSoFar: -Infinity,
        expSumSoFar: 0,
        outputAccum: []
      }
    });
  }
}));
