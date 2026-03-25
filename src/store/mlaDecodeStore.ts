// MLA Decode Store - State management for MLA + RoPE visualization

import { create } from 'zustand';
import {
  MLADecodeConfig,
  PerformanceMetrics,
  calculateFLOPs,
  calculateMemoryBandwidth
} from '../types/attention';

interface RoPEState {
  position: number;
  theta: number;
  cosValues: number[];
  sinValues: number[];
}

interface LatentState {
  originalKV: number[][];
  compressedKV: number[][];
  compressionRatio: number;
}

interface MLADecodeState {
  config: MLADecodeConfig;
  performance: PerformanceMetrics;
  currentPhase: 'idle' | 'compress' | 'stage1-qk' | 'rope' | 'stage2-v' | 'complete';
  activeTab: 'overview' | 'latent-compression' | 'rope-fusion' | 'two-stage';
  ropeState: RoPEState;
  latentState: LatentState;

  // Actions
  setConfig: (config: Partial<MLADecodeConfig>) => void;
  setActiveTab: (tab: MLADecodeState['activeTab']) => void;
  runSimulation: () => void;
  setPhase: (phase: MLADecodeState['currentPhase']) => void;
  updateRoPEState: (position: number) => void;
  initializeLatentState: () => void;
}

const defaultConfig: MLADecodeConfig = {
  batchSize: 1,
  seqLen: 256,
  numHeads: 8,
  headDim: 64,
  numKvHeads: 1,
  kvLatentDim: 32,
  ropeTheta: 10000,
  ropeScale: 1.0,
  useRoPE: true,
  nope_size: 32,
  rope_size: 32
};

export const useMLADecodeStore = create<MLADecodeState>((set, get) => ({
  config: defaultConfig,
  performance: {
    flops: calculateFLOPs(defaultConfig),
    memoryBandwidth: calculateMemoryBandwidth(defaultConfig),
    computeUtilization: 0.9,
    occupancy: 0.92,
    arithmeticIntensity: 65
  },
  currentPhase: 'idle',
  activeTab: 'overview',
  ropeState: {
    position: 0,
    theta: 10000,
    cosValues: [],
    sinValues: []
  },
  latentState: {
    originalKV: [],
    compressedKV: [],
    compressionRatio: 2
  },

  setConfig: (newConfig) => {
    set((state) => {
      const config = { ...state.config, ...newConfig };
      return { config };
    });
    get().initializeLatentState();
    get().updateRoPEState(0);
  },

  setActiveTab: (tab) => set({ activeTab: tab }),

  updateRoPEState: (position: number) => {
    const { config } = get();
    const halfDim = config.rope_size / 2;

    const cosValues: number[] = [];
    const sinValues: number[] = [];

    for (let i = 0; i < halfDim; i++) {
      const freq = 1.0 / Math.pow(config.ropeTheta, (2 * i) / config.rope_size);
      const angle = position * freq * config.ropeScale;
      cosValues.push(Math.cos(angle));
      sinValues.push(Math.sin(angle));
    }

    set({
      ropeState: {
        position,
        theta: config.ropeTheta,
        cosValues,
        sinValues
      }
    });
  },

  initializeLatentState: () => {
    const { config } = get();

    // Original KV shape: [seq_len, num_kv_heads, head_dim]
    // Compressed KV shape: [seq_len, num_kv_heads, kv_latent_dim]
    const originalKV: number[][] = [];
    const compressedKV: number[][] = [];

    for (let s = 0; s < Math.min(config.seqLen, 16); s++) {
      originalKV.push(Array.from({ length: config.headDim }, () => Math.random() * 2 - 1));
      compressedKV.push(Array.from({ length: config.kvLatentDim }, () => Math.random() * 2 - 1));
    }

    const compressionRatio = config.headDim / config.kvLatentDim;

    set({
      latentState: {
        originalKV,
        compressedKV,
        compressionRatio
      }
    });
  },

  runSimulation: () => {
    const phases: MLADecodeState['currentPhase'][] = [
      'compress', 'stage1-qk', 'rope', 'stage2-v', 'complete'
    ];

    let phaseIndex = 0;

    const runNextPhase = () => {
      if (phaseIndex < phases.length) {
        set({ currentPhase: phases[phaseIndex] });
        phaseIndex++;
        setTimeout(runNextPhase, 1000);
      }
    };

    get().initializeLatentState();
    set({ currentPhase: 'idle' });
    setTimeout(runNextPhase, 500);
  },

  setPhase: (phase) => set({ currentPhase: phase })
}));
