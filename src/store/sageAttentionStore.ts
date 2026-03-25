// Sage Attention Store - State management for FP8/MxFP4 visualization

import { create } from 'zustand';
import {
  SageAttentionConfig,
  PerformanceMetrics,
  calculateFLOPs,
  calculateMemoryBandwidth
} from '../types/attention';

interface QuantizationState {
  originalValues: number[];
  fp8Values: number[];
  mxfp4Values: number[];
  scaleFactors: number[];
  quantizationError: number;
}

interface SageAttentionState {
  config: SageAttentionConfig;
  performance: PerformanceMetrics;
  currentPhase: 'idle' | 'quantize' | 'qk' | 'scale' | 'av' | 'dequantize' | 'complete';
  activeTab: 'overview' | 'fp8-quantization' | 'scale-factors' | 'mxfp4';
  quantizationState: QuantizationState;

  // Actions
  setConfig: (config: Partial<SageAttentionConfig>) => void;
  setActiveTab: (tab: SageAttentionState['activeTab']) => void;
  runSimulation: () => void;
  setPhase: (phase: SageAttentionState['currentPhase']) => void;
  updateQuantizationState: () => void;
}

const defaultConfig: SageAttentionConfig = {
  batchSize: 1,
  seqLen: 512,
  numHeads: 8,
  headDim: 64,
  numKvHeads: 8,
  useFP8: true,
  useMxFP4: false,
  perTokenScale: true,
  smoothK: true
};

// Simulate FP8 quantization (E4M3 format)
function quantizeToFP8(value: number, scale: number): number {
  const scaled = value / scale;
  // FP8 E4M3 range: ~-448 to 448
  const clamped = Math.max(-448, Math.min(448, scaled));
  // Simulate 4-bit mantissa precision
  const mantissaBits = 3;
  const quantum = Math.pow(2, Math.floor(Math.log2(Math.abs(clamped) + 1e-10)) - mantissaBits);
  return Math.round(clamped / quantum) * quantum * scale;
}

// Simulate MxFP4 quantization (2-bit exponent, 1-bit mantissa + sign)
function quantizeToMxFP4(value: number, scale: number): number {
  const scaled = value / scale;
  // Very coarse quantization
  const levels = [-6, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 6];
  let closest = levels[0];
  let minDist = Math.abs(scaled - closest);

  for (const level of levels) {
    const dist = Math.abs(scaled - level);
    if (dist < minDist) {
      minDist = dist;
      closest = level;
    }
  }

  return closest * scale;
}

export const useSageAttentionStore = create<SageAttentionState>((set, get) => ({
  config: defaultConfig,
  performance: {
    flops: calculateFLOPs(defaultConfig),
    memoryBandwidth: calculateMemoryBandwidth(defaultConfig),
    computeUtilization: 0.93,
    occupancy: 0.95,
    arithmeticIntensity: 70
  },
  currentPhase: 'idle',
  activeTab: 'overview',
  quantizationState: {
    originalValues: [],
    fp8Values: [],
    mxfp4Values: [],
    scaleFactors: [],
    quantizationError: 0
  },

  setConfig: (newConfig) => {
    set((state) => {
      const config = { ...state.config, ...newConfig };
      return { config };
    });
    get().updateQuantizationState();
  },

  setActiveTab: (tab) => set({ activeTab: tab }),

  updateQuantizationState: () => {
    const { config } = get();

    // Generate sample values
    const originalValues = Array.from({ length: 16 }, () => Math.random() * 4 - 2);

    // Calculate per-token or per-tensor scale
    const scaleFactors: number[] = [];
    if (config.perTokenScale) {
      // Per-token scaling
      for (let i = 0; i < 4; i++) {
        const tokenValues = originalValues.slice(i * 4, (i + 1) * 4);
        const absMax = Math.max(...tokenValues.map(Math.abs));
        scaleFactors.push(absMax / 127); // FP8 max
      }
    } else {
      // Per-tensor scaling
      const absMax = Math.max(...originalValues.map(Math.abs));
      scaleFactors.push(absMax / 127);
    }

    // Quantize values
    const fp8Values: number[] = [];
    const mxfp4Values: number[] = [];

    for (let i = 0; i < originalValues.length; i++) {
      const scaleIdx = config.perTokenScale ? Math.floor(i / 4) : 0;
      const scale = scaleFactors[scaleIdx];

      fp8Values.push(quantizeToFP8(originalValues[i], scale));
      mxfp4Values.push(quantizeToMxFP4(originalValues[i], scale));
    }

    // Calculate quantization error
    const fp8Error = originalValues.reduce((sum, v, i) =>
      sum + Math.abs(v - fp8Values[i]), 0) / originalValues.length;

    set({
      quantizationState: {
        originalValues,
        fp8Values,
        mxfp4Values,
        scaleFactors,
        quantizationError: fp8Error
      }
    });
  },

  runSimulation: () => {
    const phases: SageAttentionState['currentPhase'][] = [
      'quantize', 'qk', 'scale', 'av', 'dequantize', 'complete'
    ];

    let phaseIndex = 0;

    const runNextPhase = () => {
      if (phaseIndex < phases.length) {
        set({ currentPhase: phases[phaseIndex] });
        phaseIndex++;
        setTimeout(runNextPhase, 900);
      }
    };

    get().updateQuantizationState();
    set({ currentPhase: 'idle' });
    setTimeout(runNextPhase, 500);
  },

  setPhase: (phase) => set({ currentPhase: phase })
}));
