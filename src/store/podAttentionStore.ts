// POD Attention Store - State management for simultaneous prefill+decode visualization

import { create } from 'zustand';
import {
  PODAttentionConfig,
  PerformanceMetrics,
  calculateFLOPs,
  calculateMemoryBandwidth
} from '../types/attention';

interface WorkloadDistribution {
  prefillCUs: number[];
  decodeCUs: number[];
  prefillUtilization: number;
  decodeUtilization: number;
}

interface PODAttentionState {
  config: PODAttentionConfig;
  performance: PerformanceMetrics;
  currentPhase: 'idle' | 'allocate' | 'prefill' | 'decode' | 'sync' | 'complete';
  activeTab: 'overview' | 'dual-workload' | 'cu-allocation' | 'ratio-tuning';
  workloadDistribution: WorkloadDistribution;
  prefillProgress: number;
  decodeProgress: number;

  // Actions
  setConfig: (config: Partial<PODAttentionConfig>) => void;
  setActiveTab: (tab: PODAttentionState['activeTab']) => void;
  runSimulation: () => void;
  setPhase: (phase: PODAttentionState['currentPhase']) => void;
  updateWorkloadDistribution: () => void;
  setPrefillRatio: (ratio: number) => void;
}

const defaultConfig: PODAttentionConfig = {
  batchSize: 8,
  seqLen: 512,
  numHeads: 8,
  headDim: 64,
  numKvHeads: 8,
  prefillRatio: 0.6,
  decodeRatio: 0.4,
  prefillSeqLen: 1024,
  decodeNumTokens: 64
};

const TOTAL_CUS = 120; // Typical MI300X CU count

export const usePODAttentionStore = create<PODAttentionState>((set, get) => ({
  config: defaultConfig,
  performance: {
    flops: calculateFLOPs(defaultConfig),
    memoryBandwidth: calculateMemoryBandwidth(defaultConfig),
    computeUtilization: 0.88,
    occupancy: 0.92,
    arithmeticIntensity: 55
  },
  currentPhase: 'idle',
  activeTab: 'overview',
  workloadDistribution: {
    prefillCUs: [],
    decodeCUs: [],
    prefillUtilization: 0,
    decodeUtilization: 0
  },
  prefillProgress: 0,
  decodeProgress: 0,

  setConfig: (newConfig) => {
    set((state) => {
      const config = { ...state.config, ...newConfig };
      // Ensure ratios sum to 1
      if ('prefillRatio' in newConfig) {
        config.decodeRatio = 1 - config.prefillRatio;
      } else if ('decodeRatio' in newConfig) {
        config.prefillRatio = 1 - config.decodeRatio;
      }
      return { config };
    });
    get().updateWorkloadDistribution();
  },

  setActiveTab: (tab) => set({ activeTab: tab }),

  updateWorkloadDistribution: () => {
    const { config } = get();

    const numPrefillCUs = Math.floor(TOTAL_CUS * config.prefillRatio);
    const numDecodeCUs = TOTAL_CUS - numPrefillCUs;

    const prefillCUs = Array.from({ length: numPrefillCUs }, (_, i) => i);
    const decodeCUs = Array.from({ length: numDecodeCUs }, (_, i) => numPrefillCUs + i);

    // Calculate utilization based on workload characteristics
    // Prefill is compute-bound, decode is memory-bound
    const prefillWork = config.prefillSeqLen * config.prefillSeqLen * config.numHeads;
    const decodeWork = config.decodeNumTokens * config.seqLen * config.numHeads;

    const prefillPerCU = prefillWork / numPrefillCUs;
    const decodePerCU = decodeWork / numDecodeCUs;

    const maxWork = Math.max(prefillPerCU, decodePerCU);
    const prefillUtilization = Math.min(1, prefillPerCU / maxWork);
    const decodeUtilization = Math.min(1, decodePerCU / maxWork);

    set({
      workloadDistribution: {
        prefillCUs,
        decodeCUs,
        prefillUtilization,
        decodeUtilization
      }
    });
  },

  setPrefillRatio: (ratio: number) => {
    set((state) => ({
      config: {
        ...state.config,
        prefillRatio: ratio,
        decodeRatio: 1 - ratio
      }
    }));
    get().updateWorkloadDistribution();
  },

  runSimulation: () => {
    const phases: PODAttentionState['currentPhase'][] = [
      'allocate', 'prefill', 'decode', 'sync', 'complete'
    ];

    let phaseIndex = 0;

    const runNextPhase = () => {
      if (phaseIndex < phases.length) {
        const phase = phases[phaseIndex];
        set({ currentPhase: phase });

        // Animate progress for prefill/decode phases
        if (phase === 'prefill') {
          let progress = 0;
          const interval = setInterval(() => {
            progress += 10;
            set({ prefillProgress: progress });
            if (progress >= 100) clearInterval(interval);
          }, 100);
        } else if (phase === 'decode') {
          let progress = 0;
          const interval = setInterval(() => {
            progress += 15;
            set({ decodeProgress: progress });
            if (progress >= 100) clearInterval(interval);
          }, 80);
        }

        phaseIndex++;
        setTimeout(runNextPhase, 1200);
      }
    };

    get().updateWorkloadDistribution();
    set({ currentPhase: 'idle', prefillProgress: 0, decodeProgress: 0 });
    setTimeout(runNextPhase, 500);
  },

  setPhase: (phase) => set({ currentPhase: phase })
}));
