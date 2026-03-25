// Lean Attention Store - State management for StreamK visualization

import { create } from 'zustand';
import {
  LeanAttentionConfig,
  PerformanceMetrics,
  calculateFLOPs,
  calculateMemoryBandwidth
} from '../types/attention';

interface WorkUnit {
  workId: number;
  cuId: number;
  startTime: number;
  endTime: number;
  stolen: boolean;
  qBlock: number;
  kvBlock: number;
}

interface CUState {
  cuId: number;
  currentWork: number | null;
  completedWork: number[];
  isIdle: boolean;
}

interface LeanAttentionState {
  config: LeanAttentionConfig;
  workUnits: WorkUnit[];
  cuStates: CUState[];
  performance: PerformanceMetrics;
  currentPhase: 'idle' | 'schedule' | 'execute' | 'steal' | 'complete';
  activeTab: 'overview' | 'streamk-schedule' | 'work-stealing' | 'persistent';
  currentTime: number;

  // Actions
  setConfig: (config: Partial<LeanAttentionConfig>) => void;
  setActiveTab: (tab: LeanAttentionState['activeTab']) => void;
  runSimulation: () => void;
  setPhase: (phase: LeanAttentionState['currentPhase']) => void;
  initializeWorkUnits: () => void;
  stepSimulation: () => void;
  toggleWorkStealing: () => void;
}

const defaultConfig: LeanAttentionConfig = {
  batchSize: 1,
  seqLen: 1024,
  numHeads: 8,
  headDim: 64,
  numKvHeads: 8,
  numCUs: 8,
  tilesPerCU: 4,
  enableWorkStealing: true,
  persistentKernel: true
};

export const useLeanAttentionStore = create<LeanAttentionState>((set, get) => ({
  config: defaultConfig,
  workUnits: [],
  cuStates: [],
  performance: {
    flops: calculateFLOPs(defaultConfig),
    memoryBandwidth: calculateMemoryBandwidth(defaultConfig),
    computeUtilization: 0.92,
    occupancy: 0.95,
    arithmeticIntensity: 60
  },
  currentPhase: 'idle',
  activeTab: 'overview',
  currentTime: 0,

  setConfig: (newConfig) => {
    set((state) => {
      const config = { ...state.config, ...newConfig };
      return { config };
    });
    get().initializeWorkUnits();
  },

  setActiveTab: (tab) => set({ activeTab: tab }),

  initializeWorkUnits: () => {
    const { config } = get();

    // Calculate total work units
    const numQBlocks = Math.ceil(config.seqLen / 64);
    const numKVBlocks = Math.ceil(config.seqLen / 64);
    const totalWork = numQBlocks * numKVBlocks * config.numHeads;

    // Initialize CU states
    const cuStates: CUState[] = [];
    for (let i = 0; i < config.numCUs; i++) {
      cuStates.push({
        cuId: i,
        currentWork: null,
        completedWork: [],
        isIdle: true
      });
    }

    // Create work units with StreamK scheduling
    const workUnits: WorkUnit[] = [];
    let workId = 0;

    for (let q = 0; q < numQBlocks; q++) {
      for (let kv = 0; kv < numKVBlocks; kv++) {
        const cuId = workId % config.numCUs;
        const baseTime = Math.floor(workId / config.numCUs) * 10;
        const duration = 8 + Math.random() * 4; // Variable work duration

        workUnits.push({
          workId,
          cuId,
          startTime: baseTime,
          endTime: baseTime + duration,
          stolen: false,
          qBlock: q,
          kvBlock: kv
        });

        workId++;
      }
    }

    set({ workUnits, cuStates });
  },

  stepSimulation: () => {
    set((state) => {
      const { config, workUnits, cuStates, currentTime } = state;

      // Update CU states based on current time
      const newCuStates = cuStates.map(cu => {
        const currentWork = workUnits.find(
          w => w.cuId === cu.cuId &&
               w.startTime <= currentTime &&
               w.endTime > currentTime
        );

        const completed = workUnits
          .filter(w => w.cuId === cu.cuId && w.endTime <= currentTime)
          .map(w => w.workId);

        return {
          ...cu,
          currentWork: currentWork?.workId ?? null,
          completedWork: completed,
          isIdle: !currentWork
        };
      });

      // Simulate work stealing
      if (config.enableWorkStealing) {
        const idleCUs = newCuStates.filter(cu => cu.isIdle);
        const busyCUs = newCuStates.filter(cu => !cu.isIdle);

        // Find work to steal (incomplete work from busy CUs)
        // This is a simplified simulation
      }

      return {
        cuStates: newCuStates,
        currentTime: currentTime + 1
      };
    });
  },

  toggleWorkStealing: () => {
    set((state) => ({
      config: {
        ...state.config,
        enableWorkStealing: !state.config.enableWorkStealing
      }
    }));
  },

  runSimulation: () => {
    get().initializeWorkUnits();

    const phases: LeanAttentionState['currentPhase'][] = [
      'schedule', 'execute', 'steal', 'complete'
    ];

    let phaseIndex = 0;

    const runNextPhase = () => {
      if (phaseIndex < phases.length) {
        set({ currentPhase: phases[phaseIndex] });
        phaseIndex++;
        setTimeout(runNextPhase, 1200);
      }
    };

    set({ currentPhase: 'idle', currentTime: 0 });
    setTimeout(runNextPhase, 500);
  },

  setPhase: (phase) => set({ currentPhase: phase })
}));
