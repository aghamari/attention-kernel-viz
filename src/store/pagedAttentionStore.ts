// Paged Attention Store - State management for paged attention visualization

import { create } from 'zustand';
import {
  PagedAttentionConfig,
  BlockTableEntry,
  PerformanceMetrics,
  calculateFLOPs,
  calculateMemoryBandwidth
} from '../types/attention';

interface PhysicalBlock {
  blockId: number;
  isUsed: boolean;
  seqIdx: number | null;
  logicalIdx: number | null;
}

interface PagedAttentionState {
  config: PagedAttentionConfig;
  blockTable: BlockTableEntry[];
  physicalBlocks: PhysicalBlock[];
  performance: PerformanceMetrics;
  currentPhase: 'idle' | 'allocate' | 'lookup' | 'compute' | 'partition' | 'reduce' | 'complete';
  activeTab: 'overview' | 'block-table' | 'kv-cache' | 'v1-vs-v2' | 'reduction';
  activeSequence: number;

  // Actions
  setConfig: (config: Partial<PagedAttentionConfig>) => void;
  setActiveTab: (tab: PagedAttentionState['activeTab']) => void;
  setActiveSequence: (seq: number) => void;
  runSimulation: () => void;
  setPhase: (phase: PagedAttentionState['currentPhase']) => void;
  initializeBlocks: () => void;
  allocateBlock: (seqIdx: number) => void;
}

const defaultConfig: PagedAttentionConfig = {
  batchSize: 4,
  seqLen: 256,
  numHeads: 8,
  headDim: 64,
  numKvHeads: 8,
  blockSize: 16,
  numBlocks: 32,
  maxContextLen: 2048,
  useV2: false,
  numPartitions: 4
};

export const usePagedAttentionStore = create<PagedAttentionState>((set, get) => ({
  config: defaultConfig,
  blockTable: [],
  physicalBlocks: [],
  performance: {
    flops: calculateFLOPs(defaultConfig),
    memoryBandwidth: calculateMemoryBandwidth(defaultConfig),
    computeUtilization: 0.8,
    occupancy: 0.85,
    arithmeticIntensity: 45
  },
  currentPhase: 'idle',
  activeTab: 'overview',
  activeSequence: 0,

  setConfig: (newConfig) => {
    set((state) => {
      const config = { ...state.config, ...newConfig };
      return { config };
    });
    get().initializeBlocks();
  },

  setActiveTab: (tab) => set({ activeTab: tab }),

  setActiveSequence: (seq) => set({ activeSequence: seq }),

  initializeBlocks: () => {
    const { config } = get();

    // Initialize physical blocks
    const physicalBlocks: PhysicalBlock[] = [];
    for (let i = 0; i < config.numBlocks; i++) {
      physicalBlocks.push({
        blockId: i,
        isUsed: false,
        seqIdx: null,
        logicalIdx: null
      });
    }

    // Create block table entries for each sequence
    const blockTable: BlockTableEntry[] = [];
    let nextFreeBlock = 0;

    for (let seq = 0; seq < config.batchSize; seq++) {
      const numBlocksNeeded = Math.ceil(config.seqLen / config.blockSize);

      for (let logical = 0; logical < numBlocksNeeded; logical++) {
        if (nextFreeBlock < config.numBlocks) {
          blockTable.push({
            logicalBlockIdx: logical,
            physicalBlockIdx: nextFreeBlock,
            seqIdx: seq
          });

          physicalBlocks[nextFreeBlock].isUsed = true;
          physicalBlocks[nextFreeBlock].seqIdx = seq;
          physicalBlocks[nextFreeBlock].logicalIdx = logical;

          nextFreeBlock++;
        }
      }
    }

    set({ physicalBlocks, blockTable });
  },

  allocateBlock: (seqIdx: number) => {
    set((state) => {
      const { physicalBlocks, blockTable, config } = state;

      // Find first free block
      const freeBlockIdx = physicalBlocks.findIndex(b => !b.isUsed);
      if (freeBlockIdx === -1) return state;

      // Find logical index for this sequence
      const seqBlocks = blockTable.filter(e => e.seqIdx === seqIdx);
      const nextLogical = seqBlocks.length;

      // Update physical block
      const newPhysicalBlocks = [...physicalBlocks];
      newPhysicalBlocks[freeBlockIdx] = {
        ...newPhysicalBlocks[freeBlockIdx],
        isUsed: true,
        seqIdx,
        logicalIdx: nextLogical
      };

      // Add to block table
      const newBlockTable = [...blockTable, {
        logicalBlockIdx: nextLogical,
        physicalBlockIdx: freeBlockIdx,
        seqIdx
      }];

      return { physicalBlocks: newPhysicalBlocks, blockTable: newBlockTable };
    });
  },

  runSimulation: () => {
    const { config } = get();
    const phases: PagedAttentionState['currentPhase'][] = config.useV2
      ? ['allocate', 'lookup', 'compute', 'partition', 'reduce', 'complete']
      : ['allocate', 'lookup', 'compute', 'complete'];

    let phaseIndex = 0;

    const runNextPhase = () => {
      if (phaseIndex < phases.length) {
        set({ currentPhase: phases[phaseIndex] });
        phaseIndex++;
        setTimeout(runNextPhase, 1000);
      }
    };

    set({ currentPhase: 'idle' });
    get().initializeBlocks();
    setTimeout(runNextPhase, 500);
  },

  setPhase: (phase) => set({ currentPhase: phase })
}));
