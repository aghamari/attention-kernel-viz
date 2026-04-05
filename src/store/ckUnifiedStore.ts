import { create } from 'zustand';
import {
  CKUnifiedConfig,
  GridLaunchInfo,
  calculateCKUAGrid,
  selectCKUATier,
  shouldUseCKUA,
} from '../types/attention';

interface CKUnifiedState {
  config: CKUnifiedConfig;
  gridLaunch: GridLaunchInfo;

  avgQ: number;
  totalNumQBlocks: number;
  shouldUse: boolean;

  setConfig: (config: Partial<CKUnifiedConfig>) => void;
  recalculate: () => void;
}

const defaultConfig: CKUnifiedConfig = {
  batchSize: 256,
  seqLen: 1,
  numHeads: 64,
  headDim: 64,
  numKvHeads: 8,
  kBlockM: 64,
  kBlockQ: 8,
  pageBlockSize: 64,
  maskType: 2,
  tier: 'small',
  scaleS: 1.0 / Math.sqrt(64),
  scaleK: 1.0,
  scaleV: 1.0,
  isDecodeGrid: true,
  numQueriesPerKv: 8
};

export const useCKUnifiedStore = create<CKUnifiedState>((set, get) => ({
  config: defaultConfig,
  gridLaunch: calculateCKUAGrid(
    defaultConfig.numKvHeads,
    defaultConfig.batchSize,
    Math.ceil(defaultConfig.batchSize * defaultConfig.seqLen / (defaultConfig.kBlockM / defaultConfig.numQueriesPerKv)),
    defaultConfig.isDecodeGrid
  ),
  avgQ: defaultConfig.seqLen,
  totalNumQBlocks: Math.ceil(defaultConfig.batchSize * defaultConfig.seqLen / (defaultConfig.kBlockM / defaultConfig.numQueriesPerKv)),
  shouldUse: shouldUseCKUA(
    defaultConfig.seqLen,
    defaultConfig.batchSize,
    defaultConfig.numKvHeads,
    [-1, -1],
    defaultConfig.pageBlockSize,
    4096,
    defaultConfig.headDim,
    defaultConfig.numQueriesPerKv,
    256
  ),

  setConfig: (newConfig) => set((state) => {
    const config = { ...state.config, ...newConfig };

    const numQueriesPerKv = Math.max(1, Math.floor(config.numHeads / config.numKvHeads));
    config.numQueriesPerKv = numQueriesPerKv;

    const avgQ = (config.seqLen * config.batchSize) / config.batchSize;
    const tierInfo = selectCKUATier(avgQ, config.seqLen, config.pageBlockSize);
    config.tier = tierInfo.tier;
    config.kBlockM = tierInfo.kBlockM;
    config.kBlockQ = Math.max(1, Math.floor(tierInfo.kBlockM / numQueriesPerKv));

    config.isDecodeGrid = config.seqLen === 1;
    config.scaleS = 1.0 / Math.sqrt(config.headDim);

    const totalNumQBlocks = Math.ceil(config.batchSize * config.seqLen / config.kBlockQ);
    const gridLaunch = calculateCKUAGrid(
      config.numKvHeads,
      config.batchSize,
      totalNumQBlocks,
      config.isDecodeGrid
    );

    const shouldUse = shouldUseCKUA(
      config.seqLen,
      config.batchSize,
      config.numKvHeads,
      [-1, -1],
      config.pageBlockSize,
      4096,
      config.headDim,
      numQueriesPerKv,
      256
    );

    return {
      config,
      gridLaunch,
      avgQ,
      totalNumQBlocks,
      shouldUse,
    };
  }),

  recalculate: () => {
    const { config } = get();
    get().setConfig(config);
  }
}));
