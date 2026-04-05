import { create } from 'zustand';
import {
  Triton2DConfig,
  SimulationData,
  WorkgroupInfo,
  QTileRow,
  KVTileStep,
  DEFAULT_CONFIG,
  generateSimulationData,
  buildQTile,
  computeAllTileSteps,
} from '../utils/triton2dSimulator';

export type VizStage =
  | 'idle'
  | 'setup'
  | 'q_tile_packing'
  | 'kv_tile'
  | 'output'
  | 'complete';

interface Triton2DVizState {
  config: Triton2DConfig;
  simData: SimulationData | null;
  selectedWG: WorkgroupInfo | null;
  qTile: QTileRow[];
  kvTileSteps: KVTileStep[];
  currentStage: VizStage;
  currentTileIdx: number;
  qTilePackingRow: number;
  isPlaying: boolean;

  initialize: () => void;
  selectWorkgroup: (kvHead: number, qBlockGlobal: number) => void;
  nextStep: () => void;
  prevStep: () => void;
  reset: () => void;
  play: () => void;
  pause: () => void;
}

export const useTriton2DVizStore = create<Triton2DVizState>((set, get) => ({
  config: DEFAULT_CONFIG,
  simData: null,
  selectedWG: null,
  qTile: [],
  kvTileSteps: [],
  currentStage: 'idle',
  currentTileIdx: 0,
  qTilePackingRow: -1,
  isPlaying: false,

  initialize: () => {
    const config = get().config;
    const simData = generateSimulationData(config);
    const defaultWG = simData.workgroups[0];
    const qTile = buildQTile(simData, defaultWG);
    const kvTileSteps = computeAllTileSteps(simData, defaultWG, qTile);
    set({
      simData,
      selectedWG: defaultWG,
      qTile,
      kvTileSteps,
      currentStage: 'setup',
      currentTileIdx: 0,
      qTilePackingRow: -1,
    });
  },

  selectWorkgroup: (kvHead, qBlockGlobal) => {
    const { simData } = get();
    if (!simData) return;
    const wg = simData.workgroups.find(
      w => w.kvHeadIdx === kvHead && w.qBlockGlobal === qBlockGlobal
    );
    if (!wg) return;
    const qTile = buildQTile(simData, wg);
    const kvTileSteps = computeAllTileSteps(simData, wg, qTile);
    set({
      selectedWG: wg,
      qTile,
      kvTileSteps,
      currentStage: 'setup',
      currentTileIdx: 0,
      qTilePackingRow: -1,
      isPlaying: false,
    });
  },

  nextStep: () => {
    const { currentStage, currentTileIdx, kvTileSteps, qTilePackingRow, config } = get();
    switch (currentStage) {
      case 'idle':
        get().initialize();
        break;
      case 'setup':
        set({ currentStage: 'q_tile_packing', qTilePackingRow: config.numQueriesPerKV - 1 });
        break;
      case 'q_tile_packing': {
        const nextRow = qTilePackingRow + config.numQueriesPerKV;
        if (nextRow < config.BLOCK_M) {
          set({ qTilePackingRow: nextRow });
        } else {
          set({ currentStage: 'kv_tile', currentTileIdx: 0 });
        }
        break;
      }
      case 'kv_tile':
        if (currentTileIdx < kvTileSteps.length - 1) {
          set({ currentTileIdx: currentTileIdx + 1 });
        } else {
          set({ currentStage: 'output' });
        }
        break;
      case 'output':
        set({ currentStage: 'complete', isPlaying: false });
        break;
      case 'complete':
        break;
    }
  },

  prevStep: () => {
    const { currentStage, currentTileIdx, qTilePackingRow } = get();
    switch (currentStage) {
      case 'complete':
        set({ currentStage: 'output' });
        break;
      case 'output':
        set({ currentStage: 'kv_tile', currentTileIdx: get().kvTileSteps.length - 1 });
        break;
      case 'kv_tile':
        if (currentTileIdx > 0) {
          set({ currentTileIdx: currentTileIdx - 1 });
        } else {
          set({ currentStage: 'q_tile_packing', qTilePackingRow: get().config.BLOCK_M - 1 });
        }
        break;
      case 'q_tile_packing': {
        const prevRow = qTilePackingRow - get().config.numQueriesPerKV;
        if (prevRow >= 0) {
          set({ qTilePackingRow: prevRow });
        } else {
          set({ currentStage: 'setup' });
        }
        break;
      }
      case 'setup':
        set({ currentStage: 'idle' });
        break;
      default:
        break;
    }
  },

  reset: () => {
    set({
      currentStage: 'idle',
      currentTileIdx: 0,
      qTilePackingRow: -1,
      isPlaying: false,
    });
  },

  play: () => {
    set({ isPlaying: true });
  },

  pause: () => {
    set({ isPlaying: false });
  },
}));
