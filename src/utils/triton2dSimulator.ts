// Pure simulation functions for Triton 2D Unified Attention visualization.
// Uses small dimensions (hdim=8, BLOCK_N=8) so matrices fit on screen.

export interface Triton2DConfig {
  numSeqs: number;
  nheadQ: number;
  nheadKV: number;
  hdim: number;
  numQueriesPerKV: number;
  BLOCK_M: number;
  BLOCK_Q: number;
  BLOCK_N: number;
  pageSize: number;
  seqLensK: number[];
  seqLensQ: number[];
}

export interface PagedKVCache {
  numPages: number;
  data: number[][][];       // [page][slot][hdim] -- just K values for display
}

export interface BlockTable {
  table: number[][];         // [seq][page_col] -> phys_page_id
}

export interface WorkgroupInfo {
  kvHeadIdx: number;
  qBlockGlobal: number;
  seqIdx: number;
  qBlockLocal: number;
  qLen: number;
  kvLen: number;
}

export interface QTileRow {
  row: number;
  tok: number;
  head: number;
  qHeadGlobal: number;
  values: number[];
  isReal: boolean;
}

export interface KVTileStep {
  tileIdx: number;
  kvStart: number;
  kvEnd: number;
  physPage: number;
  pageCol: number;
  kTile: number[][];         // [BLOCK_N][hdim]
  vTile: number[][];
  scores: number[][];        // [BLOCK_M][BLOCK_N] raw scores
  masked: boolean[][];       // [BLOCK_M][BLOCK_N] true if masked
  mAfter: number[];          // [BLOCK_M] running max after this tile
  lAfter: number[];          // [BLOCK_M] running exp-sum after this tile
}

export interface SimulationData {
  config: Triton2DConfig;
  qFlat: number[][][];       // [total_tokens][nhead_q][hdim]
  cuSeqlensQ: number[];
  cumQBlocks: number[];
  kvCache: PagedKVCache;
  blockTable: BlockTable;
  workgroups: WorkgroupInfo[];
  totalQBlocks: number;
}

export const DEFAULT_CONFIG: Triton2DConfig = {
  numSeqs: 4,
  nheadQ: 8,
  nheadKV: 2,
  hdim: 8,
  numQueriesPerKV: 4,
  BLOCK_M: 16,
  BLOCK_Q: 4,
  BLOCK_N: 8,
  pageSize: 8,
  seqLensK: [12, 20, 8, 6],
  seqLensQ: [1, 1, 1, 1],
};

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return (s / 0x7fffffff) * 2 - 1;
  };
}

export function generateSimulationData(config: Triton2DConfig): SimulationData {
  const rand = seededRandom(42);
  const totalQTokens = config.seqLensQ.reduce((a, b) => a + b, 0);

  // cu_seqlens_q
  const cuSeqlensQ = [0];
  for (const len of config.seqLensQ) {
    cuSeqlensQ.push(cuSeqlensQ[cuSeqlensQ.length - 1] + len);
  }

  // cum_q_blocks: cumulative block counts per sequence
  const cumQBlocks = [0];
  for (const len of config.seqLensQ) {
    cumQBlocks.push(cumQBlocks[cumQBlocks.length - 1] + Math.ceil(len / config.BLOCK_Q));
  }
  const totalQBlocks = cumQBlocks[cumQBlocks.length - 1];

  // Q_flat: [total_tokens, nhead_q, hdim] with small random values
  const qFlat: number[][][] = [];
  for (let t = 0; t < totalQTokens; t++) {
    const heads: number[][] = [];
    for (let h = 0; h < config.nheadQ; h++) {
      const vec: number[] = [];
      for (let d = 0; d < config.hdim; d++) {
        vec.push(Math.round(rand() * 10) / 10);
      }
      heads.push(vec);
    }
    qFlat.push(heads);
  }

  // Block table and KV cache
  const totalPages = config.seqLensK.reduce(
    (sum, kl) => sum + Math.ceil(kl / config.pageSize), 0
  );
  const allPageIds = Array.from({ length: totalPages + 4 }, (_, i) => i);
  // Shuffle for non-contiguous allocation
  for (let i = allPageIds.length - 1; i > 0; i--) {
    const j = Math.floor(Math.abs(rand()) * (i + 1)) % (i + 1);
    [allPageIds[i], allPageIds[j]] = [allPageIds[j], allPageIds[i]];
  }

  const table: number[][] = [];
  let pageIdx = 0;
  for (let seq = 0; seq < config.numSeqs; seq++) {
    const numPagesForSeq = Math.ceil(config.seqLensK[seq] / config.pageSize);
    const seqPages: number[] = [];
    for (let p = 0; p < numPagesForSeq; p++) {
      seqPages.push(allPageIds[pageIdx++]);
    }
    table.push(seqPages);
  }

  // KV cache: [page][slot][hdim]
  const numPages = allPageIds.length;
  const kvData: number[][][] = [];
  for (let p = 0; p < numPages; p++) {
    const page: number[][] = [];
    for (let s = 0; s < config.pageSize; s++) {
      const vec: number[] = [];
      for (let d = 0; d < config.hdim; d++) {
        vec.push(Math.round(rand() * 10) / 10);
      }
      page.push(vec);
    }
    kvData.push(page);
  }

  // Build workgroup list
  const workgroups: WorkgroupInfo[] = [];
  for (let kvHead = 0; kvHead < config.nheadKV; kvHead++) {
    for (let gIdx = 0; gIdx < totalQBlocks; gIdx++) {
      // Binary search for seq
      let seqIdx = 0;
      for (let s = 0; s < config.numSeqs; s++) {
        if (cumQBlocks[s + 1] <= gIdx) seqIdx = s + 1;
      }
      const qBlockLocal = gIdx - cumQBlocks[seqIdx];
      const qStart = cuSeqlensQ[seqIdx];
      const qLen = cuSeqlensQ[seqIdx + 1] - qStart;
      const kvLen = config.seqLensK[seqIdx];
      workgroups.push({ kvHeadIdx: kvHead, qBlockGlobal: gIdx, seqIdx, qBlockLocal, qLen, kvLen });
    }
  }

  return {
    config,
    qFlat,
    cuSeqlensQ,
    cumQBlocks,
    kvCache: { numPages, data: kvData },
    blockTable: { table },
    workgroups,
    totalQBlocks,
  };
}

export function buildQTile(
  data: SimulationData,
  wg: WorkgroupInfo
): QTileRow[] {
  const { config, qFlat, cuSeqlensQ } = data;
  const rows: QTileRow[] = [];
  const qStart = cuSeqlensQ[wg.seqIdx];

  for (let row = 0; row < config.BLOCK_M; row++) {
    const tok = wg.qBlockLocal * config.BLOCK_Q + Math.floor(row / config.numQueriesPerKV);
    const hWithin = row % config.numQueriesPerKV;
    const qHeadGlobal = wg.kvHeadIdx * config.numQueriesPerKV + hWithin;
    const isReal = tok < wg.qLen;
    const values = isReal ? qFlat[qStart + tok][qHeadGlobal] : new Array(config.hdim).fill(0);
    rows.push({ row, tok, head: hWithin, qHeadGlobal, values, isReal });
  }
  return rows;
}

function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

export function computeKVTileStep(
  data: SimulationData,
  wg: WorkgroupInfo,
  qTile: QTileRow[],
  tileIdx: number,
  prevM: number[],
  prevL: number[]
): KVTileStep {
  const { config, kvCache, blockTable } = data;
  const kvStart = tileIdx * config.BLOCK_N;
  const kvEnd = Math.min(kvStart + config.BLOCK_N, wg.kvLen);
  const numReal = Math.max(0, kvEnd - kvStart);

  // Page lookup
  const pageCol = Math.floor(kvStart / config.pageSize);
  const physPage = pageCol < blockTable.table[wg.seqIdx].length
    ? blockTable.table[wg.seqIdx][pageCol] : -1;

  // Load K and V tiles
  const kTile: number[][] = [];
  const vTile: number[][] = [];
  for (let t = 0; t < config.BLOCK_N; t++) {
    const seqOffset = kvStart + t;
    if (seqOffset < wg.kvLen && physPage >= 0) {
      const pg = blockTable.table[wg.seqIdx][Math.floor(seqOffset / config.pageSize)];
      const slot = seqOffset % config.pageSize;
      kTile.push(kvCache.data[pg]?.[slot] ?? new Array(config.hdim).fill(0));
      vTile.push(kvCache.data[pg]?.[slot] ?? new Array(config.hdim).fill(0));
    } else {
      kTile.push(new Array(config.hdim).fill(0));
      vTile.push(new Array(config.hdim).fill(0));
    }
  }

  // Compute scores: S = Q_tile @ K_tile.T / sqrt(hdim)
  const scale = 1 / Math.sqrt(config.hdim);
  const scores: number[][] = [];
  const masked: boolean[][] = [];
  for (let r = 0; r < config.BLOCK_M; r++) {
    const sRow: number[] = [];
    const mRow: boolean[] = [];
    for (let c = 0; c < config.BLOCK_N; c++) {
      const kvPos = kvStart + c;
      const qRow = qTile[r];
      const raw = dotProduct(qRow.values, kTile[c]) * scale;
      const isMasked = kvPos >= wg.kvLen || !qRow.isReal;
      sRow.push(isMasked ? -Infinity : Math.round(raw * 100) / 100);
      mRow.push(isMasked);
    }
    scores.push(sRow);
    masked.push(mRow);
  }

  // Online softmax update
  const mAfter = [...prevM];
  const lAfter = [...prevL];
  for (let r = 0; r < config.BLOCK_M; r++) {
    const rowMax = Math.max(...scores[r].filter(s => s !== -Infinity), -Infinity);
    const mNew = Math.max(mAfter[r], rowMax);
    if (mNew === -Infinity) continue;
    const correction = mAfter[r] === -Infinity ? 0 : Math.exp(mAfter[r] - mNew);
    let pSum = 0;
    for (let c = 0; c < config.BLOCK_N; c++) {
      if (!masked[r][c]) pSum += Math.exp(scores[r][c] - mNew);
    }
    lAfter[r] = lAfter[r] * correction + pSum;
    mAfter[r] = mNew;
  }

  return {
    tileIdx,
    kvStart,
    kvEnd: Math.min(kvEnd, wg.kvLen),
    physPage,
    pageCol,
    kTile,
    vTile,
    scores,
    masked,
    mAfter: mAfter.map(v => Math.round(v * 100) / 100),
    lAfter: lAfter.map(v => Math.round(v * 100) / 100),
  };
}

export function computeAllTileSteps(
  data: SimulationData,
  wg: WorkgroupInfo,
  qTile: QTileRow[]
): KVTileStep[] {
  const numTiles = Math.ceil(wg.kvLen / data.config.BLOCK_N);
  const steps: KVTileStep[] = [];
  let m = new Array(data.config.BLOCK_M).fill(-Infinity);
  let l = new Array(data.config.BLOCK_M).fill(0);

  for (let t = 0; t < numTiles; t++) {
    const step = computeKVTileStep(data, wg, qTile, t, m, l);
    steps.push(step);
    m = step.mAfter;
    l = step.lAfter;
  }
  return steps;
}
