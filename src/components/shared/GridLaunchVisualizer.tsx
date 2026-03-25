import React from 'react';
import { motion } from 'framer-motion';
import { GridLaunchInfo } from '../../types/attention';

interface GridLaunchVisualizerProps {
  gridInfo: GridLaunchInfo;
  is3D?: boolean;
  highlightedCell?: { x: number; y: number; z: number } | null;
  onCellHover?: (cell: { x: number; y: number; z: number } | null) => void;
  title?: string;
  colorScheme?: { primary: string; secondary: string };
}

const GridLaunchVisualizer: React.FC<GridLaunchVisualizerProps> = ({
  gridInfo,
  is3D = false,
  highlightedCell,
  onCellHover,
  title = 'Grid Launch',
  colorScheme = { primary: '#e94560', secondary: '#0f3460' }
}) => {
  const maxDisplay = 8; // Max cells to display per dimension
  const displayX = Math.min(gridInfo.gridX, maxDisplay);
  const displayY = Math.min(gridInfo.gridY, maxDisplay);
  const displayZ = is3D ? Math.min(gridInfo.gridZ, 4) : 1;

  const getCellColor = (x: number, y: number, z: number) => {
    if (highlightedCell && highlightedCell.x === x && highlightedCell.y === y && highlightedCell.z === z) {
      return colorScheme.primary;
    }
    // Gradient based on position
    const t = (x + y + z) / (displayX + displayY + displayZ);
    return `hsl(${340 - t * 60}, 70%, ${45 + t * 20}%)`;
  };

  return (
    <div className="grid-visualizer">
      <h3>{title}</h3>
      <div className="grid-info" style={{ marginBottom: '15px', fontSize: '14px', color: '#666' }}>
        <span>Grid: ({gridInfo.gridX}, {gridInfo.gridY}, {gridInfo.gridZ})</span>
        <span style={{ marginLeft: '20px' }}>Block: ({gridInfo.blockX}, {gridInfo.blockY}, {gridInfo.blockZ})</span>
        <span style={{ marginLeft: '20px' }}>Total Threads: {gridInfo.totalThreads.toLocaleString()}</span>
      </div>

      {is3D ? (
        // 3D visualization with layers
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          {Array.from({ length: displayZ }, (_, z) => (
            <div key={z} style={{ flex: '0 0 auto' }}>
              <div style={{ fontSize: '12px', color: '#888', marginBottom: '5px' }}>
                Segment {z}
              </div>
              <div className="grid-container">
                {Array.from({ length: displayY }, (_, y) => (
                  <div key={y} className="grid-row">
                    {Array.from({ length: displayX }, (_, x) => (
                      <motion.div
                        key={x}
                        className="grid-cell"
                        style={{ background: getCellColor(x, y, z) }}
                        onMouseEnter={() => onCellHover?.({ x, y, z })}
                        onMouseLeave={() => onCellHover?.(null)}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: (x + y + z) * 0.02 }}
                      >
                        {x},{y}
                      </motion.div>
                    ))}
                    {gridInfo.gridX > maxDisplay && (
                      <div className="grid-cell" style={{ background: '#ddd', color: '#666' }}>...</div>
                    )}
                  </div>
                ))}
                {gridInfo.gridY > maxDisplay && (
                  <div className="grid-row">
                    <div className="grid-cell" style={{ background: '#ddd', color: '#666' }}>...</div>
                  </div>
                )}
              </div>
            </div>
          ))}
          {gridInfo.gridZ > 4 && (
            <div style={{ display: 'flex', alignItems: 'center', color: '#888' }}>
              +{gridInfo.gridZ - 4} more segments
            </div>
          )}
        </div>
      ) : (
        // 2D visualization
        <div className="grid-container">
          {Array.from({ length: displayY }, (_, y) => (
            <div key={y} className="grid-row">
              <div style={{ width: '60px', fontSize: '12px', color: '#888', display: 'flex', alignItems: 'center' }}>
                KV Head {y}
              </div>
              {Array.from({ length: displayX }, (_, x) => (
                <motion.div
                  key={x}
                  className="grid-cell"
                  style={{ background: getCellColor(x, y, 0) }}
                  onMouseEnter={() => onCellHover?.({ x, y, z: 0 })}
                  onMouseLeave={() => onCellHover?.(null)}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: (x + y) * 0.03 }}
                >
                  Q{x}
                </motion.div>
              ))}
              {gridInfo.gridX > maxDisplay && (
                <div className="grid-cell" style={{ background: '#ddd', color: '#666' }}>...</div>
              )}
            </div>
          ))}
          {gridInfo.gridY > maxDisplay && (
            <div className="grid-row">
              <div style={{ width: '60px' }}></div>
              <div className="grid-cell" style={{ background: '#ddd', color: '#666' }}>...</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default GridLaunchVisualizer;
