import React, { useState } from 'react';
import { motion } from 'framer-motion';

interface MatrixBoxProps {
  label: string;
  dims: string;
  shape?: string;          // e.g., "[B×S×D]"
  actualDims?: string;     // e.g., "2×1024×768"
  memorySize?: string;     // e.g., "1.57 MB"
  description?: string;    // Explanation of what this tensor represents
  rows?: number;
  cols?: number;
  color?: string;
  showGrid?: boolean;
  highlightCells?: { row: number; col: number }[];
  isActive?: boolean;      // Whether this tensor is active in current step
  sampleValues?: number[][]; // Small sample of actual values
}

const MatrixBox: React.FC<MatrixBoxProps> = ({
  label,
  dims,
  shape,
  actualDims,
  memorySize,
  description,
  rows = 4,
  cols = 4,
  color = '#e94560',
  showGrid = true,
  highlightCells = [],
  isActive = false,
  sampleValues
}) => {
  const [showTooltip, setShowTooltip] = useState(false);

  const isHighlighted = (row: number, col: number) =>
    highlightCells.some(c => c.row === row && c.col === col);

  // Logarithmic scaling for box size based on memory
  const getBoxScale = () => {
    if (!memorySize) return 1;
    const bytes = parseFloat(memorySize);
    const unit = memorySize.split(' ')[1];
    let totalBytes = bytes;
    if (unit === 'MB') totalBytes *= 1024 * 1024;
    else if (unit === 'GB') totalBytes *= 1024 * 1024 * 1024;
    else if (unit === 'KB') totalBytes *= 1024;

    // Logarithmic scale: log10(bytes) mapped to 0.8-1.5 range
    const logScale = Math.log10(totalBytes);
    return Math.min(1.5, Math.max(0.8, 0.8 + (logScale - 3) * 0.1));
  };

  const boxScale = getBoxScale();

  return (
    <motion.div
      className="matrix-box"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{
        opacity: 1,
        scale: isActive ? boxScale * 1.1 : boxScale,
        boxShadow: isActive ? '0 0 20px rgba(255, 200, 0, 0.6)' : '0 2px 8px rgba(0,0,0,0.1)'
      }}
      transition={{ duration: 0.3 }}
      style={{
        position: 'relative',
        border: isActive ? `3px solid ${color}` : `2px solid ${color}80`,
        cursor: description ? 'pointer' : 'default'
      }}
      onMouseEnter={() => description && setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <span className="matrix-label">{label}</span>
      <span className="matrix-dims">{shape || dims}</span>

      {actualDims && (
        <span style={{
          fontSize: '11px',
          color: '#666',
          display: 'block',
          marginTop: '2px'
        }}>
          {actualDims}
        </span>
      )}

      {memorySize && (
        <span style={{
          fontSize: '10px',
          color: '#999',
          display: 'block',
          marginTop: '2px',
          fontWeight: 'bold'
        }}>
          {memorySize}
        </span>
      )}

      {showGrid && !sampleValues && (
        <div
          className="matrix-grid"
          style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
        >
          {Array.from({ length: rows * cols }, (_, i) => {
            const row = Math.floor(i / cols);
            const col = i % cols;
            const highlighted = isHighlighted(row, col);

            return (
              <motion.div
                key={i}
                className="matrix-cell"
                style={{
                  background: highlighted ? color : `${color}40`,
                  opacity: highlighted ? 1 : 0.6
                }}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: i * 0.01 }}
              />
            );
          })}
        </div>
      )}

      {sampleValues && (
        <div
          className="matrix-grid"
          style={{ gridTemplateColumns: `repeat(${sampleValues[0]?.length || cols}, 1fr)` }}
        >
          {sampleValues.flatMap((row, rowIdx) =>
            row.map((val, colIdx) => (
              <div
                key={`${rowIdx}-${colIdx}`}
                className="matrix-cell"
                style={{
                  background: `${color}${Math.floor(Math.abs(val) * 255).toString(16).padStart(2, '0')}`,
                  fontSize: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff'
                }}
                title={val.toFixed(3)}
              >
                {val.toFixed(2)}
              </div>
            ))
          )}
        </div>
      )}

      {showTooltip && description && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            position: 'absolute',
            top: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginTop: '8px',
            padding: '10px',
            background: '#333',
            color: '#fff',
            borderRadius: '6px',
            fontSize: '12px',
            maxWidth: '250px',
            zIndex: 1000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            whiteSpace: 'normal'
          }}
        >
          {description}
          <div style={{
            position: 'absolute',
            top: '-6px',
            left: '50%',
            transform: 'translateX(-50%)',
            width: 0,
            height: 0,
            borderLeft: '6px solid transparent',
            borderRight: '6px solid transparent',
            borderBottom: '6px solid #333'
          }} />
        </motion.div>
      )}
    </motion.div>
  );
};

export default MatrixBox;
