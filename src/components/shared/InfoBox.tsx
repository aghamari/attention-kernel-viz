import React from 'react';
import { motion } from 'framer-motion';
import { Info, AlertCircle, CheckCircle, AlertTriangle } from 'lucide-react';

interface InfoBoxProps {
  type?: 'concept' | 'important' | 'note' | 'warning' | 'success';
  color?: string;
  title?: string;
  children: React.ReactNode;
}

const InfoBox: React.FC<InfoBoxProps> = ({
  type = 'note',
  color,
  title,
  children
}) => {
  const getTypeConfig = () => {
    switch (type) {
      case 'concept':
        return {
          bg: '#e3f2fd',
          border: '#2196f3',
          icon: <Info size={20} color="#2196f3" />,
          defaultTitle: 'Concept'
        };
      case 'important':
        return {
          bg: '#fff3e0',
          border: '#ff9800',
          icon: <AlertCircle size={20} color="#ff9800" />,
          defaultTitle: 'Important'
        };
      case 'warning':
        return {
          bg: '#ffebee',
          border: '#f44336',
          icon: <AlertTriangle size={20} color="#f44336" />,
          defaultTitle: 'Warning'
        };
      case 'success':
        return {
          bg: '#e8f5e9',
          border: '#4caf50',
          icon: <CheckCircle size={20} color="#4caf50" />,
          defaultTitle: 'Success'
        };
      default:
        return {
          bg: '#f5f5f5',
          border: '#9e9e9e',
          icon: <Info size={20} color="#9e9e9e" />,
          defaultTitle: 'Note'
        };
    }
  };

  const config = getTypeConfig();
  const finalBg = color ? `${color}20` : config.bg;
  const finalBorder = color || config.border;

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      style={{
        background: finalBg,
        border: `2px solid ${finalBorder}`,
        borderRadius: '8px',
        padding: '15px',
        marginBottom: '15px',
        display: 'flex',
        gap: '12px'
      }}
    >
      <div style={{ flexShrink: 0, marginTop: '2px' }}>
        {config.icon}
      </div>
      <div style={{ flex: 1 }}>
        {(title || config.defaultTitle) && (
          <div style={{
            fontWeight: 'bold',
            fontSize: '14px',
            marginBottom: '8px',
            color: '#333'
          }}>
            {title || config.defaultTitle}
          </div>
        )}
        <div style={{
          fontSize: '13px',
          lineHeight: '1.6',
          color: '#555'
        }}>
          {children}
        </div>
      </div>
    </motion.div>
  );
};

export default InfoBox;
