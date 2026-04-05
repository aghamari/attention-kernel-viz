import React from 'react';

interface ConfigParam {
  label: string;
  value: string | number;
  description?: string;
}

interface ConfigDisplayProps {
  title?: string;
  params: ConfigParam[];
}

const ConfigDisplay: React.FC<ConfigDisplayProps> = ({ title = 'Configuration', params }) => {
  return (
    <div style={{ padding: '15px', background: '#f8f9fa', borderRadius: '8px' }}>
      <h3 style={{ marginBottom: '15px', fontSize: '16px', color: '#333' }}>{title}</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {params.map((p) => (
          <div key={p.label} style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '6px 10px',
            background: 'white',
            borderRadius: '6px',
            fontSize: '13px'
          }}>
            <span style={{ color: '#555' }}>{p.label}</span>
            <code style={{
              background: '#e8e8e8',
              padding: '2px 8px',
              borderRadius: '4px',
              fontWeight: 600,
              color: '#333'
            }}>
              {p.value}
            </code>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ConfigDisplay;
