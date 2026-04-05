import React from 'react';
import { Highlight, themes } from 'prism-react-renderer';

interface PseudoCodeProps {
  code: string;
  title?: string;
  subtitle?: string;
}

const PseudoCode: React.FC<PseudoCodeProps> = ({ code, title, subtitle }) => {
  return (
    <div style={{ padding: '20px', background: 'white', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)' }}>
      {title && <h3 style={{ marginBottom: subtitle ? '5px' : '15px', color: '#333' }}>{title}</h3>}
      {subtitle && <p style={{ color: '#666', marginBottom: '15px' }}>{subtitle}</p>}
      <Highlight theme={themes.vsDark} code={code.trim()} language="python">
        {({ style, tokens, getLineProps, getTokenProps }) => (
          <pre style={{
            ...style,
            padding: '15px',
            borderRadius: '8px',
            fontSize: '13px',
            lineHeight: '1.6',
            overflowX: 'auto',
            margin: 0,
          }}>
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        )}
      </Highlight>
    </div>
  );
};

export default PseudoCode;
