import React, { useState } from 'react';

export interface GlossaryEntry {
  name: string;
  type: 'int' | 'float' | 'bool' | 'enum' | 'string';
  category: 'common' | 'specific' | 'technical' | 'formula';
  range: string;
  currentValue: any;
  defaultValue: any;
  description: string;
  sampleValues: { value: any; explanation: string }[];
  formula?: string;
}

export interface ParameterGlossaryProps {
  entries: GlossaryEntry[];
  title?: string;
  collapsible?: boolean;
}

const ParameterGlossary: React.FC<ParameterGlossaryProps> = ({
  entries,
  title = 'Parameter Reference',
  collapsible = true
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Group entries by category
  const commonEntries = entries.filter(e => e.category === 'common');
  const specificEntries = entries.filter(e => e.category === 'specific');
  const technicalEntries = entries.filter(e => e.category === 'technical');
  const formulaEntries = entries.filter(e => e.category === 'formula');

  const renderEntry = (entry: GlossaryEntry) => (
    <div key={entry.name} className={`glossary-entry ${entry.category}`}>
      <div className="glossary-entry-header">
        <span className="glossary-entry-name">{entry.name}</span>
        <div className="glossary-entry-meta">
          <span className="glossary-type">Type: {entry.type}</span>
        </div>
      </div>

      <div className="glossary-entry-values">
        <span className="glossary-current-value">Current: {String(entry.currentValue)}</span>
        <span className="glossary-default">Default: {String(entry.defaultValue)}</span>
        <span className="glossary-range">Range: {entry.range}</span>
      </div>

      <p className="glossary-description">{entry.description}</p>

      {entry.formula && (
        <pre className="glossary-formula">{entry.formula}</pre>
      )}

      {entry.sampleValues.length > 0 && (
        <div className="glossary-sample-values">
          <strong>Sample Values:</strong>
          <ul>
            {entry.sampleValues.map((sample, idx) => (
              <li key={idx}>
                <code>{String(sample.value)}</code> - {sample.explanation}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );

  const renderSection = (title: string, entries: GlossaryEntry[]) => {
    if (entries.length === 0) return null;

    return (
      <div className="glossary-section-group">
        <h3 className="glossary-section-title">{title}</h3>
        <div className="glossary-section-entries">
          {entries.map(renderEntry)}
        </div>
      </div>
    );
  };

  return (
    <div className="glossary-section">
      <div
        className="glossary-header"
        onClick={() => collapsible && setIsExpanded(!isExpanded)}
        style={{ cursor: collapsible ? 'pointer' : 'default' }}
      >
        <h2>{title}</h2>
        {collapsible && (
          <span className="glossary-toggle">{isExpanded ? '▼' : '▶'}</span>
        )}
      </div>

      {(!collapsible || isExpanded) && (
        <div className="glossary-content">
          {renderSection('Common Parameters', commonEntries)}
          {renderSection('App-Specific Parameters', specificEntries)}
          {renderSection('Technical Terms', technicalEntries)}
          {renderSection('Formulas', formulaEntries)}
        </div>
      )}
    </div>
  );
};

export default ParameterGlossary;
