import React from 'react';

interface SliderConfig {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}

interface SelectConfig {
  label: string;
  value: string | boolean;
  options: { value: string | boolean; label: string }[];
  onChange: (value: string | boolean) => void;
}

interface InputPanelProps {
  title: string;
  sliders?: SliderConfig[];
  selects?: SelectConfig[];
  children?: React.ReactNode;
}

const InputPanel: React.FC<InputPanelProps> = ({ title, sliders = [], selects = [], children }) => {
  return (
    <div className="input-panel">
      <div className="panel-header">
        <h2>{title}</h2>
      </div>

      <div className="config-controls">
        {sliders.map((slider, index) => (
          <div key={index} className="control-group">
            <label>
              <span>{slider.label}</span>
              <span>{slider.value}</span>
            </label>
            <input
              type="range"
              min={slider.min}
              max={slider.max}
              step={slider.step || 1}
              value={slider.value}
              onChange={(e) => slider.onChange(Number(e.target.value))}
            />
          </div>
        ))}

        {selects.map((select, index) => (
          <div key={index} className="control-group">
            <label>
              <span>{select.label}</span>
            </label>
            <select
              value={String(select.value)}
              onChange={(e) => {
                const val = e.target.value;
                if (val === 'true') select.onChange(true);
                else if (val === 'false') select.onChange(false);
                else select.onChange(val);
              }}
            >
              {select.options.map((opt, i) => (
                <option key={i} value={String(opt.value)}>{opt.label}</option>
              ))}
            </select>
          </div>
        ))}

        {children}
      </div>
    </div>
  );
};

export default InputPanel;
