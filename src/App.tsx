import React, { useState } from 'react';
import './App.css';
import LandingPage from './LandingPage';
import UnifiedAttentionApp from './apps/UnifiedAttentionApp';
import FlashAttentionApp from './apps/FlashAttentionApp';
import PagedAttentionApp from './apps/PagedAttentionApp';
import LeanAttentionApp from './apps/LeanAttentionApp';
import HSTUAttentionApp from './apps/HSTUAttentionApp';
import MLADecodeApp from './apps/MLADecodeApp';
import SageAttentionApp from './apps/SageAttentionApp';
import PODAttentionApp from './apps/PODAttentionApp';
import { AttentionType } from './types/attention';

function App() {
  const [currentApp, setCurrentApp] = useState<AttentionType | 'landing'>('landing');

  const renderApp = () => {
    switch (currentApp) {
      case 'unified':
        return <UnifiedAttentionApp onBack={() => setCurrentApp('landing')} />;
      case 'flash':
        return <FlashAttentionApp onBack={() => setCurrentApp('landing')} />;
      case 'paged':
        return <PagedAttentionApp onBack={() => setCurrentApp('landing')} />;
      case 'lean':
        return <LeanAttentionApp onBack={() => setCurrentApp('landing')} />;
      case 'hstu':
        return <HSTUAttentionApp onBack={() => setCurrentApp('landing')} />;
      case 'mla':
        return <MLADecodeApp onBack={() => setCurrentApp('landing')} />;
      case 'sage':
        return <SageAttentionApp onBack={() => setCurrentApp('landing')} />;
      case 'pod':
        return <PODAttentionApp onBack={() => setCurrentApp('landing')} />;
      default:
        return <LandingPage onSelectKernel={(type) => setCurrentApp(type)} />;
    }
  };

  return (
    <div className="App">
      {renderApp()}
    </div>
  );
}

export default App;
