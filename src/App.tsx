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
import CKUnifiedApp from './apps/CKUnifiedApp';
import CKSplitKVApp from './apps/CKSplitKVApp';
import CKPagedKVApp from './apps/CKPagedKVApp';
import CKForwardApp from './apps/CKForwardApp';
import FeatureMatrixApp from './apps/FeatureMatrixApp';
import Triton2DVizApp from './apps/Triton2DVizApp';
import { AttentionType } from './types/attention';

function App() {
  const [currentApp, setCurrentApp] = useState<AttentionType | 'landing'>('landing');

  const renderApp = () => {
    switch (currentApp) {
      case 'unified':
        return <UnifiedAttentionApp onBack={() => setCurrentApp('landing')} onNavigate={(type) => setCurrentApp(type as AttentionType)} />;
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
      case 'ck-ua':
        return <CKUnifiedApp onBack={() => setCurrentApp('landing')} />;
      case 'ck-sk':
        return <CKSplitKVApp onBack={() => setCurrentApp('landing')} />;
      case 'ck-pk':
        return <CKPagedKVApp onBack={() => setCurrentApp('landing')} />;
      case 'ck-fwd':
        return <CKForwardApp onBack={() => setCurrentApp('landing')} />;
      case 'triton2d-viz':
        return <Triton2DVizApp onBack={() => setCurrentApp('landing')} />;
      case 'comparison':
        return <FeatureMatrixApp onBack={() => setCurrentApp('landing')} onNavigate={(type) => setCurrentApp(type)} />;
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
