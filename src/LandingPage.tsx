import React from 'react';
import { motion } from 'framer-motion';
import {
  Layers,
  Zap,
  Grid3X3,
  Activity,
  Cpu,
  Sparkles,
  Split,
  Gauge
} from 'lucide-react';
import { AttentionType, ATTENTION_COLORS } from './types/attention';

interface LandingPageProps {
  onSelectKernel: (type: AttentionType) => void;
}

interface KernelCard {
  type: AttentionType;
  title: string;
  description: string;
  features: string[];
  icon: React.ReactNode;
}

const kernelCards: KernelCard[] = [
  {
    type: 'unified',
    title: 'Unified Attention (2D/3D)',
    description: 'Flash Attention variant with adaptive kernel selection. Uses the same core algorithm (online softmax + tiling) but automatically chooses between 2D grid launch (optimal for ≤2048 tokens, single-pass) and 3D grid launch (optimal for >2048 tokens, segment parallelism with reduction). Same math, different GPU execution strategies for different workloads.',
    features: ['Flash Algorithm', '2D/3D Adaptive', 'Segment Reduction', 'Auto Selection'],
    icon: <Layers size={24} />
  },
  {
    type: 'flash',
    title: 'Flash Attention (MHA)',
    description: 'Standard Flash Attention implementation with memory-efficient online softmax algorithm. Computes attention in tiles to minimize HBM accesses, supports causal masks, ALiBi positional bias, and FP8 quantization. Foundation for many modern attention optimizations.',
    features: ['Online Softmax', 'Causal Mask', 'ALiBi', 'FP8 Support'],
    icon: <Zap size={24} />
  },
  {
    type: 'paged',
    title: 'Paged Attention',
    description: 'vLLM-style block-based KV cache management for serving workloads. Maps logical KV cache blocks to physical memory blocks via indirection table. V1 for short contexts (≤8K), V2 with multi-partition reduction for long contexts. Enables efficient batching and memory sharing.',
    features: ['Block Table', 'KV Cache', 'V1 vs V2', 'Partition Reduce'],
    icon: <Grid3X3 size={24} />
  },
  {
    type: 'lean',
    title: 'Lean Attention (StreamK)',
    description: 'Advanced parallelization using StreamK scheduling for better load balancing across GPU compute units. Persistent kernels stay resident to reduce launch overhead. Dynamic work stealing redistributes unfinished tiles to idle CUs. Optimal for irregular workloads and maximum throughput.',
    features: ['StreamK Schedule', 'Work Stealing', 'Persistent Kernel', 'Load Balance'],
    icon: <Activity size={24} />
  },
  {
    type: 'hstu',
    title: 'HSTU Attention',
    description: 'Specialized attention using SiLU activation (x * sigmoid(x)) instead of softmax, as used in HSTU architecture. Handles jagged tensors for variable-length sequences without padding. Alpha parameter controls gating strength. Different semantics from standard attention—outputs unbounded values.',
    features: ['SiLU vs Softmax', 'Jagged Tensors', 'Alpha Scaling', 'Variable Length'],
    icon: <Cpu size={24} />
  },
  {
    type: 'mla',
    title: 'MLA Decode + RoPE',
    description: 'Multi-Head Latent Attention compresses KV cache dimensions for memory efficiency (e.g., 512→64 latent dim). Fuses RoPE (Rotary Position Embeddings) into attention kernel. Two-stage computation: stage1 computes QK scores, stage2 applies to V. Used in DeepSeek models for long-context efficiency.',
    features: ['Latent Compression', 'RoPE Fusion', 'Two-Stage', 'KV Reduction'],
    icon: <Sparkles size={24} />
  },
  {
    type: 'sage',
    title: 'Sage Attention',
    description: 'High-precision FP8 quantized attention with advanced scaling strategies. Per-tensor or per-token scale factors maintain accuracy despite low precision. Smooth-K technique pre-processes keys to reduce quantization error. MxFP4 variant for ultra-low precision (4-bit). Targets inference speed without sacrificing quality.',
    features: ['FP8 Quantization', 'Per-Token Scale', 'MxFP4 Variant', 'Smooth K'],
    icon: <Gauge size={24} />
  },
  {
    type: 'pod',
    title: 'POD Attention',
    description: 'Prefill-Optimized Decode attention runs both prefill (context processing) and decode (token generation) workloads simultaneously on the same GPU. Configurable CU allocation ratios (e.g., 70% prefill, 30% decode) balance throughput. Improves GPU utilization in serving scenarios with mixed request types.',
    features: ['Dual Workload', 'CU Allocation', 'Prefill Ratio', 'Decode Ratio'],
    icon: <Split size={24} />
  }
];

const LandingPage: React.FC<LandingPageProps> = ({ onSelectKernel }) => {
  return (
    <div className="landing-page">
      <motion.div
        className="landing-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1>Attention Kernel Visualizer</h1>
        <p>Interactive visualization of 21+ attention kernel implementations from the aiter library</p>
      </motion.div>

      <div className="kernel-grid">
        {kernelCards.map((card, index) => (
          <motion.div
            key={card.type}
            className="kernel-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            onClick={() => onSelectKernel(card.type)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="kernel-card-header">
              <div
                className="kernel-card-icon"
                style={{ background: ATTENTION_COLORS[card.type].gradient }}
              >
                {card.icon}
              </div>
              <h3>{card.title}</h3>
            </div>
            <p>{card.description}</p>
            <div className="kernel-card-features">
              {card.features.map((feature, i) => (
                <span key={i} className="feature-tag">{feature}</span>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      <motion.footer
        className="app-footer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.8 }}
        style={{ marginTop: '40px' }}
      >
        <p>Attention Kernel Visualizer - Based on aiter library implementations</p>
        <p>Unified 2D/3D | Flash MHA | Paged | Lean StreamK | HSTU | MLA | Sage | POD</p>
      </motion.footer>
    </div>
  );
};

export default LandingPage;
