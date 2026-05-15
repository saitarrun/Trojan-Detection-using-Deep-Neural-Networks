"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import {
  Zap, Upload, Activity, FileText, ChevronRight,
  CheckCircle2, AlertTriangle, Loader2, Download,
  ShieldCheck, Server, Cpu, BarChart3, GitBranch,
  TrendingUp, Eye, Clock, Layers, RefreshCw, Info, X,
  CloudUpload, Microscope, BrainCircuit, FlaskConical,
  Fingerprint, ScanLine, LayoutDashboard,
  History, LineChart, Settings, Bell, Search, Plus
} from 'lucide-react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip as RechartsTooltip, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Cell
  // Cell is deprecated in recharts v3 but still functional
} from 'recharts';

// ─── Types ───────────────────────────────────────────────────────────────────

interface ScanHistoryEntry {
  id: string;
  fileName: string;
  score: number;
  verdict: string;
  timestamp: string;
  result: any;
}

type ResultTab = 'overview' | 'forensics' | 'visual' | 'report';
type NavItem = 'dashboard' | 'scan' | 'history' | 'analytics' | 'settings';

// ─── API ─────────────────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

function apiFetch(url: string, options: RequestInit = {}): Promise<Response> {
  return fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'X-Correlation-ID': crypto.randomUUID(),
      'X-API-Key': process.env.NEXT_PUBLIC_API_TOKEN || '',
    },
  });
}

// ─── Signal Gauge ─────────────────────────────────────────────────────────────

function SignalGauge({ label, value, max = 1.0, icon }: { label: string; value: number; max?: number; icon?: React.ReactNode }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = pct > 70 ? '#ef4444' : pct > 40 ? '#f59e0b' : '#3cd7ff';
  return (
    <div style={{ marginBottom: '0.9rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          {icon && <span style={{ opacity: 0.6, color: 'var(--on-surface-variant)' }}>{icon}</span>}
          <span style={{ fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.05em', color: 'var(--on-surface-variant)', fontFamily: 'var(--font-headline)' }}>{label}</span>
        </div>
        <span style={{ fontSize: '0.7rem', fontWeight: 800, color, fontFamily: 'monospace' }}>{pct.toFixed(1)}%</span>
      </div>
      <div style={{ height: '5px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: `linear-gradient(90deg, ${color}80, ${color})`,
          borderRadius: '3px',
          transition: 'width 1.2s cubic-bezier(0.4,0,0.2,1)',
          boxShadow: `0 0 8px ${color}55`
        }} />
      </div>
    </div>
  );
}

// ─── Radar Chart ──────────────────────────────────────────────────────────────

function RiskRadarChart({ details }: { details: any }) {
  const data = [
    { signal: 'Black-Box\nSweep', risk: +((details.blackbox_sweep_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Behavior\nProbe', risk: +((details.behavioral_backdoor_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Neural\nCleanse', risk: +((details.neural_cleanse_risk ?? 0) * 100).toFixed(1) },
    { signal: 'STRIP', risk: +((details.strip_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Clustering', risk: +((details.clustering_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Weight\nAudit', risk: +((details.weight_analysis_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Natural\nTrojan', risk: +((details.natural_trojan_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Gradient\nSimilarity', risk: +((details.gradient_similarity_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Spectral', risk: +((details.spectral_signatures_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Confidence', risk: +((details.confidence_distribution_risk ?? 0) * 100).toFixed(1) },
  ];
  return (
    <ResponsiveContainer width="100%" height={260}>
      <RadarChart data={data} margin={{ top: 10, right: 35, bottom: 10, left: 35 }}>
        <PolarGrid stroke="rgba(255,255,255,0.06)" />
        <PolarAngleAxis dataKey="signal" tick={{ fill: '#bbc9cf', fontSize: 10, fontWeight: 700 }} />
        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#475569', fontSize: 8 }} tickCount={4} />
        <Radar name="Risk %" dataKey="risk" stroke="#3cd7ff" fill="#3cd7ff" fillOpacity={0.15} strokeWidth={2} dot={{ fill: '#3cd7ff', r: 3 }} />
        <RechartsTooltip
          contentStyle={{ background: '#0f131c', border: '1px solid #3c494e', borderRadius: '10px', fontSize: '0.8rem' }}
          formatter={(v: any) => [`${v}%`, 'Risk Signal']}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}

// ─── Bar Chart ────────────────────────────────────────────────────────────────

function SignalBarChart({ details }: { details: any }) {
  const data = [
    { name: 'Sweep', value: +((details.blackbox_sweep_risk ?? 0) * 100).toFixed(1) },
    { name: 'Behavior', value: +((details.behavioral_backdoor_risk ?? 0) * 100).toFixed(1) },
    { name: 'NC', value: +((details.neural_cleanse_risk ?? 0) * 100).toFixed(1) },
    { name: 'STRIP', value: +((details.strip_risk ?? 0) * 100).toFixed(1) },
    { name: 'Cluster', value: +((details.clustering_risk ?? 0) * 100).toFixed(1) },
    { name: 'Weight', value: +((details.weight_analysis_risk ?? 0) * 100).toFixed(1) },
    { name: 'Natural', value: +((details.natural_trojan_risk ?? 0) * 100).toFixed(1) },
    { name: 'Gradient', value: +((details.gradient_similarity_risk ?? 0) * 100).toFixed(1) },
    { name: 'Spectral', value: +((details.spectral_signatures_risk ?? 0) * 100).toFixed(1) },
    { name: 'Confidence', value: +((details.confidence_distribution_risk ?? 0) * 100).toFixed(1) },
  ];
  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} />
        <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
        <RechartsTooltip
          contentStyle={{ background: '#0f131c', border: '1px solid #3c494e', borderRadius: '8px', fontSize: '0.8rem' }}
          formatter={(v: any) => [`${v}%`]}
        />
        <Bar dataKey="value" radius={[3, 3, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.value > 70 ? '#ef4444' : entry.value > 40 ? '#f59e0b' : '#3cd7ff'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ─── Audit Step ───────────────────────────────────────────────────────────────

function AuditStep({ label, status, subtext, isLast }: { label: string; status: 'pending' | 'active' | 'complete'; subtext?: string; isLast?: boolean }) {
  const isActive = status === 'active';
  const isComplete = status === 'complete';
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start' }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '28px', marginRight: '0.75rem', flexShrink: 0 }}>
        <div
          className={isActive ? 'pulsate' : ''}
          style={{
            width: '26px', height: '26px', borderRadius: '50%',
            background: isComplete ? 'var(--success)' : isActive ? 'var(--accent)' : 'rgba(255,255,255,0.04)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            border: `2px solid ${isComplete ? 'var(--success)' : isActive ? 'var(--accent)' : 'var(--card-border)'}`,
            boxShadow: isActive ? '0 0 16px var(--accent-glow)' : isComplete ? '0 0 8px rgba(16,185,129,0.3)' : 'none',
            transition: 'all 0.4s ease',
          }}
        >
          {isComplete
            ? <CheckCircle2 size={13} color="white" />
            : isActive
              ? <Loader2 className="animate-spin" size={13} color="white" />
              : <div style={{ width: '5px', height: '5px', background: 'rgba(255,255,255,0.2)', borderRadius: '50%' }} />
          }
        </div>
        {!isLast && <div style={{ width: '2px', minHeight: '24px', background: isComplete ? 'rgba(16,185,129,0.35)' : 'rgba(255,255,255,0.06)', margin: '3px 0', flex: 1 }} />}
      </div>
      <div style={{ paddingTop: '2px', paddingBottom: '0.6rem', opacity: isActive ? 1 : isComplete ? 0.7 : 0.3, transition: 'opacity 0.4s ease' }}>
        <p style={{ fontWeight: 700, fontSize: '0.82rem', fontFamily: 'var(--font-headline)', color: isActive ? 'var(--foreground)' : isComplete ? '#cbd5e1' : '#64748b', margin: 0 }}>{label}</p>
        {subtext && isActive && <p style={{ fontSize: '0.68rem', color: 'var(--accent)', marginTop: '0.15rem', fontWeight: 600 }}>{subtext}</p>}
      </div>
    </div>
  );
}

// ─── Drag Drop Zone ───────────────────────────────────────────────────────────

function DragDropZone({ onFile, selectedFile }: { onFile: (f: File) => void; selectedFile: File | null }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && /\.(pt|pth|onnx)$/i.test(file.name)) onFile(file);
  }, [onFile]);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      style={{
        border: `2px dashed ${dragging ? 'var(--accent)' : selectedFile ? 'rgba(16,185,129,0.6)' : 'var(--outline-variant)'}`,
        padding: '2.5rem 1.5rem',
        textAlign: 'center',
        cursor: 'pointer',
        borderRadius: '0.75rem',
        background: dragging ? 'rgba(60,215,255,0.05)' : selectedFile ? 'rgba(16,185,129,0.04)' : 'rgba(255,255,255,0.02)',
        transition: 'all 0.25s ease',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '0.75rem',
        minHeight: '240px',
      }}
    >
      {selectedFile ? (
        <>
          <div style={{ background: 'rgba(16,185,129,0.15)', padding: '1rem', borderRadius: '50%' }}>
            <CheckCircle2 size={28} style={{ color: 'var(--success)' }} />
          </div>
          <div>
            <p style={{ fontSize: '0.875rem', color: 'var(--foreground)', fontWeight: 600, wordBreak: 'break-all' }}>{selectedFile.name}</p>
            <p style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)', marginTop: '0.25rem' }}>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
        </>
      ) : (
        <>
          <div style={{ background: dragging ? 'rgba(60,215,255,0.15)' : 'rgba(168,232,255,0.08)', padding: '1.25rem', borderRadius: '50%', transition: 'all 0.25s ease', transform: dragging ? 'scale(1.1)' : 'scale(1)' }}>
            <CloudUpload size={32} style={{ color: dragging ? 'var(--accent)' : 'var(--primary)' }} />
          </div>
          <div>
            <p style={{ fontSize: '0.95rem', fontFamily: 'var(--font-headline)', fontWeight: 700, color: 'var(--foreground)' }}>
              {dragging ? 'Drop to upload' : 'Drop model files here or click to browse'}
            </p>
            <p style={{ fontSize: '0.78rem', color: 'var(--on-surface-variant)', marginTop: '0.35rem' }}>
              Supported: <span style={{ color: 'var(--accent)' }}>.pt, .pth, .onnx</span>
            </p>
          </div>
        </>
      )}
      <input ref={inputRef} type="file" style={{ display: 'none' }} onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} accept=".pt,.pth,.onnx" />
    </div>
  );
}

// ─── Stat Card ────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="card glass-panel" style={{ padding: '1.25rem', borderLeft: `3px solid ${color || 'var(--accent)'}` }}>
      <p className="label" style={{ marginBottom: '0.5rem' }}>{label}</p>
      <p style={{ fontSize: '1.5rem', fontWeight: 900, color: color || 'var(--foreground)', fontFamily: 'monospace', letterSpacing: '-0.02em' }}>{value}</p>
      {sub && <p style={{ fontSize: '0.68rem', color: 'var(--on-surface-variant)', marginTop: '0.25rem' }}>{sub}</p>}
    </div>
  );
}

// ─── History Item ─────────────────────────────────────────────────────────────

function HistoryItem({ entry, onRestore }: { entry: ScanHistoryEntry; onRestore: (e: ScanHistoryEntry) => void }) {
  const isCritical = entry.score > 0.75;
  const isWarning = entry.score > 0.4;
  const color = isCritical ? 'var(--danger)' : isWarning ? 'var(--warning)' : 'var(--success)';
  return (
    <button
      onClick={() => onRestore(entry)}
      className="glass-hover"
      style={{
        display: 'flex', alignItems: 'center', gap: '0.75rem', width: '100%',
        background: 'rgba(255,255,255,0.02)', border: '1px solid var(--card-border)',
        borderRadius: '0.5rem', padding: '0.6rem 0.9rem', cursor: 'pointer',
        transition: 'all 0.2s ease', textAlign: 'left',
      }}
    >
      <div style={{ width: '7px', height: '7px', borderRadius: '50%', background: color, flexShrink: 0 }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--foreground)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{entry.fileName}</p>
        <p style={{ fontSize: '0.65rem', color: 'var(--on-surface-variant)', marginTop: '0.1rem' }}>{entry.timestamp} · {(entry.score * 100).toFixed(0)}% risk</p>
      </div>
      <ChevronRight size={12} color="var(--outline)" />
    </button>
  );
}

// ─── Pipeline Steps ───────────────────────────────────────────────────────────

const PIPELINE_STEPS = [
  { id: 'INIT', label: 'Model Load & Validation', subtext: 'Validating architecture & tensor integrity', keys: ['INITIALIZING', 'LOADING', 'VALIDATING', 'ACCEPTED'] },
  { id: 'NC', label: 'Neural Cleanse', subtext: 'Trigger pattern inversion & anomaly scoring', keys: ['NEURAL CLEANSE', 'TRIGGER INVERSION', 'CLASS'] },
  { id: 'STRIP', label: 'STRIP Detection', subtext: 'Entropy-based test-time robustness check', keys: ['STRIP', 'ENTROPY', 'POISONING'] },
  { id: 'AC', label: 'Activation Clustering', subtext: 'Latent feature separation analysis', keys: ['CLUSTERING', 'ACTIVATION', 'T-SNE', 'TSNE'] },
  { id: 'WA', label: 'Weight Audit', subtext: 'Static weight anomaly analysis', keys: ['WEIGHT', 'LINEAR', 'NORM'] },
  { id: 'SPECTRAL', label: 'Spectral Signatures', subtext: 'SVD outlier detection in activation space', keys: ['SPECTRAL'] },
  { id: 'BEHAVIOR', label: 'Behavior Probe', subtext: 'Triggered attack success and target lift', keys: ['TRIGGERED ATTACK', 'ATTACK SUCCESS', 'BACKDOOR'] },
  { id: 'PROFILE', label: 'Profiling & Confidence', subtext: 'Shortcut sensitivity and confidence distribution', keys: ['PROFILING', 'NATURAL', 'CONFIDENCE'] },
  { id: 'FUSION', label: 'Gradient & Fusion', subtext: 'Gradient similarity and 10-signal fusion', keys: ['GRADIENT', 'SIMILARITY', 'FUSION', 'RISK SCORE', 'GENERATING', 'VISUAL', 'COMPLETE'] },
] as const;

// ─── Main Dashboard ───────────────────────────────────────────────────────────

export default function Dashboard() {
  const [mounted, setMounted] = useState(false);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [activeNav, setActiveNav] = useState<NavItem>('scan');

  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [isScanning, setIsScanning] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [scanStatus, setScanStatus] = useState<string>("IDLE");
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ResultTab>('overview');

  const [history, setHistory] = useState<ScanHistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  useEffect(() => {
    setMounted(true);
    try {
      const saved = localStorage.getItem('gemini_scan_history');
      if (saved) setHistory(JSON.parse(saved));
    } catch { /* ignore */ }
    apiFetch(`${API_BASE}/live`)
      .then(r => setApiStatus(r.ok ? 'online' : 'offline'))
      .catch(() => setApiStatus('offline'));
  }, []);

  const saveToHistory = (res: any, fileName: string) => {
    const entry: ScanHistoryEntry = {
      id: res.task_id || crypto.randomUUID(),
      fileName,
      score: res.fusion_risk_score ?? 0,
      verdict: determineVerdict(res.fusion_risk_score ?? 0),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      result: res,
    };
    const updated = [entry, ...history].slice(0, 8);
    setHistory(updated);
    try { localStorage.setItem('gemini_scan_history', JSON.stringify(updated)); } catch { /* ignore */ }
  };

  const determineVerdict = (score: number) =>
    score > 0.75 ? 'CRITICAL' : score > 0.4 ? 'WARNING' : 'SAFE';

  const getCurrentStepIndex = () => {
    const msg = (statusMessage + ' ' + scanStatus).toUpperCase();
    return PIPELINE_STEPS.findIndex(s => s.keys.some(k => msg.includes(k)));
  };

  const getStepStatus = (stepId: string): 'pending' | 'active' | 'complete' => {
    const currentIdx = getCurrentStepIndex();
    const stepIdx = PIPELINE_STEPS.findIndex(s => s.id === stepId);
    if (currentIdx === -1) return 'pending';
    if (stepIdx < currentIdx) return 'complete';
    if (stepIdx === currentIdx) return 'active';
    return 'pending';
  };

  const startScan = async () => {
    if (!selectedFile) return;

    setIsScanning(true);
    setError(null);
    setResult(null);
    setProgress(5);
    setScanStatus("INITIALIZING");
    setStatusMessage("Dispatching model to forensic pipeline...");
    setActiveTab('overview');
    setActiveNav('scan');

    try {
      const formData = new FormData();
      formData.append('model_file', selectedFile!);
      formData.append('target_class', '-1');
      formData.append('trigger_type', 'checkerboard');
      const response = await apiFetch(`${API_BASE}/api/v1/scan-model`, { method: 'POST', body: formData, cache: 'no-store' });

      if (!response.ok) {
        let msg = `HTTP ${response.status}`;
        try {
          const ct = response.headers.get('content-type') || '';
          if (ct.includes('application/json')) {
            const errData = await response.json();
            const detail = errData.detail;
            msg = typeof detail === 'string' ? detail : typeof detail === 'object' ? JSON.stringify(detail) : errData.message || msg;
          } else {
            msg = (await response.text()) || msg;
          }
        } catch { /* use default msg */ }
        throw new Error(msg);
      }

      const data = await response.json();
      setTaskId(data.task_id);
    } catch (err: any) {
      setError((err?.message as string)?.slice(0, 200) || 'Unknown error');
      setIsScanning(false);
    }
  };

  useEffect(() => {
    if (!taskId || !isScanning) return;
    const interval = setInterval(async () => {
      try {
        const r = await apiFetch(`${API_BASE}/api/v1/scan-status/${taskId}`);
        if (!r.ok) return;
        const data = await r.json();
        setScanStatus(data.status);
        if (data.status === 'PROGRESS' && data.message) {
          setStatusMessage(data.message);
          const upperMessage = data.message.toUpperCase();
          const stageIdx = PIPELINE_STEPS.findIndex(s => s.keys.some(k => upperMessage.includes(k)));
          const stageProgress = stageIdx >= 0
            ? Math.round(((stageIdx + 1) / PIPELINE_STEPS.length) * 92)
            : 12;
          setProgress(p => Math.max(p, Math.min(stageProgress, 92)));
        } else if (data.status === 'SUCCESS') {
          const res = { ...data.result, task_id: taskId };
          setResult(res);
          saveToHistory(res, selectedFile?.name || 'model');
          setIsScanning(false);
          setTaskId(null);
          setProgress(100);
          clearInterval(interval);
        } else if (data.status === 'FAILURE') {
          setError(data.error || data.message || 'Audit failed.');
          setIsScanning(false);
          setTaskId(null);
          clearInterval(interval);
        }
      } catch { /* network blip — retry next tick */ }
    }, 2000);
    return () => clearInterval(interval);
  }, [taskId, isScanning]);

  const downloadReport = async () => {
    const id = result?.task_id || taskId;
    if (!id) return;
    try {
      const r = await apiFetch(`${API_BASE}/api/v1/audit-report/${id}`);
      if (!r.ok) throw new Error('Failed to fetch report.');
      const data = await r.json();
      const doc = new jsPDF();
      const W = doc.internal.pageSize.width;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9);
      doc.setTextColor(110, 110, 110);
      doc.text('IARPA TrojAI Detecting Initiative', 14, 18);

      doc.setFont('helvetica', 'bold');
      doc.setFontSize(20);
      doc.setTextColor(20, 20, 20);
      doc.text('Neural Network Security Audit Report', 14, 27);

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(11);
      doc.setTextColor(90, 90, 90);
      doc.text('Trojan Detection & Forensic Analysis', 14, 35);

      doc.setDrawColor(210, 210, 210);
      doc.line(14, 40, W - 14, 40);

      doc.setFontSize(8.5);
      doc.setTextColor(100, 100, 100);
      const ts = new Date(data.report_metadata?.audit_timestamp || Date.now()).toLocaleString();
      doc.text(`Task ID: ${data.report_metadata?.task_id || id}`, 14, 48);
      doc.text(`Version: ${data.report_metadata?.version || '1.0-IARPA-JAN2026'}`, 14, 53);
      doc.text(`Generated: ${ts}`, W / 2, 48);
      doc.text('Type: Comprehensive Audit', W / 2, 53);

      doc.setFont('helvetica', 'bold');
      doc.setFontSize(13);
      doc.setTextColor(20, 20, 20);
      doc.text('EXECUTIVE SUMMARY', 14, 65);

      const riskScore = (data.model_summary?.risk_fusion_score || 0) * 100;
      const verdict = data.model_summary?.verdict || 'SAFE (Cleared for Production)';
      const bad = verdict.includes('WARNING') || verdict.includes('CRITICAL');
      doc.setFillColor(...(bad ? [253, 242, 244] : [240, 253, 244]) as [number, number, number]);
      doc.setDrawColor(...(bad ? [250, 175, 185] : [187, 247, 208]) as [number, number, number]);
      doc.setLineWidth(0.4);
      doc.rect(14, 69, W - 28, 11, 'FD');
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(10);
      doc.setTextColor(...(bad ? [220, 38, 50] : [22, 163, 74]) as [number, number, number]);
      doc.text(`VERDICT: ${verdict}  |  Risk Score: ${riskScore.toFixed(1)}%`, 18, 77);

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8.5);
      doc.setTextColor(70, 70, 70);
      const blurb = `This security audit analyzed the target neural network using ten complementary trojan detection signals. Analysis conducted on ${ts.split(',')[0]}.`;
      const blurbLines = doc.splitTextToSize(blurb, W - 28);
      doc.text(blurbLines, 14, 88);
      let y = 88 + blurbLines.length * 4.5 + 8;

      doc.setFont('helvetica', 'bold');
      doc.setFontSize(12);
      doc.setTextColor(20, 20, 20);
      doc.text('MODEL INFORMATION', 14, y);

      autoTable(doc, {
        startY: y + 4,
        head: [['Property', 'Value']],
        body: [
          ['Architecture', data.model_summary?.architecture || 'ResNet-18'],
          ['Framework', data.model_summary?.framework || 'PyTorch'],
          ['Input Shape', data.model_summary?.input_shape || 'N/A'],
          ['Classes', data.model_summary?.num_classes?.toString() || 'N/A'],
          ['Parameters', data.model_summary?.parameter_count || 'N/A'],
        ],
        theme: 'grid',
        headStyles: { fillColor: [0, 150, 180], textColor: [255, 255, 255], fontStyle: 'bold' },
        styles: { fontSize: 8.5, cellPadding: 3.5 },
        columnStyles: { 0: { cellWidth: 50 } },
      });

      y = (doc as any).lastAutoTable.finalY + 10;
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(12);
      doc.setTextColor(20, 20, 20);
      doc.text('TROJAN DETECTION SIGNALS', 14, y);

      const tf = data.trojan_forensics || {};
      const getS = (v: number, t: number) => v > t ? 'HIGH RISK' : 'NOMINAL';
      const ncIdx = tf.trigger_inversion?.neural_cleanse_index || 0;
      const stripAcc = (tf.test_time_checks?.strip_false_acceptance || 0) * 100;
      const stripRej = (tf.test_time_checks?.strip_false_rejection || 0) * 100;
      const waNorm = tf.weight_analysis?.max_anomaly_l2_norm || 0;
      const shortSens = (tf.natural_vulnerability_profiling?.shortcut_sensitivity || 0) * 100;
      const sil = tf.activation_clustering?.silhouette_score || 0;
      const css = tf.gradient_similarity_score || 0;

      autoTable(doc, {
        startY: y + 4,
        head: [['Signal', 'Measurement', 'Status']],
        body: [
          ['Neural Cleanse', `Anomaly Index: ${ncIdx.toFixed(3)}`, getS(ncIdx, 2.0)],
          ['STRIP Robustness', `FA: ${stripAcc.toFixed(1)}%  FR: ${stripRej.toFixed(1)}%`, getS(stripAcc, 10)],
          ['Weight Analysis', `Max L2 Norm: ${waNorm.toFixed(3)}`, getS(waNorm, 3.0)],
          ['Natural Profiling', `Shortcut Sensitivity: ${shortSens.toFixed(1)}%`, getS(shortSens, 40)],
          ['Activation Clustering', `Silhouette: ${sil.toFixed(3)}`, getS(sil, 0.3)],
          ['Gradient Similarity', `Cosine Sim: ${css.toFixed(3)}`, getS(css, 0.7)],
        ],
        theme: 'grid',
        headStyles: { fillColor: [0, 150, 180], textColor: [255, 255, 255], fontStyle: 'bold' },
        styles: { fontSize: 8.5, cellPadding: 3.5 },
        columnStyles: { 0: { cellWidth: 48 }, 1: { cellWidth: 90 } },
        didParseCell: (d) => {
          if (d.section === 'body' && d.column.index === 2) {
            d.cell.styles.textColor = d.cell.raw === 'HIGH RISK'
              ? [220, 38, 38] as [number, number, number]
              : [100, 100, 100] as [number, number, number];
            if (d.cell.raw === 'HIGH RISK') d.cell.styles.fontStyle = 'bold';
          }
        },
      });

      y = (doc as any).lastAutoTable.finalY + 10;
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(12);
      doc.setTextColor(20, 20, 20);
      doc.text('RECOMMENDATIONS', 14, y);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8.5);
      doc.setTextColor(80, 80, 80);
      let ry = y + 6;
      (data.strategic_recommendations || []).forEach((rec: string, i: number) => {
        const lines = doc.splitTextToSize(`${i + 1}. ${rec}`, W - 28);
        doc.text(lines, 14, ry);
        ry += lines.length * 4.5 + 2;
      });

      const blob = doc.output('blob');
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `Gemini_Audit_${id.slice(0, 8)}.pdf`;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
    } catch (err: any) {
      alert('PDF error: ' + ((err?.message as string)?.slice(0, 200) || 'unknown'));
    }
  };

  if (!mounted) return null;

  const riskColor = result
    ? result.fusion_risk_score > 0.75 ? 'var(--danger)'
      : result.fusion_risk_score > 0.4 ? 'var(--warning)'
      : 'var(--success)'
    : 'var(--accent)';

  const tabs: { id: ResultTab; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <BarChart3 size={14} /> },
    { id: 'forensics', label: 'Forensics', icon: <Microscope size={14} /> },
    { id: 'visual', label: 'Visual Evidence', icon: <Eye size={14} /> },
    { id: 'report', label: 'Report', icon: <FileText size={14} /> },
  ];

  const navItems: { id: NavItem; label: string; icon: React.ReactNode }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={18} /> },
    { id: 'scan', label: 'Scan Models', icon: <ShieldCheck size={18} /> },
    { id: 'history', label: 'Results History', icon: <History size={18} /> },
    { id: 'analytics', label: 'Analytics', icon: <LineChart size={18} /> },
    { id: 'settings', label: 'System Settings', icon: <Settings size={18} /> },
  ];

  return (
    <>
      {/* ── Top Navigation ── */}
      <header className="topnav">
        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
          <span style={{ fontSize: '1.15rem', fontWeight: 800, letterSpacing: '0.15em', color: 'var(--accent)', fontFamily: 'var(--font-headline)', textTransform: 'uppercase' }}>
            Gemini Core
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(255,255,255,0.05)', padding: '0.45rem 0.9rem', borderRadius: '0.5rem' }}>
            <Search size={14} color="var(--on-surface-variant)" />
            <input
              type="text"
              placeholder="Search telemetry..."
              style={{ background: 'transparent', border: 'none', outline: 'none', fontSize: '0.85rem', color: 'var(--on-surface-variant)', width: '200px', fontFamily: 'var(--font-body)' }}
            />
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button style={{ padding: '0.5rem', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--on-surface-variant)', borderRadius: '0.5rem' }}
            onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.05)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'none')}>
            <Bell size={20} />
          </button>
          <button style={{ padding: '0.5rem', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--on-surface-variant)', borderRadius: '0.5rem' }}
            onClick={() => { setApiStatus('checking'); apiFetch(`${API_BASE}/live`).then(r => setApiStatus(r.ok ? 'online' : 'offline')).catch(() => setApiStatus('offline')); }}
            onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.05)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'none')}>
            <Settings size={20} />
          </button>
          <div style={{ width: '1px', height: '28px', background: 'rgba(255,255,255,0.08)', margin: '0 0.5rem' }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{
              width: '8px', height: '8px', borderRadius: '50%',
              background: apiStatus === 'online' ? 'var(--success)' : apiStatus === 'offline' ? 'var(--danger)' : 'var(--warning)',
              animation: apiStatus === 'checking' ? 'pulsate 1.5s infinite' : 'none',
            }} />
            <span style={{ fontSize: '0.8rem', fontWeight: 700, fontFamily: 'var(--font-headline)', color: 'var(--accent)', letterSpacing: '-0.01em' }}>
              {apiStatus === 'online' ? 'Core Active' : apiStatus === 'offline' ? 'Offline' : 'Connecting...'}
            </span>
          </div>
        </div>
      </header>

      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div style={{ padding: '1.5rem 1.5rem 1rem' }}>
          <h2 style={{ color: 'var(--accent)', fontWeight: 800, fontFamily: 'var(--font-headline)', fontSize: '1rem', textTransform: 'uppercase', letterSpacing: '0.15em' }}>
            The Sentinel
          </h2>
          <p style={{ fontSize: '0.65rem', color: '#475569', letterSpacing: '0.18em', textTransform: 'uppercase', marginTop: '0.2rem' }}>v2.4.0-Stable</p>
        </div>

        <nav style={{ flex: 1, paddingTop: '0.5rem' }}>
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveNav(item.id)}
              className={`nav-item ${activeNav === item.id ? 'active' : ''}`}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        {/* New Scan CTA */}
        <div style={{ padding: '1rem 1.5rem', borderTop: '1px solid rgba(255,255,255,0.05)', marginBottom: '0.5rem' }}>
          <button
            className="button-primary"
            onClick={() => { setActiveNav('scan'); setResult(null); setError(null); setSelectedFile(null); }}
            style={{ width: '100%' }}
          >
            <Plus size={18} />
            New Scan
          </button>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main className="main-content">
        <div className="main-inner">

          {/* ── Upload + Config Section ── */}
          {activeNav === 'scan' && (
            <section style={{ marginBottom: '2.5rem' }}>
              <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: '1rem' }}>
                <div>
                  <h2 style={{ fontSize: '1.75rem', fontWeight: 800, fontFamily: 'var(--font-headline)', letterSpacing: '-0.03em', color: 'var(--foreground)' }}>
                    Scan Models
                  </h2>
                  <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.875rem', marginTop: '0.25rem' }}>
                    Upload a neural network and run the 10-signal forensic audit
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '0.75rem' }}>
                  <div className="card glass-panel" style={{ padding: '0.6rem 1rem', borderColor: 'rgba(16,185,129,0.2)' }}>
                    <p className="label" style={{ fontSize: '0.55rem', marginBottom: '0.25rem' }}>Infrastructure</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <Server size={11} color="var(--success)" />
                      <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--success)', fontFamily: 'var(--font-headline)' }}>STABLE</span>
                    </div>
                  </div>
                  <div className="card glass-panel" style={{ padding: '0.6rem 1rem', borderColor: 'rgba(60,215,255,0.2)' }}>
                    <p className="label" style={{ fontSize: '0.55rem', marginBottom: '0.25rem' }}>Signals</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <Layers size={11} color="var(--accent)" />
                      <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--accent)', fontFamily: 'var(--font-headline)' }}>10 ACTIVE</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Bento grid: drag-drop (2/3) + config panel (1/3) */}
              <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.25rem', minHeight: '280px' }}>
                {/* Drag & Drop */}
                <div className="card glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  <DragDropZone onFile={setSelectedFile} selectedFile={selectedFile} />
                  {selectedFile && (
                    <button
                      onClick={() => setSelectedFile(null)}
                      style={{ background: 'none', border: 'none', color: 'var(--danger)', fontSize: '0.72rem', cursor: 'pointer', marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.3rem', fontFamily: 'var(--font-headline)', fontWeight: 700 }}
                    >
                      <X size={11} /> Clear file
                    </button>
                  )}
                </div>

                {/* Config + scan button */}
                <div className="card glass-panel" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', gap: '1rem' }}>
                  <div>
                    {/* Recent scans */}
                    <button
                      onClick={() => setShowHistory(!showHistory)}
                      style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', background: 'none', border: 'none', cursor: 'pointer', marginBottom: showHistory ? '0.75rem' : '1rem' }}
                    >
                      <span className="label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', fontSize: '0.6rem' }}>
                        <Clock size={10} /> Recent ({history.length})
                      </span>
                      <ChevronRight size={12} color="var(--outline)" style={{ transform: showHistory ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s' }} />
                    </button>
                    {showHistory && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', marginBottom: '1rem', maxHeight: '160px', overflowY: 'auto' }}>
                        {history.length === 0
                          ? <p style={{ fontSize: '0.72rem', color: 'var(--on-surface-variant)', textAlign: 'center', padding: '0.5rem' }}>No scans yet</p>
                          : history.map(e => <HistoryItem key={e.id} entry={e} onRestore={entry => { setResult(entry.result); setActiveTab('overview'); setShowHistory(false); }} />)
                        }
                      </div>
                    )}

                    <button
                      className="button-primary"
                      onClick={startScan}
                      disabled={!selectedFile || isScanning || apiStatus === 'offline'}
                      style={{ width: '100%' }}
                    >
                      {isScanning ? <Loader2 className="animate-spin" size={18} /> : <Zap size={18} />}
                      {isScanning ? 'Analyzing...' : 'Execute Forensic Audit'}
                    </button>
                  </div>
                </div>
              </div>
            </section>
          )}

          {/* ── Error Banner ── */}
          {error && (
            <div className="card stagger-1" style={{ background: 'rgba(239,68,68,0.07)', border: '1px solid var(--danger)', marginBottom: '2rem', padding: '1.25rem' }}>
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <div style={{ padding: '0.5rem', borderRadius: '0.5rem', background: 'var(--danger)', flexShrink: 0 }}>
                  <AlertTriangle color="white" size={18} />
                </div>
                <div style={{ flex: 1 }}>
                  <p style={{ fontWeight: 800, color: 'var(--foreground)', fontSize: '0.9rem', fontFamily: 'var(--font-headline)' }}>Audit Interrupted</p>
                  <p style={{ fontSize: '0.82rem', color: 'var(--on-surface-variant)', marginTop: '0.2rem' }}>{error}</p>
                </div>
                <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--outline)' }}>
                  <X size={16} />
                </button>
              </div>
            </div>
          )}


          {/* ── Scanning Progress ── */}
          {isScanning && (
            <div className="stagger-1">
              <div className="card glass-panel" style={{ border: '1px solid rgba(60,215,255,0.2)', padding: '2rem', marginBottom: '2rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{ background: 'rgba(60,215,255,0.12)', padding: '0.75rem', borderRadius: '0.75rem', border: '1px solid rgba(60,215,255,0.25)' }}>
                      <Cpu size={22} color="var(--accent)" />
                    </div>
                    <div>
                      <h3 style={{ fontSize: '1.1rem', fontWeight: 800, fontFamily: 'var(--font-headline)' }}>Forensic Pipeline Running</h3>
                      <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.8rem', marginTop: '0.15rem' }}>{statusMessage || 'Analyzing model tensors...'}</p>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span className="badge badge-cyan">{progress}% Analyzed</span>
                    <p style={{ fontSize: '0.6rem', color: 'var(--outline)', fontWeight: 700, marginTop: '0.4rem', fontFamily: 'monospace' }}>{taskId?.slice(0, 16)}...</p>
                  </div>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.04)', height: '6px', borderRadius: '3px', overflow: 'hidden', marginBottom: '1.75rem' }}>
                  <div style={{
                    height: '100%', width: `${progress}%`,
                    background: 'linear-gradient(90deg, var(--accent), #a8e8ff)',
                    borderRadius: '3px', transition: 'width 1s cubic-bezier(0.4,0,0.2,1)',
                    boxShadow: '0 0 10px rgba(60,215,255,0.4)',
                  }} />
                </div>
                <div className="pipeline-grid">
                  {PIPELINE_STEPS.map((step, index) => (
                    <AuditStep
                      key={step.id}
                      label={step.label}
                      status={getStepStatus(step.id)}
                      subtext={step.subtext}
                      isLast={index === PIPELINE_STEPS.length - 1}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── Results ── */}
          {result && (
            <div className="stagger-1">

              {/* Section header */}
              <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                <div>
                  <h2 style={{ fontSize: '1.75rem', fontWeight: 800, fontFamily: 'var(--font-headline)', letterSpacing: '-0.03em' }}>Analysis Results</h2>
                  <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.82rem', marginTop: '0.25rem' }}>Deep neural network security telemetry</p>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {(() => {
                    const score = result.fusion_risk_score ?? 0;
                    const isCritical = score > 0.75;
                    const isWarning = score > 0.4;
                    return (
                      <span className={`badge ${isCritical ? 'badge-danger' : isWarning ? 'badge-warning' : 'badge-success'}`}>
                        {isCritical ? 'Threat Detected' : isWarning ? 'Manual Review' : 'Model Clean'}
                      </span>
                    );
                  })()}
                  <span className="badge" style={{ background: 'var(--surface-high)', color: 'var(--on-surface-variant)', border: '1px solid var(--card-border)' }}>
                    {result.details?.architecture || 'Model'}
                  </span>
                </div>
              </div>

              {/* Verdict Banner */}
              {(() => {
                const score = result.fusion_risk_score ?? 0;
                const isCritical = score > 0.75;
                const isWarning = score > 0.4;
                const bc = isCritical ? 'var(--danger)' : isWarning ? 'var(--warning)' : 'var(--success)';
                const bg = isCritical ? 'rgba(239,68,68,0.07)' : isWarning ? 'rgba(245,158,11,0.06)' : 'rgba(16,185,129,0.06)';
                return (
                  <div style={{
                    background: bg, border: `1px solid ${bc}`, borderRadius: '0.75rem',
                    padding: '1.5rem 2rem', marginBottom: '1.5rem',
                    display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap',
                    animation: isCritical ? 'verdict-pulse 2.5s ease-in-out infinite' : 'none',
                  }}>
                    <div style={{ width: '48px', height: '48px', borderRadius: '50%', background: isCritical ? 'rgba(239,68,68,0.12)' : isWarning ? 'rgba(245,158,11,0.1)' : 'rgba(16,185,129,0.12)', border: `1px solid ${bc}`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      {isCritical || isWarning ? <AlertTriangle size={22} color={bc} /> : <ShieldCheck size={22} color={bc} />}
                    </div>
                    <div style={{ flex: 1, minWidth: '200px' }}>
                      <p className="label" style={{ color: bc, marginBottom: '0.25rem', fontSize: '0.58rem' }}>Audit Verdict</p>
                      <h2 style={{ fontSize: '1.4rem', fontWeight: 900, fontFamily: 'var(--font-headline)', letterSpacing: '-0.02em', color: 'var(--foreground)', marginBottom: '0.3rem' }}>
                        {isCritical ? 'Trojan Detected' : isWarning ? 'Manual Review Required' : 'Model Is Clean'}
                      </h2>
                      <p style={{ fontSize: '0.82rem', color: 'var(--on-surface-variant)', lineHeight: 1.5, maxWidth: '520px' }}>
                        {isCritical
                          ? `Strong trojan indicators across multiple forensic channels. Confidence: ${(score * 100).toFixed(0)}%. Do NOT deploy.`
                          : isWarning
                            ? `Borderline indicators detected. Manual review recommended before production deployment.`
                            : `No trojan implants detected. Passed all 10 forensic channels. Risk: ${(score * 100).toFixed(0)}%.`}
                      </p>
                    </div>
                    <div style={{ padding: '0.9rem 1.5rem', borderRadius: '0.75rem', background: isCritical ? 'rgba(239,68,68,0.1)' : isWarning ? 'rgba(245,158,11,0.08)' : 'rgba(16,185,129,0.08)', border: `1px solid ${bc}`, textAlign: 'center', flexShrink: 0 }}>
                      <p className="label" style={{ color: bc, marginBottom: '0.2rem', fontSize: '0.55rem' }}>Risk Score</p>
                      <p style={{ fontSize: '2.2rem', fontWeight: 900, lineHeight: 1, color: 'var(--foreground)', fontFamily: 'monospace' }}>
                        {(score * 100).toFixed(0)}<span style={{ fontSize: '1rem', opacity: 0.4 }}>%</span>
                      </p>
                    </div>
                  </div>
                );
              })()}

              {/* Tabs */}
              <div style={{ display: 'flex', gap: '0.35rem', marginBottom: '1.5rem', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--card-border)', borderRadius: '0.75rem', padding: '0.35rem' }}>
                {tabs.map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    style={{
                      flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem',
                      padding: '0.55rem', borderRadius: '0.5rem', border: 'none', cursor: 'pointer',
                      fontFamily: 'var(--font-headline)', fontWeight: 700, fontSize: '0.78rem',
                      background: activeTab === tab.id ? 'var(--primary-container)' : 'transparent',
                      color: activeTab === tab.id ? 'var(--on-primary-container)' : 'var(--on-surface-variant)',
                      transition: 'all 0.2s ease',
                      boxShadow: activeTab === tab.id ? '0 2px 10px rgba(0,212,255,0.2)' : 'none',
                    }}
                  >
                    {tab.icon} {tab.label}
                  </button>
                ))}
              </div>

              {/* ── TAB: OVERVIEW ── */}
              {activeTab === 'overview' && (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                    <StatCard label="Fusion Risk Score" value={`${(result.fusion_risk_score * 100).toFixed(1)}%`} color={riskColor} sub="10-signal weighted ensemble" />
                    <StatCard label="Attack Success" value={`${((result.details?.attack_success_rate ?? 0) * 100).toFixed(1)}%`} color={(result.details?.behavioral_backdoor_risk ?? 0) > 0.5 ? 'var(--danger)' : 'var(--success)'} sub="Triggered target rate" />
                    <StatCard label="AC Silhouette" value={(result.details?.clustering_silhouette_score ?? 0).toFixed(4)} color={(result.details?.clustering_silhouette_score ?? 0) > 0.3 ? 'var(--danger)' : 'var(--success)'} sub="Latent separation" />
                    <StatCard label="Gradient Sim" value={(result.details?.gradient_similarity ?? 0).toFixed(4)} color={(result.details?.gradient_similarity ?? 0) > 0.7 ? 'var(--danger)' : 'var(--success)'} sub="Cosine similarity" />
                  </div>

                  <div className="grid-cols-2" style={{ marginBottom: '1.5rem' }}>
                    <div className="card glass-panel">
                      <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)', marginBottom: '1rem' }}>
                        <TrendingUp size={16} color="var(--accent)" /> 10-Signal Risk Radar
                      </h3>
                      <RiskRadarChart details={result.details} />
                    </div>
                    <div className="card glass-panel">
                      <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)', marginBottom: '1.25rem' }}>
                        <BarChart3 size={16} color="var(--accent)" /> Defense Signal Breakdown
                      </h3>
                      <SignalGauge label="Black-Box Sweep" value={result.details.blackbox_sweep_risk ?? 0} icon={<ScanLine size={10} />} />
                      <SignalGauge label="Neural Cleanse" value={result.details.neural_cleanse_risk ?? 0} icon={<ScanLine size={10} />} />
                      <SignalGauge label="Behavioral Backdoor" value={result.details.behavioral_backdoor_risk ?? 0} icon={<Activity size={10} />} />
                      <SignalGauge label="STRIP Robustness" value={result.details.strip_risk ?? 0} icon={<FlaskConical size={10} />} />
                      <SignalGauge label="Activation Clustering" value={result.details.clustering_risk ?? 0} icon={<Layers size={10} />} />
                      <SignalGauge label="Linear Weight Audit" value={result.details.weight_analysis_risk ?? 0} icon={<GitBranch size={10} />} />
                      <SignalGauge label="Natural Trojan Profiler" value={result.details.natural_trojan_risk ?? 0} icon={<BrainCircuit size={10} />} />
                      <SignalGauge label="Gradient Similarity" value={result.details.gradient_similarity_risk ?? 0} icon={<TrendingUp size={10} />} />
                      <SignalGauge label="Spectral Signatures" value={result.details.spectral_signatures_risk ?? 0} icon={<Layers size={10} />} />
                      <SignalGauge label="Confidence Distribution" value={result.details.confidence_distribution_risk ?? 0} icon={<BarChart3 size={10} />} />
                    </div>
                  </div>

                  <div className="card glass-panel" style={{ marginBottom: '1.5rem' }}>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)', marginBottom: '1rem' }}>
                      <Activity size={16} color="var(--accent)" /> Signal Risk Comparison
                    </h3>
                    <SignalBarChart details={result.details} />
                  </div>
                </div>
              )}

              {/* ── TAB: FORENSICS ── */}
              {activeTab === 'forensics' && (
                <div>
                  <div className="card glass-panel" style={{ marginBottom: '1.5rem', borderLeft: `3px solid ${result.details.nc_flagged_classes?.length > 0 ? 'var(--danger)' : 'var(--success)'}` }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <p className="label" style={{ marginBottom: '0.35rem', fontSize: '0.58rem' }}>Neural Cleanse Trigger Verdict</p>
                        <p style={{ fontSize: '1rem', fontWeight: 800, fontFamily: 'var(--font-headline)', color: result.details.nc_flagged_classes?.length > 0 ? 'var(--danger)' : 'var(--success)' }}>
                          {result.details.nc_flagged_classes?.length > 0
                            ? `Trigger Signature Detected → Target Class ${result.details.nc_flagged_classes[0]}`
                            : 'No Trigger Signatures Found — Model Appears Clean'}
                        </p>
                      </div>
                      <ShieldCheck size={26} color={result.details.nc_flagged_classes?.length > 0 ? 'var(--danger)' : 'var(--success)'} style={{ opacity: 0.5 }} />
                    </div>
                  </div>

                  <div className="card glass-panel" style={{ marginBottom: '1.5rem' }}>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)', marginBottom: '1.25rem' }}>
                      <GitBranch size={16} color="var(--accent)" /> Raw Telemetry Values
                    </h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                      {[
                        { label: 'STRIP False Rejection', value: result.details.strip_fr_ratio?.toFixed(6), bad: result.details.strip_fr_ratio > 0.1 },
                        { label: 'STRIP False Acceptance', value: result.details.strip_fa_ratio?.toFixed(6), bad: result.details.strip_fa_ratio > 0.1 },
                        { label: 'Black-Box Sweep Risk', value: result.details.blackbox_sweep_risk?.toFixed(6), bad: result.details.blackbox_sweep_risk > 0.5 },
                        { label: 'Black-Box Target Lift', value: result.details.blackbox_sweep_lift?.toFixed(6), bad: result.details.blackbox_sweep_lift > 0.25 },
                        { label: 'Black-Box Trigger', value: result.details.blackbox_sweep_trigger ?? 'N/A', bad: result.details.blackbox_sweep_risk > 0.5 },
                        { label: 'Attack Success Rate', value: result.details.attack_success_rate?.toFixed(6), bad: result.details.behavioral_backdoor_risk > 0.5 },
                        { label: 'Clean Target Rate', value: result.details.clean_target_rate?.toFixed(6), bad: false },
                        { label: 'Target Lift', value: result.details.target_lift?.toFixed(6), bad: result.details.target_lift > 0.3 },
                        { label: 'AC Silhouette Score', value: result.details.clustering_silhouette_score?.toFixed(6), bad: result.details.clustering_silhouette_score > 0.3 },
                        { label: 'Weight Anomaly Index', value: result.details.weight_analysis_risk?.toFixed(6), bad: result.details.weight_analysis_risk > 0.5 },
                        { label: 'Gradient Similarity', value: result.details.gradient_similarity?.toFixed(6), bad: result.details.gradient_similarity > 0.7 },
                        { label: 'Natural Shortcut Sensitivity', value: result.details.natural_sensitivity?.toFixed(6), bad: result.details.natural_sensitivity > 0.4 },
                      ].map(({ label, value, bad }) => (
                        <div key={label} style={{ padding: '0.9rem 1.1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '0.5rem', borderLeft: `3px solid ${bad ? 'var(--danger)' : 'var(--card-border)'}` }}>
                          <p className="label" style={{ fontSize: '0.56rem', marginBottom: '0.25rem' }}>{label}</p>
                          <p style={{ fontSize: '1rem', fontWeight: 800, fontFamily: 'monospace', color: bad ? 'var(--danger)' : 'var(--foreground)' }}>{value ?? 'N/A'}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {result.details.forensic_analysis?.length > 0 && (
                    <div className="card glass-panel">
                      <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)', marginBottom: '1.25rem' }}>
                        <Microscope size={16} color="var(--accent)" /> Forensic Discovery Narrative
                      </h3>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: '1rem' }}>
                        {result.details.forensic_analysis.map((item: any, idx: number) => (
                          <div key={idx} className="glass-hover" style={{
                            padding: '1.1rem 1.25rem',
                            borderLeft: `3px solid ${item.severity === 'CRITICAL' ? 'var(--danger)' : item.severity === 'HIGH' ? 'var(--warning)' : 'var(--accent)'}`,
                            background: 'rgba(255,255,255,0.02)', borderRadius: '0 0.5rem 0.5rem 0'
                          }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                              <div>
                                <p style={{ fontSize: '0.6rem', fontWeight: 900, letterSpacing: '0.08em', color: 'var(--accent)', fontFamily: 'var(--font-headline)' }}>{item.method?.toUpperCase()} AUDIT</p>
                                <p style={{ fontSize: '0.72rem', color: 'var(--on-surface-variant)' }}>{item.layer}</p>
                              </div>
                              <span className={`badge ${item.severity === 'CRITICAL' ? 'badge-danger' : 'badge-warning'}`} style={{ fontSize: '0.56rem' }}>{item.severity}</span>
                            </div>
                            <p style={{ fontSize: '0.82rem', lineHeight: 1.6, color: 'rgba(223,226,239,0.75)' }}>{item.reasoning}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ── TAB: VISUAL EVIDENCE ── */}
              {activeTab === 'visual' && (
                <div>
                  <div className="grid-cols-2" style={{ marginBottom: '1.5rem' }}>
                    <div className="card glass-panel" style={{ padding: 0, overflow: 'hidden' }}>
                      <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid var(--card-border)' }}>
                        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)' }}>
                          <Eye size={16} color="var(--accent)" /> Raw Input Image
                        </h3>
                      </div>
                      <div style={{ padding: '1.5rem', minHeight: '280px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                        {result.raw_image_b64 ? (
                          <>
                            <div style={{ border: '1px solid var(--card-border)', borderRadius: '0.5rem', overflow: 'hidden', background: '#000', maxWidth: '260px', width: '100%' }}>
                              <img src={`data:image/jpeg;base64,${result.raw_image_b64}`} alt="Raw Input" style={{ width: '100%', display: 'block', imageRendering: 'pixelated' }} />
                            </div>
                            <p style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)', marginTop: '1rem', textAlign: 'center', lineHeight: 1.6 }}>
                              Test image fed into the model. A visible trigger pattern near borders suggests backdoor vulnerability.
                            </p>
                          </>
                        ) : (
                          <div style={{ textAlign: 'center', opacity: 0.4 }}>
                            <Eye size={44} style={{ marginBottom: '0.75rem', color: 'var(--on-surface-variant)' }} />
                            <p style={{ fontSize: '0.85rem', color: 'var(--on-surface-variant)' }}>Visual forensics unavailable</p>
                            <p style={{ fontSize: '0.72rem', color: 'var(--outline)', marginTop: '0.25rem' }}>Raw input extraction failed on this architecture</p>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="card glass-panel" style={{ padding: 0, overflow: 'hidden' }}>
                      <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid var(--card-border)' }}>
                        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)' }}>
                          <Fingerprint size={16} color="var(--accent)" /> GradCAM Attention Heatmap
                        </h3>
                      </div>
                      <div style={{ padding: '1.5rem', minHeight: '280px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                        {result.gradcam_heatmap_b64 ? (
                          <>
                            <div style={{ border: '1px solid var(--card-border)', borderRadius: '0.5rem', overflow: 'hidden', background: '#000', maxWidth: '260px', width: '100%' }}>
                              <img src={`data:image/png;base64,${result.gradcam_heatmap_b64}`} alt="GradCAM Heatmap" style={{ width: '100%', display: 'block' }} />
                            </div>
                            <p style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)', marginTop: '1rem', textAlign: 'center', lineHeight: 1.6 }}>
                              Gradient-weighted class activation map. Atypical focus near image borders suggests a trigger.
                            </p>
                          </>
                        ) : (
                          <div style={{ textAlign: 'center', opacity: 0.4 }}>
                            <Activity size={44} style={{ marginBottom: '0.75rem', color: 'var(--on-surface-variant)' }} />
                            <p style={{ fontSize: '0.85rem', color: 'var(--on-surface-variant)' }}>GradCAM unavailable</p>
                            <p style={{ fontSize: '0.72rem', color: 'var(--outline)', marginTop: '0.25rem' }}>Gradient computation failed on this architecture</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {result.details.tsne_plot_b64 && (
                    <div className="card glass-panel" style={{ padding: 0, overflow: 'hidden' }}>
                      <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid var(--card-border)' }}>
                        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)' }}>
                          <BrainCircuit size={16} color="var(--accent)" /> Activation Space — t-SNE Projection
                        </h3>
                      </div>
                      <div style={{ padding: '1.5rem', display: 'flex', gap: '2rem', alignItems: 'center', flexWrap: 'wrap' }}>
                        <div style={{ flex: 1, minWidth: '200px' }}>
                          <p style={{ fontSize: '0.85rem', color: 'var(--on-surface-variant)', lineHeight: 1.7, marginBottom: '1rem' }}>
                            High-dimensional latent activations projected to 2D. Isolated poisoned clusters confirm backdoor presence.
                          </p>
                          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                            <div className="card glass-panel" style={{ padding: '0.75rem 1rem' }}>
                              <p className="label" style={{ fontSize: '0.56rem', marginBottom: '0.25rem' }}>Silhouette Score</p>
                              <p style={{ fontWeight: 800, fontSize: '1rem', fontFamily: 'monospace', color: result.details.clustering_silhouette_score > 0.1 ? 'var(--danger)' : 'var(--success)' }}>
                                {result.details.clustering_silhouette_score?.toFixed(4)}
                              </p>
                            </div>
                            <div className="card glass-panel" style={{ padding: '0.75rem 1rem' }}>
                              <p className="label" style={{ fontSize: '0.56rem', marginBottom: '0.25rem' }}>Cluster Risk</p>
                              <p style={{ fontWeight: 800, fontSize: '1rem', color: result.details.clustering_risk > 0.4 ? 'var(--danger)' : 'var(--success)' }}>
                                {((result.details.clustering_risk ?? 0) * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>
                        </div>
                        <div style={{ flex: 2, minWidth: '260px', border: '1px solid var(--card-border)', borderRadius: '0.5rem', overflow: 'hidden', background: '#e8edf2', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <img src={`data:image/png;base64,${result.details.tsne_plot_b64}`} alt="t-SNE Latent Space" style={{ width: '100%', display: 'block', mixBlendMode: 'multiply' }} />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ── TAB: REPORT ── */}
              {activeTab === 'report' && (
                <div>
                  <div className="card glass-panel" style={{ padding: '2.5rem', textAlign: 'center', marginBottom: '1.5rem' }}>
                    <div style={{ width: '56px', height: '56px', borderRadius: '50%', background: 'rgba(60,215,255,0.1)', border: '1px solid rgba(60,215,255,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.25rem' }}>
                      <FileText size={24} color="var(--accent)" />
                    </div>
                    <h3 style={{ fontSize: '1.3rem', fontWeight: 900, fontFamily: 'var(--font-headline)', marginBottom: '0.6rem' }}>IARPA-Compliant Audit Report</h3>
                    <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.875rem', maxWidth: '460px', margin: '0 auto 1.75rem', lineHeight: 1.7 }}>
                      Generates a comprehensive PDF including executive summary, model information, 10-signal forensic analysis, and strategic recommendations.
                    </p>
                    <button className="button-primary" onClick={downloadReport} style={{ margin: '0 auto' }}>
                      <Download size={18} /> Download PDF Report
                    </button>
                  </div>

                  <div className="card glass-panel">
                    <h3 style={{ fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Info size={15} color="var(--accent)" /> Report Preview
                    </h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                      {[
                        { section: 'Executive Summary', items: [`Verdict: ${determineVerdict(result.fusion_risk_score ?? 0)}`, `Risk Score: ${((result.fusion_risk_score ?? 0) * 100).toFixed(1)}%`] },
                        { section: 'Model Information', items: [`Architecture: ${result.details?.architecture || 'ResNet-18'}`, `Framework: ${result.details?.is_onnx ? 'ONNX' : 'PyTorch'}`, `Parameters: ${result.details?.parameter_count || 'N/A'}`] },
                        { section: 'Forensic Signals', items: ['Black-Box Trigger Sweep', 'Behavioral Backdoor Probe', 'Neural Cleanse', 'STRIP Robustness', 'Activation Clustering', 'Weight Analysis', 'Gradient Similarity', 'Natural Trojan Profiling', 'Spectral Signatures', 'Confidence Distribution'] },
                        { section: 'Recommendations', items: ['Defense-in-depth across AI supply chain', 'Verify model provenance for deployments', 'Continuous monitoring for low-ASR backdoors'] },
                      ].map(({ section, items }) => (
                        <div key={section} style={{ padding: '1.1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '0.5rem', border: '1px solid var(--card-border)' }}>
                          <p style={{ fontSize: '0.65rem', fontWeight: 800, fontFamily: 'var(--font-headline)', color: 'var(--accent)', letterSpacing: '0.08em', marginBottom: '0.6rem' }}>{section.toUpperCase()}</p>
                          {items.map(item => (
                            <div key={item} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.35rem' }}>
                              <ChevronRight size={10} color="var(--outline)" />
                              <p style={{ fontSize: '0.8rem', color: 'var(--on-surface-variant)' }}>{item}</p>
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

            </div>
          )}

          {/* ── Results History ── */}
          {activeNav === 'history' && (
            <div className="stagger-1">
              <div style={{ marginBottom: '1.5rem' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 800, fontFamily: 'var(--font-headline)', letterSpacing: '-0.03em' }}>Results History</h2>
                <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.875rem', marginTop: '0.25rem' }}>Previous forensic audit results</p>
              </div>
              {history.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '4rem 2rem', opacity: 0.5 }}>
                  <History size={64} style={{ color: 'var(--on-surface-variant)', marginBottom: '1rem' }} />
                  <p style={{ fontSize: '1rem', fontWeight: 700, fontFamily: 'var(--font-headline)', color: 'var(--on-surface-variant)' }}>No scan history yet</p>
                  <p style={{ fontSize: '0.85rem', color: 'var(--outline)', marginTop: '0.5rem' }}>Run a scan to see results here</p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {history.map(entry => {
                    const isCritical = entry.score > 0.75;
                    const isWarning = entry.score > 0.4;
                    const color = isCritical ? 'var(--danger)' : isWarning ? 'var(--warning)' : 'var(--success)';
                    return (
                      <div key={entry.id} className="card glass-panel" style={{ borderLeft: `3px solid ${color}`, padding: '1.25rem 1.5rem', display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
                        <div style={{ flex: 1, minWidth: '200px' }}>
                          <p style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--foreground)', fontFamily: 'var(--font-headline)' }}>{entry.fileName}</p>
                          <p style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)', marginTop: '0.25rem' }}>{entry.timestamp}</p>
                        </div>
                        <div style={{ textAlign: 'center' }}>
                          <p className="label" style={{ fontSize: '0.55rem', marginBottom: '0.25rem' }}>Risk Score</p>
                          <p style={{ fontSize: '1.4rem', fontWeight: 900, fontFamily: 'monospace', color }}>{(entry.score * 100).toFixed(0)}%</p>
                        </div>
                        <span className={`badge ${isCritical ? 'badge-danger' : isWarning ? 'badge-warning' : 'badge-success'}`}>{entry.verdict}</span>
                        <button
                          onClick={() => { setResult(entry.result); setActiveTab('overview'); setActiveNav('scan'); }}
                          className="button-primary"
                          style={{ padding: '0.6rem 1.1rem', fontSize: '0.8rem' }}
                        >
                          <Eye size={14} /> View Details
                        </button>
                      </div>
                    );
                  })}
                  <button
                    onClick={() => { setHistory([]); try { localStorage.removeItem('gemini_scan_history'); } catch { /* ignore */ } }}
                    style={{ background: 'none', border: '1px solid var(--card-border)', borderRadius: '0.5rem', color: 'var(--danger)', fontSize: '0.78rem', cursor: 'pointer', padding: '0.6rem 1.2rem', fontFamily: 'var(--font-headline)', fontWeight: 700, alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
                  >
                    <X size={13} /> Clear History
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ── Analytics ── */}
          {activeNav === 'analytics' && (
            <div className="stagger-1">
              <div style={{ marginBottom: '1.5rem' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 800, fontFamily: 'var(--font-headline)', letterSpacing: '-0.03em' }}>Analytics</h2>
                <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.875rem', marginTop: '0.25rem' }}>Aggregate telemetry across scan sessions</p>
              </div>
              {history.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '4rem 2rem', opacity: 0.5 }}>
                  <LineChart size={64} style={{ color: 'var(--on-surface-variant)', marginBottom: '1rem' }} />
                  <p style={{ fontSize: '1rem', fontWeight: 700, fontFamily: 'var(--font-headline)', color: 'var(--on-surface-variant)' }}>No data yet</p>
                  <p style={{ fontSize: '0.85rem', color: 'var(--outline)', marginTop: '0.5rem' }}>Complete at least one scan to see analytics</p>
                </div>
              ) : (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                    {[
                      { label: 'Total Scans', value: history.length.toString(), color: 'var(--accent)' },
                      { label: 'Critical', value: history.filter(e => e.score > 0.75).length.toString(), color: 'var(--danger)' },
                      { label: 'Warning', value: history.filter(e => e.score > 0.4 && e.score <= 0.75).length.toString(), color: 'var(--warning)' },
                      { label: 'Clean', value: history.filter(e => e.score <= 0.4).length.toString(), color: 'var(--success)' },
                    ].map(({ label, value, color }) => (
                      <StatCard key={label} label={label} value={value} color={color} />
                    ))}
                  </div>
                  <div className="card glass-panel">
                    <h3 style={{ fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <BarChart3 size={16} color="var(--accent)" /> Risk Score Distribution
                    </h3>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={history.map(e => ({ name: e.fileName.slice(0, 12), score: +(e.score * 100).toFixed(1) }))} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                        <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} />
                        <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                        <RechartsTooltip contentStyle={{ background: '#0f131c', border: '1px solid #3c494e', borderRadius: '8px', fontSize: '0.8rem' }} formatter={(v: any) => [`${v}%`, 'Risk']} />
                        <Bar dataKey="score" radius={[3, 3, 0, 0]}>
                          {history.map((entry, i) => (
                            <Cell key={i} fill={entry.score > 0.75 ? '#ef4444' : entry.score > 0.4 ? '#f59e0b' : '#10b981'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── System Settings ── */}
          {activeNav === 'settings' && (
            <div className="stagger-1">
              <div style={{ marginBottom: '1.5rem' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 800, fontFamily: 'var(--font-headline)', letterSpacing: '-0.03em' }}>System Settings</h2>
                <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.875rem', marginTop: '0.25rem' }}>API configuration and system status</p>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
                <div className="card glass-panel">
                  <h3 style={{ fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Server size={16} color="var(--accent)" /> API Connection
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {[
                      { label: 'Endpoint', value: API_BASE },
                      { label: 'Status', value: apiStatus.toUpperCase(), color: apiStatus === 'online' ? 'var(--success)' : apiStatus === 'offline' ? 'var(--danger)' : 'var(--warning)' },
                    ].map(({ label, value, color }) => (
                      <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '0.5rem' }}>
                        <span className="label" style={{ fontSize: '0.65rem' }}>{label}</span>
                        <span style={{ fontSize: '0.82rem', fontFamily: 'monospace', color: color || 'var(--foreground)', fontWeight: 600 }}>{value}</span>
                      </div>
                    ))}
                    <button
                      className="button-primary"
                      onClick={() => { setApiStatus('checking'); apiFetch(`${API_BASE}/live`).then(r => setApiStatus(r.ok ? 'online' : 'offline')).catch(() => setApiStatus('offline')); }}
                      style={{ width: '100%', marginTop: '0.5rem' }}
                    >
                      <RefreshCw size={15} /> Re-check Connection
                    </button>
                  </div>
                </div>
                <div className="card glass-panel">
                  <h3 style={{ fontSize: '0.9rem', fontWeight: 800, fontFamily: 'var(--font-headline)', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Layers size={16} color="var(--accent)" /> Detection Signals
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {['Neural Cleanse', 'STRIP Robustness', 'Activation Clustering', 'Weight Audit', 'Spectral Signatures', 'Behavior Probe', 'Black-Box Sweep', 'Natural Trojan Profiler', 'Gradient Similarity', 'Confidence Distribution'].map(sig => (
                      <div key={sig} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 0.75rem', background: 'rgba(255,255,255,0.02)', borderRadius: '0.4rem' }}>
                        <span style={{ fontSize: '0.8rem', color: 'var(--foreground)' }}>{sig}</span>
                        <span className="badge badge-success" style={{ fontSize: '0.55rem', padding: '0.2rem 0.5rem' }}>Active</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── Dashboard ── */}
          {activeNav === 'dashboard' && (
            <div className="stagger-1">
              <div style={{ marginBottom: '1.5rem' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 800, fontFamily: 'var(--font-headline)', letterSpacing: '-0.03em' }}>Dashboard</h2>
                <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.875rem', marginTop: '0.25rem' }}>System overview and quick actions</p>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                {[
                  { label: 'Total Scans', value: history.length.toString(), color: 'var(--accent)', sub: 'All time' },
                  { label: 'Threats Found', value: history.filter(e => e.score > 0.75).length.toString(), color: 'var(--danger)', sub: 'Critical verdicts' },
                  { label: 'API Status', value: apiStatus === 'online' ? 'Online' : 'Offline', color: apiStatus === 'online' ? 'var(--success)' : 'var(--danger)', sub: API_BASE },
                  { label: 'Signals', value: '10', color: 'var(--accent)', sub: 'All active' },
                ].map(({ label, value, color, sub }) => (
                  <StatCard key={label} label={label} value={value} color={color} sub={sub} />
                ))}
              </div>
              {result ? (
                <div className="card glass-panel" style={{ padding: '1.5rem', borderLeft: '3px solid var(--accent)' }}>
                  <p className="label" style={{ marginBottom: '0.75rem', fontSize: '0.6rem' }}>Last Scan Result</p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1 }}>
                      <p style={{ fontWeight: 800, fontSize: '1rem', fontFamily: 'var(--font-headline)' }}>{history[0]?.fileName || 'Unknown model'}</p>
                      <p style={{ fontSize: '0.78rem', color: 'var(--on-surface-variant)', marginTop: '0.2rem' }}>Risk: {((result.fusion_risk_score ?? 0) * 100).toFixed(1)}%</p>
                    </div>
                    <button className="button-primary" onClick={() => setActiveNav('scan')} style={{ padding: '0.6rem 1.1rem', fontSize: '0.8rem' }}>
                      <Eye size={14} /> View Results
                    </button>
                  </div>
                </div>
              ) : (
                <div className="card glass-panel" style={{ padding: '2rem', textAlign: 'center', opacity: 0.6 }}>
                  <ScanLine size={48} style={{ color: 'var(--on-surface-variant)', marginBottom: '0.75rem' }} />
                  <p style={{ fontSize: '0.9rem', fontFamily: 'var(--font-headline)', fontWeight: 700, color: 'var(--on-surface-variant)' }}>No scans yet</p>
                  <button className="button-primary" onClick={() => setActiveNav('scan')} style={{ margin: '1rem auto 0' }}>
                    <Plus size={15} /> Start First Scan
                  </button>
                </div>
              )}
            </div>
          )}

        </div>
      </main>
    </>
  );
}
