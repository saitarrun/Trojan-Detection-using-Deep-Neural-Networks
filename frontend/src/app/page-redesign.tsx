"use client";

import React, { useState, useEffect } from 'react';
import {
  Shield,
  Upload,
  BarChart3,
  Settings,
  LogOut,
  Menu,
  Home,
  Clock,
  Zap,
  AlertTriangle,
  CheckCircle2,
  Download,
  Loader2
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════════════════
// GEMINI TROJAN DETECTION - ENTERPRISE UI REDESIGN
// Modern security-focused interface with refined aesthetics
// Dark theme with cyan primary, magenta alerts, green success states
// ═══════════════════════════════════════════════════════════════════════════

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  badge?: number;
}

const navItems: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', icon: <Home size={20} /> },
  { id: 'scan', label: 'Scan Model', icon: <Upload size={20} /> },
  { id: 'history', label: 'History', icon: <Clock size={20} />, badge: 3 },
  { id: 'analytics', label: 'Analytics', icon: <BarChart3 size={20} /> },
];

export default function GeminiUI() {
  const [activeNav, setActiveNav] = useState('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [useLocalPath, setUseLocalPath] = useState(false);
  const [localPath, setLocalPath] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const startScan = async () => {
    if (!selectedFile && !localPath) {
      alert('Please select a model file');
      return;
    }

    setIsScanning(true);
    try {
      const formData = new FormData();
      if (selectedFile) {
        formData.append('model_file', selectedFile);
      } else {
        formData.append('model_path', localPath);
      }
      formData.append('target_class', '-1');
      formData.append('trigger_type', 'checkerboard');

      const response = await fetch('/api/v1/scan-model', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Scan failed');
      // Handle response...
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <div className="gemini-app">
      {/* Sidebar */}
      <aside className={`gemini-sidebar ${sidebarOpen ? 'open' : 'collapsed'}`}>
        <div className="sidebar-header">
          <div className="logo">
            <Shield size={28} className="logo-icon" />
            {sidebarOpen && <span className="logo-text">Gemini</span>}
          </div>
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <Menu size={20} />
          </button>
        </div>

        <nav className="sidebar-nav">
          {navItems.map(item => (
            <button
              key={item.id}
              className={`nav-item ${activeNav === item.id ? 'active' : ''}`}
              onClick={() => setActiveNav(item.id)}
              title={sidebarOpen ? '' : item.label}
            >
              <span className="nav-icon">{item.icon}</span>
              {sidebarOpen && (
                <>
                  <span className="nav-label">{item.label}</span>
                  {item.badge && <span className="nav-badge">{item.badge}</span>}
                </>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button className="sidebar-btn" title="Settings">
            <Settings size={20} />
            {sidebarOpen && <span>Settings</span>}
          </button>
          <button className="sidebar-btn logout" title="Logout">
            <LogOut size={20} />
            {sidebarOpen && <span>Logout</span>}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="gemini-main">
        {/* Header */}
        <header className="gemini-header">
          <div className="header-left">
            <h1>Model Security Analysis</h1>
            <p className="header-subtitle">Detect trojans and malicious behaviors in neural networks</p>
          </div>
          <div className="header-right">
            <button className="header-btn">
              <Download size={20} />
              Export Report
            </button>
          </div>
        </header>

        {/* Content Area */}
        <div className="gemini-content">
          {activeNav === 'dashboard' && <DashboardView />}
          {activeNav === 'scan' && (
            <ScanView
              selectedFile={selectedFile}
              handleFileChange={handleFileChange}
              startScan={startScan}
              isScanning={isScanning}
              useLocalPath={useLocalPath}
              setUseLocalPath={setUseLocalPath}
              localPath={localPath}
              setLocalPath={setLocalPath}
            />
          )}
          {activeNav === 'history' && <HistoryView />}
          {activeNav === 'analytics' && <AnalyticsView />}
        </div>
      </main>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// VIEW COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

function DashboardView() {
  return (
    <div className="view-container">
      <div className="stats-grid">
        <StatCard
          icon={<Zap size={24} />}
          label="Models Scanned"
          value="24"
          trend="+3 this week"
        />
        <StatCard
          icon={<AlertTriangle size={24} />}
          label="Threats Detected"
          value="2"
          trend="Critical: 1"
          highlight="danger"
        />
        <StatCard
          icon={<CheckCircle2 size={24} />}
          label="Clean Models"
          value="22"
          trend="91.7% success rate"
          highlight="success"
        />
        <StatCard
          icon={<Clock size={24} />}
          label="Avg Scan Time"
          value="2.4s"
          trend="±0.3s variance"
        />
      </div>

      <div className="recent-section">
        <h2>Recent Scans</h2>
        <RecentScansTable />
      </div>
    </div>
  );
}

function ScanView(props: any) {
  return (
    <div className="view-container">
      <div className="scan-layout">
        {/* Upload Section */}
        <div className="scan-section">
          <h2>Upload Model</h2>

          <div className="form-group">
            <div className="radio-group">
              <label className="radio-option">
                <input
                  type="radio"
                  checked={!props.useLocalPath}
                  onChange={() => props.setUseLocalPath(false)}
                />
                Upload File
              </label>
              <label className="radio-option">
                <input
                  type="radio"
                  checked={props.useLocalPath}
                  onChange={() => props.setUseLocalPath(true)}
                />
                Server Path
              </label>
            </div>
          </div>

          {!props.useLocalPath ? (
            <div className="upload-box">
              <input
                id="file-input"
                type="file"
                onChange={props.handleFileChange}
                accept=".pt,.pth,.onnx"
                style={{ display: 'none' }}
              />
              <button
                className="upload-trigger"
                onClick={() => document.getElementById('file-input')?.click()}
              >
                <Upload size={32} />
                <p className="upload-title">
                  {props.selectedFile ? props.selectedFile.name : 'Click to upload model'}
                </p>
                <p className="upload-subtitle">
                  Supports .pt, .pth, .onnx files
                </p>
              </button>
            </div>
          ) : (
            <input
              type="text"
              className="form-input"
              placeholder="/path/to/model.pth"
              value={props.localPath}
              onChange={(e) => props.setLocalPath(e.target.value)}
            />
          )}

          <button
            className={`scan-button ${props.isScanning ? 'loading' : ''}`}
            onClick={props.startScan}
            disabled={props.isScanning}
          >
            {props.isScanning ? (
              <>
                <Loader2 size={20} className="spin" />
                Scanning...
              </>
            ) : (
              <>
                <Zap size={20} />
                Start Security Scan
              </>
            )}
          </button>
        </div>

        {/* Quick Info */}
        <div className="scan-info">
          <div className="info-card">
            <h3>About This Scan</h3>
            <p>
              Our advanced trojan detection system uses 6 complementary signals to identify malicious behaviors in neural networks.
            </p>
            <ul className="info-list">
              <li>Neural Cleanse detection</li>
              <li>STRIP robustness testing</li>
              <li>Activation clustering analysis</li>
              <li>Weight audit mechanisms</li>
              <li>Natural trojan profiling</li>
              <li>Gradient similarity metrics</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function HistoryView() {
  return (
    <div className="view-container">
      <h2>Scan History</h2>
      <div className="empty-state">
        <Clock size={48} />
        <p>No scan history yet</p>
      </div>
    </div>
  );
}

function AnalyticsView() {
  return (
    <div className="view-container">
      <h2>Analytics</h2>
      <div className="empty-state">
        <BarChart3 size={48} />
        <p>Analytics dashboard coming soon</p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// REUSABLE COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

function StatCard({ icon, label, value, trend, highlight }: any) {
  return (
    <div className={`stat-card ${highlight || ''}`}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-content">
        <p className="stat-label">{label}</p>
        <p className="stat-value">{value}</p>
        <p className="stat-trend">{trend}</p>
      </div>
    </div>
  );
}

function RecentScansTable() {
  const scans = [
    { id: 1, model: 'resnet50_cifar10.pth', status: 'clean', score: 12, time: '2 mins ago' },
    { id: 2, model: 'badnet_poisoned.pt', status: 'danger', score: 78, time: '5 mins ago' },
    { id: 3, model: 'vgg16_baseline.pth', status: 'clean', score: 8, time: '1 hour ago' },
  ];

  return (
    <table className="scans-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Status</th>
          <th>Risk Score</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>
        {scans.map(scan => (
          <tr key={scan.id}>
            <td className="model-name">{scan.model}</td>
            <td>
              <span className={`status-badge ${scan.status}`}>
                {scan.status === 'clean' ? '✓ Clean' : '⚠ Threat'}
              </span>
            </td>
            <td className="risk-score">{scan.score}%</td>
            <td className="time">{scan.time}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
