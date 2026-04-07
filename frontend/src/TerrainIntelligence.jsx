// TerrainIntelligence.jsx
// React + TailwindCSS + Framer Motion
// Cinematic sci-fi monitoring dashboard with live visualizations

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence, useSpring, useTransform } from "framer-motion";

const API_BASE = "https://rosaline-beeriest-camie.ngrok-free.dev";

const CLASS_META = {
  Dense_Vegetation: { color: "#22c55e", dim: "#14532d", label: "Dense Veg", trav: 0.45, risk: "MED" },
  Dry_Vegetation:   { color: "#f59e0b", dim: "#78350f", label: "Dry Veg",   trav: 0.75, risk: "LOW" },
  Ground_Objects:   { color: "#a8a29e", dim: "#292524", label: "Ground",    trav: 0.30, risk: "MED-HIGH" },
  Rocks:            { color: "#6b7280", dim: "#111827", label: "Rocks",     trav: 0.15, risk: "HIGH" },
  Landscape:        { color: "#fb923c", dim: "#431407", label: "Landscape", trav: 0.95, risk: "LOW" },
  Sky:              { color: "#38bdf8", dim: "#0c4a6e", label: "Sky",       trav: 0.00, risk: "N/A" },
};

const RISK_META = {
  HIGH:     { color: "#ef4444", glow: "rgba(239,68,68,0.3)",   label: "HIGH" },
  MED_HIGH: { color: "#f97316", glow: "rgba(249,115,22,0.3)",  label: "MED-HIGH" },
  MED:      { color: "#eab308", glow: "rgba(234,179,8,0.3)",   label: "MED" },
  LOW:      { color: "#22c55e", glow: "rgba(34,197,94,0.3)",   label: "LOW" },
  "N/A":    { color: "#475569", glow: "rgba(71,85,105,0.3)",   label: "N/A" },
};

// ── Animated number counter ────────────────────────────────────
function AnimatedNumber({ value, decimals = 0, suffix = "" }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0;
    const end = parseFloat(value) || 0;
    const duration = 1200;
    const step = 16;
    const increment = (end - start) / (duration / step);
    const timer = setInterval(() => {
      start += increment;
      if ((increment > 0 && start >= end) || (increment < 0 && start <= end)) {
        setDisplay(end);
        clearInterval(timer);
      } else {
        setDisplay(start);
      }
    }, step);
    return () => clearInterval(timer);
  }, [value]);
  return <>{display.toFixed(decimals)}{suffix}</>;
}

// ── Scanning line effect ───────────────────────────────────────
function ScanLine() {
  return (
    <motion.div
      style={{
        position: "absolute", left: 0, right: 0, height: 2,
        background: "linear-gradient(90deg, transparent, #22c55e44, #22c55e, #22c55e44, transparent)",
        zIndex: 10, pointerEvents: "none",
      }}
      animate={{ top: ["0%", "100%"] }}
      transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
    />
  );
}

// ── Hexagonal grid background ──────────────────────────────────
function HexGrid() {
  return (
    <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%", opacity: 0.04 }} xmlns="http://www.w3.org/2000/svg">
      <defs>
        <pattern id="hex" x="0" y="0" width="56" height="48" patternUnits="userSpaceOnUse">
          <polygon points="28,4 52,16 52,32 28,44 4,32 4,16" fill="none" stroke="#22c55e" strokeWidth="0.5" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#hex)" />
    </svg>
  );
}

// ── Glowing metric card ────────────────────────────────────────
function GlowCard({ label, value, unit, color = "#22c55e", decimals = 0, delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: "easeOut" }}
      style={{
        background: "#080d18",
        border: `1px solid ${color}33`,
        borderRadius: 8,
        padding: "14px 16px",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <motion.div
        style={{
          position: "absolute", inset: 0, borderRadius: 8,
          background: `radial-gradient(ellipse at 50% 100%, ${color}11, transparent 70%)`,
        }}
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 3, repeat: Infinity }}
      />
      <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 6, fontFamily: "monospace" }}>
        {label}
      </div>
      <div style={{ fontSize: 26, fontWeight: 700, color, fontFamily: "monospace", lineHeight: 1 }}>
        <AnimatedNumber value={value} decimals={decimals} suffix={unit} />
      </div>
    </motion.div>
  );
}

// ── Traversability donut ───────────────────────────────────────
function TraversabilityDonut({ value = 0 }) {
  const r = 38;
  const circ = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(1, value));
  const color = pct > 0.65 ? "#22c55e" : pct > 0.4 ? "#f59e0b" : "#ef4444";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <div style={{ position: "relative", width: 100, height: 100 }}>
        <svg viewBox="0 0 100 100" style={{ width: "100%", height: "100%", transform: "rotate(-90deg)" }}>
          <circle cx="50" cy="50" r={r} fill="none" stroke="#1e293b" strokeWidth="8" />
          <motion.circle
            cx="50" cy="50" r={r} fill="none"
            stroke={color} strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circ}
            initial={{ strokeDashoffset: circ }}
            animate={{ strokeDashoffset: circ * (1 - pct) }}
            transition={{ duration: 1.5, ease: "easeOut" }}
          />
        </svg>
        <div style={{
          position: "absolute", inset: 0, display: "flex",
          flexDirection: "column", alignItems: "center", justifyContent: "center",
        }}>
          <div style={{ fontSize: 18, fontWeight: 700, color, fontFamily: "monospace" }}>
            {Math.round(pct * 100)}%
          </div>
        </div>
      </div>
      <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", fontFamily: "monospace" }}>
        Traversability
      </div>
    </div>
  );
}

// ── Animated class bar ─────────────────────────────────────────
function ClassBar({ name, pct, delay = 0 }) {
  const meta = CLASS_META[name] || { color: "#64748b", label: name };
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.4 }}
      style={{ marginBottom: 10 }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: meta.color, fontFamily: "monospace" }}>{meta.label}</span>
        <span style={{ fontSize: 11, color: "#334155", fontFamily: "monospace" }}>{(pct * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 5, background: "#0f172a", borderRadius: 3, overflow: "hidden", position: "relative" }}>
        <motion.div
          style={{ height: "100%", background: meta.color, borderRadius: 3, position: "relative" }}
          initial={{ width: 0 }}
          animate={{ width: `${pct * 100}%` }}
          transition={{ delay: delay + 0.2, duration: 0.9, ease: "easeOut" }}
        >
          <motion.div
            style={{
              position: "absolute", right: 0, top: 0, bottom: 0, width: 20,
              background: `linear-gradient(90deg, transparent, ${meta.color}aa)`,
            }}
            animate={{ opacity: [0, 1, 0] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: delay + 0.5 }}
          />
        </motion.div>
      </div>
    </motion.div>
  );
}

// ── Risk badge ─────────────────────────────────────────────────
function RiskBadge({ severity, cls, patches, row, col, delay = 0 }) {
  const meta = RISK_META[severity] || RISK_META["N/A"];
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, type: "spring", stiffness: 300 }}
      style={{
        border: `1px solid ${meta.color}55`,
        borderLeft: `3px solid ${meta.color}`,
        borderRadius: "0 6px 6px 0",
        padding: "8px 12px",
        marginBottom: 8,
        background: `${meta.color}0a`,
        display: "flex",
        alignItems: "center",
        gap: 10,
      }}
    >
      <motion.div
        style={{ width: 8, height: 8, borderRadius: "50%", background: meta.color, flexShrink: 0 }}
        animate={{ boxShadow: [`0 0 0 0 ${meta.glow}`, `0 0 0 6px ${meta.glow}00`] }}
        transition={{ duration: 1.5, repeat: Infinity }}
      />
      <div>
        <div style={{ fontSize: 12, fontWeight: 600, color: meta.color, fontFamily: "monospace" }}>
          {severity} · {cls?.replace("_", " ")}
        </div>
        <div style={{ fontSize: 10, color: "#334155", marginTop: 2, fontFamily: "monospace" }}>
          {patches} patches · r{row} c{col}
        </div>
      </div>
    </motion.div>
  );
}

// ── Pipeline step indicator ────────────────────────────────────
function PipelineSteps({ active }) {
  const steps = [
    { id: "seg",   label: "Segmentation",   icon: "◈" },
    { id: "graph", label: "Terrain Graph",  icon: "◉" },
    { id: "path",  label: "Dijkstra Path",  icon: "◈" },
    { id: "mem",   label: "Memory Query",   icon: "◉" },
    { id: "llm",   label: "LLM Reasoning",  icon: "◈" },
  ];

  const activeIdx = steps.findIndex(s => s.id === active);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 0, marginBottom: 16, overflowX: "auto" }}>
      {steps.map((step, i) => {
        const done    = activeIdx > i;
        const current = activeIdx === i;
        return (
          <div key={step.id} style={{ display: "flex", alignItems: "center" }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <motion.div
                animate={{
                  background: done ? "#22c55e" : current ? "#3b82f6" : "#1e293b",
                  borderColor: done ? "#22c55e" : current ? "#3b82f6" : "#334155",
                }}
                style={{
                  width: 28, height: 28, borderRadius: "50%",
                  border: "1px solid",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, color: done || current ? "#fff" : "#334155",
                  fontFamily: "monospace",
                }}
              >
                {done ? "✓" : step.icon}
              </motion.div>
              <div style={{ fontSize: 9, color: current ? "#3b82f6" : done ? "#22c55e" : "#334155", fontFamily: "monospace", whiteSpace: "nowrap" }}>
                {step.label}
              </div>
            </div>
            {i < steps.length - 1 && (
              <motion.div
                style={{ height: 1, width: 28, marginBottom: 14 }}
                animate={{ background: done ? "#22c55e" : "#1e293b" }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Chart.js wrapper ───────────────────────────────────────────
function ClassDonutChart({ data }) {
  const canvasRef = useRef();
  const chartRef  = useRef();

  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const entries = Object.entries(data).filter(([, v]) => v > 0.005);
    const labels  = entries.map(([k]) => CLASS_META[k]?.label || k);
    const values  = entries.map(([, v]) => parseFloat((v * 100).toFixed(1)));
    const colors  = entries.map(([k]) => CLASS_META[k]?.color || "#64748b");

    if (chartRef.current) chartRef.current.destroy();

    const Chart = window.Chart;
    if (!Chart) return;

    chartRef.current = new Chart(canvasRef.current, {
      type: "doughnut",
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: colors.map(c => c + "cc"),
          borderColor: colors,
          borderWidth: 1,
          hoverBorderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "70%",
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%`,
            },
            backgroundColor: "#0f172a",
            borderColor: "#1e293b",
            borderWidth: 1,
          },
        },
        animation: { animateRotate: true, duration: 1200 },
      },
    });

    return () => { if (chartRef.current) chartRef.current.destroy(); };
  }, [data]);

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
      <div style={{ position: "relative", width: 110, height: 110, flexShrink: 0 }}>
        <canvas ref={canvasRef} />
      </div>
      <div style={{ flex: 1 }}>
        {data && Object.entries(data)
          .filter(([, v]) => v > 0.005)
          .sort((a, b) => b[1] - a[1])
          .map(([name, pct]) => {
            const meta = CLASS_META[name];
            return (
              <div key={name} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: meta?.color, flexShrink: 0 }} />
                <span style={{ fontSize: 10, color: "#64748b", fontFamily: "monospace", flex: 1 }}>{meta?.label || name}</span>
                <span style={{ fontSize: 10, color: meta?.color, fontFamily: "monospace" }}>{(pct * 100).toFixed(1)}%</span>
              </div>
            );
          })}
      </div>
    </div>
  );
}

// ── Traversability bar chart ───────────────────────────────────
function TravBarChart({ data }) {
  const canvasRef = useRef();
  const chartRef  = useRef();

  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const entries = Object.entries(data).filter(([, v]) => v > 0.005);
    const labels  = entries.map(([k]) => CLASS_META[k]?.label || k);
    const values  = entries.map(([k]) => CLASS_META[k]?.trav || 0);
    const colors  = entries.map(([k]) => CLASS_META[k]?.color || "#64748b");

    if (chartRef.current) chartRef.current.destroy();
    const Chart = window.Chart;
    if (!Chart) return;

    chartRef.current = new Chart(canvasRef.current, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: colors.map(c => c + "88"),
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: "#475569", font: { family: "monospace", size: 10 } },
            grid: { color: "#1e293b" },
          },
          y: {
            min: 0, max: 1,
            ticks: {
              color: "#475569",
              font: { family: "monospace", size: 10 },
              callback: v => `${Math.round(v * 100)}%`,
            },
            grid: { color: "#1e293b" },
          },
        },
        animation: { duration: 1200 },
      },
    });

    return () => { if (chartRef.current) chartRef.current.destroy(); };
  }, [data]);

  return (
    <div style={{ position: "relative", height: 140 }}>
      <canvas ref={canvasRef} />
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────
export default function TerrainIntelligence() {
  const [status,    setStatus]    = useState("idle");
  const [result,    setResult]    = useState(null);
  const [preview,   setPreview]   = useState(null);
  const [activeTab, setActiveTab] = useState("segmented");
  const [pipeStep,  setPipeStep]  = useState(null);
  const [logs,      setLogs]      = useState([]);
  const fileRef  = useRef();
  const fileObj  = useRef(null);

  function addLog(msg, type = "info") {
    const t = new Date().toLocaleTimeString("en-US", { hour12: false });
    setLogs(l => [...l.slice(-20), { t, msg, type }]);
  }

  function handleFile(file) {
    if (!file?.type.startsWith("image/")) return;
    fileObj.current = file;
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setStatus("idle");
    setLogs([]);
    addLog(`Loaded ${file.name} · ${(file.size / 1024).toFixed(1)} KB`);
  }

  async function runAnalysis() {
    if (!fileObj.current) return;
    setStatus("loading");
    setResult(null);
    setPipeStep("seg");

    const timeline = [
      { step: "seg",   msg: "DeepLabV3+ inference...",     delay: 0 },
      { step: "graph", msg: "Building terrain graph...",   delay: 700 },
      { step: "path",  msg: "Dijkstra pathfinding...",     delay: 1400 },
      { step: "mem",   msg: "Querying TigerGraph memory...",delay: 2100 },
      { step: "llm",   msg: "Generating LLM briefing...",  delay: 2800 },
    ];
    timeline.forEach(({ step, msg, delay }) => {
      setTimeout(() => { setPipeStep(step); addLog(msg); }, delay);
    });

    try {
      const form = new FormData();
      form.append("file", fileObj.current);
      const res  = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult(data);
      setStatus("done");
      setPipeStep(null);
      addLog(`Complete · ${data.total_time}s · mIoU ${(data.model_miou * 100).toFixed(1)}%`, "success");
    } catch (err) {
      setStatus("error");
      setPipeStep(null);
      addLog(`Error: ${err.message}`, "error");
    }
  }

  const images = result ? {
    original:    `${API_BASE}/image/${result.image_id}/original`,
    segmented:   `${API_BASE}/image/${result.image_id}/segmented`,
    overlay:     `${API_BASE}/image/${result.image_id}/overlay`,
    path_visual: `${API_BASE}/image/${result.image_id}/path`,
  } : {};

  const logColors = { info: "#475569", success: "#22c55e", error: "#ef4444", warn: "#f59e0b" };

  return (
    <div style={{ background: "#040810", minHeight: "100vh", color: "#e2e8f0", fontFamily: "'JetBrains Mono', monospace", position: "relative", overflow: "hidden" }}>
      <HexGrid />

      {/* ── Header ── */}
      <motion.div
        initial={{ y: -40, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        style={{
          borderBottom: "1px solid #0f172a",
          padding: "0 24px",
          height: 52,
          display: "flex",
          alignItems: "center",
          gap: 16,
          background: "#060b14",
          position: "relative",
          zIndex: 10,
        }}
      >
        {/* Pulsing status */}
        <motion.div
          animate={{ opacity: [1, 0.4, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
          style={{ width: 8, height: 8, borderRadius: "50%", background: status === "done" ? "#22c55e" : status === "loading" ? "#3b82f6" : status === "error" ? "#ef4444" : "#1e293b", flexShrink: 0 }}
        />

        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <span style={{ fontSize: 14, fontWeight: 700, letterSpacing: "0.15em", color: "#e2e8f0" }}>
            TERRAIN INTELLIGENCE
          </span>
          <span style={{ fontSize: 10, color: "#1e293b", letterSpacing: "0.1em" }}>v1.0</span>
        </div>

        <div style={{ display: "flex", gap: 16, marginLeft: "auto", fontSize: 10, color: "#1e293b", fontFamily: "monospace" }}>
          {["DeepLabV3+", "TigerGraph", "Groq LLM"].map(s => (
            <motion.span
              key={s}
              whileHover={{ color: "#22c55e" }}
              style={{ cursor: "default" }}
            >
              {s}
            </motion.span>
          ))}
        </div>

        <div style={{
          fontSize: 11, fontFamily: "monospace",
          color: status === "done" ? "#22c55e" : status === "loading" ? "#3b82f6" : status === "error" ? "#ef4444" : "#334155",
          letterSpacing: "0.1em",
        }}>
          {status === "idle" ? "READY" : status === "loading" ? "PROCESSING" : status === "done" ? "COMPLETE" : "ERROR"}
        </div>
      </motion.div>

      {/* ── Main Grid ── */}
      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 260px", gap: 12, padding: 16, position: "relative", zIndex: 1 }}>

        {/* ── LEFT ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

          {/* Upload zone */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div style={{ fontSize: 10, color: "#1e293b", letterSpacing: "0.12em", marginBottom: 8, fontFamily: "monospace" }}>
              INPUT TERMINAL
            </div>
            <motion.div
              onClick={() => fileRef.current.click()}
              onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); }}
              onDragOver={e => e.preventDefault()}
              whileHover={{ borderColor: "#22c55e55" }}
              style={{
                border: "1px dashed #1e293b",
                borderRadius: 8,
                padding: preview ? 0 : "24px 16px",
                textAlign: "center",
                cursor: "pointer",
                overflow: "hidden",
                marginBottom: 10,
              }}
            >
              {preview ? (
                <div style={{ position: "relative" }}>
                  <img src={preview} alt="input" style={{ width: "100%", display: "block", borderRadius: 7 }} />
                  <ScanLine />
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: 28, color: "#1e293b", marginBottom: 8 }}>⊕</div>
                  <div style={{ fontSize: 11, color: "#1e293b", fontFamily: "monospace" }}>
                    DROP IMAGE<br />
                    <span style={{ fontSize: 10, color: "#0f172a" }}>or click to browse</span>
                  </div>
                </div>
              )}
            </motion.div>
            <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
              onChange={e => handleFile(e.target.files[0])} />

            <motion.button
              onClick={runAnalysis}
              disabled={!preview || status === "loading"}
              whileHover={preview && status !== "loading" ? { scale: 1.02 } : {}}
              whileTap={preview && status !== "loading" ? { scale: 0.98 } : {}}
              style={{
                width: "100%",
                padding: "11px",
                background: status === "loading" ? "#0f172a" : "transparent",
                border: `1px solid ${status === "loading" ? "#1e293b" : "#22c55e55"}`,
                borderRadius: 6,
                color: status === "loading" ? "#334155" : "#22c55e",
                fontSize: 11,
                fontWeight: 700,
                cursor: !preview || status === "loading" ? "not-allowed" : "pointer",
                letterSpacing: "0.12em",
                fontFamily: "monospace",
                position: "relative",
                overflow: "hidden",
              }}
            >
              {status === "loading" && (
                <motion.div
                  style={{ position: "absolute", left: 0, top: 0, bottom: 0, background: "#22c55e15", width: "30%" }}
                  animate={{ x: ["−30%", "130%"] }}
                  transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                />
              )}
              {status === "loading" ? "ANALYZING..." : "ANALYZE TERRAIN"}
            </motion.button>
          </motion.div>

          {/* Metrics grid */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}
              >
                <GlowCard label="mIoU" value={(result.model_miou * 100).toFixed(1)} unit="%" color="#22c55e" delay={0} />
                <GlowCard label="Time" value={result.total_time} unit="s" color="#38bdf8" delay={0.1} />
                <GlowCard label="Path hops" value={result.path?.hop_count || 0} unit="" color="#a78bfa" delay={0.2} />
                <GlowCard label="Risk zones" value={result.risk_zones?.length || 0} unit="" color={result.risk_zones?.length > 0 ? "#f87171" : "#22c55e"} delay={0.3} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Traversability donut */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ type: "spring", stiffness: 200, delay: 0.4 }}
                style={{
                  background: "#060b14",
                  border: "1px solid #1e293b",
                  borderRadius: 8,
                  padding: 16,
                  display: "flex",
                  justifyContent: "center",
                }}
              >
                <TraversabilityDonut value={result.avg_trav || 0} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Similar terrain memory */}
          <AnimatePresence>
            {result?.similar?.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                style={{ background: "#060b14", border: "1px solid #1e293b", borderRadius: 8, padding: 14 }}
              >
                <div style={{ fontSize: 9, color: "#1e293b", letterSpacing: "0.12em", marginBottom: 10, fontFamily: "monospace" }}>
                  GRAPH MEMORY · {result.similar.length} MATCHES
                </div>
                {result.similar.slice(0, 3).map((t, i) => {
                  const a = t.attributes || {};
                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.5 + i * 0.1 }}
                      style={{
                        fontSize: 10, fontFamily: "monospace",
                        padding: "6px 8px",
                        background: "#080d18",
                        border: "1px solid #0f172a",
                        borderRadius: 4,
                        marginBottom: 5,
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <span style={{ color: "#38bdf8" }}>{(a.dominant_class || "?").replace("_", " ")}</span>
                      <span style={{ color: "#334155" }}>{((a.avg_traversability || 0) * 100).toFixed(0)}% safe</span>
                    </motion.div>
                  );
                })}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ── CENTER ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

          {/* Pipeline progress */}
          <AnimatePresence>
            {status === "loading" && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                style={{ background: "#060b14", border: "1px solid #1e293b", borderRadius: 8, padding: "14px 16px", overflow: "hidden" }}
              >
                <PipelineSteps active={pipeStep} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Image tabs */}
          <div style={{ background: "#060b14", border: "1px solid #1e293b", borderRadius: 8, overflow: "hidden" }}>
            {/* Tab bar */}
            <div style={{ display: "flex", borderBottom: "1px solid #0f172a", padding: "0 12px" }}>
              {[
                { key: "segmented",   label: "Segmentation" },
                { key: "overlay",     label: "Overlay" },
                { key: "path_visual", label: "Safe Path" },
                { key: "original",    label: "Original" },
              ].map(tab => (
                <motion.button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  whileHover={{ color: "#e2e8f0" }}
                  style={{
                    padding: "10px 14px",
                    background: "transparent",
                    border: "none",
                    borderBottom: activeTab === tab.key ? "2px solid #22c55e" : "2px solid transparent",
                    color: activeTab === tab.key ? "#22c55e" : "#334155",
                    fontSize: 10,
                    cursor: "pointer",
                    fontFamily: "monospace",
                    letterSpacing: "0.08em",
                    transition: "color 0.2s",
                  }}
                >
                  {tab.label.toUpperCase()}
                </motion.button>
              ))}
            </div>

            {/* Image display */}
            <div style={{ position: "relative", aspectRatio: "4/3", background: "#040810", overflow: "hidden" }}>
              <AnimatePresence mode="wait">
                {status === "loading" ? (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16 }}
                  >
                    {/* Animated radar */}
                    <div style={{ position: "relative", width: 80, height: 80 }}>
                      <svg viewBox="0 0 80 80" style={{ width: "100%", height: "100%" }}>
                        <circle cx="40" cy="40" r="35" fill="none" stroke="#1e293b" strokeWidth="1" />
                        <circle cx="40" cy="40" r="25" fill="none" stroke="#1e293b" strokeWidth="0.5" />
                        <circle cx="40" cy="40" r="15" fill="none" stroke="#1e293b" strokeWidth="0.5" />
                        <motion.path
                          d="M 40 40 L 40 5"
                          stroke="#22c55e"
                          strokeWidth="2"
                          fill="none"
                          style={{ transformOrigin: "40px 40px" }}
                          animate={{ rotate: 360 }}
                          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        />
                        <motion.path
                          d="M 40 40 L 75 40"
                          stroke="#22c55e22"
                          strokeWidth="1"
                          fill="none"
                          style={{ transformOrigin: "40px 40px" }}
                          animate={{ rotate: 360 }}
                          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        />
                      </svg>
                    </div>
                    <div style={{ fontSize: 10, color: "#334155", fontFamily: "monospace", letterSpacing: "0.1em" }}>
                      SCANNING TERRAIN
                    </div>
                  </motion.div>
                ) : images[activeTab] ? (
                  <motion.img
                    key={activeTab}
                    src={images[activeTab]}
                    alt={activeTab}
                    initial={{ opacity: 0, scale: 1.05 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.4 }}
                    style={{ width: "100%", height: "100%", objectFit: "cover", position: "absolute", inset: 0 }}
                  />
                ) : preview ? (
                  <motion.img
                    key="preview"
                    src={preview}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 0.4 }}
                    style={{ width: "100%", height: "100%", objectFit: "cover", position: "absolute", inset: 0 }}
                  />
                ) : (
                  <motion.div
                    key="empty"
                    style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "#0f172a", fontSize: 12, fontFamily: "monospace" }}
                  >
                    NO IMAGE
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Charts row */}
          <AnimatePresence>
            {result?.class_dist && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}
              >
                <div style={{ background: "#060b14", border: "1px solid #1e293b", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 9, color: "#1e293b", letterSpacing: "0.12em", marginBottom: 10, fontFamily: "monospace" }}>CLASS DISTRIBUTION</div>
                  <ClassDonutChart data={result.class_dist} />
                </div>
                <div style={{ background: "#060b14", border: "1px solid #1e293b", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 9, color: "#1e293b", letterSpacing: "0.12em", marginBottom: 10, fontFamily: "monospace" }}>TRAVERSABILITY BY CLASS</div>
                  <TravBarChart data={result.class_dist} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* System log */}
          <div style={{ background: "#040810", border: "1px solid #0f172a", borderRadius: 8, padding: "10px 14px", fontFamily: "monospace", minHeight: 80 }}>
            <div style={{ fontSize: 9, color: "#1e293b", letterSpacing: "0.12em", marginBottom: 8 }}>SYSTEM LOG</div>
            <AnimatePresence>
              {logs.map((l, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  style={{ display: "flex", gap: 10, marginBottom: 3, fontSize: 10 }}
                >
                  <span style={{ color: "#1e293b", flexShrink: 0 }}>{l.t}</span>
                  <span style={{ color: logColors[l.type] || "#475569" }}>{l.msg}</span>
                </motion.div>
              ))}
            </AnimatePresence>
            {logs.length === 0 && (
              <div style={{ fontSize: 10, color: "#0f172a" }}>awaiting input...</div>
            )}
          </div>
        </div>

        {/* ── RIGHT ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

          {/* AI Briefing */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            style={{ background: "#060b14", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}
          >
            <div style={{ fontSize: 9, color: "#1e293b", letterSpacing: "0.12em", marginBottom: 12, fontFamily: "monospace" }}>
              AI NAVIGATION BRIEFING
            </div>
            <AnimatePresence mode="wait">
              {result?.briefing ? (
                <motion.div
                  key="briefing"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  style={{ fontSize: 11, lineHeight: 1.9, color: "#94a3b8", fontFamily: "monospace" }}
                >
                  {result.briefing}
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  style={{ fontSize: 10, color: "#0f172a", fontFamily: "monospace" }}
                >
                  awaiting analysis...
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Risk zones */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            style={{ background: "#060b14", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}
          >
            <div style={{ fontSize: 9, color: "#1e293b", letterSpacing: "0.12em", marginBottom: 12, fontFamily: "monospace" }}>
              RISK ZONES {result?.risk_zones?.length > 0 ? `· ${result.risk_zones.length} DETECTED` : ""}
            </div>
            <AnimatePresence>
              {result?.risk_zones?.length > 0 ? (
                result.risk_zones.map((z, i) => {
                  const a = z.attributes || {};
                  return (
                    <RiskBadge
                      key={i}
                      severity={a.severity}
                      cls={a.class_name}
                      patches={a.patch_count}
                      row={a.center_row || 0}
                      col={a.center_col || 0}
                      delay={i * 0.1}
                    />
                  );
                })
              ) : result ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  style={{ fontSize: 11, color: "#22c55e", fontFamily: "monospace" }}
                >
                  ✓ No risk zones detected
                </motion.div>
              ) : (
                <div style={{ fontSize: 10, color: "#0f172a", fontFamily: "monospace" }}>awaiting analysis...</div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Path info */}
          <AnimatePresence>
            {result?.path && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ type: "spring", stiffness: 200, delay: 0.3 }}
                style={{ background: "#060b14", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}
              >
                <div style={{ fontSize: 9, color: "#1e293b", letterSpacing: "0.12em", marginBottom: 12, fontFamily: "monospace" }}>
                  DIJKSTRA PATH
                </div>
                <div style={{ display: "flex", gap: 16, marginBottom: 12 }}>
                  <div>
                    <div style={{ fontSize: 10, color: "#334155", marginBottom: 3, fontFamily: "monospace" }}>HOPS</div>
                    <div style={{ fontSize: 28, fontWeight: 700, color: "#38bdf8", fontFamily: "monospace", lineHeight: 1 }}>
                      <AnimatedNumber value={result.path.hop_count} />
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: "#334155", marginBottom: 3, fontFamily: "monospace" }}>COST</div>
                    <div style={{ fontSize: 28, fontWeight: 700, color: "#a78bfa", fontFamily: "monospace", lineHeight: 1 }}>
                      <AnimatedNumber value={result.path.total_cost || 0} decimals={1} />
                    </div>
                  </div>
                </div>
                {/* Path cost bar */}
                <div style={{ height: 4, background: "#0f172a", borderRadius: 2, overflow: "hidden", marginBottom: 8 }}>
                  <motion.div
                    style={{ height: "100%", background: "#a78bfa", borderRadius: 2 }}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, (result.path.hop_count / 32) * 100)}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                  />
                </div>
                <div style={{ fontSize: 10, color: "#334155", fontFamily: "monospace" }}>
                  bottom-center → top-center
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Class bars */}
          <AnimatePresence>
            {result?.class_dist && (
              <motion.div
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
                style={{ background: "#060b14", border: "1px solid #1e293b", borderRadius: 8, padding: 16 }}
              >
                <div style={{ fontSize: 9, color: "#1e293b", letterSpacing: "0.12em", marginBottom: 12, fontFamily: "monospace" }}>
                  SCENE BREAKDOWN
                </div>
                {Object.entries(result.class_dist)
                  .sort((a, b) => b[1] - a[1])
                  .filter(([, v]) => v > 0.005)
                  .map(([name, pct], i) => (
                    <ClassBar key={name} name={name} pct={pct} delay={0.4 + i * 0.07} />
                  ))}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Stack */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            style={{ background: "#040810", border: "1px solid #0f172a", borderRadius: 8, padding: 14 }}
          >
            {[
              ["Vision", "DeepLabV3+ ResNet101", "#22c55e"],
              ["Graph", "TigerGraph Savanna",   "#38bdf8"],
              ["LLM",   "Groq / Gemini / Claude","#a78bfa"],
              ["Path",  "Dijkstra 256 nodes",   "#fb923c"],
            ].map(([k, v, c]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 7, fontSize: 10, fontFamily: "monospace" }}>
                <span style={{ color: "#1e293b" }}>{k}</span>
                <span style={{ color: c }}>{v}</span>
              </div>
            ))}
          </motion.div>
        </div>
      </div>

      {/* Chart.js CDN */}
      <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js" />

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: #040810; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
        button:focus { outline: none; }
      `}</style>
    </div>
  );
}