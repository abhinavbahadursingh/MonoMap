"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  PHASE_LABELS,
  PHASE_ORDER,
  fetchPhases,
  type Phase,
  type PhaseInfo,
  type SlamResult,
} from "../lib/api";

interface Props {
  result: SlamResult | null;
  phase: Phase;
}

function fmtVal(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    return Number.isInteger(v) ? v.toLocaleString() : String(Math.round(v * 1000) / 1000);
  }
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "string") return v === "" ? "—" : v;
  if (Array.isArray(v)) {
    if (v.length === 0) return "[]";
    if (v.length <= 12) return v.map((x) => fmtVal(x)).join(", ");
    return `${v.slice(0, 12).map((x) => fmtVal(x)).join(", ")} … (${v.length})`;
  }
  if (typeof v === "object") {
    const entries = Object.entries(v as Record<string, unknown>);
    if (entries.length === 0) return "{}";
    return entries.map(([k, val]) => `${k}: ${fmtVal(val)}`).join(" · ");
  }
  return String(v);
}

const STATUS_PILL: Record<PhaseInfo["status"], string> = {
  pending: "pill-idle",
  running: "pill-busy",
  done: "pill-done",
  failed: "pill-error",
};

/**
 * Per-phase parameters panel.
 * While SLAM runs it polls GET /api/slam/phases and streams each phase's
 * REAL terminal output + status live. After completion the structured
 * backend numbers (result.phase_params) render as parameter chips.
 */
export default function PhaseParameters({ result, phase }: Props) {
  const busy = phase === "uploading" || phase === "processing";
  const [live, setLive] = useState<PhaseInfo[] | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const runningRef = useRef<string | null>(null);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      const p = await fetchPhases();
      if (!stop && p) setLive(p);
    };
    // Poll live while busy; fetch once more right after completion.
    if (busy) {
      tick();
      const t = setInterval(tick, 700);
      return () => {
        stop = true;
        clearInterval(t);
      };
    }
    if (phase === "done" || phase === "error") tick();
    return () => {
      stop = true;
    };
  }, [busy, phase]);

  const phases: PhaseInfo[] = useMemo(() => {
    if (live && live.length) return live;
    return PHASE_ORDER.map((name, i) => ({
      name,
      label: PHASE_LABELS[name] ?? name,
      index: i + 1,
      total: PHASE_ORDER.length,
      status: result ? "done" : ("pending" as const),
      log: "",
      elapsed_sec: result?.phase_times?.[name] ?? 0,
    }));
  }, [live, result]);

  // Auto-scroll the list to the running phase.
  useEffect(() => {
    const running = phases.find((p) => p.status === "running")?.name ?? null;
    runningRef.current = running;
    if (running && listRef.current) {
      const el = listRef.current.querySelector(`[data-phase="${running}"]`);
      el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [phases]);

  const doneCount = phases.filter((p) => p.status === "done").length;

  return (
    <section className="card" aria-label="Phase parameters">
      <div className="metrics-head">
        <h2>Phase Parameters</h2>
        <span className={"pill " + (busy ? "pill-busy" : result ? "pill-done" : "pill-idle")}>
          <span className="pill-dot" />
          {busy
            ? `RUNNING ${doneCount}/${phases.length}`
            : result
              ? `COMPLETE ${doneCount}/${phases.length}`
              : `11 PHASES`}
        </span>
      </div>

      {!result && !busy && (
        <p className="muted">
          Run SLAM — every phase&apos;s reported parameters stream here live, straight from the
          pipeline&apos;s own output.
        </p>
      )}

      <div className="phase-list" ref={listRef}>
        {phases.map((p) => {
          const params = result?.phase_params?.[p.name] as Record<string, unknown> | undefined;
          const entries = params ? Object.entries(params).filter(([, v]) => v !== undefined) : [];
          const elapsed = p.elapsed_sec || result?.phase_times?.[p.name] || 0;
          return (
            <article key={p.name} data-phase={p.name} className={`phase-item phase-${p.status}`}>
              <header className="phase-head">
                <span className="phase-idx">{p.index}</span>
                <div className="phase-title">
                  <strong>{p.label}</strong>
                  <span className="muted phase-name">{p.name}</span>
                </div>
                <span className={"pill pill-sm " + STATUS_PILL[p.status]}>{p.status}</span>
                <span className="muted phase-time">
                  {elapsed ? `${Number(elapsed).toFixed(1)}s` : "—"}
                </span>
              </header>

              {entries.length > 0 && (
                <dl className="param-grid">
                  {entries.map(([k, v]) => (
                    <div key={k} className="param-cell">
                      <dt>{k.replace(/_/g, " ")}</dt>
                      <dd title={typeof v === "object" ? JSON.stringify(v) : String(v ?? "")}>
                        {fmtVal(v)}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}

              {p.log ? (
                <details className="phase-log">
                  <summary>Terminal output</summary>
                  <pre>{p.log}</pre>
                </details>
              ) : (
                busy &&
                p.status === "pending" && <p className="muted phase-wait">Waiting for this phase…</p>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
