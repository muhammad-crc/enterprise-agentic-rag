"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowUp,
  BarChart3,
  Database,
  FileSearch,
  Loader2,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  RefreshCw,
  Settings2,
  Sparkles,
  Trash2,
  Zap,
} from "lucide-react";
import { ChatMessage, HealthResponse } from "@/types/rag";
import { ChatItem } from "./ChatItem";
import { IngestPanel } from "./IngestPanel";

const DEFAULT_PORT = "8000";
const suggestions = [
  "Summarize the most important findings",
  "What risks or open questions should I know about?",
  "Find the implementation details and cite the pages",
];

export function RagChatApp() {
  const [backendUrl, setBackendUrl] = useState(() => process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000");
  const [showConfig, setShowConfig] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [inputQuery, setInputQuery] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(() => `session-${Date.now()}`);

  // Resizable split-pane state for Document library and Chat
  const [ingestWidth, setIngestWidth] = useState(380);
  const [isResizing, setIsResizing] = useState(false);
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const checkHealth = async (targetUrl?: string) => {
    const url = targetUrl || backendUrl;
    try {
      const response = await fetch(`${url}/health`);
      setHealth(response.ok ? await response.json() : null);
    } catch {
      setHealth(null);
    }
  };

  useEffect(() => {
    const url = process.env.NEXT_PUBLIC_API_BASE_URL || `http://${window.location.hostname || "localhost"}:${DEFAULT_PORT}`;
    const ping = () => {
      fetch(`${url}/health`).then((response) => (response.ok ? response.json() : null)).then(setHealth).catch(() => setHealth(null));
    };
    ping();
    const timer = window.setInterval(ping, 5000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Handle Dragging Splitter to resize IngestPanel and Chat
  const startResizing = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing || !splitContainerRef.current) return;
      const containerRect = splitContainerRef.current.getBoundingClientRect();
      const asideEl = splitContainerRef.current.querySelector("aside");
      const asideWidth = asideEl ? asideEl.getBoundingClientRect().width : 0;

      const rawWidth = e.clientX - containerRect.left - asideWidth;
      // Allow dragging between 240px and leaving at least 380px for chat
      const maxAllowed = containerRect.width - asideWidth - 380;
      const clamped = Math.max(240, Math.min(rawWidth, Math.max(260, maxAllowed)));
      setIngestWidth(clamped);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    }

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isResizing]);

  const clearConversation = () => {
    setMessages([]);
    setSessionId(`session-${Date.now()}`);
  };

  const handleSend = async (event?: React.FormEvent) => {
    event?.preventDefault();
    const query = inputQuery.trim();
    if (!query || loading) return;
    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    setMessages((current) => [...current, { id: Date.now().toString(), role: "user", content: query, timestamp }]);
    setInputQuery("");
    setLoading(true);
    try {
      const response = await fetch(`${backendUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          session_id: sessionId,
          conversation_history: messages.map(({ role, content }) => ({ role, content })),
        }),
      });
      if (!response.ok) throw new Error(`API Error (${response.status}): ${await response.text()}`);
      const data = await response.json();
      setMessages((current) => [
        ...current,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: data.answer || "No response received.",
          sources: data.sources || [],
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Error contacting backend.";
      setMessages((current) => [
        ...current,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: `⚠️ **Agent Error**: ${detail}`,
          isError: true,
          timestamp,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#fcfcfc] text-[#151515]">
      <header className="flex h-[72px] shrink-0 items-center justify-between border-b border-[#e5e5e5] bg-white px-5">
        <div className="flex items-center gap-5">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-[#151515] text-sm font-bold text-white">
            A
          </div>
          <span className="text-xl text-zinc-300">/</span>
          <span className="text-sm font-semibold">Atlas workspace</span>
          <span className="rounded-full border border-zinc-200 px-2 py-0.5 text-xs text-zinc-500">RAG</span>
          <span className="hidden text-xl text-zinc-300 sm:inline">/</span>
          <span className="hidden text-sm text-zinc-700 sm:inline">Knowledge agent</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="hidden items-center gap-2 text-xs text-zinc-500 sm:flex">
            <span className={`h-2 w-2 rounded-full ${health ? "bg-emerald-500" : "bg-amber-400"}`} />
            {health ? "Trained" : "Connecting"}
          </div>
          <button
            onClick={() => setShowConfig((value) => !value)}
            className="grid h-9 w-9 place-items-center rounded-lg text-zinc-500 hover:bg-zinc-100"
          >
            <Settings2 className="h-4 w-4" />
          </button>
        </div>
      </header>

      <div ref={splitContainerRef} className="flex min-h-0 flex-1 relative overflow-hidden">
        <aside className="flex w-[260px] lg:w-[280px] shrink-0 flex-col border-r border-zinc-200 bg-[#fbfbfb] p-3">
          <nav className="space-y-1">
            {[
              [Play, "Playground"],
              [Activity, "Activity"],
              [BarChart3, "Analytics"],
              [Database, "Data sources"],
              [Zap, "Actions"],
              [Settings2, "Settings"],
            ].map(([Icon, label]) => (
              <button
                key={String(label)}
                className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm ${
                  label === "Playground"
                    ? "border border-zinc-200 bg-white font-medium shadow-sm"
                    : "text-zinc-500 hover:bg-white"
                }`}
              >
                <Icon className="h-4 w-4" />
                {String(label)}
              </button>
            ))}
          </nav>
          <div className="mt-auto rounded-xl bg-[#191919] p-4 text-white">
            <p className="text-sm font-medium">Getting started ›</p>
            <p className="mt-1 text-xs text-zinc-400">Your knowledge agent is ready</p>
            <div className="mt-4 flex gap-1">
              {[1, 2, 3, 4, 5].map((item) => (
                <span
                  key={item}
                  className={`h-1 flex-1 rounded-full ${item < 5 ? "bg-white" : "bg-zinc-700"}`}
                />
              ))}
            </div>
          </div>
        </aside>

        {showSidebar && (
          <>
            <IngestPanel backendUrl={backendUrl} width={ingestWidth} />
            {/* Draggable Divider to resize Document Library & Chat */}
            <div
              onMouseDown={startResizing}
              className={`group relative flex w-2 cursor-col-resize select-none items-center justify-center transition-colors shrink-0 z-20 ${
                isResizing ? "bg-emerald-500" : "bg-zinc-200 hover:bg-emerald-400"
              }`}
              title="Drag left/right to resize Document Library & Chat panel"
            >
              <div className="h-9 w-1 rounded-full bg-zinc-400/80 group-hover:bg-white group-active:bg-white transition-colors" />
            </div>
          </>
        )}

        <main className="relative flex min-w-0 flex-1 flex-col dot-canvas overflow-hidden">
          <header className="flex h-[72px] shrink-0 items-center justify-between border-b border-zinc-200 bg-white/80 px-6 backdrop-blur-xl">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowSidebar((value) => !value)}
                className="grid h-9 w-9 place-items-center rounded-lg text-zinc-500 hover:bg-zinc-100"
                title="Toggle setup panel"
              >
                {showSidebar ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeftOpen className="h-4 w-4" />}
              </button>
              <div>
                <h1 className="text-sm font-semibold">Knowledge agent</h1>
                <p className="mt-0.5 text-[11px] text-zinc-500">Ask questions grounded in your documents</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {messages.length > 0 && (
                <button
                  onClick={clearConversation}
                  className="grid h-9 w-9 place-items-center rounded-lg text-zinc-500 hover:bg-rose-50 hover:text-rose-600"
                  title="New conversation"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              )}
            </div>
          </header>

          {showConfig && (
            <div className="flex items-center gap-3 border-b border-[#e5ece7] bg-white px-5 py-3 text-xs md:px-8">
              <label className="font-medium text-slate-600">Server</label>
              <input
                value={backendUrl}
                onChange={(event) => setBackendUrl(event.target.value)}
                className="min-w-0 flex-1 rounded-lg border border-[#dce6df] bg-[#fafcfb] px-3 py-2 text-slate-700 outline-none focus:border-emerald-500"
              />
              <button
                onClick={() => checkHealth()}
                className="flex items-center gap-1.5 rounded-lg border border-[#dce6df] px-3 py-2 font-medium text-slate-600 hover:bg-slate-50"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Check
              </button>
            </div>
          )}

          <section className="flex-1 overflow-y-auto px-4 py-7 md:px-8">
            {messages.length === 0 ? (
              <div className="mx-auto flex min-h-full max-w-2xl flex-col justify-center pb-24">
                <div className="rounded-3xl border border-zinc-200 bg-white p-7 shadow-[0_8px_30px_rgba(0,0,0,0.04)]">
                  <div className="mb-6 inline-grid h-11 w-11 place-items-center rounded-full bg-black text-white">
                    <Sparkles className="h-5 w-5" />
                  </div>
                  <h2 className="text-2xl font-semibold tracking-[-0.04em]">Hi! What can I help you with?</h2>
                  <p className="mt-3 text-sm leading-6 text-zinc-500">
                    Upload knowledge sources in the setup panel, then ask your agent anything about them.
                  </p>
                  <div className="mt-7 grid gap-2">
                    {suggestions.map((suggestion, index) => (
                      <button
                        key={suggestion}
                        onClick={() => setInputQuery(suggestion)}
                        className="group flex items-center gap-3 rounded-xl border border-zinc-200 p-3 text-left text-sm text-zinc-600 transition hover:border-zinc-400 hover:bg-zinc-50"
                      >
                        <span className="grid h-7 w-7 place-items-center rounded-lg bg-zinc-100 text-zinc-700">
                          {index === 0 ? (
                            <FileSearch className="h-3.5 w-3.5" />
                          ) : (
                            <MessageSquarePlus className="h-3.5 w-3.5" />
                          )}
                        </span>
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="mx-auto max-w-4xl w-full rounded-3xl border border-zinc-200 bg-white p-4 sm:p-6 md:p-8 shadow-[0_8px_30px_rgba(0,0,0,0.04)] overflow-hidden">
                {messages.map((message) => (
                  <ChatItem key={message.id} message={message} />
                ))}
                {loading && (
                  <div className="my-5 flex items-center gap-3 rounded-xl bg-zinc-50 p-4 text-sm text-zinc-500">
                    <Loader2 className="h-4 w-4 animate-spin text-emerald-600" />
                    Searching your knowledge base…
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </section>

          <div className="shrink-0 px-4 pb-5 md:px-8">
            <form
              onSubmit={handleSend}
              className="mx-auto max-w-4xl w-full rounded-2xl border border-[#dce6df] bg-white p-2 shadow-[0_12px_32px_rgba(21,36,30,0.08)] focus-within:border-emerald-500"
            >
              <div className="flex items-end gap-2">
                <textarea
                  rows={1}
                  value={inputQuery}
                  onChange={(event) => setInputQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Ask anything about your documents…"
                  className="max-h-32 min-h-[44px] flex-1 resize-none bg-transparent px-3 py-3 text-sm text-slate-800 outline-none placeholder:text-slate-400"
                  disabled={loading}
                />
                <button
                  type="submit"
                  disabled={!inputQuery.trim() || loading}
                  className="grid h-10 w-10 place-items-center rounded-xl bg-[#173c2d] text-white transition hover:bg-[#0d2d20] disabled:cursor-not-allowed disabled:opacity-35"
                >
                  <ArrowUp className="h-4 w-4" />
                </button>
              </div>
              <div className="flex justify-between px-3 pb-1 text-[10px] text-slate-400">
                <span>Answers include supporting citations</span>
                <span>Enter to send · Shift + Enter for a new line</span>
              </div>
            </form>
          </div>
        </main>
      </div>
    </div>
  );
}
