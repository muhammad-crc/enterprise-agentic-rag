"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, FileCode, BookOpen } from "lucide-react";
import { DocumentSource } from "@/types/rag";

interface SourceListProps {
  sources: DocumentSource[];
}

export function SourceList({ sources }: SourceListProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 border border-slate-800 bg-slate-900/60 rounded-xl overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3.5 py-2 text-xs font-medium text-slate-300 hover:text-white bg-slate-800/40 hover:bg-slate-800/80 transition-colors"
      >
        <div className="flex items-center gap-2">
          <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
          <span>Retrieved Citations & Sources ({sources.length})</span>
        </div>
        {isOpen ? (
          <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
        ) : (
          <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
        )}
      </button>

      {isOpen && (
        <div className="p-3 space-y-2 text-xs border-t border-slate-800 divide-y divide-slate-800/60">
          {sources.map((src, idx) => (
            <div key={idx} className="pt-2 first:pt-0">
              <div className="flex items-center justify-between gap-2 text-slate-400 mb-1">
                <span className="font-mono text-[11px] text-cyan-400 truncate">
                  📄 {src.source}
                </span>
                <div className="flex items-center gap-1.5 shrink-0">
                  {src.chunk_id && (
                    <span className="px-1.5 py-0.5 rounded bg-indigo-900/70 border border-indigo-700/50 text-[10px] text-indigo-300 shrink-0 font-mono">
                      {src.chunk_id}
                    </span>
                  )}
                  {src.page !== null && src.page !== undefined && (
                    <span className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] text-slate-300 shrink-0 font-medium">
                      Page {src.page}
                    </span>
                  )}
                </div>
              </div>
              <p className="text-slate-300 font-mono text-[11px] bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/70 whitespace-pre-wrap leading-relaxed break-words [overflow-wrap:anywhere] overflow-hidden">
                {src.preview}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
