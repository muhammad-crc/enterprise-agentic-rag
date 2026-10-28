"use client";

import React, { useState } from "react";
import {
  FileText,
  Upload,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Trash2,
  Database,
  FolderOpen,
} from "lucide-react";
import { IngestResponse } from "@/types/rag";

interface IngestPanelProps {
  backendUrl: string;
  onIngestSuccess?: (msg: string) => void;
  width?: number;
}

export function IngestPanel({ backendUrl, onIngestSuccess, width }: IngestPanelProps) {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{
    type: "success" | "error" | "info";
    text: string;
  } | null>(null);

  const addFiles = (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const newFiles = fileArray.filter((file) =>
      file.name.toLowerCase().endsWith(".pdf")
    );
    if (newFiles.length === 0) {
      setStatusMessage({ type: "error", text: "Please select valid PDF documents (.pdf)." });
      return;
    }
    // Avoid duplicates by name
    setSelectedFiles((prev) => {
      const existingNames = new Set(prev.map((f) => f.name));
      const filtered = newFiles.filter((f) => !existingNames.has(f.name));
      return [...prev, ...filtered];
    });
    setStatusMessage(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(e.target.files);
    }
    e.target.value = "";
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleIngest = async () => {
    if (selectedFiles.length === 0) return;

    setUploading(true);
    setStatusMessage({ type: "info", text: "Uploading PDFs and building vector index..." });

    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      const res = await fetch(`${backendUrl}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Upload failed (${res.status}): ${errorText}`);
      }

      const data: IngestResponse = await res.json();
      setStatusMessage({
        type: "success",
        text: data.message || `Indexed ${data.files_processed.length} document(s) successfully!`,
      });
      setSelectedFiles([]);
      if (onIngestSuccess) {
        onIngestSuccess(data.message);
      }
    } catch (err: unknown) {
      const isFetchFail = err instanceof Error && err.message.toLowerCase().includes("fetch");
      const msg = isFetchFail
        ? `Could not reach backend at ${backendUrl}. If hosted on Render free tier, it may be waking up from sleep (~50s). Please wait a moment and try again.`
        : (err instanceof Error ? err.message : "Document upload failed.");
      setStatusMessage({ type: "error", text: msg });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div
      style={width ? { width: `${width}px` } : undefined}
      className={`flex h-full ${width ? "" : "w-[420px]"} shrink-0 flex-col border-r border-zinc-200 bg-white p-5 lg:p-7 overflow-y-auto`}
    >
      <div className="flex items-center gap-2.5 mb-6">
        <div className="rounded-xl border border-emerald-100 bg-emerald-50 p-2 text-emerald-700">
          <Database className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-sm font-semibold tracking-[-0.02em] text-slate-800">Document library</h2>
          <p className="text-xs text-slate-500">Add sources for this workspace</p>
        </div>
      </div>

      {/* Robust Upload Container with Direct Full-Area Invisible Input */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`border-2 border-dashed rounded-2xl p-5 text-center transition-all select-none flex flex-col items-center justify-center relative overflow-hidden group ${
          isDragOver
            ? "border-emerald-400 bg-emerald-50 scale-[1.02]"
            : "border-[#dce6df] hover:border-emerald-400 bg-[#fbfdfc] hover:bg-emerald-50/30"
        }`}
      >
        {/* Full coverage native file input spanning entire box with z-index 20 */}
        <input
          type="file"
          accept=".pdf,application/pdf"
          multiple
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
          title="Click or drag PDF files here"
        />

        {/* Visual UI inside box */}
        <div className="pointer-events-none flex flex-col items-center justify-center">
          <div className="mb-3 rounded-full bg-emerald-50 p-3 text-emerald-700 transition-colors">
            <Upload className="h-5 w-5" />
          </div>
          <p className="text-xs font-semibold text-slate-700">
            Click to Browse or Drag PDFs
          </p>
          <p className="mt-1 text-[11px] text-slate-500">PDF files, up to multiple at once</p>

          <div className="mt-3 flex items-center gap-1.5 rounded-lg border border-[#dce6df] bg-white px-3 py-1.5 text-[11px] text-slate-600 transition-colors">
            <FolderOpen className="h-3.5 w-3.5 text-emerald-700" />
            <span>Choose PDF Files</span>
          </div>
        </div>
      </div>

      {/* Staged Files List */}
      <div className="mt-4 flex-1 overflow-y-auto space-y-2 pr-1">
        <div className="flex items-center justify-between px-1 text-xs text-slate-500">
          <span>Ready to index ({selectedFiles.length})</span>
          {selectedFiles.length > 0 && (
            <button
              onClick={() => setSelectedFiles([])}
              className="text-[11px] text-rose-400 hover:text-rose-300 hover:underline cursor-pointer"
            >
              Clear all
            </button>
          )}
        </div>

        {selectedFiles.length === 0 ? (
          <div className="py-8 text-center text-xs leading-5 text-slate-400">
            Your uploaded documents will appear here.
          </div>
        ) : (
          selectedFiles.map((file, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between gap-2 rounded-xl border border-[#e5ece7] bg-[#fbfdfc] p-2.5 text-xs"
            >
              <div className="flex items-center gap-2 min-w-0">
                <FileText className="h-4 w-4 shrink-0 text-emerald-700" />
                <span className="truncate font-medium text-slate-700">{file.name}</span>
              </div>
              <button
                onClick={() => removeFile(idx)}
                className="text-slate-400 hover:text-rose-400 transition-colors shrink-0 p-1 cursor-pointer"
                title="Remove file"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))
        )}
      </div>

      {/* Status Feedback Notification */}
      {statusMessage && (
        <div
          className={`mt-3 p-3 rounded-xl text-xs flex items-start gap-2 border ${
            statusMessage.type === "success"
              ? "bg-emerald-950/40 border-emerald-800 text-emerald-300"
              : statusMessage.type === "error"
              ? "bg-rose-950/40 border-rose-800 text-rose-300"
              : "bg-blue-950/40 border-blue-800 text-blue-300"
          }`}
        >
          {statusMessage.type === "success" && <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />}
          {statusMessage.type === "error" && <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />}
          {statusMessage.type === "info" && <Loader2 className="w-4 h-4 shrink-0 mt-0.5 animate-spin" />}
          <span className="break-words leading-relaxed">{statusMessage.text}</span>
        </div>
      )}

      {/* Submit / Ingest Button */}
      <button
        onClick={handleIngest}
        disabled={uploading || selectedFiles.length === 0}
        className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-[#173c2d] px-4 py-2.5 text-xs font-medium text-white shadow-sm transition-all hover:bg-[#0d2d20] disabled:cursor-not-allowed disabled:opacity-40"
      >
        {uploading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Chunking & Indexing...</span>
          </>
        ) : (
          <>
            <Upload className="w-4 h-4" />
            <span>Ingest {selectedFiles.length > 0 ? `(${selectedFiles.length})` : ""} to Knowledge Base</span>
          </>
        )}
      </button>
    </div>
  );
}
