"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, User, Copy, Check } from "lucide-react";
import { ChatMessage } from "@/types/rag";
import { SourceList } from "./SourceList";

interface ChatItemProps {
  message: ChatMessage;
}

export function ChatItem({ message }: ChatItemProps) {
  const isUser = message.role === "user";
  const [copied, setCopied] = React.useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={`flex gap-3.5 my-4 px-2 ${
        isUser ? "justify-end" : "justify-start"
      }`}
    >
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-[#173c2d] text-white shadow-sm">
          <Bot className="w-4 h-4" />
        </div>
      )}

      <div
        className={`max-w-[92%] sm:max-w-[86%] md:max-w-[82%] min-w-0 rounded-2xl p-4 text-sm leading-relaxed transition-all relative group overflow-hidden break-words [overflow-wrap:anywhere] ${
          isUser
            ? "bg-[#173c2d] text-white rounded-tr-none shadow-sm"
            : message.isError
            ? "bg-rose-950/40 border border-rose-800 text-rose-200 rounded-tl-none"
            : "border border-[#e1e9e3] bg-white text-slate-700 rounded-tl-none shadow-sm"
        }`}
      >
        <div className="flex items-center justify-between gap-4 mb-1">
          <span className={`text-[11px] font-semibold tracking-wider uppercase ${isUser ? "text-emerald-100" : "text-emerald-700"}`}>
            {isUser ? "You" : "Atlas"}
          </span>
          <div className="flex items-center gap-2 shrink-0">
            <span className={`text-[10px] ${isUser ? "text-emerald-100/70" : "text-slate-400"}`}>{message.timestamp}</span>
            <button
              onClick={copyToClipboard}
              className="p-1 opacity-0 transition-opacity group-hover:opacity-100 text-slate-400 hover:text-emerald-700"
              title="Copy message"
            >
              {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
            </button>
          </div>
        </div>

        <div className={`prose prose-sm max-w-none break-words [overflow-wrap:anywhere] ${isUser ? "prose-invert" : "prose-slate"}`}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed break-words [overflow-wrap:anywhere]">{children}</p>,
              pre: ({ children }) => <pre className="p-3 my-2 bg-slate-900 text-slate-100 rounded-lg overflow-x-auto text-xs break-words [overflow-wrap:anywhere] max-w-full">{children}</pre>,
              code: ({ children, className }) => <code className={`${className || ""} bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200 px-1 py-0.5 rounded text-[12px] break-words [overflow-wrap:anywhere]`}>{children}</code>,
              blockquote: ({ children }) => <blockquote className="border-l-4 border-emerald-600 pl-3 my-2 text-slate-600 italic break-words [overflow-wrap:anywhere]">{children}</blockquote>,
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {message.sources && message.sources.length > 0 && (
          <SourceList sources={message.sources} />
        )}
      </div>

      {isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-[#dce6df] bg-white text-slate-500">
          <User className="w-4 h-4" />
        </div>
      )}
    </div>
  );
}
