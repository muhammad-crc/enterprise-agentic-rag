export interface DocumentSource {
  source: string;
  page?: number | null;
  preview: string;
  chunk_id?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sources?: DocumentSource[];
  timestamp: string;
  isError?: boolean;
}

export interface IngestResponse {
  status: string;
  files_processed: string[];
  message: string;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  llm_provider: string;
  vector_store: string;
}
