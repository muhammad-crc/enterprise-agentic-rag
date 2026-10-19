import { RagChatApp } from "@/components/RagChatApp";

export default function Home() {
  return (
    <main suppressHydrationWarning className="flex-1 flex flex-col h-full">
      <RagChatApp />
    </main>
  );
}
