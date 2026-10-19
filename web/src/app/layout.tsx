import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Enterprise Agentic RAG",
  description: "Self-correcting Agentic RAG Platform with LangGraph & Next.js",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  // Filter out console hydration errors caused by third-party browser extensions (Bitdefender, Floto, Grammarly, etc.)
                  var _origConsoleError = console.error;
                  console.error = function() {
                    var fullMsg = '';
                    for (var i = 0; i < arguments.length; i++) {
                      fullMsg += ' ' + String(arguments[i] || '');
                    }
                    if (
                      fullMsg.indexOf('bis_skin_checked') !== -1 ||
                      fullMsg.indexOf('__next_metadata_boundary__') !== -1 ||
                      (fullMsg.indexOf('hydrated') !== -1 && fullMsg.indexOf('bis_skin_checked') !== -1)
                    ) {
                      return;
                    }
                    return _origConsoleError.apply(console, arguments);
                  };

                  // Prevent extensions from adding bis_skin_checked attribute to any DOM elements
                  if (typeof Element !== 'undefined' && Element.prototype) {
                    var _origSetAttr = Element.prototype.setAttribute;
                    Element.prototype.setAttribute = function(name, val) {
                      if (name === 'bis_skin_checked') return;
                      return _origSetAttr.apply(this, arguments);
                    };
                  }

                  // Continuously strip bis_skin_checked if already injected
                  if (typeof MutationObserver !== 'undefined' && typeof document !== 'undefined') {
                    var obs = new MutationObserver(function(mutations) {
                      for (var j = 0; j < mutations.length; j++) {
                        var m = mutations[j];
                        if (m.type === 'attributes' && m.attributeName === 'bis_skin_checked' && m.target && m.target.removeAttribute) {
                          m.target.removeAttribute('bis_skin_checked');
                        }
                      }
                    });
                    if (document.documentElement) {
                      obs.observe(document.documentElement, {
                        attributes: true,
                        subtree: true,
                        attributeFilter: ['bis_skin_checked']
                      });
                    }
                  }
                } catch(e) {}
              })();
            `,
          }}
        />
      </head>
      <body
        className="min-h-full flex flex-col bg-slate-950 text-slate-100 antialiased"
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
