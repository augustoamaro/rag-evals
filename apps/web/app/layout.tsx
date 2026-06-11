import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { Sora, Hanken_Grotesk, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const display = Sora({ subsets: ["latin"], weight: ["500", "700"], variable: "--font-display" });
const body = Hanken_Grotesk({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-body" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "rag-evals — retrieval & answer-quality lab",
  description: "Dashboard for RAG retrieval metrics, answer-quality evals, cost and latency.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>
        <header className="topbar">
          <Link href="/" className="brand">
            rag<span className="brand__accent">·</span>evals
          </Link>
          <nav className="nav">
            <Link href="/">Runs</Link>
            <Link href="/try">Query</Link>
          </nav>
        </header>
        <main className="shell">{children}</main>
      </body>
    </html>
  );
}
