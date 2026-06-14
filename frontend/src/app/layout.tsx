import type { Metadata } from "next";
import Link from "next/link";
import InspectionLogo from "@/components/InspectionLogo";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vígil — Inspeção de Soja",
  description: "Vígil: inspeção visual de grãos de soja com YOLO11",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pt-BR">
      <body className="min-h-screen bg-[#0a0a0b] font-sans text-neutral-200 antialiased">
        <header className="border-b border-white/5">
          <nav className="mx-auto flex max-w-5xl items-center gap-6 px-5 py-3.5">
            <Link href="/" className="flex items-center gap-2.5">
              <InspectionLogo className="w-6 text-neutral-200" />
              <span className="text-sm font-medium tracking-tight text-neutral-100">
                Vígil<span className="text-brand">.ia</span>
              </span>
              <span className="font-mono text-[9px] uppercase tracking-[0.2em] text-neutral-500">
                beta
              </span>
            </Link>
            <div className="ml-auto flex items-center gap-5 text-sm">
              <Link
                href="/"
                className="text-neutral-400 transition-colors hover:text-neutral-100"
              >
                Inspecionar
              </Link>
              <Link
                href="/sobre"
                className="text-neutral-400 transition-colors hover:text-neutral-100"
              >
                Sobre
              </Link>
            </div>
          </nav>
        </header>
        <main className="mx-auto max-w-5xl px-5 py-8">{children}</main>
      </body>
    </html>
  );
}
