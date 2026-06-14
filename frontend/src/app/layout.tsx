import type { Metadata } from "next";
import Link from "next/link";
import InspectionLogo from "@/components/InspectionLogo";
import "./globals.css";

export const metadata: Metadata = {
  title: "Inspeção de Soja",
  description: "Inspeção visual de grãos de soja com YOLO11",
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
                Inspeção de Soja
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
                href="/dashboard"
                className="text-neutral-400 transition-colors hover:text-neutral-100"
              >
                Dashboard
              </Link>
            </div>
          </nav>
        </header>
        <main className="mx-auto max-w-5xl px-5 py-8">{children}</main>
      </body>
    </html>
  );
}
