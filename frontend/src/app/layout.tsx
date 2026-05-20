import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Inspeção de Soja",
  description: "Sistema de inspeção visual de grãos de soja com YOLOv8",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pt-BR">
      <body className="bg-gray-50 min-h-screen font-sans antialiased">
        <header className="bg-brand text-white px-6 py-4 shadow-md">
          <nav className="max-w-5xl mx-auto flex items-center gap-6">
            <h1 className="text-xl font-bold tracking-tight">Inspeção de Soja</h1>
            <a href="/" className="text-green-100 hover:text-white text-sm">
              Nova Inspeção
            </a>
            <a href="/dashboard" className="text-green-100 hover:text-white text-sm">
              Dashboard
            </a>
          </nav>
        </header>
        <main className="max-w-5xl mx-auto px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
