import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "judgekit",
  description: "Most eval tools measure your model. This one measures your judge.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b border-[var(--color-line)]">
          <div className="mx-auto flex max-w-6xl items-baseline gap-4 px-4 py-4">
            <Link href="/" className="text-lg font-semibold tracking-tight">
              judgekit
            </Link>
            <p className="text-sm text-[var(--color-muted)]">
              Most eval tools measure your model. This one measures your judge.
            </p>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
