"use client";

import "./globals.css";
import type { ReactNode } from "react";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <title>Recipe Vector Search</title>
      </head>
      <body>{children}</body>
    </html>
  );
}
