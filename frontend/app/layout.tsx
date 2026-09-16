import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "videoSparse SLAM",
  description: "Monocular visual-odometry SLAM: upload video, inspect the 3D map.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
