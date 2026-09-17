/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Standalone output keeps the Docker image small and self-contained
  // (required by frontend/Dockerfile).
  output: "standalone",
};

export default nextConfig;
