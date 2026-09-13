import type { NextConfig } from "next";

const config: NextConfig = {
  // Standalone output keeps the runtime image small: Next traces the files it
  // actually needs rather than shipping the whole node_modules tree.
  output: "standalone",
  reactStrictMode: true,
};

export default config;
