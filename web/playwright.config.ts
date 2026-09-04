import { defineConfig } from "@playwright/test";

// Chromium headless has no reliable WebGPU: the e2e measures the wasm floor
// (ADR 0005). The dev server sends COOP/COEP, so wasm threads are on.
export default defineConfig({
  testDir: "e2e",
  timeout: 240_000,
  retries: 0,
  reporter: "line",
  use: {
    baseURL: "http://localhost:5173",
    headless: true,
    launchOptions: { args: ["--autoplay-policy=no-user-gesture-required"] },
  },
  webServer: {
    command: "npx vite --port 5173 --strictPort",
    url: "http://localhost:5173",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
