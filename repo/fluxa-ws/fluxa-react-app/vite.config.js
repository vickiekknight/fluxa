// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  envPrefix: ['VITE_', 'DEDALUS_'], 
  server: {
    host: "localhost",
    port: 5173,
  },
});
