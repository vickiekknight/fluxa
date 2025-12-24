/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        electric: {
          purple: '#BF00FF',
          light: '#C94DFF',
          dim: '#BF00FF33',
          dark: '#1b0036', // for backgrounds if you want that deep cosmic purple
        },
      },
    },
  },
  plugins: [],
}
