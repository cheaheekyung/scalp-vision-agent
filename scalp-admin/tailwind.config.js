/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",                 // ✅ Vite는 public/index.html이 아니라 루트 index.html
    "./src/**/*.{js,jsx,ts,tsx}",   // src 안의 모든 컴포넌트
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
