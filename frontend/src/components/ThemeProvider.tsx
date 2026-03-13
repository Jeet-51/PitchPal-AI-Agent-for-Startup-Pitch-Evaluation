"use client";

import { createContext, useContext, useEffect, useState } from "react";

type Theme = "light" | "dark";

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType>({
  theme: "dark",
  toggleTheme: () => { },
});

export function useTheme() {
  return useContext(ThemeContext);
}

export default function ThemeProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  // Default is dark — best first impression
  const [theme, setTheme] = useState<Theme>("dark");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);

    // Priority: 1) saved preference, 2) OS preference, 3) dark (default)
    const saved = localStorage.getItem("pitchpal-theme") as Theme | null;
    if (saved === "light" || saved === "dark") {
      setTheme(saved);
      document.documentElement.setAttribute("data-theme", saved);
    } else if (window.matchMedia("(prefers-color-scheme: light)").matches) {
      // Only switch to light if user explicitly prefers light at OS level
      // (dark is still our product default)
      setTheme("light");
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      // Default: dark
      document.documentElement.setAttribute("data-theme", "dark");
    }
  }, []);

  const toggleTheme = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("pitchpal-theme", next);
    document.documentElement.setAttribute("data-theme", next);
  };

  // Before mount: render children without flicker
  // We set data-theme="dark" in globals via :root defaults so SSR is dark by default
  if (!mounted) return <>{children}</>;

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}
