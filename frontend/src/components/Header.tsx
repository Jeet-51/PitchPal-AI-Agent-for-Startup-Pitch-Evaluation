"use client";

import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Rocket, Github, Activity, Sun, Moon,
  History, GitCompareArrows, ArrowLeftRight, Sparkles,
} from "lucide-react";
import { fetchHealth } from "@/lib/api";
import { useTheme } from "./ThemeProvider";
import { getRole, clearRole } from "@/lib/auth";
import { UserRole } from "@/types";

// ── Nav links shown in the grouped pill ────────────────────
const NAV_LINKS = [
  { href: "/history", icon: History, label: "History" },
  { href: "/compare", icon: GitCompareArrows, label: "Compare" },
  { href: "/why", icon: Sparkles, label: "Why?" },
];

export default function Header() {
  const { theme, toggleTheme } = useTheme();
  const router = useRouter();
  const pathname = usePathname();

  const [status, setStatus] = useState<"checking" | "connected" | "offline">("checking");
  const [role, setRoleState] = useState<UserRole | null>(null);
  const [scrolled, setScrolled] = useState(false);

  /* Ping backend */
  useEffect(() => {
    fetchHealth()
      .then(() => setStatus("connected"))
      .catch(() => setStatus("offline"));
    setRoleState(getRole());
  }, []);

  /* Elevate nav shadow on scroll */
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const handleSwitchRole = () => { clearRole(); router.push("/"); };

  const statusColor = {
    checking: "text-[var(--muted)]",
    connected: "text-[var(--success)]",
    offline: "text-[var(--danger)]",
  }[status];

  const statusLabel = {
    checking: "Checking…",
    connected: "API Connected",
    offline: "API Offline",
  }[status];

  /* Shared small icon-btn style for utility controls */
  const iconBtn =
    "p-2 rounded-xl text-[var(--muted)] hover:text-[var(--foreground)] hover:bg-[var(--hover-bg)] transition-all duration-200";

  return (
    <motion.header
      initial={{ y: -64, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] as any }}
      className="sticky top-0 z-50 glass-nav"
      style={{
        boxShadow: scrolled
          ? "0 8px 32px rgba(0,0,0,0.12), 0 1px 0 var(--nav-border)"
          : "none",
        transition: "box-shadow 0.3s ease",
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-4">

        {/* ── Brand ──────────────────────────────────────── */}
        <Link
          href={role ? "/evaluate" : "/"}
          className="flex items-center gap-3 group shrink-0"
        >
          <motion.div
            whileHover={{ rotate: -8, scale: 1.08 }}
            transition={{ type: "spring", stiffness: 320 }}
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{
              background: "linear-gradient(135deg, var(--primary), var(--accent))",
              boxShadow: "0 0 20px var(--primary-glow)",
            }}
          >
            <Rocket style={{ width: "1.1rem", height: "1.1rem" }} className="text-white" />
          </motion.div>
          <div className="hidden sm:block">
            <span className="text-lg font-black gradient-text-static tracking-tight">PitchPal</span>
            <p className="text-[10px] text-[var(--muted)] leading-none mt-0.5">AI Startup Evaluator</p>
          </div>
        </Link>

        {/* ── Right side ─────────────────────────────────── */}
        <div className="flex items-center gap-2">

          {/* ── Utility controls (icon-only) ─────────────── */}

          {/* Role badge */}
          <AnimatePresence>
            {role && (
              <motion.span
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.85 }}
                className={`hidden sm:flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${role === "investor"
                    ? "bg-purple-500/10 text-purple-400 border-purple-500/25"
                    : "bg-blue-500/10 text-blue-400 border-blue-500/25"
                  }`}
              >
                {role === "investor" ? "Investor" : "Startup"}
              </motion.span>
            )}
          </AnimatePresence>

          {/* Switch role */}
          {role && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleSwitchRole}
              className={iconBtn}
              title="Switch Role"
            >
              <ArrowLeftRight className="w-4 h-4" />
            </motion.button>
          )}

          {/* API status */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 glass rounded-xl">
            <Activity className={`w-3.5 h-3.5 ${statusColor}`} />
            <span className={`text-xs hidden sm:inline ${statusColor}`}>{statusLabel}</span>
          </div>

          {/* Theme toggle */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95, rotate: 15 }}
            onClick={toggleTheme}
            className={iconBtn}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            <AnimatePresence mode="wait">
              {theme === "dark" ? (
                <motion.div key="sun"
                  initial={{ opacity: 0, rotate: -30 }} animate={{ opacity: 1, rotate: 0 }}
                  exit={{ opacity: 0, rotate: 30 }} transition={{ duration: 0.2 }}
                >
                  <Sun className="w-4 h-4" />
                </motion.div>
              ) : (
                <motion.div key="moon"
                  initial={{ opacity: 0, rotate: 30 }} animate={{ opacity: 1, rotate: 0 }}
                  exit={{ opacity: 0, rotate: -30 }} transition={{ duration: 0.2 }}
                >
                  <Moon className="w-4 h-4" />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.button>

          {/* ── Thin divider ─────────── */}
          <div className="h-6 w-px mx-1" style={{ background: "var(--card-border)" }} />

          {/* ── Nav pill (labeled links) ──────────────────── */}
          <div
            className="flex items-center rounded-xl overflow-hidden"
            style={{
              background: "var(--hover-bg)",
              border: "1px solid var(--card-border)",
            }}
          >
            {NAV_LINKS.map(({ href, icon: Icon, label }, idx) => {
              const isActive = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className="relative flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-all duration-200"
                  style={{
                    color: isActive ? "var(--accent-primary)" : "var(--muted)",
                    background: isActive ? "var(--accent-primary)15" : "transparent",
                    borderRight: idx < NAV_LINKS.length - 1 ? "1px solid var(--card-border)" : "none",
                  }}
                  onMouseEnter={e => {
                    if (!isActive) (e.currentTarget as HTMLElement).style.color = "var(--foreground)";
                  }}
                  onMouseLeave={e => {
                    if (!isActive) (e.currentTarget as HTMLElement).style.color = "var(--muted)";
                  }}
                >
                  <Icon className="w-3.5 h-3.5 shrink-0" />
                  <span className="hidden sm:inline">{label}</span>
                  {isActive && (
                    <motion.div
                      layoutId="nav-active"
                      className="absolute inset-0 rounded-none"
                      style={{ background: "var(--accent-primary)10" }}
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    />
                  )}
                </Link>
              );
            })}

            {/* GitHub — treated as external so outside pill or as last item */}
            <motion.a
              href="https://github.com/Jeet-51"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-all duration-200"
              style={{ color: "var(--muted)", borderLeft: "1px solid var(--card-border)" }}
              whileHover={{ color: "var(--foreground)" } as any}
              title="GitHub"
            >
              <Github className="w-3.5 h-3.5 shrink-0" />
              <span className="hidden sm:inline">GitHub</span>
            </motion.a>
          </div>

        </div>
      </div>
    </motion.header>
  );
}
