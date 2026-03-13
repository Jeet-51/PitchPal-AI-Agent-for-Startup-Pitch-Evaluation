// PitchPal v2 - Role & Auth Management

import { UserRole } from "@/types";

const ROLE_KEY = "pitchpal-role";
const INVESTOR_TOKEN_KEY = "pitchpal-investor-token";
const INVESTOR_TOKEN_EXPIRY_KEY = "pitchpal-investor-token-expiry";

// ── Role Management ──────────────────────────────────────────

export function getRole(): UserRole | null {
  if (typeof window === "undefined") return null;
  const role = localStorage.getItem(ROLE_KEY);
  if (role === "startup" || role === "investor") return role;
  return null;
}

export function setRole(role: UserRole): void {
  localStorage.setItem(ROLE_KEY, role);
}

export function clearRole(): void {
  localStorage.removeItem(ROLE_KEY);
  clearInvestorToken();
}

// ── Investor Token Management ────────────────────────────────

export function getInvestorToken(): string | null {
  if (typeof window === "undefined") return null;

  const token = localStorage.getItem(INVESTOR_TOKEN_KEY);
  const expiry = localStorage.getItem(INVESTOR_TOKEN_EXPIRY_KEY);

  if (!token || !expiry) return null;

  // Check expiry
  if (Date.now() > parseInt(expiry, 10)) {
    clearInvestorToken();
    return null;
  }

  return token;
}

export function setInvestorToken(token: string, expiresIn: number): void {
  localStorage.setItem(INVESTOR_TOKEN_KEY, token);
  localStorage.setItem(
    INVESTOR_TOKEN_EXPIRY_KEY,
    String(Date.now() + expiresIn * 1000)
  );
}

export function clearInvestorToken(): void {
  localStorage.removeItem(INVESTOR_TOKEN_KEY);
  localStorage.removeItem(INVESTOR_TOKEN_EXPIRY_KEY);
}

export function isInvestorAuthenticated(): boolean {
  return getInvestorToken() !== null;
}
