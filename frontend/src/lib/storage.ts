// PitchPal v2 - localStorage persistence for evaluation history (role-specific)

import { SavedEvaluation, UserRole } from "@/types";

function getStorageKey(role: UserRole): string {
  return `pitchpal-${role}-history`;
}

export function getHistory(role: UserRole): SavedEvaluation[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(getStorageKey(role));
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveEvaluation(entry: SavedEvaluation): void {
  const role = entry.role || "startup";
  const history = getHistory(role);
  history.unshift(entry); // newest first
  // Keep max 50 evaluations
  if (history.length > 50) history.pop();
  localStorage.setItem(getStorageKey(role), JSON.stringify(history));
}

export function deleteEvaluation(id: string, role: UserRole): void {
  const history = getHistory(role).filter((e) => e.id !== id);
  localStorage.setItem(getStorageKey(role), JSON.stringify(history));
}

export function clearHistory(role: UserRole): void {
  localStorage.removeItem(getStorageKey(role));
}
