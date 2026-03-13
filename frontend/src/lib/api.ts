// PitchPal v2 - API Client

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = API_BASE.replace("http", "ws");

export async function fetchSamplePitches() {
  const res = await fetch(`${API_BASE}/sample-pitches`);
  if (!res.ok) throw new Error("Failed to fetch sample pitches");
  const data = await res.json();
  return data.pitches;
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("API is not reachable");
  return res.json();
}

export function getWebSocketURL(): string {
  return `${WS_BASE}/ws/evaluate`;
}

export async function clearBackendCache(): Promise<void> {
  try {
    await fetch(`${API_BASE}/cache/clear`, { method: "DELETE" });
  } catch {
    // Silently fail — cache clear is best-effort
    console.warn("Failed to clear backend cache");
  }
}

export async function deleteCacheEntry(
  pitchText: string,
  role: string
): Promise<void> {
  try {
    await fetch(`${API_BASE}/cache/entry`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pitch_text: pitchText, role }),
    });
  } catch {
    console.warn("Failed to delete backend cache entry");
  }
}

export async function verifyInvestorCode(
  code: string
): Promise<{ success: boolean; token?: string; expires_in?: number; error?: string }> {
  const res = await fetch(`${API_BASE}/verify-code`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    return { success: false, error: data.detail || "Invalid access code" };
  }

  const data = await res.json();
  return { success: true, token: data.token, expires_in: data.expires_in };
}

export async function uploadDeck(
  file: File,
): Promise<import("@/types").DeckUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/upload-deck`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Deck upload failed");
  }

  return res.json();
}
