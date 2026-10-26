// All communication with the FastAPI backend lives here.
// Set VITE_API_BASE_URL in frontend/.env (e.g. https://your-backend.onrender.com) when you deploy.
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

export type Prediction = { crop: string; confidence: number; reason: string };

export type SourceInfo = {
  status: "live" | "cached" | "user";
  distance_km?: number;
  age_hours?: number;
};

export type PredictResponse = {
  location: { name: string | null; lat: number; lon: number };
  readings: { temperature: number; humidity: number; rainfall: number; ph: number };
  sources: { climate: SourceInfo; soil: SourceInfo };
  used_fallback: boolean;
  model_used: "basic" | "full";
  model: string;
  macro_f1: number;
  status: "ok" | "low_confidence" | "out_of_range";
  message: string | null;
  predictions: Prediction[];
};

export type PredictRequest = {
  lat: number;
  lon: number;
  name?: string;
  ph?: number;
  npk?: { N: number; P: number; K: number };
  simulate_failure?: boolean;
};

export type Meta = {
  npk_ranges: { N: [number, number]; P: [number, number]; K: [number, number] };
  npk_note: string;
  rainfall_basis: string;
};

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(`Can't reach the backend at ${API_BASE}. Is it running?`, 0);
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    // FastAPI sends {detail: "text"} or, for validation errors, {detail: [{msg: ...}]}
    const detail = body?.detail;
    const message = Array.isArray(detail)
      ? detail.map((d: { msg: string }) => d.msg).join("; ")
      : typeof detail === "string"
        ? detail
        : `Request failed (${response.status}).`;
    throw new ApiError(message, response.status);
  }
  return body as T;
}

export const getMeta = () => request<Meta>("/meta");

export const geocode = (q: string) =>
  request<{ name: string; lat: number; lon: number }>(`/geocode?q=${encodeURIComponent(q)}`);

export const predict = (payload: PredictRequest) =>
  request<PredictResponse>("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });