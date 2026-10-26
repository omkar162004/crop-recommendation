import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ExternalLink,
  Loader2,
  MapPin,
  Search,
  Sprout,
} from "lucide-react";
import {
  API_BASE,
  ApiError,
  geocode,
  getMeta,
  predict,
  type Meta,
  type PredictRequest,
  type PredictResponse,
  type Prediction,
  type SourceInfo,
} from "./lib/api";

// Replace with your DagsHub MLflow link once it exists
const MLFLOW_UI_URL = "https://dagshub.com/YOUR_USERNAME/YOUR_REPO.mlflow";

const INPUT =
  "w-full px-3.5 py-2.5 rounded-md border border-[#C9B79C] bg-white focus:outline-none focus:ring-2 focus:ring-[#3F6B44] focus:border-transparent";

export default function App() {
  // fonts, loaded here so the file works without touching index.html
  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href =
      "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap";
    document.head.appendChild(link);
    return () => {
      document.head.removeChild(link);
    };
  }, []);

  const [mode, setMode] = useState<"place" | "coords">("place");
  const [placeInput, setPlaceInput] = useState("");
  const [latInput, setLatInput] = useState("");
  const [lonInput, setLonInput] = useState("");

  const [useNpk, setUseNpk] = useState(false);
  const [npkInput, setNpkInput] = useState({ N: "", P: "", K: "" });
  const [usePh, setUsePh] = useState(false);
  const [phInput, setPhInput] = useState("");
  const [simulateFailure, setSimulateFailure] = useState(false);

  const [loading, setLoading] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [showHood, setShowHood] = useState(false);

  // on load: ask the backend for the N-P-K ranges (also tells us if it's reachable)
  useEffect(() => {
    getMeta()
      .then(setMeta)
      .catch((err: Error) => setBackendError(err.message));
  }, []);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setResult(null);

    try {
      // 1. work out the coordinates
      setLoading("Finding the location");
      let lat: number;
      let lon: number;
      let name: string | undefined;

      if (mode === "place") {
        if (!placeInput.trim()) throw new Error("Type a place name first.");
        const place = await geocode(placeInput.trim());
        ({ lat, lon, name } = place);
      } else {
        lat = parseFloat(latInput);
        lon = parseFloat(lonInput);
        if (Number.isNaN(lat) || Number.isNaN(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
          throw new Error("Enter latitude (-90 to 90) and longitude (-180 to 180).");
        }
        name = `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
      }

      // 2. build the request, adding only what the user chose to provide
      const payload: PredictRequest = { lat, lon, name, simulate_failure: simulateFailure };

      if (usePh) {
        const ph = parseFloat(phInput);
        if (Number.isNaN(ph) || ph < 3 || ph > 10) throw new Error("Soil pH must be between 3 and 10.");
        payload.ph = ph;
      }
      if (useNpk) {
        const N = parseFloat(npkInput.N);
        const P = parseFloat(npkInput.P);
        const K = parseFloat(npkInput.K);
        if ([N, P, K].some(Number.isNaN)) throw new Error("Fill in all three of N, P and K, or untick the box.");
        payload.npk = { N, P, K };
      }

      // 3. ask the backend
      setLoading("Reading current conditions and matching crops");
      setResult(await predict(payload));
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
        // soil services are sometimes down: steer the user to typing their own pH
        if (err.status === 503 && /soil/i.test(err.message)) setUsePh(true);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setLoading(null);
    }
  }

  const rangeText = (key: "N" | "P" | "K") =>
    meta ? `${key} (${meta.npk_ranges[key][0]}–${meta.npk_ranges[key][1]})` : key;

  return (
    <div className="min-h-screen bg-[#EEF0E7] text-[#23261F]" style={{ fontFamily: "'IBM Plex Sans', sans-serif" }}>
      <div className="flex min-h-screen flex-col md:flex-row">
        {/* soil-core strip: horizon colours, dark humus at the top down to pale parent rock */}
        <div
          className="h-3 w-full shrink-0 md:h-auto md:w-3"
          style={{ background: "linear-gradient(180deg, #3B2E22 0%, #5B4630 28%, #A9744F 62%, #C9B79C 100%)" }}
          aria-hidden="true"
        />

        <div className="flex-1">
          <header className="flex items-center justify-between px-6 pb-6 pt-8 md:px-12">
            <div className="flex items-center gap-2.5">
              <Sprout className="h-6 w-6 text-[#3F6B44]" />
              <span className="text-xl tracking-tight" style={{ fontFamily: "'Fraunces', serif", fontWeight: 600 }}>
                Groundtruth
              </span>
            </div>
            <button
              type="button"
              onClick={() => setShowHood((v) => !v)}
              className="flex items-center gap-1 text-sm text-[#5B4630] transition-colors hover:text-[#3B2E22]"
            >
              Under the hood
              <ChevronDown className={`h-4 w-4 transition-transform ${showHood ? "rotate-180" : ""}`} />
            </button>
          </header>

          {showHood && (
            <div className="mx-6 mb-6 max-w-[640px] rounded-md border border-[#C9B79C] bg-[#F6F4EC] p-5 md:mx-12">
              <p className="mb-4 max-w-[60ch] text-sm text-[#5B4630]">
                Each answer comes from a model with a tracked training history. Climate comes from Open-Meteo,
                soil pH from SoilGrids, and every call is timed and logged. When a service is down, the last saved
                reading for that area is used and labelled.
              </p>
              <div className="mb-5 flex flex-col gap-2">
                <HoodLink href={MLFLOW_UI_URL} label="Experiment tracking" />
                <HoodLink href={`${API_BASE}/health`} label="Service health and model info" />
                <HoodLink href={`${API_BASE}/docs`} label="API documentation" />
              </div>
              <label className="flex cursor-pointer items-center gap-2.5 text-sm text-[#5B4630]">
                <input
                  type="checkbox"
                  checked={simulateFailure}
                  onChange={(e) => setSimulateFailure(e.target.checked)}
                  className="h-4 w-4 accent-[#5B4630]"
                />
                Simulate a data-service outage on the next lookup
              </label>
            </div>
          )}

          <main className="px-6 md:px-12">
            {backendError && (
              <p className="mb-4 max-w-[560px] rounded-md border border-[#D9A441] bg-[#FBF1DC] px-4 py-3 text-sm text-[#6B5220]">
                {backendError}
              </p>
            )}

            <div className="max-w-[560px]" style={{ animation: "fadeIn 0.5s ease-out" }}>
              <h1 className="mb-2 text-3xl leading-tight md:text-4xl" style={{ fontFamily: "'Fraunces', serif", fontWeight: 500 }}>
                Where are we growing?
              </h1>
              <p className="mb-6 max-w-[52ch] text-[#5B4630]">
                Live climate for the exact spot, matched against what each crop needs.
              </p>

              <form onSubmit={handleSubmit}>
                <div className="mb-3 flex gap-1 text-sm">
                  {(["place", "coords"] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setMode(m)}
                      className={`rounded-full px-3 py-1.5 transition-colors ${
                        mode === m ? "bg-[#3F6B44] text-white" : "text-[#5B4630] hover:bg-[#E4E1D3]"
                      }`}
                    >
                      {m === "place" ? "Place name" : "Coordinates"}
                    </button>
                  ))}
                </div>

                {mode === "place" ? (
                  <div className="relative">
                    <Search className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-[#A9744F]" />
                    <input
                      value={placeInput}
                      onChange={(e) => setPlaceInput(e.target.value)}
                      placeholder="Nagpur"
                      aria-label="Place name"
                      className={`${INPUT} pl-10`}
                    />
                  </div>
                ) : (
                  <div className="flex gap-2">
                    <input value={latInput} onChange={(e) => setLatInput(e.target.value)} placeholder="Latitude" aria-label="Latitude" className={INPUT} />
                    <input value={lonInput} onChange={(e) => setLonInput(e.target.value)} placeholder="Longitude" aria-label="Longitude" className={INPUT} />
                  </div>
                )}

                {/* optional soil details the user can supply */}
                <div className="mt-5 flex flex-col gap-4">
                  <div>
                    <label className="flex cursor-pointer items-center gap-2.5 text-sm">
                      <input type="checkbox" checked={usePh} onChange={(e) => setUsePh(e.target.checked)} className="h-4 w-4 accent-[#3F6B44]" />
                      I know my soil pH
                    </label>
                    {usePh && (
                      <input
                        type="number"
                        step="0.1"
                        min={3}
                        max={10}
                        value={phInput}
                        onChange={(e) => setPhInput(e.target.value)}
                        placeholder="e.g. 6.5"
                        aria-label="Soil pH"
                        className={`${INPUT} mt-2 max-w-[160px]`}
                      />
                    )}
                  </div>

                  <div>
                    <label className="flex cursor-pointer items-center gap-2.5 text-sm">
                      <input type="checkbox" checked={useNpk} onChange={(e) => setUseNpk(e.target.checked)} className="h-4 w-4 accent-[#3F6B44]" />
                      I know my soil nitrogen, phosphorus and potassium
                    </label>
                    {useNpk && (
                      <>
                        <div className="mt-2 flex gap-2">
                          {(["N", "P", "K"] as const).map((k) => (
                            <input
                              key={k}
                              type="number"
                              min={0}
                              value={npkInput[k]}
                              onChange={(e) => setNpkInput({ ...npkInput, [k]: e.target.value })}
                              placeholder={rangeText(k)}
                              aria-label={k}
                              className={INPUT}
                            />
                          ))}
                        </div>
                        <p className="mt-1.5 max-w-[52ch] text-xs text-[#5B4630]">
                          Use the same scale as the training data shown above. Its units aren't documented, so soil
                          test values in other units won't match.
                        </p>
                      </>
                    )}
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading !== null}
                  className="mt-6 rounded-md bg-[#3F6B44] px-5 py-3 font-medium text-white transition-colors hover:bg-[#345A39] disabled:opacity-60"
                >
                  Get recommendation
                </button>
              </form>

              {error && <p className="mt-3 max-w-[56ch] text-sm text-[#B3401F]">{error}</p>}
            </div>

            {loading && (
              <div className="mt-8 flex items-center gap-2.5 text-sm text-[#5B4630]">
                <Loader2 className="h-4 w-4 animate-spin" />
                {loading}
              </div>
            )}

            {result && <Results result={result} />}
          </main>

          <footer className="px-6 pb-10 pt-4 text-xs text-[#5B4630] md:px-12">
            <p className="max-w-[70ch]">
              Trained on a curated crop dataset, so treat results as guidance rather than an agronomic verdict.
            </p>
          </footer>
        </div>
      </div>

      <style>{`@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }`}</style>
    </div>
  );
}

/* ----------------------------- results ----------------------------- */

function describeSource(label: string, s: SourceInfo): string {
  if (s.status === "live") return `${label}: live`;
  if (s.status === "user") return `${label}: entered by you`;
  const hours = s.age_hours ?? 0;
  const age = hours < 1 ? "under an hour old" : `${Math.round(hours)} hours old`;
  return `${label}: saved reading from ${s.distance_km ?? 0} km away, ${age}`;
}

function Results({ result }: { result: PredictResponse }) {
  const { readings, sources, predictions } = result;
  return (
    <div className="mb-8 mt-8 max-w-[640px]">
      <div className="mb-3 flex items-center gap-1.5 text-sm text-[#5B4630]">
        <MapPin className="h-4 w-4" />
        {result.location.name ?? `${result.location.lat}, ${result.location.lon}`}
      </div>

      <p className="mb-1 text-sm text-[#5B4630]">
        {readings.temperature} °C, {readings.humidity}% humidity, {readings.rainfall} mm rain per month, soil pH{" "}
        {readings.ph}
      </p>
      <p className="mb-4 text-xs text-[#5B4630]">
        {describeSource("Climate", sources.climate)}. {describeSource("Soil pH", sources.soil)}.
      </p>

      {result.used_fallback && (
        <div className="mb-4 flex items-start gap-2.5 rounded-md border border-[#D9A441] bg-[#FBF1DC] px-4 py-3 text-sm text-[#6B5220]">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-[#B98226]" />
          Live data wasn't reachable, so this uses the last saved reading for the area.
        </div>
      )}

      {result.status === "out_of_range" && (
        <p className="rounded-md border border-[#C9B79C] bg-[#F6F4EC] px-4 py-3 text-sm">{result.message}</p>
      )}

      {result.status === "low_confidence" && result.message && (
        <p className="mb-4 max-w-[60ch] text-sm text-[#5B4630]">{result.message}</p>
      )}

      {predictions.length > 0 && (
        <>
          <TopPick prediction={predictions[0]} />
          <div className="mt-3 flex flex-col gap-3">
            {predictions.slice(1).map((p, i) => (
              <RunnerUp key={p.crop} rank={i + 2} prediction={p} />
            ))}
          </div>
          <p className="mt-4 text-xs text-[#5B4630]">
            {result.model_used === "full"
              ? "Full model using your N, P and K"
              : "Basic model using climate and soil pH only"}
            , {Math.round(result.macro_f1 * 100)}% macro F1 on held-out test data.
          </p>
        </>
      )}
    </div>
  );
}

function TopPick({ prediction }: { prediction: Prediction }) {
  const pct = Math.round(prediction.confidence * 100);
  return (
    <div className="rounded-lg border-2 border-[#3F6B44] bg-white px-6 py-5">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-2xl capitalize" style={{ fontFamily: "'Fraunces', serif", fontWeight: 600 }}>
          {prediction.crop}
        </span>
        <span className="text-sm text-[#5B4630]">{pct}% confidence</span>
      </div>
      <div className="mb-3 h-1.5 w-full overflow-hidden rounded-full bg-[#E4E1D3]">
        <div className="h-full rounded-full bg-[#3F6B44]" style={{ width: `${pct}%` }} />
      </div>
      <p className="max-w-[60ch] text-sm text-[#5B4630]">{prediction.reason}</p>
    </div>
  );
}

function RunnerUp({ rank, prediction }: { rank: number; prediction: Prediction }) {
  const pct = Math.round(prediction.confidence * 100);
  return (
    <div className="rounded-md border border-[#DAD6C4] bg-[#F6F4EC] px-4 py-3">
      <div className="flex items-center gap-4">
        <span className="w-4 text-sm text-[#A9744F]">{rank}</span>
        <span className="w-28 shrink-0 font-medium capitalize">{prediction.crop}</span>
        <div className="hidden h-1 flex-1 overflow-hidden rounded-full bg-[#E4E1D3] sm:block">
          <div className="h-full rounded-full bg-[#A9744F]" style={{ width: `${pct}%` }} />
        </div>
        <span className="ml-auto w-10 shrink-0 text-right text-sm text-[#5B4630]">{pct}%</span>
      </div>
      <p className="mt-1.5 max-w-[60ch] pl-8 text-xs text-[#5B4630]">{prediction.reason}</p>
    </div>
  );
}

function HoodLink({ href, label }: { href: string; label: string }): ReactNode {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="flex items-center gap-1.5 text-sm text-[#3E6E8E] underline decoration-[#A9744F] underline-offset-4"
    >
      {label} <ExternalLink className="h-3.5 w-3.5" />
    </a>
  );
}