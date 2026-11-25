"""Local HTTP Server & AWS Lambda Function URL Handler for TrainiumPulse.

Routes served:
- GET  /                   -> web/index.html
- GET  /api/health         -> runtime health + Bedrock configuration
- GET  /api/benchmarks     -> benchmarks/latest_results.json + live compile cache stats
- POST /api/autograd/verify -> Multi-rank TP + Ulysses SP + ZeRO-1 forward/backward parity
- POST /api/schedule/1f1b  -> Exact 0/1 knapsack checkpointing + GPipe vs. 1F1B schedule
- POST /api/fp8/train      -> Bit-level BF16/FP8/MXFP8 inspector + 40-step SR vs. RNE training
- POST /api/compiler/cache-bisect -> Alpha-canonical 24-layer cache + Bugpoint pass bisector
- POST /api/advisor/analyze -> Live Amazon Bedrock (us.amazon.nova-lite-v1:0) bottleneck advisor
- POST /api/reset          -> Clears compile cache and resets ephemeral /tmp state
"""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import traceback
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

from trainium_pulse.checkpoint_1f1b import verify_1f1b_and_checkpointing
from trainium_pulse.dist_compile_cache import (
    ContentAddressedCompileCache,
    WORKLOAD_PROFILES,
    analyze_training_bottleneck_bedrock,
    run_compile_cache_and_bisect,
)
from trainium_pulse.fp8_stochastic import run_fp8_convergence_lab
from trainium_pulse.sharded_autograd import verify_sharded_autograd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_INDEX_PATH = PROJECT_ROOT / "web" / "index.html"
BENCHMARKS_JSON_PATH = PROJECT_ROOT / "benchmarks" / "latest_results.json"


def _cors_headers(content_type: str = "application/json; charset=utf-8") -> Dict[str, str]:
    return {
        "Content-Type": content_type,
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Accept",
        "Cache-Control": "no-store",
    }


def _load_benchmarks() -> Dict[str, Any]:
    if BENCHMARKS_JSON_PATH.exists():
        try:
            return json.loads(BENCHMARKS_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def dispatch_api_request(method: str, path: str, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    clean_path = path.rstrip("/") or "/"
    method_u = method.upper()

    if method_u == "OPTIONS":
        return 200, {"ok": True}

    if clean_path == "/api/health" and method_u == "GET":
        cache = ContentAddressedCompileCache()
        return 200, {
            "status": "ok",
            "service": "trainium-pulse",
            "version": "1.0.0",
            "delivery": os.environ.get("TRAINIUM_PULSE_DELIVERY", "local-http"),
            "bedrock_model": os.environ.get(
                "TRAINIUM_PULSE_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"
            ),
            "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
            "compile_cache": cache.stats(),
            "workload_profiles": list(WORKLOAD_PROFILES.keys()),
        }

    if clean_path == "/api/benchmarks" and method_u == "GET":
        cache = ContentAddressedCompileCache()
        return 200, {
            "status": "ok",
            "benchmarks": _load_benchmarks(),
            "compile_cache": cache.stats(),
            "workloads": WORKLOAD_PROFILES,
        }

    if clean_path == "/api/autograd/verify" and method_u == "POST":
        res = verify_sharded_autograd(
            world_size=int(body.get("world_size", 4)),
            batch_size=int(body.get("batch_size", 2)),
            seq_len=int(body.get("seq_len", 16)),
            d_model=int(body.get("d_model", 32)),
            n_heads=int(body.get("n_heads", 4)),
            d_ff=int(body.get("d_ff", 64)),
            vocab_size=int(body.get("vocab_size", 32)),
            architecture=str(body.get("architecture", "llama_causal")),
            seed=int(body.get("seed", 42)),
        )
        return 200, res

    if clean_path == "/api/schedule/1f1b" and method_u == "POST":
        res = verify_1f1b_and_checkpointing(
            num_stages=int(body.get("num_stages", 4)),
            num_microbatches=int(body.get("num_microbatches", 8)),
            microbatch_size=int(body.get("microbatch_size", 1)),
            seq_len=int(body.get("seq_len", 16)),
            d_model=int(body.get("d_model", 16)),
            n_heads=int(body.get("n_heads", 4)),
            d_ff=int(body.get("d_ff", 32)),
            memory_budget_ratio=float(body.get("memory_budget_ratio", 0.45)),
            seed=int(body.get("seed", 7)),
        )
        return 200, res

    if clean_path == "/api/fp8/train" and method_u == "POST":
        res = run_fp8_convergence_lab(
            steps=int(body.get("steps", 40)),
            learning_rate=float(body.get("learning_rate", 1e-4)),
            sample_float=float(body.get("sample_float", -1.375)),
            outlier_magnitude=float(body.get("outlier_magnitude", 1024.0)),
            seed=int(body.get("seed", 123)),
        )
        return 200, res

    if clean_path == "/api/compiler/cache-bisect" and method_u == "POST":
        res = run_compile_cache_and_bisect(
            num_layers=int(body.get("num_layers", 24)),
            seq_len=int(body.get("seq_len", 16)),
            d_model=int(body.get("d_model", 32)),
            d_ff=int(body.get("d_ff", 64)),
            hybrid_moe_every=int(body.get("hybrid_moe_every", 0)),
            reset_cache_first=bool(body.get("reset_cache_first", True)),
        )
        if body.get("include_bedrock"):
            res["bedrock_analysis"] = analyze_training_bottleneck_bedrock(
                workload_id=str(body.get("workload_id", "deepseek_moe_fp8")),
                custom_note=(
                    f"Subgraph cache hit {res['cache_hits']}/{res['num_layers']} "
                    f"({res['cache_hit_rate_pct']}%) and isolated {res['bugpoint_bisector']['isolated_pass_name']} "
                    f"to {res['bugpoint_bisector']['minimal_reproducer_op_count']} SSA ops."
                ),
            )
        return 200, res

    if clean_path == "/api/advisor/analyze" and method_u == "POST":
        res = analyze_training_bottleneck_bedrock(
            workload_id=str(body.get("workload_id", "llama3_70b_tp4_zero1")),
            custom_note=str(body.get("custom_note", "")),
        )
        return 200, res

    if clean_path == "/api/reset" and method_u == "POST":
        cache = ContentAddressedCompileCache()
        cache.clear()
        return 200, {"status": "reset", "compile_cache": cache.stats()}

    return 404, {"error": f"Unknown route: {method_u} {clean_path}"}


def lambda_handler(event: Dict[str, Any], _context: Any = None) -> Dict[str, Any]:
    """AWS Lambda Function URL (payload format 2.0) + API Gateway handler."""
    try:
        http_ctx = event.get("requestContext", {}).get("http", {})
        method = http_ctx.get("method") or event.get("httpMethod") or "GET"
        raw_path = event.get("rawPath") or event.get("path") or "/"

        if method.upper() == "OPTIONS":
            return {"statusCode": 200, "headers": _cors_headers(), "body": "{}"}

        if raw_path in ("/", "/index.html") and method.upper() == "GET":
            html = (
                WEB_INDEX_PATH.read_text(encoding="utf-8")
                if WEB_INDEX_PATH.exists()
                else "<h1>TrainiumPulse</h1>"
            )
            return {
                "statusCode": 200,
                "headers": _cors_headers("text/html; charset=utf-8"),
                "body": html,
            }

        raw_body = event.get("body") or ""
        if event.get("isBase64Encoded") and raw_body:
            raw_body = base64.b64decode(raw_body).decode("utf-8")
        parsed_body: Dict[str, Any] = json.loads(raw_body) if raw_body.strip() else {}

        status, payload = dispatch_api_request(method, raw_path, parsed_body)
        return {
            "statusCode": status,
            "headers": _cors_headers(),
            "body": json.dumps(payload),
        }
    except Exception as exc:
        return {
            "statusCode": 400,
            "headers": _cors_headers(),
            "body": json.dumps(
                {"error": str(exc), "trace": traceback.format_exc().splitlines()[-3:]}
            ),
        }


class _LocalHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self._send(200, {"ok": True})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            data = (
                WEB_INDEX_PATH.read_bytes()
                if WEB_INDEX_PATH.exists()
                else b"<h1>TrainiumPulse</h1>"
            )
            self.send_response(200)
            for k, v in _cors_headers("text/html; charset=utf-8").items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
            return
        status, payload = dispatch_api_request("GET", parsed.path, {})
        self._send(status, payload)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else ""
        try:
            body = json.loads(raw) if raw.strip() else {}
            status, payload = dispatch_api_request("POST", parsed.path, body)
        except Exception as exc:
            status, payload = 400, {"error": str(exc)}
        self._send(status, payload)

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _fmt: str, *_args: Any) -> None:
        return


def main() -> None:
    port = int(os.environ.get("PORT", "3099"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), _LocalHandler)
    print(f"TrainiumPulse local server listening on http://127.0.0.1:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
