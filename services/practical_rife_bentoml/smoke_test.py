from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the Practical-RIFE BentoML MVP service.")
    parser.add_argument("--url", required=True, help="Full interpolate endpoint URL, e.g. http://localhost:3000/interpolate_video")
    parser.add_argument("--input-path", required=True, help="Absolute input video path visible to the service container")
    parser.add_argument("--output-path", required=True, help="Absolute output video path visible to the service container")
    parser.add_argument("--interpolation-factor", required=True, type=int, choices=(2, 3, 4))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--output-playback-mode",
        choices=("real_time", "slow_motion"),
        default="real_time",
    )
    parser.add_argument(
        "--disable-quality-evaluation",
        action="store_true",
        help="Send quality_evaluation_enabled=false.",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="HTTP timeout in seconds")
    return parser.parse_args(argv)


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    return {"request": {
        "input_path": args.input_path,
        "output_path": args.output_path,
        "interpolation_factor": args.interpolation_factor,
        "scale": args.scale,
        "output_playback_mode": args.output_playback_mode,
        "quality_evaluation_enabled": not args.disable_quality_evaluation,
    }}


def post_json(url: str, payload: dict[str, object], *, timeout: float) -> tuple[int, dict[str, Any]]:
    encoded = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            return response.status, json.loads(response_body)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach service: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Service returned invalid JSON") from exc


def validate_response(response: dict[str, Any], *, expected_output_path: str) -> None:
    status = response.get("status")
    output_path = response.get("output_path")
    if status != "completed":
        raise RuntimeError(f"Unexpected response status: {status!r}")
    if output_path != expected_output_path:
        raise RuntimeError(f"Unexpected output_path: {output_path!r}, expected {expected_output_path!r}")


def verify_output_file(output_path: str) -> None:
    path = Path(output_path)
    if not path.is_file():
        raise RuntimeError(f"Output file does not exist: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Output file is empty: {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    try:
        status_code, response = post_json(args.url, payload, timeout=args.timeout)
        if not 200 <= status_code < 300:
            raise RuntimeError(f"Unexpected HTTP status code: {status_code}")
        validate_response(response, expected_output_path=args.output_path)
        verify_output_file(args.output_path)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"OK: wrote non-empty output video to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
