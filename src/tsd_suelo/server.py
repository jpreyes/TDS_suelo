from __future__ import annotations

from dataclasses import dataclass
import html
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import PipelineConfig


def _tail(path: Path, max_bytes: int = 50000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _html_page(title: str, body: str) -> bytes:
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1d252c; }}
label {{ display: block; margin: 10px 0 4px; font-weight: 600; }}
input[type=text], input[type=password], input[type=number], select {{ width: min(760px, 100%); padding: 7px; }}
button {{ margin: 10px 8px 10px 0; padding: 8px 12px; border: 1px solid #8796a5; background: #f5f8fa; border-radius: 4px; cursor: pointer; }}
button.danger {{ border-color: #b33; color: #8d1f1f; }}
pre {{ white-space: pre-wrap; background: #111827; color: #e5e7eb; padding: 12px; max-height: 460px; overflow: auto; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }}
.card {{ border: 1px solid #d5dde5; border-radius: 6px; padding: 10px; background: #f8fafc; }}
.note {{ color: #53606d; }}
</style>
</head>
<body>
{body}
</body>
</html>""".encode("utf-8")


@dataclass
class ServeConfig:
    pipeline: PipelineConfig
    repo_dir: Path
    admin_token: str | None


class ProcessState:
    def __init__(self, admin_log: Path) -> None:
        self.admin_log = admin_log
        self.process: subprocess.Popen[bytes] | None = None
        self.label: str | None = None
        self.started_at: float | None = None
        self.last_returncode: int | None = None
        self.last_finished_at: float | None = None

    def running(self) -> bool:
        if self.process is None:
            return False
        code = self.process.poll()
        if code is None:
            return True
        self.last_returncode = code
        self.last_finished_at = self.last_finished_at or time.time()
        self.process = None
        return False

    def start(self, label: str, command: list[str], cwd: Path) -> tuple[bool, str]:
        if self.running():
            return False, f"Ya hay un proceso corriendo: {self.label}"
        self.admin_log.parent.mkdir(parents=True, exist_ok=True)
        with self.admin_log.open("ab") as log:
            log.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {label}\n".encode("utf-8"))
            log.write((" ".join(command) + "\n").encode("utf-8", errors="replace"))
        log_handle = self.admin_log.open("ab")
        self.process = subprocess.Popen(command, cwd=str(cwd), stdout=log_handle, stderr=subprocess.STDOUT)
        self.label = label
        self.started_at = time.time()
        self.last_returncode = None
        self.last_finished_at = None
        return True, f"Iniciado: {label}"

    def stop(self) -> tuple[bool, str]:
        if not self.running() or self.process is None:
            return False, "No hay proceso activo."
        self.process.terminate()
        return True, f"Se envio terminate a {self.label}."

    def snapshot(self) -> dict[str, Any]:
        is_running = self.running()
        return {
            "running": is_running,
            "label": self.label,
            "started_at": self.started_at,
            "elapsed_s": round(time.time() - self.started_at, 1) if is_running and self.started_at else None,
            "last_returncode": self.last_returncode,
            "last_finished_at": self.last_finished_at,
        }


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on", "si", "sí"}


def _build_command(form: dict[str, str], defaults: PipelineConfig) -> list[str]:
    records_dir = form.get("records_dir") or str(defaults.records_dir)
    flatfiles_dir = form.get("flatfiles_dir") or str(defaults.flatfiles_dir)
    output_dir = form.get("output_dir") or str(defaults.output_dir)
    workers = str(max(1, int(form.get("workers") or defaults.workers or 1)))
    progress_every = str(max(1, int(form.get("progress_every") or defaults.progress_every or 500)))
    command = [
        sys.executable,
        "-m",
        "tsd_suelo.cli",
        "build",
        "--records-dir",
        records_dir,
        "--flatfiles-dir",
        flatfiles_dir,
        "--output-dir",
        output_dir,
        "--workers",
        workers,
        "--progress-every",
        progress_every,
    ]
    analysis_mode = form.get("analysis_mode") or defaults.analysis_mode
    if analysis_mode in {"spatial", "spectral", "both"}:
        command.extend(["--analysis-mode", analysis_mode])
    if _truthy(form.get("reuse_products")):
        command.append("--reuse-products")
    if _truthy(form.get("skip_psa")):
        command.append("--skip-psa")
    if _truthy(form.get("no_chile_mask")):
        command.append("--no-chile-mask")
    return command


def _handler_factory(config: ServeConfig, state: ProcessState):
    output_dir = config.pipeline.output_dir

    class TsdHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output_dir), **kwargs)

        def _send_bytes(self, payload: bytes, content_type: str = "text/html; charset=utf-8", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            self._send_bytes(json.dumps(payload, indent=2).encode("utf-8"), "application/json; charset=utf-8", status)

        def _query(self) -> dict[str, str]:
            parsed = urlparse(self.path)
            return {key: values[-1] for key, values in parse_qs(parsed.query).items()}

        def _authorized(self, form: dict[str, str] | None = None) -> bool:
            if not config.admin_token:
                return False
            token = self.headers.get("X-TSD-Admin-Token") or self._query().get("token") or (form or {}).get("token")
            return bool(token) and hmac.compare_digest(str(token), config.admin_token)

        def _read_form(self) -> dict[str, str]:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            return {key: values[-1] for key, values in parse_qs(raw).items()}

        def _redirect_admin(self, token: str | None = None) -> None:
            suffix = f"?token={token}" if token else ""
            self.send_response(303)
            self.send_header("Location", f"/admin{suffix}")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                target = "/results_report.html" if (output_dir / "results_report.html").exists() else "/admin"
                self.send_response(302)
                self.send_header("Location", target)
                self.end_headers()
                return
            if parsed.path == "/admin":
                self._send_bytes(self._admin_html())
                return
            if parsed.path == "/api/status":
                if not self._authorized():
                    self._send_json({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                    return
                self._send_json(state.snapshot())
                return
            if parsed.path == "/api/log":
                if not self._authorized():
                    self._send_bytes(b"unauthorized\n", "text/plain; charset=utf-8", HTTPStatus.UNAUTHORIZED)
                    return
                kind = self._query().get("kind", "run")
                path = state.admin_log if kind == "admin" else output_dir / "run.log"
                self._send_bytes(_tail(path).encode("utf-8"), "text/plain; charset=utf-8")
                return
            return super().do_GET()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            form = self._read_form()
            if parsed.path.startswith("/admin/") and not self._authorized(form):
                self._send_bytes(_html_page("No autorizado", "<h1>No autorizado</h1><p>Token admin invalido.</p>"), status=HTTPStatus.UNAUTHORIZED)
                return
            if parsed.path == "/admin/build":
                ok, message = state.start("tsd-suelo build", _build_command(form, config.pipeline), config.repo_dir)
                self._redirect_admin(form.get("token"))
                return
            if parsed.path == "/admin/git-pull":
                ok, message = state.start("git pull --ff-only", ["git", "pull", "--ff-only"], config.repo_dir)
                self._redirect_admin(form.get("token"))
                return
            if parsed.path == "/admin/install":
                ok, message = state.start("pip install -e .", [sys.executable, "-m", "pip", "install", "-e", "."], config.repo_dir)
                self._redirect_admin(form.get("token"))
                return
            if parsed.path == "/admin/stop":
                ok, message = state.stop()
                self._redirect_admin(form.get("token"))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _admin_html(self) -> bytes:
            query = self._query()
            token = query.get("token", "")
            authorized = self._authorized({"token": token})
            if not config.admin_token:
                body = """
<h1>TSD-Suelo Admin</h1>
<p class="note">Admin deshabilitado. Inicia el servidor con <code>--admin-token</code> o define <code>TSD_SUELO_ADMIN_TOKEN</code>.</p>
<p><a href="/results_report.html">Volver al reporte</a></p>
"""
                return _html_page("TSD-Suelo Admin", body)
            if not authorized:
                body = """
<h1>TSD-Suelo Admin</h1>
<p class="note">Ingresa el token admin para ver controles de ejecucion.</p>
<form method="get" action="/admin">
<label>Token</label>
<input type="password" name="token" autocomplete="current-password">
<button type="submit">Entrar</button>
</form>
<p><a href="/results_report.html">Volver al reporte</a></p>
"""
                return _html_page("TSD-Suelo Admin", body)

            snap = state.snapshot()
            defaults = config.pipeline
            body = f"""
<h1>TSD-Suelo Admin</h1>
<p><a href="/results_report.html">Reporte</a> | <a href="/kozyrev_heatmap.geojson">kozyrev_heatmap.geojson</a> | <a href="/run.log">run.log</a></p>
<div class="grid">
  <div class="card"><strong>Proceso activo</strong><br>{html.escape(str(snap.get("running")))}</div>
  <div class="card"><strong>Comando</strong><br>{html.escape(str(snap.get("label") or ""))}</div>
  <div class="card"><strong>Ultimo retorno</strong><br>{html.escape(str(snap.get("last_returncode")))}</div>
</div>

<h2>Build</h2>
<form method="post" action="/admin/build">
<input type="hidden" name="token" value="{html.escape(token)}">
<label>records-dir</label>
<input type="text" name="records_dir" value="{html.escape(str(defaults.records_dir))}">
<label>flatfiles-dir</label>
<input type="text" name="flatfiles_dir" value="{html.escape(str(defaults.flatfiles_dir))}">
<label>output-dir</label>
<input type="text" name="output_dir" value="{html.escape(str(defaults.output_dir))}">
<label>workers</label>
<input type="number" name="workers" min="1" value="{defaults.workers}">
<label>progress-every</label>
<input type="number" name="progress_every" min="1" value="{defaults.progress_every}">
<label>analysis-mode</label>
<select name="analysis_mode">
  <option value="both" {'selected' if defaults.analysis_mode == 'both' else ''}>both: espacial + espectral</option>
  <option value="spatial" {'selected' if defaults.analysis_mode == 'spatial' else ''}>spatial: grilla espacial</option>
  <option value="spectral" {'selected' if defaults.analysis_mode == 'spectral' else ''}>spectral: red dinamica en frecuencia</option>
</select>
<p>
<label><input type="checkbox" name="reuse_products" checked> Reusar productos existentes</label>
<label><input type="checkbox" name="skip_psa"> Omitir PSA</label>
<label><input type="checkbox" name="no_chile_mask"> Sin mascara Chile</label>
</p>
<button type="submit">Lanzar build</button>
</form>

<h2>Mantenimiento</h2>
<form method="post" action="/admin/git-pull" style="display:inline">
<input type="hidden" name="token" value="{html.escape(token)}">
<button type="submit">git pull --ff-only</button>
</form>
<form method="post" action="/admin/install" style="display:inline">
<input type="hidden" name="token" value="{html.escape(token)}">
<button type="submit">pip install -e .</button>
</form>
<form method="post" action="/admin/stop" style="display:inline">
<input type="hidden" name="token" value="{html.escape(token)}">
<button class="danger" type="submit">Detener proceso</button>
</form>

<h2>Logs</h2>
<p><a href="/api/log?kind=run&token={html.escape(token)}">run.log</a> | <a href="/api/log?kind=admin&token={html.escape(token)}">admin log</a> | <a href="/api/status?token={html.escape(token)}">status JSON</a></p>
<h3>run.log</h3>
<pre>{html.escape(_tail(output_dir / "run.log"))}</pre>
<h3>admin log</h3>
<pre>{html.escape(_tail(state.admin_log))}</pre>
"""
            return _html_page("TSD-Suelo Admin", body)

    return TsdHandler


def serve(config: PipelineConfig, host: str, port: int, repo_dir: Path, admin_token: str | None = None) -> int:
    cfg = config.resolved()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    token = admin_token or os.environ.get("TSD_SUELO_ADMIN_TOKEN")
    serve_config = ServeConfig(pipeline=cfg, repo_dir=repo_dir.expanduser().resolve(), admin_token=token)
    state = ProcessState(cfg.output_dir / "admin_process.log")
    handler = _handler_factory(serve_config, state)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving TSD-Suelo on http://{host}:{port}/ from {cfg.output_dir}")
    if token:
        print("Admin habilitado en /admin con token.")
    else:
        print("Admin deshabilitado: define TSD_SUELO_ADMIN_TOKEN o usa --admin-token.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()
    return 0
