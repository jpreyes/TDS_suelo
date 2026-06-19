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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: light;
  --ink: #17212b;
  --muted: #607080;
  --line: #d6dee6;
  --panel: #ffffff;
  --soft: #f4f7fa;
  --brand: #205c6b;
  --brand-strong: #174754;
  --danger: #9f2f2f;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background: #eef3f6;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
}}
a {{ color: var(--brand); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 24px;
  background: #102832;
  color: #ffffff;
}}
.brand {{ font-weight: 700; letter-spacing: 0; }}
.nav {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.nav a {{ color: #dcecf0; font-size: 0.92rem; }}
.shell {{ max-width: 1220px; margin: 0 auto; padding: 24px; }}
.page-head {{
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 16px;
  align-items: end;
  margin-bottom: 18px;
}}
h1 {{ margin: 0; font-size: 1.8rem; }}
h2 {{ margin: 0 0 12px; font-size: 1.12rem; }}
h3 {{ margin: 18px 0 8px; font-size: 0.98rem; }}
.note {{ color: var(--muted); margin: 6px 0 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 16px 0; }}
.card, .panel {{
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  box-shadow: 0 1px 2px rgba(16, 40, 50, 0.05);
}}
.card {{ padding: 14px; }}
.card span {{ display: block; color: var(--muted); font-size: 0.82rem; }}
.card strong {{ display: block; margin-top: 6px; font-size: 1.15rem; }}
.panel {{ padding: 18px; margin: 16px 0; }}
.form-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
label {{ display: block; margin: 0 0 5px; font-weight: 650; font-size: 0.88rem; }}
input[type=text], input[type=password], input[type=number], select {{
  width: 100%;
  padding: 9px 10px;
  border: 1px solid #b9c5cf;
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
}}
.check-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 12px 0 2px; }}
.check-row label {{ font-weight: 500; }}
.actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
button {{
  padding: 9px 13px;
  border: 1px solid var(--brand);
  background: var(--brand);
  color: #ffffff;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 650;
}}
button.secondary {{ background: #ffffff; color: var(--brand); }}
button.danger {{ border-color: var(--danger); background: #ffffff; color: var(--danger); }}
.status {{ display: inline-flex; align-items: center; gap: 7px; padding: 6px 9px; border-radius: 999px; background: #e8f3ee; color: #1f6a46; font-weight: 650; }}
.status.idle {{ background: #edf1f5; color: #455665; }}
pre {{
  white-space: pre-wrap;
  background: #101923;
  color: #e5edf4;
  border-radius: 8px;
  padding: 14px;
  max-height: 460px;
  overflow: auto;
  font-size: 0.84rem;
}}
@media (max-width: 760px) {{
  .topbar, .page-head {{ grid-template-columns: 1fr; align-items: start; }}
  .shell {{ padding: 16px; }}
}}
</style>
</head>
<body>
<div class="topbar">
  <div class="brand">TSD-Suelo</div>
  <nav class="nav">
    <a href="/results_report.html">Reporte</a>
    <a href="/admin">Admin</a>
    <a href="/run.log">run.log</a>
  </nav>
</div>
<main class="shell">
{body}
</main>
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


def _forward_command(form: dict[str, str], defaults: PipelineConfig) -> list[str]:
    output_dir = form.get("output_dir") or str(defaults.output_dir)
    top_n = str(max(1, int(form.get("top_n") or 50)))
    command = [
        sys.executable,
        "-m",
        "tsd_suelo.cli",
        "forward",
        "--output-dir",
        output_dir,
        "--top-n",
        top_n,
    ]
    mask_geojson = form.get("mask_geojson") or (str(defaults.mask_geojson) if defaults.mask_geojson else "")
    if mask_geojson:
        command.extend(["--mask-geojson", mask_geojson])
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
            if parsed.path == "/admin/forward":
                ok, message = state.start("tsd-suelo forward", _forward_command(form, config.pipeline), config.repo_dir)
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
<section class="page-head">
  <div>
    <h1>Admin</h1>
    <p class="note">Admin deshabilitado. Inicia el servidor con <code>--admin-token</code> o define <code>TSD_SUELO_ADMIN_TOKEN</code>.</p>
  </div>
</section>
<section class="panel">
  <a href="/results_report.html">Volver al reporte</a>
</section>
"""
                return _html_page("TSD-Suelo Admin", body)
            if not authorized:
                body = """
<section class="page-head">
  <div>
    <h1>Admin</h1>
    <p class="note">Ingresa el token admin para ver controles de ejecucion.</p>
  </div>
</section>
<section class="panel">
  <form method="get" action="/admin">
    <label>Token</label>
    <input type="password" name="token" autocomplete="current-password">
    <div class="actions">
      <button type="submit">Entrar</button>
      <a href="/results_report.html">Volver al reporte</a>
    </div>
  </form>
</section>
"""
                return _html_page("TSD-Suelo Admin", body)

            snap = state.snapshot()
            defaults = config.pipeline
            running = bool(snap.get("running"))
            status_class = "" if running else " idle"
            status_text = "corriendo" if running else "sin proceso activo"
            elapsed = snap.get("elapsed_s")
            forward_ready = all(
                (output_dir / name).exists()
                for name in (
                    "geo_targets_observed.parquet",
                    "geo_residuals.parquet",
                    "latent_modes.parquet",
                    "kozyrev_graph_fields.parquet",
                    "fault_candidates.parquet",
                )
            )
            body = f"""
<section class="page-head">
  <div>
    <h1>Consola TSD-Suelo</h1>
    <p class="note">Ejecucion operacional sobre parquets observados y productos derivados.</p>
  </div>
  <div class="status{status_class}">{html.escape(status_text)}</div>
</section>

<div class="grid">
  <div class="card"><span>Proceso</span><strong>{html.escape(str(snap.get("label") or "ninguno"))}</strong></div>
  <div class="card"><span>Tiempo activo</span><strong>{html.escape(str(elapsed if elapsed is not None else "-"))} s</strong></div>
  <div class="card"><span>Ultimo retorno</span><strong>{html.escape(str(snap.get("last_returncode")))}</strong></div>
  <div class="card"><span>Forward listo</span><strong>{html.escape("si" if forward_ready else "no")}</strong></div>
</div>

<section class="panel">
  <h2>Forward condicionado</h2>
  <p class="note">Recalcula solo <code>compatible_dynamics</code>, perfiles de condicionamiento y reporte desde parquets observados existentes. No relee H5.</p>
  <form method="post" action="/admin/forward">
    <input type="hidden" name="token" value="{html.escape(token)}">
    <div class="form-grid">
      <div>
        <label>output-dir</label>
        <input type="text" name="output_dir" value="{html.escape(str(defaults.output_dir))}">
      </div>
      <div>
        <label>top-n reporte</label>
        <input type="number" name="top_n" min="1" value="80">
      </div>
      <div>
        <label>mask-geojson opcional</label>
        <input type="text" name="mask_geojson" value="{html.escape(str(defaults.mask_geojson or ""))}">
      </div>
    </div>
    <div class="actions">
      <button type="submit">Ejecutar forward</button>
      <a href="/forward_conditioning_template.json">Contrato forward</a>
      <a href="/forward_conditioning_profiles.parquet">Perfiles parquet</a>
    </div>
  </form>
</section>

<section class="panel">
  <h2>Build completo</h2>
  <form method="post" action="/admin/build">
    <input type="hidden" name="token" value="{html.escape(token)}">
    <div class="form-grid">
      <div>
        <label>records-dir</label>
        <input type="text" name="records_dir" value="{html.escape(str(defaults.records_dir))}">
      </div>
      <div>
        <label>flatfiles-dir</label>
        <input type="text" name="flatfiles_dir" value="{html.escape(str(defaults.flatfiles_dir))}">
      </div>
      <div>
        <label>output-dir</label>
        <input type="text" name="output_dir" value="{html.escape(str(defaults.output_dir))}">
      </div>
      <div>
        <label>workers</label>
        <input type="number" name="workers" min="1" value="{defaults.workers}">
      </div>
      <div>
        <label>progress-every</label>
        <input type="number" name="progress_every" min="1" value="{defaults.progress_every}">
      </div>
      <div>
        <label>analysis-mode</label>
        <select name="analysis_mode">
          <option value="both" {'selected' if defaults.analysis_mode == 'both' else ''}>both: espacial + espectral</option>
          <option value="spatial" {'selected' if defaults.analysis_mode == 'spatial' else ''}>spatial: grilla espacial</option>
          <option value="spectral" {'selected' if defaults.analysis_mode == 'spectral' else ''}>spectral: red dinamica en frecuencia</option>
        </select>
      </div>
    </div>
    <div class="check-row">
      <label><input type="checkbox" name="reuse_products" checked> Reusar productos existentes</label>
      <label><input type="checkbox" name="skip_psa"> Omitir PSA</label>
      <label><input type="checkbox" name="no_chile_mask"> Sin mascara Chile</label>
    </div>
    <div class="actions">
      <button type="submit">Lanzar build</button>
    </div>
  </form>
</section>

<section class="panel">
  <h2>Mantenimiento</h2>
  <div class="actions">
    <form method="post" action="/admin/git-pull">
      <input type="hidden" name="token" value="{html.escape(token)}">
      <button class="secondary" type="submit">git pull --ff-only</button>
    </form>
    <form method="post" action="/admin/install">
      <input type="hidden" name="token" value="{html.escape(token)}">
      <button class="secondary" type="submit">pip install -e .</button>
    </form>
    <form method="post" action="/admin/stop">
      <input type="hidden" name="token" value="{html.escape(token)}">
      <button class="danger" type="submit">Detener proceso</button>
    </form>
  </div>
</section>

<section class="panel">
  <h2>Logs</h2>
  <p><a href="/api/log?kind=run&token={html.escape(token)}">run.log</a> | <a href="/api/log?kind=admin&token={html.escape(token)}">admin log</a> | <a href="/api/status?token={html.escape(token)}">status JSON</a></p>
  <h3>run.log</h3>
  <pre>{html.escape(_tail(output_dir / "run.log"))}</pre>
  <h3>admin log</h3>
  <pre>{html.escape(_tail(state.admin_log))}</pre>
</section>
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
