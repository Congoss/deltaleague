import json
import os
import sys
from pathlib import Path
import time
import uuid
import webbrowser
import httpx
import typer
from .local_commands import run, evaluate
from .compare import compare
from .artifacts import build_sparse, validate_manifest, file_hash

app = typer.Typer(help="Delta League: submit a patch, prove an improvement.", no_args_is_help=True)
models_app = typer.Typer()
model_app = typer.Typer()
patch_app = typer.Typer()
submission_app = typer.Typer()
competition_app = typer.Typer(help="Discover model competitions and initialize a reproducible workspace.")
app.add_typer(models_app, name="models")
app.add_typer(model_app, name="model")
app.add_typer(patch_app, name="patch")
app.add_typer(submission_app, name="submission")
app.add_typer(competition_app, name="competition")
API = "https://deltaleague.ai"


def config_path():
    return Path(os.getenv("DELTA_CONFIG", str(Path.home() / ".config/deltaleague/auth.json")))


def credentials():
    p = config_path()
    return json.loads(p.read_text()) if p.exists() else {}


def client():
    saved = credentials()
    token = os.getenv("DELTA_TOKEN") or (saved.get("token") if saved.get("api") == API else None)
    return httpx.Client(base_url=API, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=60)


def call(method, path, **kwargs):
    try:
        with client() as c:
            r = c.request(method, "/api/v1/" + path, **kwargs)
    except httpx.HTTPError:
        typer.echo("Could not reach the Delta League API. Check the connection and --api URL, then retry.", err=True)
        raise typer.Exit(1)
    if not r.is_success:
        if r.status_code in (401, 403):
            typer.echo("Authentication or permission required. Run `delta login`, then check your account access.", err=True)
        else:
            typer.echo(f"API request failed (HTTP {r.status_code}).", err=True)
            try:
                show(r.json())
            except ValueError:
                pass
        raise typer.Exit(1)
    return r.json()


def show(value):
    typer.echo(json.dumps(value, indent=2))


@app.callback()
def options(api: str = typer.Option(os.getenv("DELTA_API", "https://deltaleague.ai"), help="Delta League API origin")):
    global API
    API = api.rstrip("/")
    if not API.startswith("https://") and not API.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise typer.BadParameter("Use HTTPS outside localhost")


@app.command()
def login(no_browser: bool = typer.Option(False, "--no-browser", help="Print the authorization URL without opening a browser.")):
    data = call("POST", "cli-auth")
    typer.echo(f"Authorize this CLI: {data['verification_url']}")
    if data.get("user_code"):
        typer.echo(f"Then enter this code in the browser: {data['user_code']}")
        typer.echo("Only enter it on a page you opened yourself. Never share this code.")
    if not no_browser:
        webbrowser.open(data["verification_url"])
    for _ in range(120):
        time.sleep(data["interval"])
        result = call("POST", f"cli-auth/{data['id']}", json={"secret": data["secret"]})
        if "token" in result:
            path = config_path()
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump({"api": API, "token": result["token"]}, f)
            typer.echo("Signed in.")
            return
    raise typer.Exit(1)


@app.command()
def logout():
    path = config_path()
    try:
        # Best effort: an expired token or an unreachable API must not keep the local credential.
        with client() as http:
            http.post("/api/v1/logout")
    except (httpx.HTTPError, OSError, ValueError):
        pass
    finally:
        path.unlink(missing_ok=True)
    typer.echo("Signed out.")


@app.command()
def whoami():
    show(call("GET", "me"))


@app.command()
def quota():
    show(call("GET", "me"))


@models_app.command("list")
def models_list():
    show(call("GET", "models"))


def download_artifact(artifact, output, force=False):
    name = artifact["name"]
    if Path(name).name != name:
        raise typer.BadParameter("Invalid artifact filename from server")
    path = output / name
    if path.exists():
        if path.stat().st_size == artifact["size_bytes"] and file_hash(path) == artifact["sha256"]:
            typer.echo(f"Reusing verified {name}", err=True)
            return
        if not force:
            raise typer.BadParameter(f"{path} exists with different contents. Use a new directory or --force to replace it.")
    temporary = path.with_name(path.name + ".part")
    typer.echo(f"Downloading {name} ({artifact['size_bytes'] / 1024**2:.1f} MiB)", err=True)
    try:
        with httpx.stream("GET", artifact["url"], timeout=300) as response:
            response.raise_for_status()
            total = 0
            with temporary.open("wb") as target, typer.progressbar(length=artifact["size_bytes"], label=name, file=sys.stderr) as progress:
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > artifact["size_bytes"]:
                        raise ValueError("download_size")
                    target.write(chunk)
                    progress.update(len(chunk))
        if temporary.stat().st_size != artifact["size_bytes"] or file_hash(temporary) != artifact["sha256"]:
            raise ValueError("download_integrity")
        temporary.replace(path)
    except (httpx.HTTPError, OSError, ValueError):
        typer.echo(f"Download failed or integrity check failed for {name}. Retry the command; verified files are reused.", err=True)
        raise typer.Exit(1)
    finally:
        temporary.unlink(missing_ok=True)


def pull_model(version, output, force=False):
    data = call("GET", f"model-versions/{version}")
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "delta-model.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("id") != data["id"]:
            raise typer.BadParameter("This directory contains a different parent version. Choose a new output directory.")
    for artifact in data["files"]:
        download_artifact(artifact, output, force=force)
    # Signed download URLs are temporary credentials, not reproducibility metadata.
    data["files"] = [{k: f[k] for k in ("name", "sha256", "size_bytes")} for f in data["files"]]
    metadata_path.write_text(json.dumps(data, indent=2))
    return data


@model_app.command("pull")
def model_pull(version: str, output: Path = typer.Option(Path("model")), force: bool = False):
    """Download a parent checkpoint. Reuses hash-verified files; --force replaces damaged/modified files."""
    pull_model(version, output, force=force)
    typer.echo(str(output))


@competition_app.command("list")
def competitions_list():
    rows = []
    for model in call("GET", "models"):
        board = call("GET", "leaderboard", params={"model": model["id"]})
        parent = next((r["version_id"] for r in board["rows"] if r["parent_id"] is None), None)
        rows.append({"model_id": model["id"], "name": model["name"], "config_id": board["config_id"],
                     "default_parent_version_id": parent, "ready": bool(board["config_id"] and parent)})
    show(rows)


@competition_app.command("init")
def competition_init(model: str, output: Path = typer.Option(Path("experiment")), parent: str | None = None):
    """Pull a model parent, public validation data, and the active evaluation configuration."""
    board = call("GET", "leaderboard", params={"model": model})
    if not board["config_id"]:
        raise typer.BadParameter("This model has no active competition. Run `delta competition list` to find available models.")
    parent = parent or next((r["version_id"] for r in board["rows"] if r["parent_id"] is None), None)
    if not parent:
        raise typer.BadParameter("No original checkpoint is registered yet. Choose --parent VERSION_ID or wait for setup.")
    version = call("GET", f"model-versions/{parent}")
    if version["model_id"] != model:
        raise typer.BadParameter("The selected parent belongs to a different model.")
    config = call("GET", f"evaluation-configs/{board['config_id']}")
    if not config.get("validation_download") or not config.get("validation_sha256"):
        raise typer.BadParameter("The server does not provide a verifiable public validation download for this configuration yet.")
    identity = {"api": API, "model_id": model, "parent_version_id": parent, "config_id": config["id"]}
    workspace_path = output / "delta-workspace.json"
    if workspace_path.exists():
        previous = json.loads(workspace_path.read_text())
        if any(previous.get(k) != value for k, value in identity.items()):
            raise typer.BadParameter("This workspace belongs to another model, parent, API, or evaluation version. Use a new output directory.")
    pull_model(parent, output / "model")
    download_artifact({"name": "validation.json", "url": config["validation_download"],
                       "sha256": config["validation_sha256"], "size_bytes": config["validation_size_bytes"]}, output)
    public_config = {k: v for k, v in config.items() if k != "validation_download"}
    (output / "evaluation.json").write_text(json.dumps(public_config, indent=2))
    workspace_path.write_text(json.dumps(identity, indent=2))
    show({**identity, "directory": str(output), "parent_directory": str(output / "model"),
          "validation_file": str(output / "validation.json"), "validation_benchmarks": config["validation_names"]})


@patch_app.command("build")
def patch_build(parent: Path = typer.Option(...), modified: Path = typer.Option(...), output: Path = typer.Option(Path("submission"))):
    """Build an exact sparse replacement patch. Parent is a `delta model pull` directory."""
    metadata = json.loads((parent / "delta-model.json").read_text())
    manifest = build_sparse(parent / "model.safetensors", modified, output, metadata["id"], metadata["content_hash"], metadata["model_id"])
    typer.echo(f"Built {len(manifest['tensors'])} modified tensors in {output}")


@patch_app.command("inspect")
def inspect(path: Path):
    show(json.loads((path / "manifest.json").read_text()))


@app.command()
def validate(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    validate_manifest(manifest)
    for f in manifest["artifacts"]:
        if file_hash(path / f["name"]) != f["sha256"] or (path / f["name"]).stat().st_size != f["size_bytes"]:
            raise typer.BadParameter("Artifact integrity failure")
    typer.echo("Manifest and artifact hashes are valid. Server checks parent compatibility and tensors.")


@app.command()
def submit(path: Path, title: str = "Weight modification", pr: str | None = None):
    validate(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if pr:
        manifest["research"] = {"pr_url": pr}
    # Persist intent before posting: retries after a lost response reuse the same idempotency key.
    state_path = path / ".delta-upload.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    fingerprint = __import__("hashlib").sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if state.get("fingerprint") != fingerprint or state.get("api") != API:
        upload = call("POST", "upload-sessions", json={"manifest": manifest})
        for f in upload["files"]:
            file = path / f["name"]
            with file.open("rb") as stream:
                response = httpx.put(f["url"], content=stream, headers={"Content-Length": str(file.stat().st_size)}, timeout=600)
                response.raise_for_status()
        state = {"upload_id": upload["id"], "idempotency_key": str(uuid.uuid4()), "fingerprint": fingerprint, "api": API}
        state_path.write_text(json.dumps(state))
    show(call("POST", "submissions", json={"upload_id": state["upload_id"], "title": title}, headers={"Idempotency-Key": state["idempotency_key"]}))


@submission_app.command("status")
def status(submission: str, watch: bool = False):
    while True:
        data = call("GET", f"submissions/{submission}")
        show(data)
        if not watch or data["state"] in {"verified", "rejected", "failed", "canceled"}:
            return
        time.sleep(10)


@submission_app.command("results")
def results(submission: str, json_output: bool = typer.Option(False, "--json")):
    show(call("GET", f"submissions/{submission}")["result"])


@app.command()
def leaderboard(model: str | None = None):
    show(call("GET", "leaderboard", params={"model": model} if model else {}))


app.command("run")(run)
app.command("evaluate")(evaluate)
app.command("compare")(compare)
