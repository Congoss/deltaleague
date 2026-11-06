import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid
import webbrowser
import httpx
import typer
from .local_commands import run, evaluate
from .public_eval.cli import app as public_eval_app
from .compare import compare
from .artifacts import build_sparse, validate_manifest, file_hash, verify_artifact_bytes

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
app.add_typer(public_eval_app, name="public-eval")
API = "https://deltaleague.ai"
POLL_RETRIES = 5


def config_path():
    return Path(os.getenv("DELTA_CONFIG", str(Path.home() / ".config/deltaleague/auth.json")))


def credentials():
    p = config_path()
    return json.loads(p.read_text()) if p.exists() else {}


def client():
    saved = credentials()
    token = os.getenv("DELTA_TOKEN") or (saved.get("token") if saved.get("api") == API else None)
    return httpx.Client(base_url=API, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=60)


def retryable(response):
    return response.status_code == 429 or response.status_code >= 500


def request(method, path, retries=0, **kwargs):
    """One attempt, or a bounded retry of transport errors, 429s and 5xx responses."""
    bound = max(retries, 0)
    with client() as http:
        for attempt in range(bound + 1):
            try:
                response = http.request(method, "/api/v1/" + path, **kwargs)
            except httpx.HTTPError:
                if attempt == bound:
                    raise
            else:
                if attempt == bound or not retryable(response):
                    return response
            time.sleep(2)


def call(method, path, retries=0, **kwargs):
    try:
        r = request(method, path, retries, **kwargs)
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
        result = call("POST", f"cli-auth/{data['id']}", json={"secret": data["secret"]}, retries=POLL_RETRIES)
        if "token" in result:
            path = config_path()
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump({"api": API, "token": result["token"]}, f)
            typer.echo("Signed in.")
            return
    typer.echo("Sign-in timed out. Run `delta login` again.", err=True)
    raise typer.Exit(1)


@app.command()
def logout():
    call("POST", "logout")
    config_path().unlink(missing_ok=True)
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


def publish_directory(staging: Path, output: Path):
    """Replace output with staging only after the new directory is complete."""
    if not output.exists():
        staging.rename(output)
        return
    if not any(output.iterdir()):
        output.rmdir()
        staging.rename(output)
        return
    displaced = output.with_name(f".{output.name}.replaced")
    if displaced.exists():
        shutil.rmtree(displaced)
    output.rename(displaced)
    try:
        staging.rename(output)
    except OSError:
        displaced.rename(output)
        raise
    shutil.rmtree(displaced)


@patch_app.command("build")
def patch_build(parent: Path = typer.Option(...), modified: Path = typer.Option(...), output: Path = typer.Option(Path("submission")),
                force: bool = typer.Option(False, "--force", help="Replace an existing output directory.")):
    """Build an exact sparse replacement patch. Parent is a `delta model pull` directory."""
    staging = None
    try:
        if output.exists() and any(output.iterdir()) and not force:
            raise typer.BadParameter(f"{output} already exists. Choose a new directory or pass --force.")
        metadata = json.loads((parent / "delta-model.json").read_text())
        if not isinstance(metadata, dict) or any(key not in metadata for key in ("id", "content_hash", "model_id", "files")):
            raise ValueError("invalid parent metadata")
        verify_artifact_bytes(parent, metadata["files"])
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        manifest = build_sparse(parent / "model.safetensors", modified, staging, metadata["id"], metadata["content_hash"], metadata["model_id"])
        publish_directory(staging, output)
        staging = None
    except typer.BadParameter:
        raise
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from None
    finally:
        if staging and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
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


def fail(message):
    typer.echo(message, err=True)
    raise typer.Exit(1)


def write_upload_state(path: Path, state: dict):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".delta-upload-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        os.chmod(path, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def upload_url_allowed(url):
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return False
    if parsed.username or parsed.password or parsed.hostname is None:
        return False
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"} and port is not None


def put_artifact(directory: Path, name: str, url):
    """Upload one manifest artifact. Errors name the artifact and never the signed URL."""
    if not isinstance(name, str) or Path(name).name != name:
        fail("Upload failed: the server returned an unexpected filename.")
    if not upload_url_allowed(url):
        fail(f"Upload failed for {name}: the upload URL is not allowed.")
    file = directory / name
    try:
        with file.open("rb") as stream:
            response = httpx.put(url, content=stream, headers={"Content-Length": str(file.stat().st_size)}, timeout=600, follow_redirects=False)
    except (httpx.HTTPError, httpx.InvalidURL, OSError):
        fail(f"Upload failed for {name}. Check the connection and rerun the same submit command.")
    if response.status_code < 200 or response.status_code >= 300:
        fail(f"Upload failed for {name} (HTTP {response.status_code}).")


def pending_upload_files(manifest: dict, upload):
    """Map URLs to manifest names without opening server-selected paths."""
    if not isinstance(upload, dict) or not isinstance(upload.get("id"), str) or not upload["id"]:
        fail("Upload failed: the server returned an invalid upload session.")
    files = upload.get("files")
    if not isinstance(files, list):
        fail("Upload failed: the server returned an invalid upload session.")
    names = {artifact["name"] for artifact in manifest["artifacts"]}
    urls = {}
    for item in files:
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"] in names:
            if item["name"] in urls:
                fail("Upload failed: duplicate artifact URLs in upload session.")
            urls[item["name"]] = item.get("url")
    return [{"name": artifact["name"], "url": urls.get(artifact["name"])}
            for artifact in manifest["artifacts"]]


@app.command()
def submit(path: Path, title: str = "Weight modification", pr: str | None = None):
    validate(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if pr:
        manifest["research"] = {"pr_url": pr}
    # Persist intent before posting: retries after a lost response reuse the same idempotency key.
    state_path = path / ".delta-upload.json"
    try:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    pending = state.get("pending_files", [])
    reusable = (state.get("fingerprint") == fingerprint and state.get("api") == API
                and isinstance(state.get("upload_id"), str) and bool(state["upload_id"])
                and isinstance(state.get("idempotency_key"), str) and bool(state["idempotency_key"])
                and isinstance(pending, list))
    if pending and reusable:
        expires = state.get("expires_at", 0)
        reusable = type(expires) in (int, float) and expires > time.time()
    if not reusable:
        expires_at = time.time() + 3600
        upload = call("POST", "upload-sessions", json={"manifest": manifest})
        pending = pending_upload_files(manifest, upload)
        state = {"upload_id": upload["id"], "idempotency_key": str(uuid.uuid4()),
                 "fingerprint": fingerprint, "api": API, "title": title,
                 "pending_files": pending, "expires_at": expires_at, "submission_started": False}
    elif "title" in state and state["title"] != title:
        if state.get("submission_started", not bool(pending)):
            fail("Submission title is already fixed. Retry with the original title.")
        state.update(title=title, idempotency_key=str(uuid.uuid4()))
    # Legacy completed state keeps its key, including when its title is unknown.
    state.setdefault("title", title)
    write_upload_state(state_path, state)
    names = {artifact["name"] for artifact in manifest["artifacts"]}
    while state.get("pending_files"):
        item = state["pending_files"][0]
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or item["name"] not in names:
            fail("Upload failed: saved session contains an unexpected artifact.")
        put_artifact(path, item["name"], item.get("url"))
        state["pending_files"].pop(0)
        write_upload_state(state_path, state)
    # Persist before POST: a lost response may already have finalized the upload.
    state["submission_started"] = True
    write_upload_state(state_path, state)
    show(call("POST", "submissions", json={"upload_id": state["upload_id"], "title": title}, headers={"Idempotency-Key": state["idempotency_key"]}))



@submission_app.command("status")
def status(submission: str, watch: bool = False):
    while True:
        data = call("GET", f"submissions/{submission}", retries=POLL_RETRIES if watch else 0)
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
