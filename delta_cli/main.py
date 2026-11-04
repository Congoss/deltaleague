import json
import os
from pathlib import Path
import time
import uuid
import webbrowser
import httpx
import typer
from .artifacts import build_sparse, validate_manifest, file_hash

app = typer.Typer(help="Delta League: submit a patch, prove an improvement.", no_args_is_help=True)
models_app = typer.Typer()
model_app = typer.Typer()
patch_app = typer.Typer()
submission_app = typer.Typer()
app.add_typer(models_app, name="models")
app.add_typer(model_app, name="model")
app.add_typer(patch_app, name="patch")
app.add_typer(submission_app, name="submission")
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
    with client() as c:
        r = c.request(method, "/api/v1/" + path, **kwargs)
    if not r.is_success:
        typer.echo(f"HTTP {r.status_code}: {r.text}", err=True)
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
def login():
    data = call("POST", "cli-auth")
    typer.echo(f"Authorize this CLI: {data['verification_url']}")
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


@model_app.command("pull")
def model_pull(version: str, output: Path = typer.Option(Path("model"))):
    data = call("GET", f"model-versions/{version}")
    output.mkdir(parents=True, exist_ok=True)
    for f in data["files"]:
        if Path(f["name"]).name != f["name"]:
            raise typer.BadParameter("Invalid filename from server")
        path = output / f["name"]
        with httpx.stream("GET", f["url"], timeout=300) as response:
            response.raise_for_status()
            with path.open("wb") as target:
                for chunk in response.iter_bytes():
                    target.write(chunk)
        if path.stat().st_size != f["size_bytes"] or file_hash(path) != f["sha256"]:
            raise typer.BadParameter("Downloaded artifact integrity failure")
    (output / "delta-model.json").write_text(json.dumps(data, indent=2))
    typer.echo(str(output))


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
