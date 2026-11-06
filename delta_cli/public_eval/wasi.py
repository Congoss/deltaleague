"""CPython/WASI sandbox: no network, host environment, writable mounts or native code."""
import os
from pathlib import Path
import threading
import time
import uuid

ARCHIVE_URL = 'https://github.com/brettcannon/cpython-wasi-build/releases/download/v3.12.10/python-3.12.10-wasi_sdk-20.zip'
ARCHIVE_SHA256 = '872808f341c1cc8c6795d1617b716300fbf79b79a8d2a45fc9f2d31b086bbc67'
SANDBOX_ID = 'cpython-3.12.10-wasi-wasmtime-50.0.1'
_lock = threading.Lock()
_runtime = None


def runtime():
    global _runtime
    with _lock:
        if _runtime is None:
            import wasmtime
            from importlib.metadata import version
            if version('wasmtime') != '50.0.1':
                raise RuntimeError('Unexpected Wasmtime version')
            root = Path(os.environ.get('DELTA_WASI_ROOT', '/opt/python-wasi'))
            config = wasmtime.Config()
            config.consume_fuel = True
            config.epoch_interruption = True
            engine = wasmtime.Engine(config)
            module = wasmtime.Module.from_file(engine, root / 'python.wasm')
            def clock():
                while True:
                    time.sleep(.1)
                    engine.increment_epoch()
            threading.Thread(target=clock, daemon=True).start()
            _runtime = engine, module, root
        return _runtime


def sandbox(source, *, timeout=12):
    import wasmtime
    engine, module, root = runtime()  # Infrastructure failures must propagate.
    nonce = uuid.uuid4().hex
    output = bytearray()
    def capture(data):
        if len(output) + len(data) > 1024 * 1024:
            return -1
        output.extend(data)
        return len(data)
    wasi = wasmtime.WasiConfig()
    wasi.argv = ['python', '-B', '-s', '-c', source + f'\nprint({nonce!r}, flush=True)\n']
    wasi.env = [('PYTHONHOME', '/python'), ('PYTHONHASHSEED', '0')]
    wasi.preopen_dir(str(root / 'lib'), '/python/lib', fs_mutable=False)
    wasi.stdout_custom = capture
    wasi.stderr_custom = lambda data: len(data)  # No unbounded diagnostic output.
    store = wasmtime.Store(engine)
    store.set_limits(memory_size=256 * 1024**2, instances=1, memories=1)
    store.set_fuel(2_000_000_000)
    store.set_epoch_deadline(max(1, int(timeout * 10)))
    store.set_wasi(wasi)
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    instance = linker.instantiate(store, module)
    reason = 'test_failure'
    try:
        instance.exports(store)['_start'](store)
    except wasmtime.ExitTrap as exc:
        if exc.code != 0:
            return {'passed': False, 'reason': reason}
    except wasmtime.Trap:
        return {'passed': False, 'reason': 'resource_limit'}
    finally:
        store.close()
    passed = nonce.encode() in output.splitlines()
    return {'passed': passed, 'reason': 'passed' if passed else reason}
