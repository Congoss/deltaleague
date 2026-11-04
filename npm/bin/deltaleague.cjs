#!/usr/bin/env node
'use strict';
const {spawn, spawnSync} = require('node:child_process');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const VERSION = '0.1.0';

function needsLocal(args) {
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--api') { i++; continue; }
    if (args[i].startsWith('--api=')) continue;
    if (args[i].startsWith('-')) continue;
    return ['run', 'evaluate'].includes(args[i]);
  }
  return false;
}
function pythonCommand(env = process.env, check = spawnSync) {
  const choices = env.DELTA_PYTHON ? [env.DELTA_PYTHON] : ['python3.12', 'python3.13', 'python3'];
  for (const binary of choices) {
    const result = check(binary, ['-c', 'import sys; print(str(sys.version_info.major)+"."+str(sys.version_info.minor)); sys.exit(sys.version_info < (3,12))'], {encoding: 'utf8'});
    if (result.status === 0) return {binary, version: result.stdout.trim()};
  }
  throw new Error('Python 3.12+ is required. Install Python first (on Mac: brew install python@3.12), or set DELTA_PYTHON to its executable.');
}
function execute(binary, args, {install = false} = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(binary, args, {stdio: ['inherit', install ? 2 : 1, 2], shell: false});
    const forward = signal => { if (!child.killed) child.kill(signal); };
    const interrupt = () => forward('SIGINT');
    const terminate = () => forward('SIGTERM');
    process.on('SIGINT', interrupt); process.on('SIGTERM', terminate);
    const cleanup = () => { process.off('SIGINT', interrupt); process.off('SIGTERM', terminate); };
    child.once('error', error => { cleanup(); reject(error); });
    child.once('exit', (code, signal) => { cleanup(); resolve(code ?? (signal === 'SIGINT' ? 130 : 143)); });
  });
}
async function ready(directory, spec) {
  try { return (await fs.readFile(path.join(directory, '.delta-ready'), 'utf8')) === spec; }
  catch { return false; }
}
async function ensureEnvironment(python, directory, spec, run = (binary, args) => execute(binary, args, {install: true})) {
  if (await ready(directory, spec)) return;
  await fs.mkdir(path.dirname(directory), {recursive: true});
  const lock = directory + '.lock';
  const deadline = Date.now() + 10 * 60 * 1000;
  while (true) {
    try { await fs.mkdir(lock); break; }
    catch (error) {
      if (error.code !== 'EEXIST') throw error;
      if (await ready(directory, spec)) return;
      // Recover a lock left behind by a killed launcher, without interrupting a live installer.
      try {
        const pid = Number(await fs.readFile(path.join(lock, 'pid'), 'utf8'));
        if (Number.isInteger(pid) && pid > 0) {
          try { process.kill(pid, 0); }
          catch (e) { if (e.code === 'ESRCH') { await fs.rm(lock, {recursive: true, force: true}); continue; } }
        }
      } catch (e) {
        if (e.code === 'ENOENT') {
          const stat = await fs.stat(lock).catch(() => null);
          if (stat && Date.now() - stat.mtimeMs > 30000) { await fs.rm(lock, {recursive: true, force: true}); continue; }
        }
      }
      if (Date.now() > deadline) throw new Error('Another Delta League installation is still running. Retry after it finishes.');
      await new Promise(resolve => setTimeout(resolve, 250));
    }
  }
  await fs.writeFile(path.join(lock, 'pid'), String(process.pid), {mode: 0o600});
  try {
    if (await ready(directory, spec)) return;
    console.error(`Preparing ${spec} in an isolated environment. The first run downloads Python dependencies.`);
    if (await run(python.binary, ['-m', 'venv', directory])) throw new Error('Could not create a virtual environment. Install the Python venv component for your OS.');
    const bin = path.join(directory, 'bin', 'python');
    if (await run(bin, ['-m', 'pip', 'install', '--disable-pip-version-check', '--index-url', 'https://pypi.org/simple', spec])) throw new Error('Dependency installation failed. Retry the command to finish setup.');
    await fs.writeFile(path.join(directory, '.delta-ready'), spec, {mode: 0o600});
  } finally { await fs.rm(lock, {recursive: true, force: true}); }
}
async function main(args = process.argv.slice(2)) {
  if (!['darwin', 'linux'].includes(process.platform)) throw new Error('This launcher supports macOS and Linux. Use pip install deltaleague on other platforms.');
  const python = pythonCommand();
  const local = needsLocal(args);
  const cache = process.env.XDG_CACHE_HOME || (process.platform === 'darwin' ? path.join(os.homedir(), 'Library', 'Caches') : path.join(os.homedir(), '.cache'));
  const directory = path.join(cache, 'deltaleague', 'npx', VERSION, `py${python.version}-${local ? 'local' : 'base'}`);
  await ensureEnvironment(python, directory, `deltaleague${local ? '[local]' : ''}==${VERSION}`);
  process.exitCode = await execute(path.join(directory, 'bin', 'delta'), args);
}
if (require.main === module) main().catch(error => { console.error(`deltaleague: ${error.message}`); process.exitCode = 1; });
module.exports = {needsLocal, pythonCommand, ensureEnvironment, execute};
