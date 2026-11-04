'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {needsLocal, pythonCommand, ensureEnvironment, execute} = require('../bin/deltaleague.cjs');

test('selects local extra only for inference commands, after global API arguments', () => {
  assert.equal(needsLocal(['--api','https://example.test','run','hello']), true);
  assert.equal(needsLocal(['--api=https://example.test','evaluate']), true);
  assert.equal(needsLocal(['submit','run']), false);
  assert.equal(needsLocal(['--help']), false);
});
test('Python overrides are executed as one argument, never shell text', () => {
  let seen;
  assert.deepEqual(pythonCommand({DELTA_PYTHON:'/path with spaces/python'}, binary => {
    seen=binary; return {status:0,stdout:'3.12\n'};
  }), {binary:'/path with spaces/python',version:'3.12'});
  assert.equal(seen,'/path with spaces/python');
  assert.throws(() => pythonCommand({}, () => ({status:1})), /Python 3.12/);
});
test('concurrent launches install once and incomplete installation is retryable', async () => {
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'delta-npm-test-'));
  try {
    const directory=path.join(root,'venv');let calls=0;
    const installer=async () => {calls++;await fs.mkdir(directory,{recursive:true});await new Promise(r=>setTimeout(r,30));return 0;};
    await Promise.all([ensureEnvironment({binary:'python'},directory,'deltaleague==0.1.0',installer),ensureEnvironment({binary:'python'},directory,'deltaleague==0.1.0',installer)]);
    assert.equal(calls,2);
    await ensureEnvironment({binary:'python'},directory,'deltaleague==0.1.0',()=>{throw Error('unexpected install');});
    const failed=path.join(root,'failed');
    await assert.rejects(ensureEnvironment({binary:'python'},failed,'spec',async()=>1),/virtual environment/);
    await assert.rejects(fs.stat(failed+'.lock'),{code:'ENOENT'});
    await ensureEnvironment({binary:'python'},failed,'spec',async()=>{await fs.mkdir(failed,{recursive:true});return 0;});
  } finally {await fs.rm(root,{recursive:true,force:true});}
});
test('preserves child exit status and argument boundaries', async () => {
  assert.equal(await execute(process.execPath,['-e','process.exit(process.argv[1] === "a b;echo no" ? 7 : 9)','a b;echo no']),7);
});
