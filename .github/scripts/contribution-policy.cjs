'use strict';
const CONTROL = new Set(['AGENTS.md', 'CONTRIBUTING.md', 'ARTIFACTS.md', 'PROTOCOL.md', 'research/README.md']);
const artifact = /(?:\.(?:safetensors|bin|pt|pth|ckpt|gguf|npz)|\.generations\.jsonl)$/i;
const sensitive = /(?:^|\/)(?:\.env(?:\..*)?|auth\.json|\.delta-upload\.json)$|\.private\./i;

function classify({files, author, maintainer = false, tree = [], truncated = false}) {
  const errors = [];
  const paths = [...new Set(files.flatMap(f => [f.filename, f.previous_filename].filter(Boolean)))];
  const researchPaths = paths.filter(p => p.startsWith('research/') && p !== 'research/README.md');
  const otherPaths = paths.filter(p => !researchPaths.includes(p));
  const category = researchPaths.length ? 'research' : paths.some(p => CONTROL.has(p) || p.startsWith('benchmarks/') || p.startsWith('datasets/') || p === 'delta_cli/scoring.py') ? 'competition' : 'tooling';
  for (const p of paths) {
    if (p.startsWith('/') || p.split('/').some(part => !part || part === '.' || part === '..')) errors.push(`Invalid path: ${p}`);
    if (!maintainer && p.split('/')[0].startsWith('.')) errors.push(`External PR cannot change root hidden path: ${p}`);
  }
  if (files.length > 3000 || truncated) errors.push('File inventory is incomplete or too large; split this PR.');
  if (researchPaths.length && otherPaths.length) errors.push('Split research and tooling/rules changes into separate PRs.');
  if (researchPaths.length) {
    const experimentKey = path => {
      const parts = path.split('/');
      return parts.length >= 4 && (parts[1].toLowerCase() === author.toLowerCase() || (maintainer && !tree.some(entry => entry.path === `research/${parts[1]}/README.md`))) ? `${parts[1]}/${parts[2]}` : parts[1];
    };
    const directories = [...new Set(researchPaths.map(experimentKey))];
    if (directories.length !== 1) errors.push('A research PR must change exactly one experiment directory.');
    for (const p of researchPaths) {
      const parts = p.split('/');
      if (parts.length < 3 || !/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(parts[1])) errors.push(`Invalid experiment path: ${p}`);
      const login = author.toLowerCase();
      const directory = parts[1].toLowerCase();
      const owned = directory === login || directory.startsWith(login + '-');
      if (!maintainer && !owned) errors.push(`Research must stay in research/${author}/<experiment>/ or research/${author}-<experiment>/: ${p}`);
      if (/(^|\/)\.gitattributes$/.test(p) || p.endsWith('.lfsconfig')) errors.push(`Git LFS configuration cannot be committed in research: ${p}`);
      if (artifact.test(p) || sensitive.test(p)) errors.push(`Weights, credentials, private-marked data, and raw generations cannot be committed in research: ${p}`);
    }
    const entries = new Map(tree.map(entry => [entry.path, entry]));
    const changed = files.filter(f => f.status !== 'removed').map(f => f.filename);
    for (const p of changed) {
      const entry = entries.get(p);
      if (!entry) errors.push(`Missing file metadata: ${p}`);
      else if (!['100644', '100755'].includes(entry.mode) || entry.type !== 'blob') errors.push(`Symlinks and submodules are not allowed in research: ${p}`);
    }
    for (const directory of directories) {
      const prefix = `research/${directory}/`;
      const contents = tree.filter(entry => entry.path.startsWith(prefix));
      const readme = entries.get(prefix + 'README.md');
      if (!readme || readme.type !== 'blob' || !['100644', '100755'].includes(readme.mode)) errors.push(`Experiment needs ${prefix}README.md.`);
      let size = 0;
      for (const entry of contents) {
        size += entry.size || 0;
        if ((entry.size || 0) > 1024 * 1024) errors.push(`Research file exceeds 1 MiB: ${entry.path}`);
        if (typeof entry.content === 'string' && entry.content.startsWith('version https://git-lfs.github.com/spec/v1')) errors.push(`Git LFS pointer cannot be committed in research: ${entry.path}`);
      }
      if (size > 5 * 1024 * 1024) errors.push(`Experiment exceeds 5 MiB: ${prefix}`);
    }
  }
  return {category, errors, paths};
}

async function inspect(github, context, number) {
  const repo = context.repo;
  const {data: pr} = await github.rest.pulls.get({...repo, pull_number: number});
  if (pr.state !== 'open') throw new Error('PR must be open.');
  let maintainer = false;
  try {
    const {data} = await github.rest.repos.getCollaboratorPermissionLevel({...repo, username: pr.user.login});
    maintainer = ['admin', 'maintain'].includes(data.role_name || data.permission);
  } catch (error) { if (error.status !== 404) throw error; }
  // Files come from the SHAs captured above, not the live pull-request file list.
  const {data: compared} = await github.rest.repos.compareCommits({...repo, base: pr.base.sha, head: pr.head.sha});
  const files = compared.files || [];
  let tree = [], truncated = Boolean(compared.truncated) || files.length !== pr.changed_files || files.length >= 300;
  if (files.some(f => [f.filename, f.previous_filename].some(p => p?.startsWith('research/') && p !== 'research/README.md'))) {
    const {data} = await github.rest.git.getTree({...repo, tree_sha: pr.head.sha, recursive: '1'});
    tree = data.tree; truncated ||= data.truncated;
  }
  const preliminary = classify({files, author: pr.user.login, maintainer, tree, truncated});
  if (preliminary.category === 'research' && !preliminary.errors.length) {
    // Read immutable blobs only after metadata passes the file and experiment limits.
    const changed = new Set(files.filter(f => f.status !== 'removed').map(f => f.filename));
    for (const entry of tree.filter(entry => changed.has(entry.path) && entry.type === 'blob')) {
      const {data: blob} = await github.rest.git.getBlob({...repo, file_sha: entry.sha});
      if (blob.encoding !== 'base64' || blob.size > 1024 * 1024 || typeof blob.content !== 'string') {
        throw new Error('Invalid or oversized research blob');
      }
      const bytes = Buffer.from(blob.content, 'base64');
      if (bytes.length !== entry.size || bytes.length !== blob.size) throw new Error('Research blob size mismatch');
      entry.content = bytes.toString('utf8');
    }
  }
  const {data: current} = await github.rest.pulls.get({...repo, pull_number: number});
  if (current.head.sha !== pr.head.sha) throw new Error('PR changed during inspection; rerun for the latest commit.');
  return {pr, ...classify({files, author: pr.user.login, maintainer, tree, truncated})};
}
module.exports = {classify, inspect};
