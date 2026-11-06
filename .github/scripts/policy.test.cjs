const {test} = require('node:test');
const assert = require('node:assert/strict');
const {classify, inspect} = require('./contribution-policy.cjs');
const regular = path => ({path, type:'blob', mode:'100644',size:20,sha:'blobsha'});
function check(files, extra={}) {return classify({author:'alice',files:files.map(filename=>typeof filename==='string'?{filename,status:'modified'}:filename),...extra});}
test('external root dot paths are blocked for additions, edits, deletions and both sides of renames',()=>{
 for(const status of ['added','modified','removed']) assert.ok(check([{filename:'.github/workflows/ci.yml',status}]).errors.length);
 for(const [filename,previous_filename] of [['README.md','.gitignore'],['.config/settings','README.md']]) assert.ok(check([{filename,previous_filename,status:'renamed'}]).errors.length);
 assert.ok(check(['.env']).errors.length);
 assert.equal(check(['.github/workflows/ci.yml'],{maintainer:true}).errors.length,0);
});
test('research is confined to one author directory, with no mixed changes',()=>{
 const paths=['research/alice-edit/README.md','research/alice-edit/edit.py']; const tree=paths.map(regular);
 assert.equal(check(paths,{tree}).errors.length,0);
 assert.equal(check(paths,{tree}).category,'research');
 assert.ok(check([...paths,'delta_cli/main.py'],{tree}).errors.some(e=>e.startsWith('Split')));
 assert.ok(check(['research/bob-edit/README.md'],{tree:[regular('research/bob-edit/README.md')]}).errors.length);
 assert.ok(check([...paths,'research/alice-other/README.md'],{tree}).errors.length);
});
test('reject research symlinks, missing README, private data, artifacts and incomplete inventories',()=>{
 const readme='research/alice-edit/README.md';
 for(const name of ['weights.safetensors','heldout.private.json','.env','auth.json','answer.generations.jsonl']) assert.ok(check([readme,'research/alice-edit/'+name],{tree:[regular(readme),regular('research/alice-edit/'+name)]}).errors.length);
 assert.ok(check([readme],{tree:[{...regular(readme),mode:'120000'}]}).errors.length);
 assert.ok(check(['research/alice-edit/code.py'],{tree:[regular('research/alice-edit/code.py')]}).errors.length);
 assert.ok(check([readme],{tree:[regular(readme)],truncated:true}).errors.length);
 assert.ok(check([readme],{tree:[{...regular(readme),size:2*1024*1024}]}).errors.length);
});
test('classification comes from paths rather than labels or PR title',()=>{
 assert.equal(check(['delta_cli/main.py','tests/test_cli.py']).category,'tooling');
 assert.equal(check(['delta_cli/scoring.py']).category,'competition');
 assert.equal(check(['PROTOCOL.md']).category,'competition');
});

function githubFor({state='open', sha='abc', changed=1, permission='read', roleName, permissionStatus, files, tree, truncated=false, secondSha}={}) {
 const pr = {state, user:{login:'alice'}, head:{sha}, base:{sha:'basesha'}, changed_files:changed};
 let reads = 0;
 const github = {
  rest:{
   pulls:{
    get: async () => ({data: reads++ ? {...pr, head:{sha: secondSha || sha}} : pr}),
    listFiles: async () => { throw new Error('listFiles must be paginated'); },
   },
   repos:{
    getCollaboratorPermissionLevel: async () => {
     if (permissionStatus) throw Object.assign(new Error('permission'), {status: permissionStatus});
     return {data: roleName ? {role_name: roleName} : {permission}};
    },
    compareCommits: async (args) => { github.compareArgs = args; return {data:{files, truncated:false}}; },
   },
   git:{
    getBlob: async () => ({data:{encoding:"base64",size:20,content:Buffer.from("x".repeat(20)).toString("base64")}}),
    getTree: async (args) => { github.treeCalls++; github.treeArgs = args; return {data:{tree: tree || [], truncated}}; },
   },
  },
  paginate: async () => files,
  treeCalls: 0,
 };
 return github;
}
const context = {repo:{owner:'o', repo:'deltaleague'}};

test('inspect fails closed on closed PRs, permission 404, and head-change', async () => {
 const dot = [{filename:'.github/workflows/ci.yml', status:'modified'}];
 await assert.rejects(inspect(githubFor({state:'closed', files:dot}), context, 1), /PR must be open/);
 const denied = await inspect(githubFor({permissionStatus:404, files:dot}), context, 1);
 assert.ok(denied.errors.some(error => error.includes('.github/workflows/ci.yml')));
 assert.equal(denied.pr.user.login, 'alice');
 await assert.rejects(inspect(githubFor({permissionStatus:500, files:dot}), context, 1), /permission/);
 await assert.rejects(inspect(githubFor({files:dot, secondSha:'def'}), context, 1), /changed during inspection/);
 const tooling = [{filename:'delta_cli/main.py', status:'modified'}];
 const partial = await inspect(githubFor({files:tooling, changed:2}), context, 1);
 assert.ok(partial.errors.some(error => error.includes('incomplete')));
 const github = githubFor({files:tooling, changed:1});
 const clean = await inspect(github, context, 1);
 assert.equal(clean.errors.length, 0);
 assert.equal(github.treeCalls, 0);
 const research = [{filename:'research/alice-edit/README.md', status:'added'}];
 const fetched = githubFor({files:research, changed:1, tree:[regular('research/alice-edit/README.md')]});
 const recorded = await inspect(fetched, context, 1);
 assert.equal(recorded.category, 'research');
 assert.equal(fetched.treeCalls, 1);
});

test('inspect lets a maintainer edit hidden paths and still blocks write access', async () => {
 const dot = [{filename:'.github/workflows/ci.yml', status:'modified'}];
 assert.equal((await inspect(githubFor({permission:'admin', files:dot}), context, 1)).errors.length, 0);
 assert.equal((await inspect(githubFor({roleName:'maintain', files:dot}), context, 1)).errors.length, 0);
 const write = await inspect(githubFor({permission:'write', files:dot}), context, 1);
 assert.ok(write.errors.some(error => error.includes('.github/workflows/ci.yml')));
});

test('inspect fails closed when the git tree is truncated', async () => {
 const research = [{filename:'research/alice-edit/README.md', status:'added'}];
 const result = await inspect(githubFor({files:research, changed:1, truncated:true, tree:[regular('research/alice-edit/README.md')]}), context, 1);
 assert.ok(result.errors.some(error => error.includes('incomplete')));
});

test('inspect fetches the head tree when a research path is renamed away', async () => {
 const files = [{filename:'docs/x.md', previous_filename:'research/alice-edit/x.md', status:'renamed'}];
 const github = githubFor({files, changed:1, sha:'headsha', tree:[regular('research/alice-edit/README.md'), regular('docs/x.md')]});
 await inspect(github, context, 1);
 assert.equal(github.treeCalls, 1);
 assert.equal(github.treeArgs.tree_sha, 'headsha');
 assert.equal(github.compareArgs.head, 'headsha');
 assert.equal(github.compareArgs.base, 'basesha');
});

test('exact login directories are owned only by that login', () => {
 const paths = ['research/alice/edit/README.md', 'research/alice/edit/edit.py'];
 const tree = paths.map(regular);
 assert.equal(check(paths, {tree, author:'alice'}).errors.length, 0);
 assert.ok(check(paths, {tree, author:'bob'}).errors.some(error => error.includes('research/bob/')));
});

test('research rejects git lfs attributes and pointer blobs', () => {
 const readme = 'research/alice-edit/README.md';
 const attributes = 'research/alice-edit/.gitattributes';
 assert.ok(check([readme, attributes], {tree:[regular(readme), regular(attributes)]}).errors.some(error => error.includes('LFS')));
 const pointer = {...regular('research/alice-edit/weights.bin'), content:'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n'};
 assert.ok(check([readme, pointer.path], {tree:[regular(readme), pointer]}).errors.some(error => error.includes('pointer')));
});

test('inspect reads immutable blob contents to reject disguised LFS pointers', async () => {
 const path = 'research/alice-edit/README.md';
 const content = 'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n';
 const github = githubFor({files:[{filename:path,status:'added'}], tree:[{...regular(path),size:Buffer.byteLength(content)}]});
 github.rest.git.getBlob = async args => {
  assert.equal(args.file_sha, 'blobsha');
  return {data:{encoding:'base64',size:Buffer.byteLength(content),content:Buffer.from(content).toString('base64')}};
 };
 const result = await inspect(github, context, 1);
 assert.ok(result.errors.some(error => error.includes('LFS pointer')));
});

test('maintainers can update another authors nested experiment', () => {
 const paths = ['research/bob/edit/README.md', 'research/bob/edit/edit.py'];
 assert.equal(check(paths, {tree:paths.map(regular),maintainer:true}).errors.length,0);
});
