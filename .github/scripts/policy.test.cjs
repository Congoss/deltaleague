const {test} = require('node:test');
const assert = require('node:assert/strict');
const {classify, inspect} = require('./contribution-policy.cjs');
const regular = path => ({path, type:'blob', mode:'100644',size:20});
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

function githubFor({state='open', sha='abc', changed=1, permission='read', permissionStatus, files, tree, secondSha}={}) {
 const pr = {state, user:{login:'alice'}, head:{sha}, changed_files:changed};
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
     return {data:{permission}};
    },
   },
   git:{
    getTree: async () => { github.treeCalls++; return {data:{tree: tree || [], truncated:false}}; },
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
