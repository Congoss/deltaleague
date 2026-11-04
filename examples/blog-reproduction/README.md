# Rebuild the published weight-space edit

Source: [Rikiya's weight-space programming post](https://rikiyat.github.io/blog/weight-space-programming/) and [published experiment](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit).

Initialize a Qwen competition workspace using the main README. From that workspace:

```sh
mkdir -p upstream
curl --fail --location https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json --output upstream/manifest.json
curl --fail --location https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/edit.npz --output upstream/edit.npz
python /path/to/deltaleague/examples/blog-reproduction/rebuild.py
delta patch build --parent model --modified edited.safetensors --output submission
delta validate submission
# Open a research PR first and replace PR_URL with its actual URL.
delta submit submission --title "Published weight-space edit reproduction" --pr PR_URL
```

Before the submit command, open a research PR in [the public repository](https://github.com/ilblackdragon/deltaleague) recording the upstream revision, reproduction code, checked tensor hashes, parent/evaluation IDs, and public validation findings. Attach that PR with `--pr`; the CLI does not create it. Add the submission link and verified results to the PR afterward. Omit `--pr` only when choosing a source-free submission.

The self-contained rebuild script uses NumPy, PyTorch, and safetensors (install NumPy separately if needed). Run Python normally, without `-O`, because the reproduction checks use assertions. It reads the upstream data with pickle disabled and checks the original and resulting tensor hashes against the pinned manifest. It does not run upstream code or fit to evaluation examples.

The edit changes 130,188 BF16 entries across layers 15 and 19 MLP down-projection matrices. During the initial CLI trial, both original tensor hashes and both resulting tensor hashes matched. This reproduces the released **weights**, not the complete research process or published benchmark scores. Delta League's starter validation and provisional test samples differ from the post's full benchmark protocol. An official starter score cannot establish reproduction of the post's reported results.
