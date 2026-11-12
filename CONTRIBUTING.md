# Working on the pipeline

How changes get into the code, how releases are numbered, and how a result is
traced back to the code that made it. The same rules hold for
`micropattern_pipeline` and `analysis_pipeline_v2`.

## The copies of the code

| Copy | Where | What it is for |
| --- | --- | --- |
| **Deployed copy** | `KR/<pipeline>/` on the lab share (`/data/Warmflash/KR/<pipeline>` on the cluster) | What the GUI, `hpc/run.sh` and every lab member run. Always on `main`, never edited by hand. |
| **Development clone** | anywhere you like, e.g. `~/code/<pipeline>` on your workstation | Where changes are made and tested. |
| **GitHub** (when set up) | private repository | Backup, history browser, pull requests, automatic tests (`.github/workflows/tests.yml`). |

Edit in a development clone, not in the deployed copy: a job submitted while
the deployed copy sits on a half-finished branch runs that half-finished code.

```bash
git clone /data/Warmflash/KR/micropattern_pipeline ~/code/micropattern_pipeline   # on the cluster
git clone "/run/user/$UID/gvfs/smb-share:server=bmiisi.chmcres.cchmc.org,share=warmflash/KR/micropattern_pipeline" \
          ~/code/micropattern_pipeline                                            # on a Linux workstation
cd ~/code/micropattern_pipeline
git config core.hooksPath .githooks        # the pre-commit check, once per clone
```

**Git on the share from a Linux workstation (GNOME/gvfs mount).** gvfs cannot
change file permissions, so `git init`, `git config`, `git remote add` and
`git push -u` fail there with "chmod ... Operation not supported". Commit,
branch, log, diff, pull and push all work. Run the few configuration commands on
the cluster instead (`ssh bmicluster`, `module load git`, `cd /data/Warmflash/KR/<pipeline>`),
where the same folder is an ordinary NFS directory.

## Making a change

1. **Start from an up-to-date `main`** and make a branch named for the change:
   `git switch -c fix/plate-map-ranges` or `feat/exclude-markers`.
2. **Keep `src/` generic.** No experiment names, paths or doses in the code;
   those live in the experiment's `config.yaml` and `plate_map.yaml`. A new
   option gets a default that keeps old configs working, an entry in
   `templates/config.yaml` and a line in `docs/REFERENCE.md`.
3. **Add or update a synthetic test** with a known answer (`tests/`).
4. **Run the tests**: `python -m pytest -q -p no:cacheprovider tests/`.
5. **Shared code.** About forty files are identical in the two pipelines
   (listed in `tools/check_shared.sh`). After changing one, copy it to the other
   repository, run both test suites, commit in both, and confirm with
   `tools/check_shared.sh`.
6. **Write it down** under `## [Unreleased]` in `CHANGELOG.md`: what changed,
   in words a user of the pipeline understands. Say so plainly when a change
   alters numbers in existing results.
7. **Commit** in small steps, each one working. Subject line in the imperative,
   at most about 70 characters ("Exclude markers by name within a channel");
   a blank line; then why, if it is not obvious.
8. **Merge** into `main` (a pull request on GitHub, or `git switch main && git merge --no-ff <branch>`),
   then update the deployed copy (below).

## Releases and version numbers

The number in `VERSION` follows [semantic versioning](https://semver.org):

| Bump | When | Example |
| --- | --- | --- |
| **MAJOR** (2.0.0) | an existing config stops working, or an output file changes name or meaning | renaming a config key; a new definition of `% positive` |
| **MINOR** (1.1.0) | a new analysis, option or output; old configs give the same results | a new downstream step |
| **PATCH** (1.0.1) | a bug fix | a plot label; a crash on an empty well |

A bug fix that changes numbers in results is still a patch, but its changelog
entry says which results to rerun.

To release:

```bash
# on main, tests passing, CHANGELOG [Unreleased] filled in
echo 1.1.0 > VERSION
# in CHANGELOG.md: rename [Unreleased] to [1.1.0] - YYYY-MM-DD, start a new empty [Unreleased]
git commit -am "Release 1.1.0"
git tag -a v1.1.0 -m "micropattern_pipeline 1.1.0"
git push && git push --tags        # when there is a remote
```

## Updating the deployed copy

The deployed copy only ever moves forward along `main`:

```bash
cd /data/Warmflash/KR/micropattern_pipeline     # or the share path on a workstation
git status                                      # must be clean: nobody edited it by hand
git pull --ff-only                              # from GitHub; or, from a development clone:
                                                # git push <deployed copy path> main
                                                # (on the cluster, `module load git` first:
                                                # the default git 1.8 cannot update a checkout on push)
```

Pushing `main` from a development clone straight into the deployed copy works
because the deployed copy is set to `receive.denyCurrentBranch updateInstead`:
the files update in place, and the push is refused if the deployed copy has
uncommitted edits.

## Which code made this result?

Every run logs a line such as `micropattern_pipeline 1.1.0 (v1.1.0-2-g3f9c2ab)`
to `pipeline.log`; every downstream run writes `pipeline_version.json` into the
results folder and prints the same line in the header of `report.html`.

| The line says | Meaning |
| --- | --- |
| `v1.1.0` | exactly release 1.1.0 |
| `v1.1.0-2-g3f9c2ab` | two commits after 1.1.0, at commit `3f9c2ab` |
| `uncommitted changes` | the code had edits that were never committed; the result cannot be reproduced exactly |

To rerun with the code that made an old result, check out that version in a
separate folder, never in the deployed copy:

```bash
git worktree add ~/code/micropattern_pipeline-1.0.0 v1.0.0
python ~/code/micropattern_pipeline-1.0.0/main.py downstream --config config.yaml --out analysis_rerun_1.0.0
```

## What never goes in the repository

Images, masks, per-cell tables, results, experiment configs and plate maps,
`hpc_job.env` files, logs. `.gitignore` keeps them out, and the pre-commit hook
refuses data files and anything over 1 MB.
