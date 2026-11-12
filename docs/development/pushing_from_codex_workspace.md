# Publishing repository changes

The repository is [flengyel/SimplicialTensors](https://github.com/flengyel/SimplicialTensors).
Run these commands from its local checkout with GitHub write access.

## Inspect the checkout

```sh
git status --short --branch
git remote -v
git log -3 --oneline
git fetch origin
```

Verify that `origin` refers to `flengyel/SimplicialTensors`. The clone
directory and local branch name depend on the checkout.

Configure the commit identity for this repository when necessary.

```sh
git config user.name "Full Name"
git config user.email "email-associated-with-GitHub"
```

Use the actual author identity. Authentication is handled separately
through the configured HTTPS credential helper or an authorized SSH key.

## Commit a focused change

Inspect the diff and stage the intended paths.

```sh
git diff
git diff --check
git add path/to/changed-file
git diff --cached
git commit -m "Describe the verified change"
```

Run the checks appropriate to the change before publishing.
Library changes require `python -m pytest -q`.
Documentation changes require checking their commands, paths, and links.

## Publish and verify

For a review branch, push that branch and open a pull request.

```sh
git push -u origin HEAD
```

When a change is approved for `main`, fetch its current head, integrate
any new commits, and push a normal fast-forward update.

```sh
git fetch origin
git log --oneline --left-right HEAD...origin/main
git push origin HEAD:main
git fetch origin
git rev-parse HEAD
git rev-parse origin/main
```

The final two hashes identify the local commit and fetched remote
commit. Resolve a rejected push against the new remote head before
retrying.

On another machine, update a clean checkout with `git pull --ff-only`.
The manuscript build is documented in [tex/README.md](../../tex/README.md).
