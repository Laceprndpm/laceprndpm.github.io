# Repository Guidelines

## Project Structure & Module Organization

- `source/_posts/` contains published Markdown and article-named image directories.
- `tools/sync_posts.py` and `rsync.sh` import external notes and prepare image paths.
- `scripts/` contains Hexo hooks; `layout/`, `source/css/`, `source/images/`, and `source/_data/` hold site customizations.
- `_config.yml` configures Hexo; `_config.keep.yml` configures Keep. `scaffolds/` contains article templates.
- `themes/keep/` is an upstream Git submodule. Prefer site-level extensions over theme edits. Generated `public/`, `db.json`, and dependencies are ignored.

## Build, Test, and Development Commands

Use Node.js 26 (CI), Python 3.9+, and rsync.

- `git submodule update --init --recursive`: initialize the Keep theme.
- `npm ci`: install locked dependencies.
- `npm run sync`: synchronize notes and prepare per-post assets.
- `npm run sync -- /path/to/notes`: use an alternative notes directory.
- `npm run clean && npm run build`: rebuild and validate the site.
- `npm run check`: validate generated links; add `-- --external` for warning-only external checks.
- `npm run server -- --ip 127.0.0.1`: preview at `http://127.0.0.1:4000/`.

## Content & Synchronization

The default source is `~/Voile/AI/AI-Infra-note`. Synchronization preserves originals but uses `rsync --delete` on blog copies: inspect local changes first. Durable article edits belong in the original notes when authorized, followed by synchronization.

Keep shared images in the notes' `assets/` directory; the importer creates article-named folders and rewrites blog-only links. Preserve compressed image filenames and keep original-image backups outside this repository.

Keep article links relative; builds resolve published targets through Hexo metadata.

Articles should include title, date, tags, categories, and `mathjax: true` when needed. Place a one-sentence summary before `<!-- more -->`; put audience and scope notes afterward.

## Coding Style & Naming Conventions

Follow existing styles: two-space indentation for JavaScript, EJS, CSS, JSON, and YAML; four spaces and snake_case for Python. JavaScript uses semicolons. Use kebab-case article and asset filenames. No formatter or linter is configured.

## Testing Guidelines

There is no test suite or coverage threshold. Builds run `npm run check`, blocking broken internal links, missing assets/anchors, private filesystem paths, and MathJax errors. External checks are opt-in and non-blocking. Retain AMS support; run `git diff --check`. Check desktop/mobile layouts for UI changes and source preservation/idempotence for synchronization changes.

## Commit & Pull Request Guidelines

Follow history's concise `docs:`, `fix:`, and `feat:` prefixes. Keep commits focused. PRs should include context, relevant issues, validation results, and screenshots for visual changes. Commit or push only when requested; pushes to `main` trigger GitHub Pages deployment through `.github/workflows/pages.yml`.
