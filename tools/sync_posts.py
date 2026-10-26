import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".avif", ".bmp"}
ATTACHMENT_EXTENSIONS = {".py", ".json", ".ipynb", ".pdf", ".txt", ".csv", ".zip"}
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
INLINE_ASSET = re.compile(
    r"(?P<ticks>`+)(?!`)[\s\S]*?(?<!`)(?P=ticks)(?!`)"
    r"|(?P<prefix>!?\[(?:\\.|[^\]\\\n])*\]\(\s*)"
    r"(?P<url><[^>\n]+>|(?:\\.|[^\s()])+)"
    r"(?P<suffix>\s*(?:(?:\"[^\"\n]*\"|'[^'\n]*')\s*)?\))"
)


def copy_asset(source, destination):
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise ValueError(f"Conflicting asset names: {source} -> {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_posts(snapshot):
    shared_assets = snapshot / "assets"
    posts = sorted(
        post for post in snapshot.rglob("*.md")
        if not post.is_relative_to(shared_assets)
    )
    rewritten_count = 0
    for post in posts:
        asset_dir = post.with_suffix("")
        shared_dir = shared_assets / post.stem
        if shared_dir.is_dir():
            for image in sorted(shared_dir.rglob("*")):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                    copy_asset(image, asset_dir / image.relative_to(shared_dir))

        def replace_asset(match):
            nonlocal rewritten_count
            if match.group("ticks"):
                return match.group(0)
            href = match.group("url").strip("<>")
            address = urlsplit(href)
            if address.scheme or address.netloc or href.startswith(("/", "#")):
                return match.group(0)
            if not match.group("prefix").startswith("![") and Path(address.path).suffix.lower() not in ATTACHMENT_EXTENSIONS:
                return match.group(0)
            if address.query or address.fragment:
                raise ValueError(f"Local asset queries/fragments are unsupported: {post}: {href}")
            filename = re.sub(r"\\([\\ ()])", r"\1", unquote(address.path))
            image = (post.parent / filename).resolve()
            if not image.is_relative_to(snapshot) or not image.is_file():
                raise ValueError(f"Missing or out-of-tree asset: {post.relative_to(snapshot)}: {href}")
            if image.is_relative_to(shared_dir):
                relative = image.relative_to(shared_dir)
            elif image.is_relative_to(asset_dir):
                relative = image.relative_to(asset_dir)
            else:
                relative = Path(image.name)
            copy_asset(image, asset_dir / relative)
            replacement = relative.as_posix()
            if re.search(r"[\s()]", replacement) or match.group("url").startswith("<"):
                replacement = f"<{replacement}>"
            rewritten_count += 1
            return match.group("prefix") + replacement + match.group("suffix")

        timestamps = post.stat()
        original = post.read_text(encoding="utf-8")
        front_matter = re.match(r"\A---\r?\n.*?\r?\n---(?:\r?\n|$)", original, re.DOTALL)
        prefix = front_matter.group(0) if front_matter else ""
        body = original[len(prefix):]
        output = [prefix]
        prose = []
        fence_marker = None
        for line in body.splitlines(keepends=True):
            marker = FENCE.match(line)
            if fence_marker:
                output.append(line)
                if marker and marker[1][0] == fence_marker[0] and len(marker[1]) >= len(fence_marker) and not marker[2].strip():
                    fence_marker = None
            elif marker or line.startswith(("    ", "\t")):
                output.append(INLINE_ASSET.sub(replace_asset, "".join(prose)))
                prose.clear()
                output.append(line)
                if marker:
                    fence_marker = marker[1]
            else:
                prose.append(line)
        output.append(INLINE_ASSET.sub(replace_asset, "".join(prose)))
        transformed = "".join(output)
        if transformed != original:
            post.write_text(transformed, encoding="utf-8")
            os.utime(post, ns=(timestamps.st_atime_ns, timestamps.st_mtime_ns))

    if shared_assets.exists():
        shutil.rmtree(shared_assets)
    return len(posts), rewritten_count


def sync_posts(source, destination):
    source = source.expanduser().resolve(strict=True)
    destination = destination.resolve()
    if not source.is_dir() or not any(source.rglob("*.md")):
        raise ValueError(f"Not a notes directory containing Markdown: {source}")
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Source and destination must not overlap.")
    with tempfile.TemporaryDirectory(prefix="blog-post-sync-") as temporary:
        snapshot = Path(temporary)
        subprocess.run(
            ["rsync", "-a", "--exclude=.*", f"{source}/", f"{snapshot}/"],
            check=True,
        )
        if any(path.is_symlink() for path in snapshot.rglob("*")):
            raise ValueError("Symlinks in notes are unsupported; destination was not changed.")
        post_count, asset_count = prepare_posts(snapshot)
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-av", "--delete", f"{snapshot}/", f"{destination}/"],
            check=True,
        )
    print(f"Synced {post_count} posts; prepared {asset_count} asset references. Source notes unchanged.")


def main():
    parser = argparse.ArgumentParser(
        description="Sync notes into Hexo, copying images and linked attachments into per-post asset folders."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=Path.home() / "Voile/AI/AI-Infra-note",
        help="Notes directory (default: ~/Voile/AI/AI-Infra-note); never modified.",
    )
    arguments = parser.parse_args()
    destination = Path(__file__).resolve().parents[1] / "source/_posts"
    try:
        sync_posts(arguments.source, destination)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Sync failed: {error}\n")


if __name__ == "__main__":
    main()
