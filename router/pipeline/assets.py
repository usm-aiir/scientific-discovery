"""Prepare release downloads or get the saved router and data."""

import argparse
import gzip
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

from router_src.data import checkpoint_task, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name('release.json')
REPOSITORY = 'usm-aiir/scientific-discovery'
DIRECTORIES = {'model': 'model/modernbert', 'data': 'data'}
MODEL_FILES = (
    'README.md', 'config.json', 'deployment.json', 'model.safetensors',
    'tokenizer.json', 'tokenizer_config.json', 'training.json',
)


def file_info(path):
    """Record the size and checksum without loading the file into memory."""
    return {'size_bytes': path.stat().st_size, 'sha256': file_hash(path)}


def relative_path(name):
    """Accept only plain paths relative to the router folder."""
    relative = PurePosixPath(name)
    if (relative.is_absolute() or '..' in relative.parts or '\\' in name
            or not relative.parts or str(relative) != name):
        raise ValueError(f'Invalid release path: {name}')
    return relative


def checked_path(root, name):
    """Keep release files inside the destination and refuse symbolic links."""
    path = root
    for part in relative_path(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Release path is a symbolic link: {path}')
    return path


def verify_files(root, asset):
    """Require every released file to match its recorded bytes."""
    for name, expected in asset['files'].items():
        path = checked_path(root, name)
        if not path.is_file() or file_info(path) != expected:
            raise ValueError(f'File differs from the release: {path}')


def validate_manifest(manifest):
    """Check the pinned download description before using its paths."""
    if manifest.get('schema_version') != 1 or set(manifest.get('assets', {})) != set(DIRECTORIES):
        raise ValueError('Expected a model and data release manifest')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', manifest.get('repository', '')):
        raise ValueError('Invalid GitHub repository')
    if not isinstance(manifest.get('tag'), str) or not manifest['tag']:
        raise ValueError('Missing release tag')
    for key, asset in manifest['assets'].items():
        directory = DIRECTORIES[key]
        if asset.get('directory') != directory or not asset.get('files'):
            raise ValueError('Invalid asset directory or empty file list')
        filename = asset.get('filename', '')
        if not re.fullmatch(r'[\w.-]+\.tar\.gz', filename):
            raise ValueError('Invalid archive filename')
        for name in asset['files']:
            relative_path(name)
            if not name.startswith(directory + '/'):
                raise ValueError(f'File is outside {directory}: {name}')
        for info in [asset, *asset['files'].values()]:
            if (type(info.get('size_bytes')) is not int or info['size_bytes'] < 0
                    or not re.fullmatch(r'[0-9a-f]{64}', info.get('sha256', ''))):
                raise ValueError('Invalid file size or checksum')


def unpack(archive, staging, asset):
    """Check the archive, then copy only the expected regular files."""
    expected = {key: asset[key] for key in ('size_bytes', 'sha256')}
    if file_info(archive) != expected:
        raise ValueError(f'Archive checksum or size does not match: {archive.name}')
    seen = set()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            if (member.name not in asset['files'] or member.name in seen
                    or not member.isfile()
                    or member.size != asset['files'][member.name]['size_bytes']):
                raise ValueError(f'Unexpected archive entry: {member.name}')
            path = checked_path(staging, member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, path.open('xb') as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            seen.add(member.name)
    if seen != set(asset['files']):
        raise ValueError('Archive is missing release files')
    verify_files(staging, asset)


def prepare(root, output, tag):
    """Package unchanged files into a new folder for manual release upload."""
    checkpoint = root / DIRECTORIES['model']
    checkpoint_task(SimpleNamespace(**json.loads((checkpoint / 'config.json').read_text())))
    if json.loads((checkpoint / 'training.json').read_text()).get('status') != 'complete':
        raise ValueError('The router checkpoint is not complete')
    output = output.resolve()
    for directory in DIRECTORIES.values():
        if output.is_relative_to(root / directory):
            raise ValueError('Release output must be outside the model and data folders')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'repository': REPOSITORY, 'tag': tag, 'assets': {}}
    for key, directory in DIRECTORIES.items():
        paths = ([checkpoint / name for name in MODEL_FILES] if key == 'model'
                 else sorted((root / directory).rglob('*')))
        files = {}
        for path in paths:
            name = path.relative_to(root).as_posix()
            checked_path(root, name)
            if key == 'model' or path.is_file():
                files[name] = file_info(path)
        if key == 'model':
            deployment = json.loads((checkpoint / 'deployment.json').read_text())
            for name, digest in deployment['asset_sha256'].items():
                if files[f'{directory}/{name}']['sha256'] != digest:
                    raise ValueError(f'Checkpoint differs from its deployment record: {name}')
        filename = 'modernbert-router.tar.gz' if key == 'model' else 'router-data.tar.gz'
        archive = output / filename
        with archive.open('xb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w') as stream:
                    for name in sorted(files):
                        path = root / name
                        entry = tarfile.TarInfo(name)
                        entry.size = files[name]['size_bytes']
                        entry.mode = 0o644
                        with path.open('rb') as source:
                            stream.addfile(entry, source)
        asset = {'directory': directory, 'filename': filename, **file_info(archive), 'files': files}
        if asset['size_bytes'] >= 2 * 1024**3:
            raise ValueError(f'{filename} exceeds the release asset size limit')
        verify_files(root, asset)
        manifest['assets'][key] = asset
        print(f"Prepared {filename}: {asset['size_bytes'] / 1e6:.1f} MB", flush=True)
    validate_manifest(manifest)
    write_json(output / 'release.json', manifest)
    (output / 'SHA256SUMS').write_text(''.join(
        f"{asset['sha256']}  {asset['filename']}\n" for asset in manifest['assets'].values()
    ), encoding='utf-8')
    return manifest


def download(manifest, root, *, source_dir=None, only=None):
    """Restore the pinned release without replacing existing files."""
    validate_manifest(manifest)
    selected = {key: asset for key, asset in manifest['assets'].items() if only in (None, key)}
    pending = {}
    for key, asset in selected.items():
        directory = checked_path(root, asset['directory'])
        if directory.exists():
            verify_files(root, asset)
            print(f'{key}: already matches {manifest["tag"]}', flush=True)
        else:
            pending[key] = asset
    if not pending:
        return
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.router-download-', dir=root) as temporary:
        staging = Path(temporary)
        for key, asset in pending.items():
            archive = staging / asset['filename']
            if source_dir is not None:
                archive = source_dir / asset['filename']
            else:
                url = (f'https://github.com/{manifest["repository"]}/releases/download/'
                       f'{quote(manifest["tag"], safe="")}/{asset["filename"]}')
                print(f'Downloading {asset["filename"]} ...', flush=True)
                try:
                    with urlopen(url, timeout=60) as response, archive.open('xb') as target:
                        shutil.copyfileobj(response, target, length=1024 * 1024)
                except HTTPError as error:
                    if error.code == 404:
                        raise ValueError(
                            f'Release {manifest["tag"]} is unavailable. Publish its assets first; '
                            'for a private repository, download them manually and use --from-dir.'
                        ) from error
                    raise
            unpack(archive, staging, asset)
        # All downloads must pass before installing either folder.
        for key, asset in pending.items():
            destination = checked_path(root, asset['directory'])
            if destination.exists():
                raise FileExistsError(f'Refusing to replace {destination}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            (staging / asset['directory']).rename(destination)
            print(f'{key}: restored to {destination}', flush=True)


def main(argv=None):
    """Prepare downloads for a release, or restore them on another machine."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    package = commands.add_parser('prepare', help='Create files for manual GitHub release upload')
    package.add_argument('--output', type=Path, required=True, help='New output folder')
    package.add_argument('--tag', default='router-v1')
    restore = commands.add_parser('download', help='Get the exact saved router and data')
    restore.add_argument('--only', choices=['model', 'data'])
    restore.add_argument('--destination', type=Path, default=ROOT)
    restore.add_argument('--from-dir', type=Path, help='Use archives already downloaded to this folder')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            prepare(ROOT, args.output, args.tag)
        else:
            download(json.loads(MANIFEST.read_text()), args.destination.resolve(),
                     source_dir=args.from_dir, only=args.only)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
