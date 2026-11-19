# Packages or downloads the saved table index and selected cell model.
import argparse
import gzip
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

from table_retrieval.data import digest, load_json, write_json

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'release.json'
REPOSITORY = 'usm-aiir/scientific-discovery'
LIMIT = 2 * 1024**3
DIRECTORIES = {'index': 'model/index', 'model': 'model/cell_model'}
MODEL_FILES = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json')
INDEX_FILES = ('prepared.json', 'index.json', 'table_ids.json', 'table_inputs.json',
               'truncated_tables.json', 'embeddings.npy', 'bm25_full.npz',
               'bm25_matched.npz', 'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json')


# Records a file's size and SHA-256 checksum.
def file_info(path):
    return {'size_bytes': path.stat().st_size, 'sha256': digest(path)}


# Accepts only plain relative release paths.
def relative_path(name):
    if not isinstance(name, str):
        raise ValueError('Release paths must be strings')
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or '\\' in name
            or not path.parts or str(path) != name):
        raise ValueError(f'Invalid release path: {name}')
    return path


# Keeps release paths inside their root and refuses symbolic links.
def checked_path(root, name):
    root = Path(root).absolute()
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise ValueError(f'Release path is a symbolic link: {parent}')
    path = root
    for part in relative_path(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Release path is a symbolic link: {path}')
    return path


# Requires an exact byte size and SHA-256 checksum.
def check_info(info):
    if (not isinstance(info, dict) or type(info.get('size_bytes')) is not int
            or info['size_bytes'] < 0 or not isinstance(info.get('sha256'), str)
            or re.fullmatch(r'[0-9a-f]{64}', info['sha256']) is None):
        raise ValueError('Invalid file size or checksum')


# Validates the pinned files and download parts before touching destinations.
def validate_manifest(manifest):
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
            or set(manifest.get('assets', {})) != set(DIRECTORIES)):
        raise ValueError('Expected a table index and cell model release')
    if (not isinstance(manifest.get('repository'), str)
            or re.fullmatch(r'[\w.-]+/[\w.-]+', manifest['repository']) is None
            or not isinstance(manifest.get('tag'), str)
            or re.fullmatch(r'[\w.-]+', manifest['tag']) is None):
        raise ValueError('Invalid release repository or tag')
    filenames = set()
    for key, asset in manifest['assets'].items():
        directory = DIRECTORIES[key]
        if (not isinstance(asset, dict) or asset.get('directory') != directory
                or not isinstance(asset.get('files'), dict) or not asset['files']
                or not isinstance(asset.get('parts'), list) or not asset['parts']):
            raise ValueError('Invalid release directory, files or download parts')
        check_info(asset)
        filename = asset.get('filename', '')
        if not isinstance(filename, str) or re.fullmatch(r'[\w.-]+\.tar\.gz', filename) is None:
            raise ValueError('Invalid archive filename')
        for name, info in asset['files'].items():
            relative_path(name)
            if not name.startswith(directory + '/'):
                raise ValueError(f'File is outside its release directory: {name}')
            check_info(info)
        for part in asset['parts']:
            check_info(part)
            name = part.get('filename', '')
            if (not isinstance(name, str) or re.fullmatch(r'[\w.-]+', name) is None
                    or name in filenames or part['size_bytes'] >= LIMIT):
                raise ValueError('Invalid or duplicate download part')
            if len(relative_path(name).parts) != 1:
                raise ValueError('Download parts need plain filenames')
            filenames.add(name)
        if sum(part['size_bytes'] for part in asset['parts']) != asset['size_bytes']:
            raise ValueError('Download parts do not match the archive size')
        required = INDEX_FILES if key == 'index' else (*MODEL_FILES, 'selection.json')
        if set(asset['files']) != {directory + '/' + name for name in required}:
            raise ValueError('Release does not contain the required artifact files')


# Checks that every installed file matches the pinned release.
def verify_files(root, asset):
    for name, expected in asset['files'].items():
        path = checked_path(root, name)
        if not path.is_file() or file_info(path) != expected:
            raise ValueError(f'File differs from the release: {path}')


# Verifies index and selected-model provenance without reading question labels.
def source_files(index, checkpoint, training_record):
    training_record = checked_path(training_record.parent, training_record.name)
    prepared = load_json(checked_path(index, 'prepared.json'))
    metadata = load_json(checked_path(index, 'index.json'))
    ids_path = checked_path(index, 'table_ids.json')
    ids = load_json(ids_path)
    if (metadata.get('prepared_sha256') != digest(index / 'prepared.json')
            or prepared.get('ids_sha256') != digest(ids_path)
            or not isinstance(ids, list) or not ids
            or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(ids) != len(set(ids))
            or prepared.get('table_count') != len(ids) or metadata.get('table_count') != len(ids)
            or not isinstance(prepared.get('corpus_sha256'), str)
            or re.fullmatch(r'[0-9a-f]{64}', prepared['corpus_sha256']) is None):
        raise ValueError('The saved table index has inconsistent provenance')
    training = load_json(training_record)
    if (training.get('checkpoint') != checkpoint.name
            or not str(training.get('config', {}).get('selection_metric', '')).startswith('validation')
            or set(training.get('checkpoint_sha256', {})) != set(MODEL_FILES)):
        raise ValueError('The cell checkpoint is not the recorded validation-selected model')
    record = checked_path(training_record.parent, 'data_manifest.json')
    if not record.is_file() or digest(record) != training.get('data_manifest_sha256'):
        raise ValueError('The training data manifest differs from its recorded checksum')
    data = load_json(record)
    if (data.get('corpus_sha256') != prepared['corpus_sha256']
            or data.get('validation', {}).get('stage1_prepared_sha256') != digest(index / 'prepared.json')
            or data.get('test_data_used') is not False):
        raise ValueError('Selected model provenance does not match this index and corpus')
    files = {'index': {}, 'model': {}}
    for name in INDEX_FILES:
        files['index'][DIRECTORIES['index'] + '/' + name] = checked_path(index, name)
    for name in MODEL_FILES:
        path = checked_path(checkpoint, name)
        if not path.is_file() or digest(path) != training['checkpoint_sha256'][name]:
            raise ValueError(f'Checkpoint differs from its selection record: {name}')
        files['model'][DIRECTORIES['model'] + '/' + name] = path
    files['model'][DIRECTORIES['model'] + '/selection.json'] = training_record
    return files, prepared, training


# Writes a deterministic archive without altering any source file.
def make_archive(path, sources, expected):
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for name in sorted(sources):
                    entry = tarfile.TarInfo(name)
                    entry.size = expected[name]['size_bytes']
                    entry.mode = 0o644
                    with sources[name].open('rb') as source:
                        archive.addfile(entry, source)


# Packages the recorded index and selected cell model for manual upload.
def prepare(index, checkpoint, training_record, output, tag='table-retrieval-v1'):
    index, checkpoint = Path(index).absolute(), Path(checkpoint).absolute()
    training_record, output = Path(training_record).absolute(), Path(output).absolute()
    if not isinstance(tag, str) or re.fullmatch(r'[\w.-]+', tag) is None:
        raise ValueError('Invalid release tag')
    for source in (index, checkpoint):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError('Release output must be outside source artifacts')
    sources, prepared, training = source_files(index, checkpoint, training_record)
    original = {key: {name: file_info(path) for name, path in paths.items()}
                for key, paths in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'repository': REPOSITORY, 'tag': tag,
                'provenance': {'source_index': str(index), 'source_checkpoint': str(checkpoint),
                               'corpus_sha256': prepared['corpus_sha256'],
                               'training_data_manifest_sha256': training['data_manifest_sha256'],
                               'selected_checkpoint': training['checkpoint']}, 'assets': {}}
    for key, paths in sources.items():
        filename = 'table-index.tar.gz' if key == 'index' else 'table-cell-model.tar.gz'
        archive = output / filename
        make_archive(archive, paths, original[key])
        if archive.stat().st_size >= LIMIT:
            raise ValueError(f'{filename} exceeds the 2 GiB release attachment limit')
        info = file_info(archive)
        asset = {'directory': DIRECTORIES[key], 'filename': filename, **info,
                 'files': original[key], 'parts': [{'filename': filename, **info}]}
        for name, path in paths.items():
            if file_info(path) != original[key][name]:
                raise ValueError(f'Source changed during packaging: {path}')
        manifest['assets'][key] = asset
        print(f'Prepared {filename}: {asset["size_bytes"] / 1e6:.1f} MB', flush=True)
    validate_manifest(manifest)
    write_json(output / 'release.json', manifest)
    (output / 'SHA256SUMS').write_text(''.join(
        f'{part["sha256"]}  {part["filename"]}\n'
        for asset in manifest['assets'].values() for part in asset['parts']), encoding='utf-8')
    return manifest


# Copies or downloads one release file and verifies it before assembly.
def fetch_part(part, destination, manifest, source_dir):
    if source_dir is not None:
        source = checked_path(source_dir, part['filename'])
        with source.open('rb') as stream, destination.open('xb') as target:
            shutil.copyfileobj(stream, target, length=1024 * 1024)
    else:
        url = (f'https://github.com/{manifest["repository"]}/releases/download/'
               f'{quote(manifest["tag"], safe="")}/{part["filename"]}')
        print(f'Downloading {part["filename"]} ...', flush=True)
        try:
            with urlopen(url, timeout=60) as stream, destination.open('xb') as target:
                shutil.copyfileobj(stream, target, length=1024 * 1024)
        except HTTPError as error:
            if error.code == 404:
                raise ValueError('Table release unavailable. Publish its assets first; for a private '
                                 'repository, download them manually and use --from-dir.') from error
            raise
    expected = {key: part[key] for key in ('size_bytes', 'sha256')}
    if file_info(destination) != expected:
        raise ValueError(f'Download part differs from the release: {part["filename"]}')


# Extracts only the expected files after checking the assembled archive.
def unpack(archive, staging, asset):
    if file_info(archive) != {key: asset[key] for key in ('size_bytes', 'sha256')}:
        raise ValueError('Archive checksum or size does not match')
    seen = set()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            if (member.name not in asset['files'] or member.name in seen or not member.isfile()
                    or member.size != asset['files'][member.name]['size_bytes']):
                raise ValueError(f'Unexpected archive entry: {member.name}')
            path = checked_path(staging, member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            seen.add(member.name)
    if seen != set(asset['files']):
        raise ValueError('Archive is missing release files')
    verify_files(staging, asset)


# Restores checked artifacts into a new location without replacing local files.
def download(manifest, root=ROOT, *, source_dir=None, only=None):
    validate_manifest(manifest)
    if only not in (None, *DIRECTORIES):
        raise ValueError('Choose only index or model')
    root = Path(root).absolute()
    checked_path(root, 'model')
    pending = {}
    for key, asset in manifest['assets'].items():
        if only not in (None, key):
            continue
        destination = checked_path(root, asset['directory'])
        if destination.exists():
            verify_files(root, asset)
            if any(path.is_symlink() for path in destination.rglob('*')):
                raise ValueError(f'Existing artifact contains symbolic links: {destination}')
            expected = {str(Path(name).relative_to(asset['directory'])) for name in asset['files']}
            actual = {str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file()}
            if actual != expected:
                raise ValueError(f'Existing artifact has unexpected files: {destination}')
            print(f'{key}: already matches {manifest["tag"]}', flush=True)
        else:
            pending[key] = asset
    if not pending:
        return
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.table-download-', dir=root) as temporary:
        staging = Path(temporary)
        for key, asset in pending.items():
            archive = staging / (key + '.tar.gz')
            with archive.open('xb') as target:
                for position, part in enumerate(asset['parts']):
                    piece = staging / f'{key}.{position}.part'
                    fetch_part(part, piece, manifest, source_dir)
                    with piece.open('rb') as stream:
                        shutil.copyfileobj(stream, target, length=1024 * 1024)
                    piece.unlink()
            unpack(archive, staging, asset)
            archive.unlink()
        # Every selected artifact must pass before any is installed.
        for asset in pending.values():
            destination = checked_path(root, asset['directory'])
            if destination.exists():
                raise FileExistsError(f'Refusing to replace {destination}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            (staging / asset['directory']).rename(destination)
            print(f'Restored {destination}', flush=True)


# Prepares a manual release or downloads its pinned table artifacts.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    package = commands.add_parser('prepare', help='Create files for manual release upload')
    package.add_argument('--index', type=Path, required=True)
    package.add_argument('--checkpoint', type=Path, required=True)
    package.add_argument('--training-record', type=Path, required=True)
    package.add_argument('--output', type=Path, required=True, help='New release folder')
    package.add_argument('--tag', default='table-retrieval-v1')
    restore = commands.add_parser('download', help='Get the saved table index and cell model')
    restore.add_argument('--only', choices=['index', 'model'])
    restore.add_argument('--destination', type=Path, default=ROOT)
    restore.add_argument('--from-dir', type=Path, help='Use already downloaded release files')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            return prepare(args.index, args.checkpoint, args.training_record, args.output, args.tag)
        return download(load_json(MANIFEST), args.destination, source_dir=args.from_dir, only=args.only)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
