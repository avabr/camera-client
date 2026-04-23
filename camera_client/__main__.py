"""Command-line interface for camera-client package."""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlparse


def is_url(text: str) -> bool:
    """
    Check if a string is a valid URL.

    Args:
        text: String to check

    Returns:
        True if the string looks like a URL, False otherwise
    """
    text = text.strip()
    return text.startswith('http://') or text.startswith('https://')


def download_archive(url: str, output_dir: str = ".", silent: bool = False) -> bool:
    """
    Download camera calibration archive from URL to the specified directory.

    Args:
        url: URL to download the archive from
        output_dir: Directory to save the downloaded file (default: current directory)
        silent: If True, suppress success messages (errors still printed)

    Returns:
        True if download succeeded, False otherwise
    """
    try:
        # Parse the URL to extract filename from path or Content-Disposition header
        parsed_url = urlparse(url)

        # Create request with headers
        request = Request(url, headers={'User-Agent': 'camera-client'})

        # Open URL and get response
        if not silent:
            print(f"Downloading from: {url}")
        with urlopen(request) as response:
            # Try to get filename from Content-Disposition header
            content_disposition = response.headers.get('Content-Disposition', '')
            filename = None

            if 'filename=' in content_disposition:
                # Extract filename from Content-Disposition header
                parts = content_disposition.split('filename=')
                if len(parts) > 1:
                    filename = parts[1].strip('"')

            # Fallback to URL path if no Content-Disposition
            if not filename:
                path_parts = parsed_url.path.split('/')
                filename = path_parts[-1] if path_parts[-1] else 'camera_archive.npz'

            # Ensure .npz extension
            if not filename.endswith('.npz'):
                filename += '.npz'

            # Create output path
            output_path = Path(output_dir) / filename

            # Download file
            if not silent:
                print(f"Saving to: {output_path}")
            with open(output_path, 'wb') as f:
                f.write(response.read())

            if not silent:
                print(f"Successfully downloaded: {filename}")
                print(f"File size: {output_path.stat().st_size} bytes")

            return True

    except Exception as e:
        print(f"Error downloading archive from {url}: {e}", file=sys.stderr)
        return False


def download_from_file(file_path: str, output_dir: str = ".", camera_id: int = None) -> None:
    """
    Download camera calibration archives from a .txt file with URLs or a .json config.

    For .txt files: one URL per line, non-URL lines are ignored.
    For .json files: expects a list of objects with "archive_url" key.
        Optionally filter by camera_id.

    Args:
        file_path: Path to .txt or .json file
        output_dir: Directory to save the downloaded files (default: current directory)
        camera_id: If provided, only download archives for this camera_id (JSON only)
    """
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                configs = json.load(f)

            if not isinstance(configs, list):
                configs = [configs]

            if camera_id is not None:
                configs = [c for c in configs if c.get('camera_id') == camera_id]

            urls = [c['archive_url'] for c in configs if 'archive_url' in c]

            if not urls:
                msg = f"No matching entries found in {file_path}"
                if camera_id is not None:
                    msg += f" for camera_id={camera_id}"
                print(msg, file=sys.stderr)
                sys.exit(1)
        else:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if is_url(line)]

            if not urls:
                print(f"No URLs found in {file_path}", file=sys.stderr)
                sys.exit(1)

        print(f"Found {len(urls)} URL(s) in {file_path}")
        print(f"Downloading to: {output_dir}\n")

        success_count = 0
        failed_count = 0

        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Processing: {url}")
            if download_archive(url, output_dir, silent=False):
                success_count += 1
            else:
                failed_count += 1
            print()  # Empty line between downloads

        # Summary
        print("=" * 50)
        print(f"Download complete: {success_count} succeeded, {failed_count} failed")

        if failed_count > 0:
            sys.exit(1)

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for camera-client CLI."""
    parser = argparse.ArgumentParser(
        prog='camera-client',
        description='Camera calibration client utilities'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # get_camera_archive command
    download_parser = subparsers.add_parser(
        'get_camera_archive',
        help='Get camera calibration archive(s) from URL or file'
    )
    download_parser.add_argument(
        'url',
        nargs='?',
        help='URL to download the archive from'
    )
    download_parser.add_argument(
        '-f', '--file',
        help='File containing URLs (one per line)'
    )
    download_parser.add_argument(
        '-o', '--output-dir',
        default='.',
        help='Output directory (default: current directory)'
    )
    download_parser.add_argument(
        '--camera_id',
        type=int,
        default=None,
        help='Filter by camera_id (only used with JSON config files)'
    )

    args = parser.parse_args()

    if args.command == 'get_camera_archive':
        # Check that either url or file is provided (but not both)
        if args.url and args.file:
            print("Error: Cannot specify both URL and file. Use either positional URL or -f/--file option.", file=sys.stderr)
            sys.exit(1)
        elif not args.url and not args.file:
            print("Error: Must specify either URL or file with -f/--file option.", file=sys.stderr)
            download_parser.print_help()
            sys.exit(1)

        # Process based on input type
        if args.file:
            download_from_file(args.file, args.output_dir, camera_id=args.camera_id)
        else:
            success = download_archive(args.url, args.output_dir)
            sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
