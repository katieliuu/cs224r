"""
protein/fetch/fetch.py

Download PDBx/mmCIF coordinate files from RCSB PDB and verify that the
atom_site table still contains per-atom B-factor values.

The official RCSB file download service exposes uncompressed mmCIF files at:
    https://files.rcsb.org/download/<pdb_id>.cif

Example:
    python protein/fetch/fetch.py 4hhb --output-dir data/proteins
"""

from __future__ import annotations

import argparse
import gzip
import re
import shlex
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


DEFAULT_DOWNLOAD_ROOT = "https://files.rcsb.org/download"
DEFAULT_TIMEOUT_SECONDS = 30.0
MISSING_MMCIF_VALUES = frozenset({"?", "."})
FLOAT_WITH_OPTIONAL_ESD = re.compile(
    r"^(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\(\d+\))?$"
)


class FetchError(RuntimeError):
    """Raised when an mmCIF file cannot be downloaded from RCSB."""


class BFactorError(FetchError):
    """Raised when the downloaded mmCIF file does not contain usable B-factors."""


@dataclass(frozen=True, slots=True)
class BFactorSummary:
    """Summary statistics for the atom_site B-factor column."""

    atom_rows: int
    numeric_values: int
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    column_name: str = "_atom_site.B_iso_or_equiv"

    @property
    def has_numeric_values(self) -> bool:
        """Return True when at least one atom row has a numeric B-factor."""
        return self.numeric_values > 0


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Result object returned after a successful download."""

    pdb_id: str
    output_path: Path
    source_url: str
    b_factors: BFactorSummary | None = None


def normalize_pdb_id(pdb_id: str) -> str:
    """
    Normalize a user-supplied PDB identifier or file stem.

    Supports classic 4-character PDB IDs like ``4hhb`` and explicit file stems
    like ``pdb_00004hhb``.
    """
    token = pdb_id.strip()
    if not token:
        raise ValueError("PDB ID cannot be empty.")

    lower_token = token.lower()
    for suffix in (".cif.gz", ".cif"):
        if lower_token.endswith(suffix):
            token = token[: -len(suffix)]
            lower_token = token.lower()
            break

    if re.fullmatch(r"[a-z0-9]{4}", lower_token):
        return lower_token

    if re.fullmatch(r"pdb_[a-z0-9]{8,12}", lower_token):
        return lower_token

    raise ValueError(
        "Unsupported PDB identifier. Use a classic 4-character ID like "
        "'4hhb' or an explicit file stem like 'pdb_00004hhb'."
    )


def default_output_path(pdb_id: str, output_dir: str | Path | None = None) -> Path:
    """Build the default output path for a downloaded mmCIF file."""
    normalized = normalize_pdb_id(pdb_id)
    directory = Path.cwd() if output_dir is None else Path(output_dir)
    return directory / f"{normalized}.cif"


def build_download_urls(pdb_id: str) -> list[tuple[str, bool]]:
    """
    Build candidate RCSB download URLs.

    The short ``<pdb_id>.cif`` URL is tried first, followed by the explicit
    ``pdb_0000<pdb_id>.cif`` style that RCSB also documents for classic IDs.
    Gzipped variants are used as fallbacks and transparently decompressed.
    """
    normalized = normalize_pdb_id(pdb_id)
    stems = [normalized]

    if len(normalized) == 4:
        stems.append(f"pdb_0000{normalized}")

    urls: list[tuple[str, bool]] = []
    for stem in stems:
        urls.append((f"{DEFAULT_DOWNLOAD_ROOT}/{stem}.cif", False))
        urls.append((f"{DEFAULT_DOWNLOAD_ROOT}/{stem}.cif.gz", True))
    return urls


def fetch_mmcif(
    pdb_id: str,
    output_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    overwrite: bool = False,
    ensure_b_factors: bool = True,
) -> FetchResult:
    """
    Download an RCSB PDBx/mmCIF file and optionally verify B-factors.

    Args:
        pdb_id: Classic 4-character PDB ID or explicit RCSB file stem.
        output_path: Full path to the output ``.cif`` file.
        output_dir: Directory used when ``output_path`` is not provided.
        timeout: Network timeout in seconds.
        overwrite: Whether to replace an existing file.
        ensure_b_factors: If True, verify ``_atom_site.B_iso_or_equiv`` and
            collect summary statistics.

    Returns:
        FetchResult describing the downloaded file.
    """
    normalized = normalize_pdb_id(pdb_id)
    target_path = (
        Path(output_path)
        if output_path is not None
        else default_output_path(normalized, output_dir=output_dir)
    )

    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file without overwrite=True: {target_path}"
        )

    payload, source_url = _download_mmcif_bytes(normalized, timeout=timeout)
    cif_text = payload.decode("utf-8")
    b_factor_summary = summarize_b_factors(cif_text) if ensure_b_factors else None

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(payload)

    return FetchResult(
        pdb_id=normalized,
        output_path=target_path,
        source_url=source_url,
        b_factors=b_factor_summary,
    )


def summarize_b_factors(cif_text: str) -> BFactorSummary:
    """
    Summarize ``_atom_site.B_iso_or_equiv`` values from an mmCIF file.

    Raises:
        BFactorError: If the atom_site loop or B-factor column is missing, or if
            the column exists but contains no numeric values.
    """
    lines = cif_text.splitlines()
    line_count = len(lines)
    index = 0

    while index < line_count:
        stripped = lines[index].strip()
        if stripped != "loop_":
            index += 1
            continue

        index += 1
        columns: list[str] = []

        while index < line_count:
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if stripped.startswith("_"):
                columns.append(stripped.split()[0])
                index += 1
                continue
            break

        if not columns or not any(column.startswith("_atom_site.") for column in columns):
            continue

        if "_atom_site.B_iso_or_equiv" not in columns:
            raise BFactorError(
                "Downloaded mmCIF contains an atom_site loop but no "
                "'_atom_site.B_iso_or_equiv' column."
            )

        column_count = len(columns)
        b_factor_index = columns.index("_atom_site.B_iso_or_equiv")
        atom_rows = 0
        values: list[float] = []
        token_buffer: list[str] = []

        while index < line_count:
            raw_line = lines[index]
            stripped = raw_line.strip()

            if not stripped:
                index += 1
                continue

            if stripped == "#":
                break

            if stripped == "loop_" or stripped.startswith("data_"):
                break

            if stripped.startswith("_") and not token_buffer:
                break

            if raw_line.startswith(";"):
                raise BFactorError(
                    "Encountered unexpected multiline text while parsing atom_site."
                )

            token_buffer.extend(_tokenize_mmcif_line(raw_line))

            while len(token_buffer) >= column_count:
                row = token_buffer[:column_count]
                del token_buffer[:column_count]
                atom_rows += 1
                parsed_value = _parse_cif_float(row[b_factor_index])
                if parsed_value is not None:
                    values.append(parsed_value)

            index += 1

        if token_buffer:
            raise BFactorError("atom_site loop appears truncated or malformed.")

        if not values:
            raise BFactorError(
                "Downloaded mmCIF has a B-factor column, but no numeric "
                "B-factor values were found."
            )

        return BFactorSummary(
            atom_rows=atom_rows,
            numeric_values=len(values),
            min_value=min(values),
            max_value=max(values),
            mean_value=mean(values),
        )

    raise BFactorError("No atom_site loop was found in the downloaded mmCIF file.")


def _download_mmcif_bytes(pdb_id: str, *, timeout: float) -> tuple[bytes, str]:
    """Try documented RCSB download URLs until one succeeds."""
    errors: list[str] = []

    for url, is_gzipped in build_download_urls(pdb_id):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "small-molecule-mod-toolkit-cs273b/1.0 "
                    "(protein fetcher; contact local user)"
                )
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            errors.append(f"{url} -> HTTP {exc.code}")
            continue
        except urllib.error.URLError as exc:
            errors.append(f"{url} -> {exc.reason}")
            continue

        if is_gzipped:
            try:
                payload = gzip.decompress(payload)
            except OSError as exc:
                raise FetchError(f"Failed to decompress {url}: {exc}") from exc

        return payload, url

    raise FetchError(
        "Unable to download an mmCIF file from RCSB for "
        f"{pdb_id}. Attempted URLs: {', '.join(errors)}"
    )


def _tokenize_mmcif_line(line: str) -> list[str]:
    """Split one mmCIF data line into tokens while respecting quotes."""
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _parse_cif_float(token: str) -> float | None:
    """Parse a numeric mmCIF token, including values with uncertainty suffixes."""
    if token in MISSING_MMCIF_VALUES:
        return None

    match = FLOAT_WITH_OPTIONAL_ESD.fullmatch(token)
    if not match:
        return None

    return float(match.group("value"))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the fetch CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Download a PDBx/mmCIF coordinate file from RCSB and verify that "
            "the file contains atom-level B-factors."
        )
    )
    parser.add_argument(
        "pdb_id",
        help="Classic 4-character PDB ID such as 4hhb.",
    )

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Exact path for the downloaded .cif file.",
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where <pdb_id>.cif will be written.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Network timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    parser.add_argument(
        "--skip-b-factor-check",
        action="store_true",
        help="Download the file without validating the atom_site B-factor column.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    result = fetch_mmcif(
        args.pdb_id,
        output_path=args.output,
        output_dir=args.output_dir,
        timeout=args.timeout,
        overwrite=args.overwrite,
        ensure_b_factors=not args.skip_b_factor_check,
    )

    print(f"Downloaded {result.pdb_id} from {result.source_url}")
    print(f"Saved mmCIF to {result.output_path.resolve()}")

    if result.b_factors is not None:
        summary = result.b_factors
        print(
            "B-factor summary: "
            f"rows={summary.atom_rows}, "
            f"numeric={summary.numeric_values}, "
            f"min={summary.min_value:.2f}, "
            f"mean={summary.mean_value:.2f}, "
            f"max={summary.max_value:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
