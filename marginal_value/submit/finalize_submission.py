from __future__ import annotations

import argparse
import csv
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from marginal_value.data.split_manifest import hash_manifest_url, read_manifest_urls


def finalize_submission_ids(
    *,
    submission_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    input_id_column: str = "worker_id",
    output_id_column: str = "worker_id",
) -> dict[str, object]:
    submission_rows = _read_csv(Path(submission_path))
    id_map = manifest_hash_to_url_stem(read_manifest_urls(manifest_path))
    remapped_rows = []
    missing_ids = []
    for row in submission_rows:
        internal_id = str(row.get(input_id_column, "")).strip()
        external_id = id_map.get(internal_id)
        if external_id is None:
            missing_ids.append(internal_id)
            continue
        remapped = dict(row)
        if output_id_column != input_id_column:
            remapped.pop(input_id_column, None)
        remapped[output_id_column] = external_id
        remapped_rows.append(remapped)
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(f"{len(missing_ids)} submission IDs were not present in the manifest hash map: {preview}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(submission_rows, input_id_column=input_id_column, output_id_column=output_id_column)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(remapped_rows)
    return {
        "input_rows": len(submission_rows),
        "output_rows": len(remapped_rows),
        "manifest_ids": len(id_map),
        "output_path": str(output),
        "output_id_column": output_id_column,
    }


def manifest_hash_to_url_stem(urls: list[str]) -> dict[str, str]:
    return {hash_manifest_url(url): _url_stem(url) for url in urls}


def _url_stem(url: str) -> str:
    parsed = urlparse(url)
    name = PurePosixPath(parsed.path).name
    return Path(name).stem


def _fieldnames(rows: list[dict[str, str]], *, input_id_column: str, output_id_column: str) -> list[str]:
    if not rows:
        return [output_id_column]
    fields = list(rows[0].keys())
    if output_id_column == input_id_column:
        return fields
    return [output_id_column if field == input_id_column else field for field in fields]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Map internal manifest-hash submission IDs back to manifest URL IDs.")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-id-column", default="worker_id")
    parser.add_argument("--output-id-column", default="worker_id")
    args = parser.parse_args()
    result = finalize_submission_ids(
        submission_path=args.submission,
        manifest_path=args.manifest,
        output_path=args.out,
        input_id_column=args.input_id_column,
        output_id_column=args.output_id_column,
    )
    print(result)


if __name__ == "__main__":
    main()
