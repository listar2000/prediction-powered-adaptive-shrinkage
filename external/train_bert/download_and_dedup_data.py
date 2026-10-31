#!/usr/bin/env python3
"""Download and deduplicate all SNAP Amazon Fine Foods reviews before splitting.

Uses only the Python standard library. See README.md for the selection rules.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterator
from urllib.request import urlopen


SOURCE_URL = "https://snap.stanford.edu/data/finefoods.txt.gz"
CSV_COLUMNS = [
    "Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator",
    "HelpfulnessDenominator", "Score", "Time", "Summary", "Text",
]
# Exact strings, with the rating converted to an integer. No text normalization.
ReviewKey = tuple[str, str, int, str]  # title, text, rating, user ID
Metadata = tuple[str, int, int, int]  # profile name, helpful votes, total votes, time


@dataclass
class Corpus:
    keys: list[ReviewKey]
    products: dict[str, dict[int, Metadata]]
    raw_counts: Counter
    helpfulness_anomalies: int = 0


@dataclass
class Deduplication:
    products: dict[str, dict[int, Metadata]]
    canonical: dict[str, str]
    dropped: dict[str, tuple[str, int]]
    sparse_removed: dict[str, int]
    summary: dict


def download(destination: Path) -> None:
    """Cache the archive, publishing it only after a successful download."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as out:
            temporary = Path(out.name)
            with urlopen(SOURCE_URL, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def iter_snap_records(source: Path, encoding: str = "latin-1") -> Iterator[dict[str, str]]:
    """Read records, including unescaped multiline profile names and review text.

    A new product field starts a record. Blank lines alone cannot delimit records:
    they can occur inside a multiline value. Blank lines preceding the next
    product field (or EOF) are record separators, not part of the final value.
    """
    fields = {"product/productId", "review/userId", "review/profileName",
              "review/helpfulness", "review/score", "review/time",
              "review/summary", "review/text"}
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding=encoding, errors="strict", newline="\n") as stream:
        record: dict[str, str] = {}
        previous_field = None
        pending_blank_lines = 0
        for line_number, line in enumerate(stream, 1):
            line = line.rstrip("\r\n")
            if not line:
                pending_blank_lines += 1
                continue
            name, separator, value = line.partition(": ")
            if separator and name == "product/productId":
                if record:
                    yield record
                record = {name: value}
                previous_field = name
            elif separator and name in fields:
                if previous_field is None or name in record:
                    raise ValueError(f"Malformed or duplicate field at line {line_number}")
                if pending_blank_lines:
                    record[previous_field] += "\n" * pending_blank_lines
                record[name] = value
                previous_field = name
            else:
                if previous_field is None:
                    raise ValueError(f"Content outside a record at line {line_number}")
                record[previous_field] += "\n" * (pending_blank_lines + 1) + line
            pending_blank_lines = 0
        if record:
            yield record


def load_corpus(records: Iterator[dict[str, str]]) -> Corpus:
    """Intern review identities and remove repeated rows within each product."""
    key_ids: dict[ReviewKey, int] = {}
    keys: list[ReviewKey] = []
    products: dict[str, dict[int, Metadata]] = {}
    raw_counts: Counter = Counter()
    helpfulness_anomalies = 0
    for row_number, row in enumerate(records, 1):
        try:
            product = row["product/productId"]
            user = row["review/userId"]
            rating_number = float(row["review/score"])
            if not product or not user or rating_number not in (1, 2, 3, 4, 5):
                raise ValueError("missing identifier or invalid rating")
            key = (row["review/summary"], row["review/text"], int(rating_number), user)
            helpful, total = map(int, row["review/helpfulness"].split("/"))
            if not 0 <= helpful <= total:
                # Preserve source metadata; helpfulness is not a deduplication key.
                helpfulness_anomalies += 1
            metadata = (row["review/profileName"], helpful, total, int(row["review/time"]))
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid source record {row_number}: {error}") from error
        review_id = key_ids.get(key)
        if review_id is None:
            review_id = len(keys)
            key_ids[key] = review_id
            keys.append(key)
        raw_counts[product] += 1
        # On repeated identities within a product, retain the first row's metadata.
        products.setdefault(product, {}).setdefault(review_id, metadata)
    if not products:
        raise ValueError("Source contains no reviews")
    return Corpus(keys, products, raw_counts, helpfulness_anomalies)


def count_summary(counts) -> dict:
    ordered = sorted(counts, reverse=True)
    if not ordered:
        return {"products": 0, "reviews": 0, "min": None, "max": None,
                "top_200_reviews": 0, "rank_200": None, "rank_201": None}
    return {
        "products": len(ordered), "reviews": sum(ordered),
        "min": ordered[-1], "max": ordered[0],
        "top_200_reviews": sum(ordered[:200]),
        "rank_200": ordered[199] if len(ordered) >= 200 else None,
        "rank_201": ordered[200] if len(ordered) >= 201 else None,
    }


def deduplicate(corpus: Corpus, large_overlap_threshold: int = 10) -> Deduplication:
    """Merge equal sets, drop large-overlap products, remove all residual sharing.

    Large means strictly more than the threshold, measured BEFORE removing shared
    rows. Products are considered by decreasing distinct-review count, breaking
    ties by ascending ProductId. Only previously kept products can cause a drop.
    """
    if large_overlap_threshold < 0:
        raise ValueError("large_overlap_threshold must be nonnegative")
    canonical: dict[str, str] = {}
    signatures: dict[frozenset[int], str] = {}
    representatives: dict[str, dict[int, Metadata]] = {}
    for product in sorted(corpus.products):
        signature = frozenset(corpus.products[product])
        representative = signatures.setdefault(signature, product)
        canonical[product] = representative
        if representative == product:
            representatives[product] = corpus.products[product]
    del signatures

    # An inverted index avoids comparing all pairs of the ~74,000 input products.
    owners: dict[int, list[str]] = defaultdict(list)
    kept: dict[str, dict[int, Metadata]] = {}
    dropped: dict[str, tuple[str, int]] = {}
    for product in sorted(representatives, key=lambda p: (-len(representatives[p]), p)):
        reviews = representatives[product]
        overlaps: Counter = Counter()
        for review_id in reviews:
            overlaps.update(owners.get(review_id, ()))
        conflicts = [other for other, count in overlaps.items()
                     if count > large_overlap_threshold]
        if conflicts:
            # Record a deterministic witness explaining why this product was dropped.
            witness = min(conflicts, key=lambda p: (-overlaps[p], -len(kept[p]), p))
            dropped[product] = (witness, overlaps[witness])
        else:
            kept[product] = reviews
            for review_id in reviews:
                owners[review_id].append(product)

    shared = {review_id for review_id, products in owners.items() if len(products) > 1}
    sparse_removed = {p: sum(r in shared for r in reviews) for p, reviews in kept.items()}
    final = {p: {r: metadata for r, metadata in reviews.items() if r not in shared}
             for p, reviews in kept.items()}
    empty_products = sum(not reviews for reviews in final.values())
    final = {p: reviews for p, reviews in final.items() if reviews}

    raw_rows = sum(corpus.raw_counts.values())
    unique_within_rows = sum(map(len, corpus.products.values()))
    merged_rows = sum(map(len, representatives.values()))
    kept_rows = sum(map(len, kept.values()))
    final_rows = sum(map(len, final.values()))
    # Fail loudly if the resulting dataset is not globally review-disjoint.
    seen: set[int] = set()
    for product, reviews in final.items():
        if seen.intersection(reviews):
            raise RuntimeError(f"Residual cross-product overlap at {product}")
        seen.update(reviews)
    if final_rows != len(seen):
        raise RuntimeError("Output contains duplicate review identities")
    removed = {
        "within_product_duplicate_rows": raw_rows - unique_within_rows,
        "identical_product_duplicate_rows": unique_within_rows - merged_rows,
        "large_overlap_product_rows": merged_rows - kept_rows,
        "remaining_shared_rows_from_all_products": sum(sparse_removed.values()),
    }
    if raw_rows != final_rows + sum(removed.values()):
        raise RuntimeError("Review accounting does not balance")
    removed_products = len(corpus.products) - len(representatives) + len(dropped) + empty_products
    if len(corpus.products) != len(final) + removed_products:
        raise RuntimeError("Product accounting does not balance")
    summary = {
        "identity_fields": ["Summary", "Text", "Score", "UserId"],
        "large_overlap_threshold": large_overlap_threshold,
        "large_overlap_comparison": "strictly greater than threshold",
        "before": count_summary(corpus.raw_counts.values()),
        "distinct_review_identities_before": len(corpus.keys),
        "source_rows_with_inconsistent_helpfulness_counts": corpus.helpfulness_anomalies,
        "after_within_product_dedup": count_summary(map(len, corpus.products.values())),
        "after_identical_product_merge": count_summary(map(len, representatives.values())),
        "after_large_overlap_drops": count_summary(map(len, kept.values())),
        "after": count_summary(map(len, final.values())),
        "removed_rows": removed,
        "merged_product_ids": len(corpus.products) - len(representatives),
        "products_dropped_for_large_overlap": len(dropped),
        "products_affected_by_remaining_sharing": sum(n > 0 for n in sparse_removed.values()),
        "distinct_remaining_shared_reviews_removed": len(shared),
        "products_emptied_by_shared_review_removal": empty_products,
        "validation": {"globally_unique_review_identities": True,
                       "review_accounting_balances": True,
                       "product_accounting_balances": True},
    }
    return Deduplication(final, canonical, dropped, sparse_removed, summary)


def write_csv(path: Path, columns: list[str], rows) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export(corpus: Corpus, result: Deduplication, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def review_rows():
        row_id = 0
        for product in sorted(result.products):
            for review_id, metadata in sorted(result.products[product].items()):
                row_id += 1
                title, text, rating, user = corpus.keys[review_id]
                profile, helpful, total, timestamp = metadata
                yield [row_id, product, user, profile, helpful, total,
                       rating, timestamp, title, text]

    def mapping_rows():
        for product in sorted(corpus.products):
            representative = result.canonical[product]
            witness, overlap = result.dropped.get(representative, ("", 0))
            status = ("dropped_large_overlap" if witness else
                      "kept" if representative in result.products else "emptied_by_shared_review_removal")
            yield [product, representative, product != representative, status,
                   witness, overlap, corpus.raw_counts[product], len(corpus.products[product]),
                   result.sparse_removed.get(representative, 0),
                   len(result.products.get(representative, {}))]

    write_csv(output_dir / "Reviews.csv", CSV_COLUMNS, review_rows())
    write_csv(output_dir / "product_mapping.csv", [
        "original_product_id", "canonical_product_id", "merged_into_canonical",
        "canonical_status", "dropped_for_overlap_with", "overlap_with_witness",
        "original_review_rows", "original_distinct_reviews",
        "canonical_shared_reviews_removed", "canonical_final_reviews",
    ], mapping_rows())
    ranked = sorted(result.products, key=lambda p: (-len(result.products[p]), p))
    write_csv(output_dir / "product_counts.csv", ["rank", "ProductId", "review_count"],
              ((i, p, len(result.products[p])) for i, p in enumerate(ranked, 1)))
    # Written last: this describes the successfully exported run.
    summary_path = output_dir / "dedup_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result.summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "assets")
    parser.add_argument("--source", type=Path, help="Existing SNAP .txt or .txt.gz; otherwise download the archive")
    parser.add_argument("--encoding", default="latin-1", help="Source text encoding (default: latin-1); output is UTF-8")
    parser.add_argument("--large-overlap-threshold", type=int, default=10,
                        help="Drop the smaller product if shared reviews exceed this count (default: 10)")
    args = parser.parse_args()
    if args.large_overlap_threshold < 0:
        parser.error("--large-overlap-threshold must be nonnegative")
    source = args.source or args.output_dir / "finefoods.txt.gz"
    if args.source is None:
        print(f"Downloading or reusing {source}", flush=True)
        download(source)
    print(f"Reading {source}", flush=True)
    corpus = load_corpus(iter_snap_records(source, args.encoding))
    print(f"Loaded {sum(corpus.raw_counts.values()):,} rows across {len(corpus.products):,} products", flush=True)
    result = deduplicate(corpus, args.large_overlap_threshold)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    result.summary.update(source_url=SOURCE_URL, source_path=str(source.resolve()),
                          source_sha256=digest.hexdigest(), source_encoding=args.encoding,
                          output_encoding="utf-8")
    export(corpus, result, args.output_dir)
    print(json.dumps(result.summary, indent=2), flush=True)
    print(f"Saved deduplicated data to {(args.output_dir / 'Reviews.csv').resolve()}", flush=True)


if __name__ == "__main__":
    main()
