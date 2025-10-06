"""Command line interface for exploring the CS 189 course corpus."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import click

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PROJECT_ROOT / "docs"


@dataclass
class Document:
    """Representation of a markdown document in the repository."""

    path: Path

    @property
    def category(self) -> str:
        relative = self.path.relative_to(DOCS_ROOT)
        if len(relative.parts) > 1:
            return relative.parts[0]
        return "misc"

    @property
    def subcategory(self) -> Optional[str]:
        relative = self.path.relative_to(DOCS_ROOT)
        if len(relative.parts) > 2:
            return "/".join(relative.parts[1:-1]) or None
        if len(relative.parts) == 2:
            return relative.parts[0]
        return None

    @property
    def slug(self) -> str:
        return self.path.stem

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def word_count(self) -> int:
        text = self.read_text()
        return len(text.split())

    def headings(self) -> List[str]:
        headings: List[str] = []
        for line in self.read_text().splitlines():
            if line.strip().startswith("#"):
                headings.append(line.strip().lstrip("# "))
        return headings


def _iter_documents() -> Iterable[Document]:
    if not DOCS_ROOT.exists():
        raise click.UsageError("The docs directory could not be found.")

    for path in sorted(DOCS_ROOT.rglob("*.md")):
        yield Document(path=path)


def _group_documents_by_category() -> Dict[str, List[Document]]:
    grouped: Dict[str, List[Document]] = defaultdict(list)
    for doc in _iter_documents():
        grouped[doc.category].append(doc)
    return dict(grouped)


def _match_document(identifier: str) -> Document:
    identifier = identifier.lower()
    matches = [
        doc
        for doc in _iter_documents()
        if identifier in doc.slug.lower() or identifier in str(doc.path.name).lower()
    ]

    if not matches:
        raise click.UsageError(f"No document found matching '{identifier}'.")

    if len(matches) > 1:
        options = "\n".join(f"  • {doc.path.relative_to(PROJECT_ROOT)}" for doc in matches)
        raise click.UsageError(
            "Multiple documents matched your query.\n"
            "Please be more specific. Candidates include:\n"
            f"{options}"
        )

    return matches[0]


@click.group()
def app() -> None:
    """Utilities for working with the CS 189 markdown corpus."""


@app.command("list")
@click.option(
    "--category",
    "category_filter",
    type=str,
    help="Filter the results by top-level category (e.g. lectures, discussions).",
)
def list_documents(category_filter: Optional[str]) -> None:
    """List the documents that are available in the repository."""

    category_filter_normalized = category_filter.lower() if category_filter else None
    grouped = _group_documents_by_category()

    if category_filter_normalized and category_filter_normalized not in grouped:
        raise click.UsageError(
            f"Category '{category_filter}' not found. Available categories: "
            + ", ".join(sorted(grouped.keys()))
        )

    for category, documents in sorted(grouped.items()):
        if category_filter_normalized and category.lower() != category_filter_normalized:
            continue

        click.echo(click.style(category.capitalize(), bold=True))
        for doc in documents:
            relative = doc.path.relative_to(PROJECT_ROOT)
            click.echo(f"  - {relative} ({doc.word_count()} words)")
        click.echo()


@app.command()
@click.argument("identifier", metavar="DOCUMENT")
def info(identifier: str) -> None:
    """Show metadata about a document (use part of the filename)."""

    doc = _match_document(identifier)
    headings_preview = doc.headings()[:5]
    click.echo(click.style(str(doc.path.relative_to(PROJECT_ROOT)), bold=True))
    click.echo(f"Words: {doc.word_count()}")
    if headings_preview:
        click.echo("Top headings:")
        for heading in headings_preview:
            click.echo(f"  • {heading}")


@app.command()
@click.argument("identifier", metavar="DOCUMENT")
@click.option("--lines", default=20, show_default=True, help="Number of lines to preview.")
def preview(identifier: str, lines: int) -> None:
    """Show the first few lines of a document."""

    doc = _match_document(identifier)
    content = doc.read_text().splitlines()
    click.echo(click.style(str(doc.path.relative_to(PROJECT_ROOT)), bold=True))
    for line in content[:lines]:
        click.echo(line)


@app.command()
@click.argument("query", metavar="QUERY")
@click.option("--limit", default=5, show_default=True, help="Maximum number of matches per file.")
def search(query: str, limit: int) -> None:
    """Search all markdown documents for a text query."""

    normalized_query = query.lower()
    results_found = False

    for doc in _iter_documents():
        matches = []
        for idx, line in enumerate(doc.read_text().splitlines(), start=1):
            if normalized_query in line.lower():
                matches.append((idx, line.strip()))
            if len(matches) >= limit:
                break

        if matches:
            results_found = True
            click.echo(click.style(str(doc.path.relative_to(PROJECT_ROOT)), bold=True))
            for line_number, text in matches:
                click.echo(f"  L{line_number:>4}: {text}")
            click.echo()

    if not results_found:
        click.echo("No matches found.")


if __name__ == "__main__":
    app()
