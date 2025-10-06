#!/usr/bin/env python3
"""Unified command line interface for document ingestion and annotation."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests
import yaml
from openai import OpenAI
from dotenv import load_dotenv

try:
    import nbformat
    from PIL import Image
except ImportError as exc:  # pragma: no cover - handled via requirements
    raise ImportError(
        "Missing optional dependency. Install project requirements first: "
        "pip install -r requirements.txt"
    ) from exc


# Load environment variables from project root .env (if present), without overriding
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
# Also respect a .env in the current working directory as a fallback
load_dotenv(override=False)

logger = logging.getLogger("ingest_cli")

MATHPIX_API_URL = "https://api.mathpix.com/v3/pdf"
MATHPIX_POLL_INTERVAL = 3
MATHPIX_MAX_WAIT_SECONDS = 300

IMAGE_PROMPT = (
    "Describe this image from an academic lecture slide. Focus on:\n"
    "1. Detailed description of diagrams including axes, labels, and what is shown.\n"
    "2. For equations, return the LaTeX enclosed in $$.\n"
    "3. For miscellaneous images, describe their purpose.\n\n"
    "Keep it technical, precise, and under 100 words so it is indexable by an LLM."
)


@dataclass
class DirectoryLayout:
    """Manage project directories used during ingestion."""

    base_dir: Path = Path("data")
    ingest: Path = field(init=False)
    processed: Path = field(init=False)
    searchable: Path = field(init=False)
    figures: Path = field(init=False)

    def __post_init__(self) -> None:
        self.ingest = self.base_dir / "ingest"
        self.processed = self.base_dir / "processed"
        self.searchable = self.base_dir / "searchable"
        self.figures = self.base_dir / "figures"

    def ensure(self) -> None:
        for path in (self.ingest, self.processed, self.searchable):
            path.mkdir(parents=True, exist_ok=True)


class MathpixProcessor:
    """Process PDFs and slide decks with the Mathpix API."""

    def __init__(self, app_id: Optional[str] = None, api_key: Optional[str] = None):
        self.app_id = app_id or os.getenv("MATHPIX_APP_ID")
        self.api_key = api_key or os.getenv("MATHPIX_API_KEY")

        if not self.app_id or not self.api_key:
            raise RuntimeError(
                "Mathpix credentials missing. Set MATHPIX_APP_ID and MATHPIX_API_KEY."
            )

        self.session = requests.Session()
        self.headers = {"app_id": self.app_id, "app_key": self.api_key}

    @staticmethod
    def _slugify(text: str) -> str:
        slug = text.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s-]+", "-", slug)
        return slug.strip("-")

    def _frontmatter(self, metadata: Dict[str, str]) -> str:
        base = {
            "course": metadata.get("course", "CS 189"),
            "semester": metadata.get("semester", "Fall 2025"),
            "type": metadata.get("type", "lecture"),
            "title": metadata.get("title", "Untitled"),
            "source_type": metadata.get("source_type", "slides"),
            "source_file": metadata.get("source_file", ""),
            "processed_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "processor": "mathpix",
        }
        extra = {k: v for k, v in metadata.items() if k not in base}
        payload = {**base, **extra}
        return f"---\n{yaml.dump(payload, sort_keys=False)}---\n\n"

    def _poll_status(self, pdf_id: str) -> Dict[str, str]:
        logger.info("Polling Mathpix status for pdf_id=%s", pdf_id)
        deadline = time.time() + MATHPIX_MAX_WAIT_SECONDS
        while time.time() < deadline:
            response = self.session.get(f"{MATHPIX_API_URL}/{pdf_id}", headers=self.headers)
            response.raise_for_status()
            body = response.json()
            status = body.get("status")
            if status == "completed":
                logger.info("Mathpix completed for pdf_id=%s", pdf_id)
                return body
            if status == "error":
                raise RuntimeError(body.get("error", "Unknown Mathpix error"))
            time.sleep(MATHPIX_POLL_INTERVAL)
        raise TimeoutError("Mathpix conversion timed out")

    def _download_markdown(self, pdf_id: str) -> str:
        response = self.session.get(f"{MATHPIX_API_URL}/{pdf_id}.md", headers=self.headers)
        response.raise_for_status()
        return response.text

    def _process(self, file_path: Path, output_dir: Path, metadata: Dict[str, str]) -> Path:
        logger.info("Submitting to Mathpix: %s", file_path.name)
        with file_path.open("rb") as handle:
            files = {"file": handle}
            options = {
                "conversion_formats": {"md": True},
                "math_inline_delimiters": ["$", "$"],
                "math_display_delimiters": ["$$", "$$"],
                "rm_spaces": True,
            }
            response = self.session.post(
                MATHPIX_API_URL,
                headers=self.headers,
                files=files,
                data={"options_json": json.dumps(options)},
            )
        response.raise_for_status()
        pdf_id = response.json()["pdf_id"]
        logger.debug("Received pdf_id=%s for %s", pdf_id, file_path.name)

        self._poll_status(pdf_id)
        markdown_body = self._download_markdown(pdf_id)

        output_path = output_dir / f"{metadata['slug']}.md"
        frontmatter = self._frontmatter(metadata)
        output_path.write_text(frontmatter + markdown_body, encoding="utf-8")
        logger.info("Wrote markdown: %s", output_path.name)
        return output_path

    def process_pptx(self, file_path: Path, output_dir: Path) -> Path:
        metadata = self._metadata_from_pptx(file_path)
        return self._process(file_path, output_dir, metadata)

    def process_pdf(self, file_path: Path, output_dir: Path) -> Path:
        metadata = self._metadata_from_pdf(file_path)
        return self._process(file_path, output_dir, metadata)

    def _metadata_from_pptx(self, file_path: Path) -> Dict[str, str]:
        stem = file_path.stem
        lecture_pattern = re.compile(r"lecture\s+(\d+)\s*[\-–—]{2}\s*(.+)", re.IGNORECASE)
        discussion_pattern = re.compile(r"discussion\s+mini\s+lecture\s+(\d+)", re.IGNORECASE)
        numbered_pattern = re.compile(r"(\d+(?:\.\d+)?)\s+(.+)")

        slug_suffix = "-processed"

        if (match := lecture_pattern.match(stem)):
            number = int(match.group(1))
            title = match.group(2).strip()
            slug = f"lec{number:02d}-{self._slugify(title)}{slug_suffix}"
            return {
                "type": "lecture",
                "number": number,
                "title": title,
                "slug": slug,
                "source_type": "slides",
                "source_file": file_path.name,
            }

        if (match := discussion_pattern.match(stem)):
            number = int(match.group(1))
            slug = f"disc{number:02d}-mini-lecture{slug_suffix}"
            return {
                "type": "discussion",
                "number": number,
                "title": f"Discussion {number}",
                "slug": slug,
                "source_type": "slides",
                "source_file": file_path.name,
            }

        if (match := numbered_pattern.match(stem)):
            section = match.group(1)
            title = match.group(2).strip()
            slug = f"{section.replace('.', '-')}-{self._slugify(title)}{slug_suffix}"
            return {
                "type": "module",
                "section": section,
                "title": title,
                "slug": slug,
                "source_type": "slides",
                "source_file": file_path.name,
            }

        slug = f"{self._slugify(stem)}{slug_suffix}"
        return {
            "type": "lecture",
            "title": stem,
            "slug": slug,
            "source_type": "slides",
            "source_file": file_path.name,
        }

    def _metadata_from_pdf(self, file_path: Path) -> Dict[str, str]:
        stem = file_path.stem
        slug = f"{self._slugify(stem)}-processed"
        return {
            "type": "pdf",
            "title": stem,
            "slug": slug,
            "source_type": "pdf",
            "source_file": file_path.name,
        }


class NotebookProcessor:
    """Convert notebooks to markdown with extracted figures."""

    def __init__(self, layout: DirectoryLayout):
        self.layout = layout

    @staticmethod
    def _slugify(text: str) -> str:
        slug = text.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s-]+", "-", slug)
        return slug.strip("-")

    def _parse_filename(self, path: Path) -> Dict[str, str]:
        match = re.match(r"lec\s*0*(\d+)", path.stem, re.IGNORECASE)
        if match:
            number = int(match.group(1))
            return {
                "type": "notebook",
                "number": number,
                "title": f"Lecture {number} Notebook",
                "slug": f"lec{number:02d}-notebook",
            }
        return {
            "type": "notebook",
            "title": path.stem,
            "slug": self._slugify(path.stem),
        }

    @staticmethod
    def _notebook_title(nb: nbformat.NotebookNode) -> Optional[str]:
        for cell in nb.cells:
            if cell.get("cell_type") == "markdown":
                for line in cell.get("source", "").splitlines():
                    if line.startswith("# "):
                        return line[2:].strip()
        return None

    def _frontmatter(self, metadata: Dict[str, str]) -> str:
        payload = {
            "course": "CS 189",
            "semester": "Fall 2025",
            "type": metadata.get("type", "notebook"),
            "number": metadata.get("number"),
            "title": metadata.get("title", "Untitled"),
            "source_type": "jupyter_notebook",
            "processed_date": datetime.utcnow().strftime("%Y-%m-%d"),
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        return f"---\n{yaml.dump(payload, sort_keys=False)}---\n\n"

    def _process_outputs(
        self,
        outputs: Sequence[nbformat.NotebookNode],
        slug: str,
        cell_index: int,
    ) -> str:
        buffer: List[str] = []
        for output in outputs:
            output_type = output.get("output_type")
            if output_type in {"stream", "execute_result"}:
                text = output.get("text")
                if text:
                    if isinstance(text, list):
                        text = "".join(text)
                    buffer.append(f"```\n{text.rstrip()}\n```\n\n")
            if output_type in {"display_data", "execute_result"}:
                data = output.get("data", {})
                if "image/png" in data:
                    encoded = data["image/png"]
                    buffer.append(f"![Output](data:image/png;base64,{encoded})\n\n")
                elif "image/jpeg" in data:
                    encoded = data["image/jpeg"]
                    buffer.append(f"![Output](data:image/jpeg;base64,{encoded})\n\n")
            if output_type == "error":
                traceback = "\n".join(output.get("traceback", []))
                buffer.append(
                    f"```\nError: {output.get('ename', 'ExecutionError')}\n{traceback}\n```\n\n"
                )
        return "".join(buffer)

    def process(self, notebook_path: Path) -> Path:
        notebook = nbformat.read(notebook_path, as_version=4)
        metadata = self._parse_filename(notebook_path)
        title = self._notebook_title(notebook)
        if title:
            metadata["title"] = title
            if "number" in metadata:
                metadata["slug"] = f"lec{metadata['number']:02d}-{self._slugify(title)}"
        slug = metadata["slug"]

        lines: List[str] = []
        code_cell_index = 0

        for cell in notebook.cells:
            cell_type = cell.get("cell_type")
            if cell_type == "markdown":
                lines.append(cell.get("source", ""))
                lines.append("\n\n")
            elif cell_type == "code":
                code = cell.get("source", "").rstrip()
                if code:
                    lines.append(f"```python\n{code}\n```\n\n")
                outputs = cell.get("outputs") or []
                if outputs:
                    code_cell_index += 1
                    lines.append(self._process_outputs(outputs, slug, code_cell_index))

        markdown_path = self.layout.searchable / f"{slug}.md"
        markdown_path.write_text(self._frontmatter(metadata) + "".join(lines), encoding="utf-8")
        return markdown_path


class VisionAnnotator:
    """Annotate images in markdown files using OpenAI Vision."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=key)

    def _image_description(self, reference: str, base_dir: Path) -> Optional[str]:
        if reference.startswith("http://") or reference.startswith("https://"):
            # Prefer downloading remote images and sending as base64 to avoid external fetch failures
            try:
                response = requests.get(reference, timeout=15)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                data_bytes = response.content
                encoded = base64.b64encode(data_bytes).decode("utf-8")
                # Infer MIME type conservatively
                if "image/png" in content_type or reference.lower().endswith(".png"):
                    mime = "image/png"
                elif "image/jpg" in content_type or "image/jpeg" in content_type or reference.lower().endswith((".jpg", ".jpeg")):
                    mime = "image/jpeg"
                elif "image/webp" in content_type or reference.lower().endswith(".webp"):
                    mime = "image/webp"
                else:
                    mime = "image/png"
                image_payload = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "low"},
                }
            except Exception:
                # Fallback to passing the URL directly
                image_payload = {"type": "image_url", "image_url": {"url": reference, "detail": "low"}}
        else:
            image_path = (base_dir / reference).resolve()
            if not image_path.exists():
                return None
            with image_path.open("rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("utf-8")
            image_payload = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "low"},
            }
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [{"type": "text", "text": IMAGE_PROMPT}, image_payload]}],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _should_skip(alt_text: str, reference: str, base_dir: Path) -> bool:
        if len(alt_text.strip()) > 50:
            return True
        if reference.startswith("http"):
            return False
        image_path = (base_dir / reference).resolve()
        if not image_path.exists():
            return True
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            return True
        return width < 100 or height < 100

    def annotate_markdown(self, path: Path) -> Tuple[int, int]:
        content = path.read_text(encoding="utf-8")
        pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        base_dir = path.parent
        updated = content
        annotated = 0
        skipped = 0

        for match in pattern.finditer(content):
            alt_text, reference = match.groups()
            if "**Image description:**" in content[match.end(): match.end() + 200]:
                skipped += 1
                continue
            if self._should_skip(alt_text, reference, base_dir):
                skipped += 1
                continue
            description = self._image_description(reference, base_dir)
            if not description:
                skipped += 1
                continue
            truncated_alt = description[:100].replace("\n", " ")
            replacement = (
                f"![{truncated_alt}]({reference})\n\n**Image description:** {description}\n"
            )
            updated = updated.replace(match.group(0), replacement, 1)
            annotated += 1

        if updated != content:
            path.write_text(updated, encoding="utf-8")
        return annotated, skipped


def move_to_processed(path: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / path.name
    path.replace(target)


def gather_files(directory: Path, pattern: str) -> List[Path]:
    return sorted(directory.glob(pattern))


def process_documents(
    layout: DirectoryLayout,
    doc_types: Sequence[str],
    annotate: bool,
) -> None:
    layout.ensure()
    mathpix: Optional[MathpixProcessor] = None
    notebook_processor: Optional[NotebookProcessor] = None
    new_markdown: List[Path] = []

    if any(t in {"pptx", "pdf"} for t in doc_types):
        mathpix = MathpixProcessor()

    if "ipynb" in doc_types:
        notebook_processor = NotebookProcessor(layout)

    if "pptx" in doc_types:
        pptx_files = gather_files(layout.ingest, "*.pptx")
        logger.info("Found %d PPTX file(s)", len(pptx_files))
        for file_path in pptx_files:
            try:
                logger.info("Processing PPTX: %s", file_path.name)
                output = mathpix.process_pptx(file_path, layout.searchable)  # type: ignore[arg-type]
                new_markdown.append(output)
                move_to_processed(file_path, layout.processed)
                logger.info("Moved processed PPTX to %s", layout.processed)
                time.sleep(2)
            except Exception as exc:
                logger.error("Failed to process %s: %s", file_path.name, exc)

    if "pdf" in doc_types:
        pdf_files = gather_files(layout.ingest, "*.pdf")
        logger.info("Found %d PDF file(s)", len(pdf_files))
        for file_path in pdf_files:
            try:
                logger.info("Processing PDF: %s", file_path.name)
                output = mathpix.process_pdf(file_path, layout.searchable)  # type: ignore[arg-type]
                new_markdown.append(output)
                move_to_processed(file_path, layout.processed)
                logger.info("Moved processed PDF to %s", layout.processed)
                time.sleep(2)
            except Exception as exc:
                logger.error("Failed to process %s: %s", file_path.name, exc)

    if "ipynb" in doc_types:
        notebook_files = gather_files(layout.ingest, "*.ipynb")
        for file_path in notebook_files:
            try:
                output = notebook_processor.process(file_path)  # type: ignore[union-attr]
                new_markdown.append(output)
                move_to_processed(file_path, layout.processed)
            except Exception as exc:
                print(f"Failed to process {file_path.name}: {exc}")

    if annotate:
        annotator = VisionAnnotator()
        targets = new_markdown if new_markdown else gather_files(layout.searchable, "*.md")
        if not targets:
            logger.info("No markdown files found to annotate.")
        else:
            logger.info("Annotating %d markdown file(s)", len(targets))
            for md_file in targets:
                try:
                    logger.info("Annotating: %s", md_file.name)
                    annotated, skipped = annotator.annotate_markdown(md_file)
                    logger.info(
                        "Annotated %d image(s) and skipped %d in %s",
                        annotated,
                        skipped,
                        md_file.name,
                    )
                except Exception as exc:
                    logger.error("Failed to annotate %s: %s", md_file.name, exc)

    if not new_markdown:
        logger.info("No new markdown files generated.")
    else:
        logger.info("Generated markdown files:")
        for path in new_markdown:
            logger.info(" - %s", path.relative_to(layout.searchable))


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ingestion CLI")
    parser.add_argument(
        "--base-dir",
        default="data",
        type=Path,
        help="Root directory containing ingest/, processed/, searchable/, and figures/",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["pptx", "pdf", "ipynb"],
        default=["pptx", "pdf", "ipynb"],
        help="Document types to process",
    )
    parser.add_argument(
        "--annotate-files",
        nargs="*",
        help="Specific markdown file names in searchable/ to annotate (e.g. aug27-processed.md)",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable annotation (annotation is enabled by default)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    # Configure logging level
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    layout = DirectoryLayout(base_dir=args.base_dir)
    annotate = not args.no_annotate
    # If user specified files, annotate only those
    if annotate and args.annotate_files:
        annotator = VisionAnnotator()
        targets = [layout.searchable / name for name in args.annotate_files]
        logger.info("Annotating %d explicitly specified file(s)", len(targets))
        for md_file in targets:
            try:
                logger.info("Annotating: %s", md_file.name)
                annotated, skipped = annotator.annotate_markdown(md_file)
                logger.info(
                    "Annotated %d image(s) and skipped %d in %s",
                    annotated,
                    skipped,
                    md_file.name,
                )
            except Exception as exc:
                logger.error("Failed to annotate %s: %s", md_file.name, exc)
        return

    process_documents(layout, args.types, annotate)


if __name__ == "__main__":
    main()
