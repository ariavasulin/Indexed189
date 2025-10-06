# Course Ingestion CLI

A lightweight command line toolchain for converting course assets (PPTX, PDF, IPYNB) into searchable Markdown. Documents are OCR'd with Mathpix and (optionally) enriched with GPT-4o Vision image annotations.

## Features

- Single entrypoint for PPTX, PDF, and IPYNB processing.
- Automatic Markdown front matter for downstream indexing.
- Consistent `-processed.md` naming convention.
- Optional GPT-4o Vision annotations for slide and figure images.
- Simple, opinionated directory structure so files are easy to manage.

## Prerequisites

- Python 3.10+
- Mathpix and OpenAI API credentials.

## Setup
0. You will need a Mathpix and OpenAI API key
1. Clone the repo and open a terminal in the project root.
2. (Optional) Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment template and fill in your credentials.
   ```bash
   cp .env.example .env
   ```
   Populate `MATHPIX_APP_ID`, `MATHPIX_API_KEY`, and `OPENAI_API_KEY`.

## Directory Layout

The CLI keeps everything under a single base directory (`data/` by default). The structure is created on first run:

```
data/
├── ingest/      # Drop PPTX, PDF, and IPYNB files here
├── processed/   # Originals are moved here after successful ingestion
├── searchable/  # Generated Markdown lives here (e.g. lec05-bayes-processed.md)
```

Use `--base-dir` to point at a different location if needed.

## Usage

Inspect available options:

```bash
python -m src.ingest_cli --help
```

Process everything in `data/ingest/`:

```bash
python -m src.ingest_cli
```

Limit to specific document types:

```bash
# Only process PowerPoint decks and PDFs
python -m src.ingest_cli --types pptx pdf
```

Annotate newly generated Markdown with GPT-4o Vision descriptions:

```bash
python -m src.ingest_cli --annotate
```

The CLI prints each generated Markdown file and moves the original sources into `processed/`. Newly created Markdown is written to `searchable/` with the `-processed.md` suffix.

## Notes

- Mathpix usage incurs API cost; results are polled for up to five minutes per document.
- GPT-4o Vision annotations can also incur cost. Use `--annotate` only when desired.
- The CLI assumes outbound network access for Mathpix and OpenAI requests.
- Existing Markdown files are updated in place when annotations are added.

## Usecases
- The easiest ways to use this is with NotebookLM or Cursor 

## Troubleshooting

- If credentials are missing, the CLI raises an error before processing any files.
- To change polling behaviour, adjust `MATHPIX_POLL_INTERVAL` and `MATHPIX_MAX_WAIT_SECONDS` constants in `src/ingest_cli.py`.
- Notebook processing relies on `nbformat` and `Pillow`; ensure the requirements were installed.

Happy indexing!
