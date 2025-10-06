# Indexed189

Indexed189 is a tidy, shareable collection of CS 189 course materials that have been
converted into Markdown for downstream retrieval-augmented generation (RAG) and
semantic search experiments. The repository now includes a small Python command
line interface (CLI) that makes it easy to browse, preview and search through the
notes directly from the terminal.

## Repository layout

```
├── docs/
│   ├── deep-learning/        # Supplemental deep learning readings
│   ├── discussions/          # Discussion mini-lecture notes
│   └── lectures/
│       ├── notes/            # Primary lecture note exports
│       └── notebooks/        # Companion Jupyter notebook exports
├── src/indexed189/          # Python package providing the CLI utilities
├── pyproject.toml           # Packaging metadata (installable via pip)
└── requirements.txt         # Minimal runtime dependencies
```

## Quick start

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-account>/Indexed189.git
   cd Indexed189
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   ```

3. **Install the CLI and dependencies**

   ```bash
   pip install -e .
   ```

   Alternatively, install directly from the requirements file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Explore the corpus with the CLI**

   ```bash
   indexed189 list
   indexed189 info lec04-k-means
   indexed189 preview lec07-linear-regression-2 --lines 15
   indexed189 search "Bayes"
   ```

## CLI reference

The CLI exposes four subcommands:

- `indexed189 list [--category <name>]` — Display the available documents grouped
  by category (e.g., `lectures`, `discussions`, `deep-learning`).
- `indexed189 info <document>` — Show metadata for the matching document,
  including an estimated word count and the first few headings.
- `indexed189 preview <document> [--lines N]` — Print the first *N* lines of a
  document for a quick glance.
- `indexed189 search <query> [--limit N]` — Search the corpus for a text query
  and preview up to *N* matches per file.

Each `<document>` argument can be any unique substring of a document's filename.
If multiple matches are found the CLI will prompt you to be more specific.

Run `indexed189 --help` or `indexed189 <command> --help` for additional details.

## Development

- Format and linting: the codebase is intentionally small and relies only on
  standard formatting conventions (PEP 8). Add your favourite linters if you
  need stricter checks.
- Testing: for lightweight validation you can import `indexed189.cli` in an
  interactive shell and exercise the helper functions directly.

Pull requests and improvements that extend the CLI or enrich the documents are
always welcome!
