#!/usr/bin/env python3
"""
Jupyter Notebook Processing Script
Converts Jupyter notebooks to searchable Markdown format for RAG/indexing.
"""

import sys
import re
import base64
from pathlib import Path
from datetime import datetime

try:
    import yaml
    import nbformat
    from PIL import Image
    import io
except ImportError as e:
    print(f"Missing module: {e.name}")
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "nbformat", "pillow"])
    import yaml
    import nbformat
    from PIL import Image
    import io


class NotebookProcessor:
    """Process Jupyter notebooks to Markdown."""
    
    def __init__(self, base_dir="Docs"):
        self.base_dir = Path(base_dir)
        self.source_dir = self.base_dir / "source"
        self.searchable_dir = self.base_dir / "searchable"
        self.figures_dir = self.base_dir / "figures"
        
        self.stats = {"processed": 0, "images": 0, "failed": 0}
    
    def _slugify(self, text):
        """Convert text to URL-friendly slug."""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')
    
    def parse_filename(self, filepath):
        """Extract metadata from notebook filename."""
        filename = filepath.stem
        
        # Match "lecXX" or "lec XX"
        match = re.match(r'lec\s*0*(\d+)', filename, re.IGNORECASE)
        
        if match:
            number = int(match.group(1))
            return {
                "type": "notebook",
                "number": number,
                "title": f"Lecture {number} Notebook",
                "slug": f"lec{number:02d}-notebook"
            }
        
        return {
            "type": "notebook",
            "title": filename,
            "slug": self._slugify(filename)
        }
    
    def extract_title_from_notebook(self, nb):
        """Try to extract title from first markdown cell."""
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                lines = cell.source.split('\n')
                for line in lines:
                    if line.startswith('# '):
                        return line[2:].strip()
        return None
    
    def generate_frontmatter(self, metadata):
        """Generate YAML frontmatter."""
        frontmatter = {
            "course": "CS 189",
            "semester": "Fall 2025",
            "type": metadata.get("type", "notebook"),
            "number": metadata.get("number", 0),
            "title": metadata.get("title", "Unknown"),
            "source_type": "jupyter_notebook",
            "processed_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        return f"---\n{yaml_str}---\n\n"
    
    def process_cell_outputs(self, outputs, fig_dir, lec_slug, cell_num):
        """Extract images and text from cell outputs."""
        output_md = []
        
        for output in outputs:
            # Text output
            if output.output_type in ['stream', 'execute_result']:
                if 'text' in output:
                    text = ''.join(output.text) if isinstance(output.text, list) else output.text
                    output_md.append(f"```\n{text.strip()}\n```\n\n")
            
            # Image output
            elif output.output_type == 'display_data' or output.output_type == 'execute_result':
                if 'data' in output:
                    data = output.data
                    
                    # PNG images
                    if 'image/png' in data:
                        img_filename = f"cell-{cell_num:03d}-output.png"
                        img_path = fig_dir / img_filename
                        
                        # Decode and save
                        img_data = base64.b64decode(data['image/png'])
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                        
                        rel_path = f"../figures/{fig_dir.name}/{img_filename}"
                        output_md.append(f"![Output]({rel_path})\n\n")
                        self.stats["images"] += 1
                    
                    # JPEG images
                    elif 'image/jpeg' in data:
                        img_filename = f"cell-{cell_num:03d}-output.jpg"
                        img_path = fig_dir / img_filename
                        
                        img_data = base64.b64decode(data['image/jpeg'])
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                        
                        rel_path = f"../figures/{fig_dir.name}/{img_filename}"
                        output_md.append(f"![Output]({rel_path})\n\n")
                        self.stats["images"] += 1
            
            # Error output
            elif output.output_type == 'error':
                traceback = '\n'.join(output.traceback)
                output_md.append(f"```\nError: {output.ename}\n{traceback}\n```\n\n")
        
        return ''.join(output_md)
    
    def process_notebook(self, nb_path, archive_dir=None):
        """Process a single notebook."""
        print(f"\n📓 Processing: {nb_path.name}")
        
        try:
            # Load notebook
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Parse metadata
            metadata = self.parse_filename(nb_path)
            
            # Try to extract title from notebook
            nb_title = self.extract_title_from_notebook(nb)
            if nb_title:
                metadata['title'] = nb_title
                if 'number' in metadata:
                    metadata['slug'] = f"lec{metadata['number']:02d}-{self._slugify(nb_title)}"
            
            # Create figures directory
            fig_dir = self.figures_dir / metadata['slug']
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            # Process cells
            markdown_lines = []
            cell_num = 0
            
            for cell in nb.cells:
                if cell.cell_type == 'markdown':
                    markdown_lines.append(cell.source)
                    markdown_lines.append("\n\n")
                
                elif cell.cell_type == 'code':
                    cell_num += 1
                    
                    # Add code
                    if cell.source.strip():
                        markdown_lines.append(f"```python\n{cell.source}\n```\n\n")
                    
                    # Add outputs
                    if cell.outputs:
                        output_md = self.process_cell_outputs(
                            cell.outputs, fig_dir, metadata['slug'], cell_num
                        )
                        if output_md:
                            markdown_lines.append(output_md)
            
            # Generate final markdown
            frontmatter = self.generate_frontmatter(metadata)
            final_content = frontmatter + ''.join(markdown_lines)
            
            # Write output
            output_file = self.searchable_dir / f"{metadata['slug']}.md"
            output_file.write_text(final_content, encoding='utf-8')
            
            print(f"   ✅ Generated: {output_file.name}")
            print(f"   📷 Extracted {self.stats['images']} images")
            
            # Move source file to archive directory if specified
            if archive_dir:
                import shutil
                archive_path = archive_dir / nb_path.name
                shutil.move(str(nb_path), str(archive_path))
                print(f"   📦 Moved source to: {archive_dir.name}/{nb_path.name}")
            
            self.stats["processed"] += 1
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.stats["failed"] += 1
            return False
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "="*60)
        print("📊 NOTEBOOK PROCESSING SUMMARY")
        print("="*60)
        print(f"✅ Processed: {self.stats['processed']}")
        print(f"🖼️  Images extracted: {self.stats['images']}")
        print(f"❌ Failed: {self.stats['failed']}")
        print("="*60)


def main():
    """Main execution."""
    print("🚀 Jupyter Notebook Processing Script")
    print("="*60)
    
    processor = NotebookProcessor(base_dir="Docs")
    
    # Setup directories
    processor.searchable_dir.mkdir(parents=True, exist_ok=True)
    processor.figures_dir.mkdir(parents=True, exist_ok=True)
    processor.source_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ingest directory
    ingest_dir = processor.base_dir / "ingest"
    
    # Find notebooks in ingest directory first, then source
    nb_files = sorted(ingest_dir.glob("*.ipynb")) if ingest_dir.exists() else []
    from_ingest = bool(nb_files)
    
    if not nb_files:
        # Fall back to source directory
        nb_files = sorted(processor.source_dir.glob("*.ipynb"))
    
    if not nb_files:
        print(f"\n⚠️  No notebooks found in {ingest_dir}/ or {processor.source_dir}/")
        print("   Looking for .ipynb files to process...")
        return
    
    source_loc = "ingest" if from_ingest else "source"
    print(f"\nFound {len(nb_files)} notebook(s) in /{source_loc}/:\n")
    for nb in nb_files:
        print(f"  • {nb.name}")
    
    print("\n" + "="*60)
    if from_ingest:
        print("Files will be moved to /source/ after processing\n")
    
    # Process each notebook
    for nb_file in nb_files:
        archive_dir = processor.source_dir if from_ingest else None
        processor.process_notebook(nb_file, archive_dir=archive_dir)
    
    # Print summary
    processor.print_summary()
    
    print(f"\n📁 Output directory: {processor.searchable_dir}")
    print("✨ Ready for RAG/semantic search!")


if __name__ == "__main__":
    main()

