#!/usr/bin/env python3
"""
Mathpix Ingestion Script
Process PowerPoint slides and PDFs with Mathpix to extract LaTeX equations and content.
Single script for all document processing.
"""

import os
import sys
import time
import requests
import base64
import re
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Installing pyyaml...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml

try:
    from openai import OpenAI
except ImportError:
    print("Installing openai...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI


class MathpixProcessor:
    """Process documents using Mathpix API."""
    
    def __init__(self, api_key=None, app_id=None):
        # Default API credentials (can be overridden by environment variables or parameters)
        self.app_id = app_id or os.getenv('MATHPIX_APP_ID') or "test_3e3c3e_472d64"
        self.api_key = api_key or os.getenv('MATHPIX_API_KEY') or "526485070e495b964ebafedf9b52d0f662f7097e35fb37e74a59863ccf7b3ff6"
        
        if not self.api_key or not self.app_id:
            print("⚠️  Mathpix credentials not found")
            print("Please set MATHPIX_API_KEY and MATHPIX_APP_ID")
            raise ValueError("Missing Mathpix API credentials")
        
        self.api_url = "https://api.mathpix.com/v3/pdf"
        self.headers = {
            "app_id": self.app_id,
            "app_key": self.api_key
        }
        
        self.stats = {
            "processed": 0,
            "failed": 0,
            "total_pages": 0
        }
        self.processed_files = []  # Track successfully processed output files
    
    def slugify(self, text):
        """Convert text to URL-friendly slug."""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')
    
    def generate_frontmatter(self, metadata):
        """Generate YAML frontmatter."""
        frontmatter = {
            "course": "CS 189",
            "semester": "Fall 2025",
            "type": metadata.get("type", "lecture"),
            "title": metadata.get("title", "Unknown"),
            "source_type": metadata.get("source_type", "slides"),
            "source_file": metadata.get("source_file", ""),
            "processed_date": datetime.now().strftime("%Y-%m-%d"),
            "processor": "mathpix"
        }
        
        # Add optional fields if they exist
        if "number" in metadata:
            frontmatter["number"] = metadata["number"]
        if "section" in metadata:
            frontmatter["section"] = metadata["section"]
        
        yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        return f"---\n{yaml_str}---\n\n"
    
    def process_file(self, file_path, output_dir, metadata, archive_dir=None):
        """Process a file with Mathpix API."""
        file_path = Path(file_path)
        print(f"\n📄 Processing: {file_path.name}")
        
        # Step 1: Upload file
        print("   Uploading to Mathpix...", end=" ")
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                self.api_url,
                headers=self.headers,
                files=files,
                data={
                    'options_json': '''{
                        "conversion_formats": {"md": true},
                        "math_inline_delimiters": ["$", "$"],
                        "math_display_delimiters": ["$$", "$$"],
                        "rm_spaces": true
                    }'''
                }
            )
        
        if response.status_code != 200:
            print(f"✗ Upload failed: {response.text[:100]}")
            self.stats["failed"] += 1
            return False
        
        result = response.json()
        pdf_id = result.get('pdf_id')
        print(f"✓ (ID: {pdf_id})")
        
        # Step 2: Poll for completion
        print("   Converting...", end=" ")
        max_wait = 300  # 5 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(
                f"{self.api_url}/{pdf_id}",
                headers=self.headers
            )
            
            if status_response.status_code != 200:
                print(f"✗ Status check failed")
                self.stats["failed"] += 1
                return False
            
            status = status_response.json()
            
            if status.get('status') == 'completed':
                print("✓")
                break
            elif status.get('status') == 'error':
                print(f"✗ Conversion error: {status.get('error')}")
                self.stats["failed"] += 1
                return False
            
            time.sleep(3)
        else:
            print("✗ Timeout")
            self.stats["failed"] += 1
            return False
        
        # Step 3: Download markdown
        print("   Downloading markdown...", end=" ")
        md_response = requests.get(
            f"{self.api_url}/{pdf_id}.md",
            headers=self.headers
        )
        
        if md_response.status_code != 200:
            print(f"✗ Download failed")
            self.stats["failed"] += 1
            return False
        
        markdown_content = md_response.text
        print("✓")
        
        # Step 4: Save with frontmatter
        frontmatter = self.generate_frontmatter(metadata)
        final_content = frontmatter + markdown_content
        
        output_file = output_dir / f"{metadata['slug']}.md"
        output_file.write_text(final_content, encoding='utf-8')
        
        print(f"   ✅ Saved to: {output_file.name}")
        
        # Step 5: Move source file to archive directory if specified
        if archive_dir:
            archive_path = archive_dir / file_path.name
            import shutil
            shutil.move(str(file_path), str(archive_path))
            print(f"   📦 Moved source to: {archive_dir.name}/{file_path.name}")
        
        self.stats["processed"] += 1
        if 'pages' in metadata:
            self.stats["total_pages"] += metadata['pages']
        
        # Track the output file for potential annotation
        self.processed_files.append(output_file)
        
        return True
    
    def process_powerpoint(self, pptx_path, output_dir, archive_dir=None):
        """Process PowerPoint file."""
        import re
        filename = pptx_path.stem
        
        # Parse filename for metadata
        lecture_match = re.match(r'Lecture\s+(\d+)\s*--\s*(.+)', filename, re.IGNORECASE)
        disc_match = re.match(r'Discussion\s+Mini\s+Lecture\s+(\d+)', filename, re.IGNORECASE)
        numbered_match = re.match(r'(\d+(?:\.\d+)?)\s+(.+)', filename)  # Matches "3.2 Similarity" or "5 Title"
        
        if lecture_match:
            number = int(lecture_match.group(1))
            title = lecture_match.group(2).strip()
            metadata = {
                "type": "lecture",
                "number": number,
                "title": title,
                "slug": f"lec{number:02d}-{self.slugify(title)}-mathpix",
                "source_type": "slides",
                "source_file": pptx_path.name
            }
        elif disc_match:
            number = int(disc_match.group(1))
            metadata = {
                "type": "discussion",
                "number": number,
                "title": f"Discussion {number}",
                "slug": f"disc{number:02d}-mini-lecture-mathpix",
                "source_type": "slides",
                "source_file": pptx_path.name
            }
        elif numbered_match:
            # Handle files like "3.2 Similarity.pptx" or "5.1 Classification.pptx"
            section = numbered_match.group(1)
            title = numbered_match.group(2).strip()
            metadata = {
                "type": "module",
                "section": section,
                "title": title,
                "slug": f"{section.replace('.', '-')}-{self.slugify(title)}-mathpix",
                "source_type": "slides",
                "source_file": pptx_path.name
            }
        else:
            # Fallback for any other naming pattern
            metadata = {
                "type": "lecture",
                "title": filename,
                "slug": self.slugify(filename) + "-mathpix",
                "source_type": "slides",
                "source_file": pptx_path.name
            }
        
        return self.process_file(pptx_path, output_dir, metadata, archive_dir)
    
    def process_pdf(self, pdf_path, output_dir, title=None, pages_info=None, archive_dir=None):
        """Process PDF file."""
        metadata = {
            "type": "textbook",
            "title": title or pdf_path.stem,
            "slug": self.slugify(title or pdf_path.stem) + "-mathpix",
            "source_type": "pdf",
            "source_file": pdf_path.name
        }
        
        if pages_info:
            metadata["title"] = f"{metadata['title']} {pages_info}"
            metadata["slug"] = metadata["slug"].replace("-mathpix", f"-{self.slugify(pages_info)}-mathpix")
        
        return self.process_file(pdf_path, output_dir, metadata, archive_dir)
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "="*60)
        print("📊 MATHPIX PROCESSING SUMMARY")
        print("="*60)
        print(f"✅ Successfully processed: {self.stats['processed']}")
        print(f"❌ Failed: {self.stats['failed']}")
        print(f"📄 Total pages: {self.stats['total_pages']}")
        print("="*60)


class VisionAnnotator:
    """Annotate images with OpenAI Vision API."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("⚠️  OPENAI_API_KEY not found in environment")
            self.api_key = input("Please enter your OpenAI API key: ").strip()
        
        self.client = OpenAI(api_key=self.api_key)
        self.stats = {"annotated": 0, "skipped": 0}
    
    def encode_image(self, image_path):
        """Encode image to base64 for API."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def describe_image(self, image_path):
        """Use GPT-4 Vision to describe the image."""
        base64_image = self.encode_image(image_path)
        
        prompt = """Describe this image from an academic lecture slide. Focus on:
1. if a diagram or image, a detailed description of it such that it is indexable and interpretable by an llm. What kind of diagram is it, what are the axis's, what does it show, etc
2. If it's an equation/formula: just return the latex of the equation in $$
3. if a misc. image describe it as such

the goal is to make them indexable and searchable for an llm to use.

Keep it under 100 words, technical and precise."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"   ⚠️  Error describing image: {e}")
            return None
    
    def download_image(self, url):
        """Download image from URL and return bytes."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading: {e}")
            return None
    
    def describe_image_from_url(self, image_url):
        """Use GPT-4 Vision to describe an image from URL."""
        prompt = """Describe this image from an academic lecture slide. Focus on:
1. if a diagram or image, a detailed description of it such that it is indexable and interpretable by an llm. What kind of diagram is it, what are the axis's, what does it show, etc
2. If it's an equation/formula: just return the latex of the equation in $$
3. if a misc. image describe it as such

the goal is to make them indexable and searchable for an llm to use.

Keep it under 100 words, technical and precise."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"   ⚠️  Error describing image: {e}")
            return None
    
    def is_image_too_small(self, img_url_or_path, min_width=100, min_height=100):
        """Check if image is too small to be worth annotating (e.g., QR codes, icons)."""
        # For Mathpix CDN URLs, check URL parameters
        if isinstance(img_url_or_path, str) and 'height=' in img_url_or_path and 'width=' in img_url_or_path:
            import re
            height_match = re.search(r'height=(\d+)', img_url_or_path)
            width_match = re.search(r'width=(\d+)', img_url_or_path)
            
            if height_match and width_match:
                height = int(height_match.group(1))
                width = int(width_match.group(1))
                return height < min_height or width < min_width
        
        # For local files, check actual dimensions
        elif Path(img_url_or_path).exists():
            try:
                from PIL import Image
                img = Image.open(img_url_or_path)
                width, height = img.size
                return height < min_height or width < min_width
            except:
                pass
        
        return False
    
    def annotate_markdown(self, md_file):
        """Find images in markdown and add Vision API descriptions."""
        print(f"\n🖼️  Annotating images in: {md_file.name}")
        
        content = md_file.read_text(encoding='utf-8')
        
        # Find all image references: ![alt](path)
        image_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        matches = list(re.finditer(image_pattern, content))
        
        if not matches:
            print("   No images found")
            return
        
        print(f"   Found {len(matches)} images")
        
        updated_content = content
        for idx, match in enumerate(matches, 1):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # Skip if already has good alt text
            if len(alt_text) > 50:
                print(f"   [{idx}/{len(matches)}] Skipping (already has description)")
                self.stats["skipped"] += 1
                continue
            
            # Skip tiny images (QR codes, icons, etc.)
            if self.is_image_too_small(img_path):
                print(f"   [{idx}/{len(matches)}] Skipping (too small - likely QR/icon)")
                self.stats["skipped"] += 1
                continue
            
            # Handle both URLs (Mathpix CDN) and local files
            if img_path.startswith('http'):
                print(f"   [{idx}/{len(matches)}] Describing from URL...", end=" ")
                description = self.describe_image_from_url(img_path)
            else:
                # Local file path
                img_full_path = md_file.parent / img_path
                if not img_full_path.exists():
                    print(f"   [{idx}/{len(matches)}] Skipping {img_path} (not found)")
                    self.stats["skipped"] += 1
                    continue
                
                print(f"   [{idx}/{len(matches)}] Describing {img_full_path.name}...", end=" ")
                description = self.describe_image(img_full_path)
            
            if description:
                # Update the markdown with description (even for equations)
                original = match.group(0)
                updated = f"{original}\n\n**Image Description:** {description}\n"
                updated_content = updated_content.replace(original, updated, 1)
                print("✓")
                self.stats["annotated"] += 1
            else:
                print("✗ (failed)")
                self.stats["skipped"] += 1
            
            time.sleep(0.5)  # Rate limiting
        
        # Write back
        md_file.write_text(updated_content, encoding='utf-8')
        print(f"   ✅ Updated {md_file.name}")
    
    def print_summary(self):
        """Print annotation summary."""
        print("\n" + "="*60)
        print("📊 VISION API ANNOTATION SUMMARY")
        print("="*60)
        print(f"🖼️  Images annotated: {self.stats['annotated']}")
        print(f"⊘  Images skipped: {self.stats['skipped']}")
        print("="*60)


def main():
    """Main execution function."""
    print("🚀 Mathpix Ingestion Script")
    print("="*60)
    
    # Initialize processor
    processor = MathpixProcessor()
    
    # Setup directories
    base_dir = Path("Docs")
    ingest_dir = base_dir / "ingest"
    source_dir = base_dir / "source"
    output_dir = base_dir / "searchable"
    
    # Create directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📁 Processing files with Mathpix API")
    print("   Files will be ingested from /Docs/ingest/")
    print("   After processing, they'll be moved to /Docs/source/\n")
    
    # Process ALL PowerPoint files
    pptx_files = sorted(ingest_dir.glob("*.pptx"))
    
    if pptx_files:
        print(f"Found {len(pptx_files)} PowerPoint files in /ingest/:")
        for idx, pptx in enumerate(pptx_files, 1):
            print(f"  {idx}. {pptx.name}")
        
        print("\nOptions:")
        print("  'all' - Process all files")
        print("  '1,3,5' - Process specific files by number")
        print("  'n' - Skip PowerPoint processing")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'all':
            files_to_process = pptx_files
        elif choice != 'n':
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                files_to_process = [pptx_files[i] for i in indices if 0 <= i < len(pptx_files)]
            except:
                print("Invalid selection, skipping...")
                files_to_process = []
        else:
            files_to_process = []
        
        for pptx_file in files_to_process:
            processor.process_powerpoint(pptx_file, output_dir, archive_dir=source_dir)
            time.sleep(2)  # Rate limiting
    
    # Process PDFs
    pdf_files = list(ingest_dir.glob("*.pdf"))
    
    if pdf_files:
        print(f"\nFound {len(pdf_files)} PDF file(s) in /ingest/:")
        for pdf_file in pdf_files:
            print(f"  • {pdf_file.name}")
        
        if input("\nProcess PDF files? (y/n): ").lower() == 'y':
            for pdf_file in pdf_files:
                processor.process_pdf(
                    pdf_file, 
                    output_dir,
                    title=pdf_file.stem,
                    archive_dir=source_dir
                )
                time.sleep(2)  # Rate limiting
    
    # Print Mathpix summary
    processor.print_summary()
    
    # Optional: Annotate images with Vision API (ONLY for newly processed files)
    if processor.stats["processed"] > 0 and processor.processed_files:
        print("\n" + "="*60)
        print("VISION API IMAGE ANNOTATION (Optional)")
        print("="*60)
        print("Mathpix has extracted equations as LaTeX.")
        print("Images (charts/diagrams) can now be annotated with AI descriptions.")
        print(f"This will ONLY annotate the {len(processor.processed_files)} newly processed file(s):")
        for pf in processor.processed_files:
            print(f"  • {pf.name}")
        
        if input("\nAnnotate images with Vision API? (y/n): ").lower() == 'y':
            annotator = VisionAnnotator()
            
            # Only annotate files that were just processed in this session
            for md_file in processor.processed_files:
                annotator.annotate_markdown(md_file)
        
            annotator.print_summary()
    
    print("\n✨ Processing complete!")
    print(f"📁 Output directory: {output_dir}")
    print("\nWhat you now have:")
    print("✅ Equations extracted as LaTeX (searchable!)")
    print("✅ Images preserved as references")
    print("✅ Image descriptions added (if Vision API was used)")
    print("\nNext steps:")
    print("1. Check the generated markdown files for quality")
    print("2. Compare with original slides to verify LaTeX extraction")
    print("3. Use for RAG/semantic search!")


if __name__ == "__main__":
    main()

