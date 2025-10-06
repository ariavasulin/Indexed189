#!/usr/bin/env python3
"""
OpenAI Vision API Annotator
Adds detailed descriptions to PowerPoint slide images using GPT-4 Vision
"""

import os
import base64
import re
from pathlib import Path
from openai import OpenAI
import time
import json

# Initialize OpenAI client
client = None


def init_openai():
    """Initialize OpenAI client with API key."""
    global client
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found in environment")
        api_key = input("Please enter your OpenAI API key: ").strip()
    
    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized")


def encode_image(image_path):
    """Encode image to base64 for API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_slide_context(md_file, slide_num):
    """Extract context around a specific slide from markdown."""
    content = md_file.read_text(encoding='utf-8')
    
    # Find the slide section
    slide_pattern = rf'## Slide {slide_num}\s*\n(.*?)(?=\n## Slide|\Z)'
    match = re.search(slide_pattern, content, re.DOTALL)
    
    if match:
        slide_text = match.group(1).strip()
        # Remove existing image references
        slide_text = re.sub(r'!\[.*?\]\([^\)]+\)', '', slide_text)
        # Clean up extra whitespace
        slide_text = re.sub(r'\n\s*\n', '\n', slide_text).strip()
        return slide_text[:500]  # Limit context length
    
    return ""


def describe_image(image_path, context=""):
    """Use GPT-4 Vision to describe the image."""
    base64_image = encode_image(image_path)
    
    # Build prompt with context
    prompt = """Describe this academic lecture slide image concisely for searchability. Focus on:
1. Main concept/topic shown
2. Type of visualization (diagram, equation, graph, chart, table, etc.)
3. Key elements, labels, or mathematical content
4. Educational purpose

Keep it under 100 words, technical and precise."""
    
    if context:
        prompt = f"Slide context: {context}\n\n{prompt}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini for cost efficiency; use "gpt-4o" for better quality
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "low"  # Use "high" for better quality but higher cost
                            }
                        }
                    ]
                }
            ],
            max_tokens=200
        )
        
        description = response.choices[0].message.content.strip()
        return description
        
    except Exception as e:
        print(f"   ❌ Error describing image: {e}")
        return None


def update_markdown_with_descriptions(md_file, annotations):
    """Update markdown file with image descriptions."""
    content = md_file.read_text(encoding='utf-8')
    
    for img_name, description in annotations.items():
        # Find image reference
        pattern = rf'(!\[.*?\]\([^\)]*{re.escape(img_name)}[^\)]*\))'
        
        def replace_with_description(match):
            img_markdown = match.group(1)
            # Update alt text with description
            updated = re.sub(
                r'!\[.*?\]',
                f'![{description[:100]}]',  # Truncate alt text
                img_markdown
            )
            # Add full description after image
            return f"{updated}\n\n**Image:** {description}\n"
        
        content = re.sub(pattern, replace_with_description, content)
    
    md_file.write_text(content, encoding='utf-8')


def process_lecture(lec_num):
    """Process all images for a specific lecture."""
    figures_dir = Path("Docs/figures")
    searchable_dir = Path("Docs/searchable")
    
    # Find figure directory and markdown file
    lec_dir = figures_dir / f"lec{lec_num:02d}"
    if not lec_dir.exists():
        print(f"   No figures found for lecture {lec_num}")
        return
    
    # Find corresponding markdown file
    md_files = list(searchable_dir.glob(f"*lec{lec_num:02d}*.md"))
    if not md_files:
        print(f"   No markdown file found for lecture {lec_num}")
        return
    
    md_file = md_files[0]
    
    # Get all slide images (not cell outputs)
    images = sorted(lec_dir.glob("slide-*.png")) + sorted(lec_dir.glob("slide-*.jpg"))
    
    if not images:
        print(f"   No slide images in lecture {lec_num}")
        return
    
    print(f"\n📚 Lecture {lec_num}: {len(images)} images to annotate")
    
    annotations = {}
    
    for idx, img_path in enumerate(images, 1):
        # Extract slide number from filename
        slide_match = re.search(r'slide-(\d+)', img_path.name)
        slide_num = int(slide_match.group(1)) if slide_match else 0
        
        # Get context
        context = get_slide_context(md_file, slide_num) if slide_num else ""
        
        print(f"   [{idx}/{len(images)}] {img_path.name}...", end=" ")
        
        # Describe image
        description = describe_image(img_path, context)
        
        if description:
            annotations[img_path.name] = description
            print(f"✓")
            print(f"      → {description[:80]}...")
        else:
            print("✗")
        
        # Rate limiting: small delay between API calls
        time.sleep(0.5)
    
    # Update markdown with all annotations
    if annotations:
        update_markdown_with_descriptions(md_file, annotations)
        print(f"   ✅ Updated {md_file.name} with {len(annotations)} descriptions")
    
    return len(annotations)


def save_progress(progress_file, completed_lectures):
    """Save progress to resume if interrupted."""
    with open(progress_file, 'w') as f:
        json.dump({"completed": completed_lectures}, f)


def load_progress(progress_file):
    """Load previous progress."""
    if Path(progress_file).exists():
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return set(data.get("completed", []))
    return set()


def main():
    print("🎨 OpenAI Vision API Image Annotator")
    print("="*60)
    
    # Initialize OpenAI
    init_openai()
    
    # Progress tracking
    progress_file = "vision_annotation_progress.json"
    completed = load_progress(progress_file)
    
    if completed:
        print(f"\n📝 Resuming... {len(completed)} lectures already completed")
    
    # Process lectures 1-10 (PowerPoint lectures only, skip discussions for now)
    lectures_to_process = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    total_annotated = 0
    
    for lec_num in lectures_to_process:
        if lec_num in completed:
            print(f"\n⏭️  Lecture {lec_num}: Already completed, skipping")
            continue
        
        try:
            count = process_lecture(lec_num)
            if count:
                total_annotated += count
                completed.add(lec_num)
                save_progress(progress_file, list(completed))
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            save_progress(progress_file, list(completed))
            print(f"Progress saved. Run again to resume.")
            break
        except Exception as e:
            print(f"   ❌ Error processing lecture {lec_num}: {e}")
            continue
    
    # Process discussions (if needed)
    print("\n" + "="*60)
    print(f"📊 Summary: {total_annotated} images annotated")
    print(f"✅ Completed lectures: {sorted(completed)}")
    
    # Cleanup progress file if all done
    if len(completed) >= len(lectures_to_process):
        Path(progress_file).unlink(missing_ok=True)
        print("\n🎉 All lectures processed! Progress file deleted.")


if __name__ == "__main__":
    main()

