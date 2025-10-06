#!/usr/bin/env python3
"""
QR Code Detector and Remover
Systematically detects and removes QR codes from extracted images
"""

import cv2
import numpy as np
from pathlib import Path
import re


def is_qr_code(image_path):
    """
    Detect if an image contains a QR code using visual heuristics.
    Returns True if QR code detected.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        
        # Visual heuristics for QR codes
        # QR codes are typically square, black/white, high contrast
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Check if image is roughly square (QR codes are square)
        aspect_ratio = width / height if height > 0 else 0
        if not (0.8 < aspect_ratio < 1.2):
            return False
        
        # Check for high contrast (QR codes are black & white)
        # Calculate standard deviation
        std_dev = np.std(gray)
        mean_val = np.mean(gray)
        
        # QR codes typically have high std dev (black and white contrast)
        # and binary-like distribution
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        binary_ratio = binary_pixels / total_pixels
        
        # QR codes have roughly 50% black, 50% white pixels
        # and high standard deviation
        if std_dev > 80 and 0.3 < binary_ratio < 0.7:
            # Additional check: look for QR code patterns (three corner squares)
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # QR codes have many small square contours
            square_contours = 0
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                if len(approx) == 4:  # Quadrilateral
                    square_contours += 1
            
            # If many square patterns, likely a QR code
            if square_contours > 20:
                return True
        
        return False
        
    except Exception as e:
        print(f"   Warning: Could not analyze {image_path.name}: {e}")
        return False


def is_likely_slido_qr(image_path, slide_text=""):
    """
    Additional heuristic: Check if this is a Slido QR code based on context.
    """
    # Check filename for common patterns
    filename_lower = str(image_path).lower()
    if 'slido' in filename_lower or 'qr' in filename_lower:
        return True
    
    # Check if slide text mentions Slido
    if 'slido' in slide_text.lower():
        # Small images on Slido slides are likely QR codes
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                height, width = img.shape[:2]
                # Small images (< 500x500) on Slido slides
                if height < 500 and width < 500:
                    return True
        except:
            pass
    
    return False


def remove_qr_code_references(md_file, deleted_images):
    """
    Remove image references from Markdown files for deleted QR codes.
    """
    if not deleted_images:
        return False
    
    content = md_file.read_text(encoding='utf-8')
    original_content = content
    
    for img_name in deleted_images:
        # Match image markdown patterns
        patterns = [
            rf'!\[.*?\]\([^\)]*{re.escape(img_name)}[^\)]*\)\s*\n*',  # Standard
            rf'<img[^>]*src="[^"]*{re.escape(img_name)}"[^>]*>\s*\n*',  # HTML
        ]
        
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    # Write back if changed
    if content != original_content:
        md_file.write_text(content, encoding='utf-8')
        return True
    return False


def main():
    print("🔍 QR Code Detection and Removal")
    print("="*60)
    
    figures_dir = Path("Docs/figures")
    searchable_dir = Path("Docs/searchable")
    
    # Find all images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.wmf']
    all_images = []
    for ext in image_extensions:
        all_images.extend(figures_dir.rglob(ext))
    
    print(f"Scanning {len(all_images)} images for QR codes...\n")
    
    qr_codes = []
    non_qr = []
    
    for img_path in all_images:
        if is_qr_code(img_path):
            qr_codes.append(img_path)
            print(f"   🔳 QR CODE: {img_path.relative_to(figures_dir)}")
        else:
            non_qr.append(img_path)
    
    print(f"\n📊 Results:")
    print(f"   QR codes found: {len(qr_codes)}")
    print(f"   Regular images: {len(non_qr)}")
    
    if qr_codes:
        print(f"\n🗑️  Deleting {len(qr_codes)} QR code images...")
        
        # Group by lecture for markdown updates
        deleted_by_lecture = {}
        
        for qr_path in qr_codes:
            # Delete the image
            lecture_dir = qr_path.parent.name
            img_name = qr_path.name
            
            if lecture_dir not in deleted_by_lecture:
                deleted_by_lecture[lecture_dir] = []
            deleted_by_lecture[lecture_dir].append(img_name)
            
            qr_path.unlink()
            print(f"   ✓ Deleted: {qr_path.relative_to(figures_dir)}")
        
        # Update Markdown files
        print(f"\n📝 Updating Markdown files to remove QR references...")
        md_files = list(searchable_dir.glob("*.md"))
        updated_count = 0
        
        for md_file in md_files:
            # Determine which lecture this is
            for lec_dir, deleted_imgs in deleted_by_lecture.items():
                if lec_dir in md_file.stem or any(lec_dir in line for line in md_file.read_text().split('\n')[:20]):
                    if remove_qr_code_references(md_file, deleted_imgs):
                        print(f"   ✓ Updated: {md_file.name}")
                        updated_count += 1
        
        print(f"\n✅ Complete!")
        print(f"   Deleted: {len(qr_codes)} QR code images")
        print(f"   Updated: {updated_count} Markdown files")
        print(f"   Remaining: {len(non_qr)} useful images")
    else:
        print("\n✅ No QR codes found!")


if __name__ == "__main__":
    main()
