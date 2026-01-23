"""Generate sample images for the gallery: pens, sofas, cups, and glasses."""

from PIL import Image, ImageDraw, ImageFont
import os
import random

GALLERY_FOLDER = '../search-app/image_gallery'

def draw_pen(draw, x, y, color, style='ballpoint'):
    """Draw a simple pen."""
    # Pen body
    draw.rectangle([x+50, y+100, x+70, y+300], fill=color, outline='black', width=2)
    # Pen tip
    draw.polygon([x+50, y+300, x+70, y+300, x+60, y+320], fill='gray', outline='black')
    # Pen clip
    if style == 'ballpoint':
        draw.rectangle([x+70, y+120, x+80, y+160], fill='silver', outline='black')
    # Cap detail
    draw.ellipse([x+48, y+95, x+72, y+105], fill=color, outline='black', width=2)

def draw_sofa(draw, x, y, color, style='modern'):
    """Draw a simple sofa."""
    # Seat
    draw.rectangle([x+50, y+200, x+350, y+260], fill=color, outline='black', width=2)
    # Backrest
    draw.rectangle([x+50, y+130, x+350, y+200], fill=color, outline='black', width=2)
    # Arms
    draw.rectangle([x+30, y+150, x+70, y+260], fill=color, outline='black', width=2)
    draw.rectangle([x+330, y+150, x+370, y+260], fill=color, outline='black', width=2)
    # Legs
    if style == 'modern':
        draw.rectangle([x+60, y+260, x+75, y+280], fill='#654321', outline='black')
        draw.rectangle([x+325, y+260, x+340, y+280], fill='#654321', outline='black')
    # Cushions
    draw.line([x+150, y+130, x+150, y+200], fill='black', width=2)
    draw.line([x+250, y+130, x+250, y+200], fill='black', width=2)

def draw_cup(draw, x, y, color, style='mug'):
    """Draw a simple cup."""
    # Cup body
    draw.rectangle([x+120, y+150, x+280, y+280], fill=color, outline='black', width=2)
    # Handle
    if style == 'mug':
        draw.arc([x+270, y+180, x+320, y+250], start=270, end=90, fill='black', width=3)
        draw.arc([x+280, y+190, x+310, y+240], start=270, end=90, fill=color, width=2)
    # Rim
    draw.ellipse([x+118, y+145, x+282, y+160], fill=color, outline='black', width=2)

def draw_glass(draw, x, y, color='lightblue', style='water'):
    """Draw a simple glass."""
    # Glass body (trapezoid)
    draw.polygon([x+130, y+150, x+270, y+150, x+280, y+290, x+120, y+290], 
                 fill=color, outline='black', width=2)
    # Rim highlight
    draw.ellipse([x+128, y+145, x+272, y+160], fill='white', outline='black', width=1)
    # Water/liquid level
    if style == 'water':
        draw.polygon([x+135, y+200, x+265, y+200, x+275, y+285, x+125, y+285], 
                     fill='lightblue', outline=None)
    # Glass shine effect
    draw.ellipse([x+160, y+180, x+180, y+220], fill='white', outline=None)

def create_image(item_type, item_name, color, style, index):
    """Create an image of the specified item."""
    img = Image.new('RGB', (400, 400), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw the item
    if item_type == 'pen':
        draw_pen(draw, 100, 20, color, style)
    elif item_type == 'sofa':
        draw_sofa(draw, 0, 50, color, style)
    elif item_type == 'cup':
        draw_cup(draw, 0, 50, color, style)
    elif item_type == 'glass':
        draw_glass(draw, 50, 50, color, style)
    
    # Add text label
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 350), f"{item_name}", fill='black', font=font)
    
    return img

# Define 10 items
items = [
    ('pen', 'Blue Ballpoint', 'blue', 'ballpoint'),
    ('pen', 'Red Pen', 'red', 'ballpoint'),
    ('sofa', 'Brown Leather Sofa', '#8B4513', 'modern'),
    ('sofa', 'Gray Fabric Sofa', 'gray', 'modern'),
    ('sofa', 'Green Couch', 'green', 'classic'),
    ('cup', 'Red Coffee Mug', 'red', 'mug'),
    ('cup', 'Blue Tea Cup', 'skyblue', 'mug'),
    ('cup', 'Yellow Mug', 'yellow', 'mug'),
    ('glass', 'Water Glass', 'lightblue', 'water'),
    ('glass', 'Empty Glass', 'lightcyan', 'empty'),
]

# Create gallery folder
os.makedirs(GALLERY_FOLDER, exist_ok=True)

# Remove old sample images
for fname in os.listdir(GALLERY_FOLDER):
    if fname.startswith('sample_'):
        try:
            os.remove(os.path.join(GALLERY_FOLDER, fname))
            print(f"Removed old image: {fname}")
        except Exception as e:
            print(f"Could not remove {fname}: {e}")

# Generate and save images
for idx, (item_type, name, color, style) in enumerate(items):
    img = create_image(item_type, name, color, style, idx)
    filename = f"{item_type}_{idx+1:02d}.jpg"
    filepath = os.path.join(GALLERY_FOLDER, filename)
    img.save(filepath, 'JPEG', quality=95)
    print(f"Created: {filename} - {name}")

print(f"\nSuccessfully created {len(items)} images in {GALLERY_FOLDER}")
