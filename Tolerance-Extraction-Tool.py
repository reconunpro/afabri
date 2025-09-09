import fitz  # PyMuPDF
from collections import defaultdict
import math
import re
import argparse
import os

# Map suspicious single chars to their actual symbols
SUSPICIOUS_CHARS_MAP = {
    'O': '⌀',  # diameter
    'o': '⌀',  # diameter
    'v': '°',  # degree
    '`': '↧',  # depth
    'j': '↧',  # depth
    '¬': '↧',  # depth
    's': '⌵',  # countersink
    'c': '⌴',  # counterbore
    'h': '⌓',  # surface profile
    'r': '▱',  # flatness
    'u': '±'   # plus/minus
}

# GD&T multi-symbol patterns
GD_T_PATTERNS = [
    (re.compile(r'^[cC][oO]$'), '⌴⌀'),  # counterbore and diameter
    (re.compile(r'^[sS][oO]$'), '⌵⌀'),  # countersink and diameter
    (re.compile(r'^⌀\d+(\.\d+)?\s*[A-Za-z]\d+$'), '⌀'),  # diameter with tolerance
    (re.compile(r'^⌀\d+(\.\d+)?\s*↧\d+(\.\d+)?$'), '⌀↧'),  # diameter with depth
]

# STRICT ISO fit definitions - exact matches only from isofits library
HOLE_ISO_FITS = {
    "E7", "F6", "F7", "F8", "G6", "G7", "H6", "H7", "H8",
    "H9", "H10", "JS6", "JS7", "K6", "K7", "M6", "M7",
    "N6", "N7", "P6", "P7", "R7"
}

SHAFT_ISO_FITS = {
    "f6", "f7", "g5", "g6", "h4", "h5", "h6", "h7",
    "h8", "h9", "js5", "js6", "js7", "k5", "k6",
    "m5", "m6", "n5", "n6", "p6", "r6"
}

# All supported ISO fits
ALL_SUPPORTED_ISO_FITS = HOLE_ISO_FITS.union(SHAFT_ISO_FITS)

def extract_tolerance_from_dimension(text):
    """Extract tolerance if it's embedded in the dimension text"""
    text = text.strip()
    
    # Look for patterns like "12 ±0,1" or "25.4±0.05"
    tolerance_pattern = r'(\d+(?:[.,]\d+)?)\s*([±]\s*\d+(?:[.,]\d+)?)'
    match = re.search(tolerance_pattern, text)
    
    if match:
        base_dimension = match.group(1)
        tolerance = match.group(2)
        return base_dimension, tolerance
    
    return text, None

def has_embedded_tolerance(text):
    """Check if dimension text has embedded tolerance"""
    _, tolerance = extract_tolerance_from_dimension(text)
    return tolerance is not None

def is_iso_fit_tolerance(text):
    """
    IMPROVED: Strict ISO fit tolerance detection using exact matching
    Only returns True for exact matches in our supported ISO fits list
    """
    text = text.strip()
    
    # First, check for exact match in our supported fits
    if text in ALL_SUPPORTED_ISO_FITS:
        return True
    
    # Check if the text contains any of our supported fits as separate words
    for fit in ALL_SUPPORTED_ISO_FITS:
        if fit in text.split():
            return True
    
    # Final check: look for exact fit within the text
    for fit in ALL_SUPPORTED_ISO_FITS:
        if fit in text:
            return True
    
    return False

def extract_iso_fit_from_text(text):
    """
    Extract the actual ISO fit code from text that contains it
    Returns the ISO fit code if found, None otherwise
    """
    text = text.strip()
    
    # Check for exact match first
    if text in ALL_SUPPORTED_ISO_FITS:
        return text
    
    # Check for ISO fit as separate word
    for fit in ALL_SUPPORTED_ISO_FITS:
        if fit in text.split():
            return fit
    
    # Check for ISO fit within text
    for fit in ALL_SUPPORTED_ISO_FITS:
        if fit in text:
            return fit
    
    return None

def calculate_iso_fit_tolerance(nominal_size, fit_code):
    """Calculate ISO fit tolerance using isofits library if available"""
    try:
        from isofits import isotol
        
        # Determine if it's a hole or shaft fit based on case
        if fit_code[0].isupper():
            fit_type = 'hole'
        else:
            fit_type = 'shaft'
        
        # Get tolerances in microns
        upper_dev, lower_dev = isotol(fit_type, nominal_size, fit_code, 'both')
        
        # Convert to mm
        upper_mm = upper_dev / 1000
        lower_mm = lower_dev / 1000
        
        return {
            'upper_deviation_um': upper_dev,
            'lower_deviation_um': lower_dev,
            'upper_deviation_mm': upper_mm,
            'lower_deviation_mm': lower_mm,
            'tolerance_range': f"{nominal_size + lower_mm:.3f} to {nominal_size + upper_mm:.3f} mm",
            'fit_type': fit_type
        }
    except ImportError:
        print("Warning: isofits library not available. Install with: pip install isofits")
        return None
    except Exception as e:
        print(f"Error calculating ISO fit tolerance for {fit_code}: {e}")
        return None

def is_dimension(text):
    text = text.strip()
    
    # Check if dimension contains embedded tolerance
    if has_embedded_tolerance(text):
        return True  # It's a dimension with embedded tolerance
    
    # Simple dimension detection (number, possibly with units or symbols)
    if re.match(r'^[+-]?\d+(?:[.,]\d+)?$', text):
        return True
    if re.match(r'^⌀?\s*\d+(?:[.,]\d+)?(?:°|mm|cm|m|in)?$', text, re.IGNORECASE):
        return True
    # Avoid catching words
    if re.search(r'[a-zA-Z]', text) and not re.match(r'^\d+(?:[.,]\d+)?\s*(mm|cm|m|in|°)$', text, re.IGNORECASE):
        return False
    return False

def is_signed_tolerance(text):
    """Detect text that looks like signed tolerance: contains + or - signs, maybe with numbers"""
    text = text.strip()
    
    # First check if it's an ISO fit tolerance (exclude these)
    if is_iso_fit_tolerance(text):
        return False
        
    # e.g. "+0.1", "-0.05", "+/-0.02", "±0.01"
    if re.match(r'^[±\+\-]\s*\d+(?:[.,]\d+)?$', text):
        return True
    if re.match(r'^\+\-\s*\d+(?:[.,]\d+)?$', text):  # for "+-" variant
        return True
    if re.match(r'^\d+(?:[.,]\d+)?\s*[±\+\-]\s*\d+(?:[.,]\d+)?$', text):  # e.g. "5 ± 0.1"
        return True
    return False

def is_plus_minus_tolerance(text):
    """Check if tolerance uses ± symbol specifically"""
    text = text.strip()
    return '±' in text

def extract_general_tolerance_table(doc):
    """Extract general tolerance values from tolerance tables in the PDF"""
    general_tolerances = {}
    
    for page in doc:
        page_text = page.get_text()
        
        # Look for X.X format patterns (common in engineering drawings)
        # Pattern: X =±0.2 mm, X.X =±0.1 mm, X.XX =±0.05 mm, X.XXX =±0.02 mm
        x_patterns = [
            (r'X\.XXX\s*=\s*[±]\s*([0-9.]+)\s*mm', 3),  # X.XXX = ±value
            (r'X\.XX\s*=\s*[±]\s*([0-9.]+)\s*mm', 2),   # X.XX = ±value  
            (r'X\.X\s*=\s*[±]\s*([0-9.]+)\s*mm', 1),    # X.X = ±value
            (r'X\s*=\s*[±]\s*([0-9.]+)\s*mm', 0)        # X = ±value
        ]
        
        for pattern, decimal_places in x_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                tolerance_value = matches[0]
                general_tolerances[decimal_places] = f"±{tolerance_value}"
                print(f"Found general tolerance: {decimal_places} decimals = ±{tolerance_value}mm")
        
        # Also look for angular tolerances
        angular_pattern = r'ANGULAR[S]?\s*[:\-=]\s*[±]\s*([0-9.]+)\s*[°]?'
        angular_matches = re.findall(angular_pattern, page_text, re.IGNORECASE)
        if angular_matches:
            general_tolerances['angular'] = f"±{angular_matches[0]}°"
            print(f"Found angular tolerance: ±{angular_matches[0]}°")
    
    # If no general tolerances found, use default values
    if not general_tolerances:
        print("No general tolerance table found. Using default values.")
        general_tolerances = {
            0: "±0.2",     # X = ±0.2mm
            1: "±0.1",     # X.X = ±0.1mm  
            2: "±0.05",    # X.XX = ±0.05mm
            3: "±0.02",    # X.XXX = ±0.02mm
            'angular': "±0.5°"
        }
    
    return general_tolerances

def get_decimal_places(dimension_text):
    """Determine number of decimal places in a dimension"""
    # Remove units and symbols first
    clean_text = re.sub(r'[⌀°a-zA-Z\s±]', '', dimension_text)
    
    # Find the main number (handle both . and , as decimal separators)
    number_match = re.search(r'\d+(?:[.,]\d+)?', clean_text)
    if number_match:
        number = number_match.group()
        # Handle both . and , as decimal separators
        if '.' in number:
            decimal_part = number.split('.')[1]
            return len(decimal_part)
        elif ',' in number:
            decimal_part = number.split(',')[1]
            return len(decimal_part)
    return 0  # No decimal places (whole number)

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def find_suspicious_chars_near_iso_fit(iso_fit_span, all_spans, font_tolerance=2.0, proximity_threshold=30):
    """
    Check for SUSPICIOUS_CHARS_MAP characters AND nearby dimensions near ISO fit tolerances
    More generous proximity to catch both symbols and dimensions
    """
    iso_pos = iso_fit_span['pos']
    iso_font = iso_fit_span['font_size']
    iso_page = iso_fit_span['page']
    
    nearby_suspicious = []
    nearby_dimensions = []
    
    for span in all_spans:
        if span['page'] != iso_page:
            continue
            
        # Skip if it's the ISO fit itself
        if span['text'] == iso_fit_span['text'] and span['pos'] == iso_fit_span['pos']:
            continue
            
        # Check font size similarity (more lenient for ISO fits)
        font_diff = abs(span['font_size'] - iso_font)
        if font_diff > font_tolerance + 1.0:  # More lenient
            continue
            
        # Check proximity (more generous for ISO fits)
        distance_to_iso = distance(span['pos'], iso_pos)
        if distance_to_iso > proximity_threshold:
            continue
            
        span_text = span['text'].strip()
        
        # Check if it's one of our suspicious characters
        if len(span_text) == 1 and span_text in SUSPICIOUS_CHARS_MAP:
            translated_symbol = SUSPICIOUS_CHARS_MAP[span_text]
            
            nearby_suspicious.append({
                'original': span_text,
                'translated': translated_symbol,
                'pos': span['pos'],
                'distance': distance_to_iso,
                'font_size': span['font_size']
            })
        
        # Also check for nearby dimensions (numbers) - more generous pattern
        elif re.match(r'^\d+(?:[.,]\d+)?(?:mm|cm|m|in)?$', span_text, re.IGNORECASE):
            # Remove units to get just the number
            clean_number = re.sub(r'(mm|cm|m|in)$', '', span_text, flags=re.IGNORECASE).strip()
            
            nearby_dimensions.append({
                'text': clean_number,
                'original_text': span_text,
                'pos': span['pos'],
                'distance': distance_to_iso,
                'font_size': span['font_size']
            })
    
    # Sort by distance (closest first)
    nearby_suspicious.sort(key=lambda x: x['distance'])
    nearby_dimensions.sort(key=lambda x: x['distance'])
    
    return nearby_suspicious, nearby_dimensions

def construct_iso_fit_with_symbols(iso_fit_span, nearby_suspicious, nearby_dimensions):
    """
    Construct complete ISO fit notation with nearby suspicious characters AND dimensions
    """
    iso_fit_code = iso_fit_span.get('extracted_fit', iso_fit_span['text'])
    
    # Start with symbols
    all_symbols = ''.join([s['translated'] for s in nearby_suspicious]) if nearby_suspicious else ''
    
    # Add closest dimension if found
    dimension_part = nearby_dimensions[0]['text'] if nearby_dimensions else ''
    
    # Construct complete notation: symbols + dimension + ISO fit
    if all_symbols and dimension_part:
        complete_notation = all_symbols + dimension_part + ' ' + iso_fit_code
    elif all_symbols:
        complete_notation = all_symbols + iso_fit_code
    elif dimension_part:
        complete_notation = dimension_part + ' ' + iso_fit_code
    else:
        complete_notation = iso_fit_code
    
    return complete_notation

def find_nearby_symbols_for_embedded_tolerance(main_dim, all_spans, font_tolerance=2.0, proximity_threshold=20, axis_tolerance=3):
    """
    Find suspicious characters specifically around dimensions with embedded ± tolerances
    This handles cases like 'o 12±0,1' which should become '⌀12 ±0,1'
    """
    main_pos = main_dim['pos']
    main_font = main_dim['font_size']
    main_page = main_dim['page']
    
    nearby_symbols = []
    
    for span in all_spans:
        if span['page'] != main_page:
            continue
            
        # Skip if it's the main dimension itself
        if span['text'] == main_dim['text'] and span['pos'] == main_dim['pos']:
            continue
            
        # Check font size similarity (allow slightly more tolerance for embedded dimensions)
        font_diff = abs(span['font_size'] - main_font)
        if font_diff > font_tolerance + 1.0:  # Slightly more lenient
            continue
            
        # Check proximity - be more generous for embedded tolerance detection
        distance_to_main = distance(span['pos'], main_pos)
        if distance_to_main > proximity_threshold + 5:  # Slightly more generous
            continue
            
        # Check if it's one of our suspicious characters
        span_text = span['text'].strip()
        
        # ONLY process single characters that are in our SUSPICIOUS_CHARS_MAP
        if len(span_text) == 1 and span_text in SUSPICIOUS_CHARS_MAP:
            translated_symbol = SUSPICIOUS_CHARS_MAP[span_text]
            
            nearby_symbols.append({
                'original': span_text,
                'translated': translated_symbol,
                'pos': span['pos'],
                'distance': distance_to_main,
                'font_size': span['font_size']
            })
    
    # Sort by distance (closest first)
    nearby_symbols.sort(key=lambda x: x['distance'])
    
    return nearby_symbols

def construct_complete_embedded_tolerance_notation(main_dim, nearby_symbols):
    """
    Construct complete notation for embedded tolerance dimensions
    Input: main_dim with text like "12 ±0,1"
    Output: complete notation like "⌀12 ±0,1" if 'o' symbol found nearby
    """
    original_text = main_dim['text']
    
    if not nearby_symbols:
        return original_text
    
    # Extract the base dimension and tolerance parts
    base_dim, tolerance_part = extract_tolerance_from_dimension(original_text)
    
    if not tolerance_part:
        # Fallback - just add symbols to the front
        all_symbols = ''.join([s['translated'] for s in nearby_symbols])
        complete_notation = all_symbols + original_text
        return complete_notation
    
    # Reconstruct with symbols at the front of the base dimension
    all_symbols = ''.join([s['translated'] for s in nearby_symbols])
    complete_notation = all_symbols + base_dim + ' ' + tolerance_part
    
    return complete_notation

def find_nearby_symbols_and_text(main_dim, all_spans, font_tolerance=2.0, proximity_threshold=20, axis_tolerance=3):
    """Find symbols and text near the main dimension with similar font size and axis alignment"""
    main_pos = main_dim['pos']
    main_font = main_dim['font_size']
    main_page = main_dim['page']
    
    nearby_elements = []
    
    for span in all_spans:
        if span['page'] != main_page:
            continue
            
        # Skip if it's the main dimension itself
        if span['text'] == main_dim['text'] and span['pos'] == main_dim['pos']:
            continue
            
        # Check font size similarity (stricter)
        font_diff = abs(span['font_size'] - main_font)
        if font_diff > font_tolerance:
            continue
            
        # Check proximity (much stricter - only very close elements)
        distance_to_main = distance(span['pos'], main_pos)
        if distance_to_main > proximity_threshold:
            continue
            
        # Check axis alignment (stricter tolerance)
        x_diff = abs(span['pos'][0] - main_pos[0])
        y_diff = abs(span['pos'][1] - main_pos[1])
        
        # Consider aligned if on same X-axis (horizontal) OR Y-axis (vertical) - but not both
        x_aligned = x_diff <= axis_tolerance
        y_aligned = y_diff <= axis_tolerance
        is_aligned = (x_aligned and not y_aligned) or (y_aligned and not x_aligned)
        
        if is_aligned:
            # Skip if it's a pure number (don't treat numbers as symbols)
            if re.match(r'^\d+(?:[.,]\d+)?$', span['text']):
                continue
            
            # ONLY process single characters that are in our SUSPICIOUS_CHARS_MAP
            if len(span['text']) == 1 and span['text'] in SUSPICIOUS_CHARS_MAP:
                translated_text = SUSPICIOUS_CHARS_MAP[span['text']]
                
                nearby_elements.append({
                    'text': span['text'],
                    'translated': translated_text,
                    'pos': span['pos'],
                    'font_size': span['font_size'],
                    'distance': distance_to_main,
                    'alignment': 'horizontal' if x_aligned else 'vertical'
                })
            
            # Check for GD&T patterns (only specific ones we defined)
            elif len(span['text']) == 2:  # Only 2-character combinations like "co", "so"
                for pattern, symbol in GD_T_PATTERNS:
                    if pattern.fullmatch(span['text']):
                        nearby_elements.append({
                            'text': span['text'],
                            'translated': symbol,
                            'pos': span['pos'],
                            'font_size': span['font_size'],
                            'distance': distance_to_main,
                            'alignment': 'horizontal' if x_aligned else 'vertical'
                        })
                        break
    
    # Sort by distance (closest first)
    nearby_elements.sort(key=lambda x: x['distance'])
    
    return nearby_elements

def construct_complete_dimension_notation(main_dim, nearby_elements):
    """Construct complete dimension notation including nearby symbols"""
    if not nearby_elements:
        return main_dim['text']
    
    # Group elements by position relative to main dimension
    left_elements = []
    right_elements = []
    top_elements = []
    bottom_elements = []
    
    main_pos = main_dim['pos']
    
    for elem in nearby_elements:
        elem_pos = elem['pos']
        
        # Determine relative position
        if elem_pos[0] < main_pos[0] - 5:  # Left
            left_elements.append(elem)
        elif elem_pos[0] > main_pos[0] + 5:  # Right
            right_elements.append(elem)
        elif elem_pos[1] < main_pos[1] - 5:  # Top
            top_elements.append(elem)
        elif elem_pos[1] > main_pos[1] + 5:  # Bottom
            bottom_elements.append(elem)
    
    # Sort each group by position
    left_elements.sort(key=lambda x: x['pos'][0], reverse=True)  # Closest to main first
    right_elements.sort(key=lambda x: x['pos'][0])  # Closest to main first
    top_elements.sort(key=lambda x: x['pos'][1], reverse=True)  # Closest to main first
    bottom_elements.sort(key=lambda x: x['pos'][1])  # Closest to main first
    
    # Construct the complete notation - symbols at the start
    complete_notation = ""
    
    # Add symbols at the start (most common practice)
    all_symbols = []
    
    # Collect all symbols from all positions
    if left_elements:
        all_symbols.extend([elem['translated'] for elem in left_elements])
    if top_elements:
        all_symbols.extend([elem['translated'] for elem in top_elements])
    if right_elements:
        all_symbols.extend([elem['translated'] for elem in right_elements])
    if bottom_elements:
        all_symbols.extend([elem['translated'] for elem in bottom_elements])
    
    # Add symbols at the start
    if all_symbols:
        complete_notation = "".join(all_symbols) + main_dim['text']
    else:
        complete_notation = main_dim['text']
    
    return complete_notation.strip()

def assign_general_tolerances(dim_groups, general_tolerances):
    """Assign general tolerances to dimensions without specific tolerances"""
    assignments = []
    
    for idx, group in dim_groups.items():
        main_dim = group['main']
        
        # Skip angular dimensions - they don't need tolerance assignments
        if '°' in main_dim['text']:
            continue
            
        # Skip dimensions that already have embedded tolerance
        if has_embedded_tolerance(main_dim['text']):
            continue
            
        # Check if dimension has NO specific tolerances assigned
        has_tolerances = (group['tolerances'] or 
                         group['signed_tolerances'] or 
                         group.get('plus_minus_tolerances', []) or
                         group.get('iso_fit_tolerances', []))
        
        if not has_tolerances:
            # Get decimal places from the main dimension
            decimal_places = get_decimal_places(main_dim['text'])
            
            assigned_tolerance = general_tolerances.get(decimal_places, '±0.1')
            
            # Create descriptive reason based on decimal places
            if decimal_places == 0:
                reason = 'Whole number (X format) - general tolerance'
            elif decimal_places == 1:
                reason = '1 decimal place (X.X format) - general tolerance'
            elif decimal_places == 2:
                reason = '2 decimal places (X.XX format) - general tolerance'
            elif decimal_places == 3:
                reason = '3 decimal places (X.XXX format) - general tolerance'
            else:
                reason = f'{decimal_places} decimal places - general tolerance'
                
            assignments.append({
                'main_dim': main_dim,
                'assigned_tolerance': assigned_tolerance,
                'reason': reason,
                'decimal_places': decimal_places
            })
    
    return assignments

def filter_abnormal_font_sizes(tolerance_list, tolerance_type):
    """Filter out font sizes that appear only once if there are multiple different sizes"""
    if len(tolerance_list) <= 1:
        return tolerance_list
    
    # Count font sizes
    font_size_counts = defaultdict(int)
    for tol in tolerance_list:
        font_size_counts[tol['font_size']] += 1
    
    # Find the most common font size
    most_common_font = max(font_size_counts.items(), key=lambda x: x[1])
    most_common_size, most_common_count = most_common_font
    
    # Filter out font sizes that appear only once if there are multiple different sizes
    filtered_list = []
    removed_count = 0
    
    for tol in tolerance_list:
        if font_size_counts[tol['font_size']] == 1 and len(font_size_counts) > 1:
            removed_count += 1
            print(f"Filtered out {tolerance_type}: '{tol['text']}' (Font {tol['font_size']:.1f}pt - appears only once)")
        else:
            filtered_list.append(tol)
    
    if removed_count > 0:
        print(f"Removed {removed_count} {tolerance_type} entries with abnormal font sizes")
    
    return filtered_list

def filter_main_dimensions_font_sizes(main_dims_list):
    """Filter out main dimensions with abnormal font sizes"""
    if len(main_dims_list) <= 2:
        return main_dims_list
    
    # Count font sizes for main dimensions
    font_size_counts = defaultdict(int)
    for dim in main_dims_list:
        font_size_counts[dim['font_size']] += 1
    
    # Find the most common font size
    most_common_font = max(font_size_counts.items(), key=lambda x: x[1])
    most_common_size, most_common_count = most_common_font
    
    # Filter out font sizes that appear only once if there are multiple different sizes
    filtered_list = []
    removed_count = 0
    
    for dim in main_dims_list:
        if font_size_counts[dim['font_size']] == 1 and len(font_size_counts) > 1:
            removed_count += 1
            print(f"Filtered out main dimension: '{dim['text']}' (Font {dim['font_size']:.1f}pt - appears only once)")
        else:
            filtered_list.append(dim)
    
    if removed_count > 0:
        print(f"Removed {removed_count} main dimension entries with abnormal font sizes")
    
    return filtered_list

def analyze_and_group_tolerances(pdf_path, tolerance_font_ratio=0.7, max_distance=30):
    """
    Enhanced tolerance analysis with IMPROVED ISO fit detection and symbol/dimension integration
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return
    
    print(f"Analyzing PDF: {pdf_path}")
    print("=" * 60)

    # Print supported ISO fits information
    print("SUPPORTED ISO FIT TOLERANCES:")
    print(f"• HOLE fits: {', '.join(sorted(HOLE_ISO_FITS))}")
    print(f"• SHAFT fits: {', '.join(sorted(SHAFT_ISO_FITS))}")
    print(f"• Total supported: {len(ALL_SUPPORTED_ISO_FITS)} fit codes")
    print("=" * 60)

    # Extract general tolerance table
    general_tolerances = extract_general_tolerance_table(doc)
    print(f"General tolerances found: {general_tolerances}\n")

    all_dims = []
    all_spans = []  # Store all spans for symbol detection
    iso_fit_tolerances = []  # Store ISO fit tolerances separately

    # Extract all dimension-like spans with their info
    print("\nSCANNING FOR DIMENSIONS, TOLERANCES AND ISO FITS...")
    print("=" * 60)
    
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                        
                    bbox = span["bbox"]
                    x = (bbox[0] + bbox[2]) / 2
                    y = (bbox[1] + bbox[3]) / 2
                    font_size = span.get("size", 10)
                    
                    span_data = {
                        'text': text,
                        'page': page_num,
                        'pos': (x, y),
                        'font_size': font_size
                    }
                    
                    # Store all spans for symbol detection
                    all_spans.append(span_data)
                    
                    # IMPROVED: Strict ISO fit detection using exact matching
                    extracted_fit = extract_iso_fit_from_text(text)
                    if extracted_fit:
                        iso_fit_tolerances.append(span_data)
                        span_data['extracted_fit'] = extracted_fit
                        fit_type = 'HOLE' if extracted_fit in HOLE_ISO_FITS else 'SHAFT'
                        print(f"Found ISO fit: '{extracted_fit}' in text '{text}' ({fit_type}) (Page {page_num}, Font {font_size:.1f}pt)")
                        
                        # Initialize defaults - symbols will be detected later in comprehensive pass
                        span_data['complete_notation'] = text
                        span_data['nearby_elements'] = []
                    
                    # Only add dimension-like texts to all_dims (exclude ISO fits already found)
                    elif is_dimension(text) or is_signed_tolerance(text):
                        all_dims.append(span_data)

    if not all_dims:
        print("No dimension-like texts found.")
        doc.close()
        return

    print(f"Found {len(iso_fit_tolerances)} ISO fit tolerances")

    # Sort by font size descending to find max font (main dims usually bigger font)
    all_dims.sort(key=lambda d: d['font_size'], reverse=True)
    max_font = all_dims[0]['font_size']

    main_dims = []
    tolerances = []
    signed_tolerances = []

    # Threshold font size for main dimension vs tolerance
    threshold_font = max_font * tolerance_font_ratio

    # Separate main dims, regular tolerances, signed tolerances
    for d in all_dims:
        if d['font_size'] >= threshold_font:
            main_dims.append(d)
        else:
            if is_signed_tolerance(d['text']):
                signed_tolerances.append(d)
            else:
                tolerances.append(d)

    # Filter out abnormal font sizes if we only have 1 entry
    if len(main_dims) == 1:
        print("Only 1 main dimension found with large font - ignoring font size filtering")
        # Re-process without font filtering
        main_dims = []
        tolerances = []
        signed_tolerances = []
        
        for d in all_dims:
            if is_dimension(d['text']) and not is_signed_tolerance(d['text']):
                main_dims.append(d)
            elif is_signed_tolerance(d['text']):
                signed_tolerances.append(d)
            else:
                tolerances.append(d)

    # Filter main dimensions, tolerances and signed tolerances
    main_dims = filter_main_dimensions_font_sizes(main_dims)
    tolerances = filter_abnormal_font_sizes(tolerances, "regular tolerance")
    signed_tolerances = filter_abnormal_font_sizes(signed_tolerances, "signed tolerance")
    iso_fit_tolerances = filter_abnormal_font_sizes(iso_fit_tolerances, "ISO fit tolerance")

    print(f"Found {len(main_dims)} main dimensions, {len(tolerances)} regular tolerances, {len(signed_tolerances)} signed tolerances, and {len(iso_fit_tolerances)} ISO fit tolerances.\n")

    # Prepare groups: each main dim with tolerances and signed tolerances
    dim_groups = defaultdict(lambda: {'main': None, 'tolerances': [], 'signed_tolerances': [], 'plus_minus_tolerances': [], 'iso_fit_tolerances': []})
    for idx, main in enumerate(main_dims):
        dim_groups[idx]['main'] = main

    unassigned_tolerances = []
    unassigned_signed_tolerances = []
    unassigned_iso_fit_tolerances = []

    # Assign regular tolerances to closest main dimension within max_distance
    for tol in tolerances:
        closest_main_idx = None
        closest_dist = float('inf')
        for idx, main in enumerate(main_dims):
            if tol['page'] != main['page']:
                continue
            dist = distance(tol['pos'], main['pos'])
            if dist < closest_dist and dist <= max_distance:
                closest_dist = dist
                closest_main_idx = idx
        if closest_main_idx is not None:
            dim_groups[closest_main_idx]['tolerances'].append(tol)
        else:
            unassigned_tolerances.append(tol)

    # Assign signed tolerances to closest main dimension within max_distance
    # Separate ± tolerances from +/- tolerances
    for tol in signed_tolerances:
        closest_main_idx = None
        closest_dist = float('inf')
        for idx, main in enumerate(main_dims):
            if tol['page'] != main['page']:
                continue
            dist = distance(tol['pos'], main['pos'])
            if dist < closest_dist and dist <= max_distance:
                closest_dist = dist
                closest_main_idx = idx
        if closest_main_idx is not None:
            if is_plus_minus_tolerance(tol['text']):
                # This is ± tolerance
                dim_groups[closest_main_idx]['plus_minus_tolerances'].append(tol)
            else:
                # This is upper/lower tolerance (+/-)
                dim_groups[closest_main_idx]['signed_tolerances'].append(tol)
        else:
            unassigned_signed_tolerances.append(tol)

    # Assign ISO fit tolerances to closest main dimension within max_distance
    for tol in iso_fit_tolerances:
        closest_main_idx = None
        closest_dist = float('inf')
        for idx, main in enumerate(main_dims):
            if tol['page'] != main['page']:
                continue
            dist = distance(tol['pos'], main['pos'])
            if dist < closest_dist and dist <= max_distance:
                closest_dist = dist
                closest_main_idx = idx
        if closest_main_idx is not None:
            dim_groups[closest_main_idx]['iso_fit_tolerances'].append(tol)
        else:
            unassigned_iso_fit_tolerances.append(tol)

    # Assign general tolerances to dimensions without specific tolerances
    general_tolerance_assignments = assign_general_tolerances(dim_groups, general_tolerances)

    # Find nearby symbols and complete dimension notations
    print("\n" + "="*80)
    print("DETECTING NEARBY SYMBOLS AND COMPLETE NOTATIONS")
    print("="*80)
    
    # Process regular dimensions
    for idx, group in dim_groups.items():
        main_dim = group['main']
        nearby_elements = find_nearby_symbols_and_text(main_dim, all_spans)
        
        if nearby_elements:
            complete_notation = construct_complete_dimension_notation(main_dim, nearby_elements)
            group['complete_notation'] = complete_notation
            group['nearby_elements'] = nearby_elements
            
            print(f"Dimension: '{main_dim['text']}' → Complete: '{complete_notation}'")
            print(f"  Nearby elements: {[elem['translated'] for elem in nearby_elements]}")
        else:
            group['complete_notation'] = main_dim['text']
            group['nearby_elements'] = []
    
    # ENHANCED: Process ISO fit tolerances for nearby SUSPICIOUS CHARACTERS AND DIMENSIONS
    print(f"\nDETECTING SUSPICIOUS CHARACTERS AND DIMENSIONS NEAR ISO FIT TOLERANCES:")
    print("-" * 70)
    
    iso_fits_with_symbols = []
    
    for iso_tol in iso_fit_tolerances:
        extracted_fit = iso_tol.get('extracted_fit', iso_tol['text'])
        
        # Check for suspicious characters AND dimensions near this ISO fit
        nearby_suspicious, nearby_dimensions = find_suspicious_chars_near_iso_fit(iso_tol, all_spans)
        
        if nearby_suspicious or nearby_dimensions:
            complete_iso_notation = construct_iso_fit_with_symbols(iso_tol, nearby_suspicious, nearby_dimensions)
            iso_tol['complete_iso_notation'] = complete_iso_notation
            iso_tol['nearby_suspicious'] = nearby_suspicious
            iso_tol['nearby_dimensions'] = nearby_dimensions
            
            print(f"ISO Fit: '{extracted_fit}' → Complete: '{complete_iso_notation}'")
            
            # Show what was found
            found_items = []
            if nearby_suspicious:
                suspicious_chars_display = [f"{s['original']}→{s['translated']}" for s in nearby_suspicious]
                found_items.append(f"Symbols: {suspicious_chars_display}")
            if nearby_dimensions:
                dimensions_display = [d['text'] for d in nearby_dimensions]
                found_items.append(f"Dimensions: {dimensions_display}")
            
            print(f"  Found nearby: {', '.join(found_items)}")
            
            iso_fits_with_symbols.append({
                'original_fit': extracted_fit,
                'complete_notation': complete_iso_notation,
                'suspicious_chars': nearby_suspicious,
                'nearby_dimensions': nearby_dimensions,
                'page': iso_tol['page']
            })
        else:
            iso_tol['complete_iso_notation'] = extracted_fit
            iso_tol['nearby_suspicious'] = []
            iso_tol['nearby_dimensions'] = []
            print(f"ISO Fit: '{extracted_fit}' → No symbols or dimensions found nearby")
    
    if iso_fits_with_symbols:
        print(f"\nSUMMARY - ISO FITS WITH DETECTED SYMBOLS AND DIMENSIONS:")
        print("-" * 60)
        for i, item in enumerate(iso_fits_with_symbols, 1):
            info_parts = []
            if item['suspicious_chars']:
                symbols_info = ", ".join([f"{s['original']}({s['translated']})" for s in item['suspicious_chars']])
                info_parts.append(f"Symbols: {symbols_info}")
            if item.get('nearby_dimensions'):
                dims_info = ", ".join([d['text'] for d in item['nearby_dimensions']])
                info_parts.append(f"Dimensions: {dims_info}")
            
            info_display = " | ".join(info_parts) if info_parts else "No additional info"
            print(f"   {i}) {item['original_fit']} → {item['complete_notation']} [{info_display}] (Page {item['page']})")
    else:
        print("No symbols or dimensions found near any ISO fit tolerances.")

    # Print comprehensive tolerance analysis results
    print("\n" + "="*80)
    print("TOLERANCE ANALYSIS RESULTS")
    print("="*80)
    
    # Collect and categorize all dimensions
    plus_minus_tolerances_list = []  # ± tolerances
    upper_lower_tolerances_list = []  # +/- tolerances  
    iso_fit_tolerances_list = []  # ISO fit tolerances like H7, g6
    general_tolerances_list = []
    
    for idx, group in dim_groups.items():
        main = group['main']
        tols = group['tolerances']
        signed_tols = group['signed_tolerances']
        plus_minus_tols = group.get('plus_minus_tolerances', [])
        iso_fit_tols = group.get('iso_fit_tolerances', [])
        complete_dim = group.get('complete_notation', main['text'])
        
        # Skip angular dimensions
        if '°' in main['text']:
            continue
            
        # Check if dimension has embedded tolerance - UPDATED WITH SYMBOL DETECTION
        base_dim, embedded_tolerance = extract_tolerance_from_dimension(main['text'])
        if embedded_tolerance:
            # Find symbols near embedded tolerance dimensions
            nearby_symbols = find_nearby_symbols_for_embedded_tolerance(main, all_spans)
            complete_embedded_notation = construct_complete_embedded_tolerance_notation(main, nearby_symbols)
            
            plus_minus_tolerances_list.append({
                'dimension': complete_embedded_notation,  # Now includes GD&T symbols!
                'tolerances': [embedded_tolerance],
                'page': main['page']
            })
        # Check if has ISO fit tolerances
        elif iso_fit_tols:
            tolerance_texts = []
            complete_iso_notations = []
            
            for t in iso_fit_tols:
                extracted_fit = t.get('extracted_fit', t['text'])
                tolerance_texts.append(extracted_fit)
                
                # Use the ORIGINAL TEXT that contains dimension + ISO fit (like "16 g6", "10 H7")
                original_text_with_dimension = t['text']
                
                # Check for symbols around the ISO fit
                nearby_suspicious, nearby_dimensions = find_suspicious_chars_near_iso_fit(t, all_spans)
                
                if nearby_suspicious:
                    # Add symbols to the original text that already contains the dimension
                    all_symbols = ''.join([s['translated'] for s in nearby_suspicious])
                    complete_iso_notation = all_symbols + original_text_with_dimension
                    complete_iso_notations.append(complete_iso_notation)
                    
                    # Store the symbol info in the ISO fit tolerance object
                    t['complete_iso_notation'] = complete_iso_notation
                    t['nearby_suspicious'] = nearby_suspicious
                    t['nearby_dimensions'] = nearby_dimensions
                else:
                    # Use original text as-is (preserves dimension numbers like "16 g6")
                    complete_iso_notations.append(original_text_with_dimension)
                    t['complete_iso_notation'] = original_text_with_dimension
                    t['nearby_suspicious'] = []
                    t['nearby_dimensions'] = []
            
            iso_fit_tolerances_list.append({
                'dimension': complete_dim,  # Main dimension (might be from bearing, shaft, etc.)
                'tolerances': tolerance_texts,
                'complete_iso_notations': complete_iso_notations,  # Store complete ISO notations with symbols
                'page': main['page']
            })
        # Check if has ± tolerances
        elif plus_minus_tols:
            tolerance_texts = [t['text'] for t in plus_minus_tols]
            plus_minus_tolerances_list.append({
                'dimension': complete_dim,
                'tolerances': tolerance_texts,
                'page': main['page']
            })
        # Check if has upper/lower tolerances (+/-) or regular tolerances
        elif signed_tols or tols:
            # Combine signed tolerances and regular tolerances
            all_tolerance_texts = []
            if signed_tols:
                all_tolerance_texts.extend([t['text'] for t in signed_tols])
            if tols:
                all_tolerance_texts.extend([t['text'] for t in tols])
            
            upper_lower_tolerances_list.append({
                'dimension': complete_dim,
                'tolerances': all_tolerance_texts,
                'page': main['page']
            })
        else:
            # Gets general tolerance
            decimal_places = get_decimal_places(main['text'])
            assigned_tol = general_tolerances.get(decimal_places, '±0.1')
            general_tolerances_list.append({
                'dimension': complete_dim,
                'tolerance': assigned_tol,
                'decimal_places': decimal_places,
                'page': main['page']
            })
    
    # Also check for unassigned ISO fit tolerances (not assigned to any main dimension)
    for iso_tol in unassigned_iso_fit_tolerances:
        extracted_fit = iso_tol.get('extracted_fit', iso_tol['text'])
        
        # Use the ORIGINAL TEXT that contained the dimension + ISO fit
        original_text_with_dimension = iso_tol['text']  # This contains "16 g6", "10 H7", etc.
        
        # Check for symbols around this ISO fit
        nearby_suspicious, nearby_dimensions = find_suspicious_chars_near_iso_fit(iso_tol, all_spans)
        
        if nearby_suspicious:
            # Add symbols to the original text
            all_symbols = ''.join([s['translated'] for s in nearby_suspicious])
            complete_iso_notation = all_symbols + original_text_with_dimension
        else:
            complete_iso_notation = original_text_with_dimension
        
        iso_fit_tolerances_list.append({
            'dimension': complete_iso_notation,  # Use complete notation with original dimensions
            'tolerances': [extracted_fit],
            'complete_iso_notations': [complete_iso_notation],  # Store complete notation with symbols
            'page': iso_tol['page']
        })
    
    # Print categorized results
    print(f"\n1) DIMENSIONS WITH ± TOLERANCES ({len(plus_minus_tolerances_list)}):")
    print("-" * 60)
    if plus_minus_tolerances_list:
        for i, item in enumerate(plus_minus_tolerances_list, 1):
            tolerance_texts = ", ".join(item['tolerances'])
            print(f"   {i}) {item['dimension']} → {tolerance_texts} (Page {item['page']})")
    else:
        print("   No dimensions with ± tolerances found.")
    
    print(f"\n2) DIMENSIONS WITH UPPER/LOWER TOLERANCES ({len(upper_lower_tolerances_list)}):")
    print("-" * 60)
    if upper_lower_tolerances_list:
        for i, item in enumerate(upper_lower_tolerances_list, 1):
            # Parse upper and lower tolerances
            tolerances = item['tolerances']
            upper_tol = ""
            lower_tol = ""
            
            for tol in tolerances:
                tol_clean = tol.strip()
                if tol_clean.startswith('+'):
                    upper_tol = tol_clean
                elif tol_clean.startswith('-'):
                    lower_tol = tol_clean
                elif re.match(r'^\d+(?:[.,]\d+)?$', tol_clean):
                    # If it's just a number (regular tolerance), assign based on what's missing
                    if not upper_tol and not lower_tol:
                        # First number found - check if we have signed tolerances to determine assignment
                        has_plus = any(t.startswith('+') for t in tolerances)
                        has_minus = any(t.startswith('-') for t in tolerances)
                        
                        if has_plus and not has_minus:
                            # We have +signed, so regular tolerance becomes lower (keep as is, no sign)
                            lower_tol = tol_clean
                        elif has_minus and not has_plus:
                            # We have -signed, so regular tolerance becomes upper (keep as is, no sign)
                            upper_tol = tol_clean
                        else:
                            # Default: first number is upper (keep as is, no sign)
                            upper_tol = tol_clean
                    elif upper_tol and not lower_tol:
                        # We already have upper, this becomes lower (keep as is, no sign)
                        lower_tol = tol_clean
                    elif not upper_tol and lower_tol:
                        # We already have lower, this becomes upper (keep as is, no sign)
                        upper_tol = tol_clean
            
            # Format output with arrow for clarity
            if upper_tol and lower_tol:
                print(f"   {i}) {item['dimension']} → Upper: {upper_tol} Lower: {lower_tol} (Page {item['page']})")
            elif upper_tol:
                print(f"   {i}) {item['dimension']} → Upper: {upper_tol} (Page {item['page']})")
            elif lower_tol:
                print(f"   {i}) {item['dimension']} → Lower: {lower_tol} (Page {item['page']})")
            else:
                print(f"   {i}) {item['dimension']} (Page {item['page']})")
    else:
        print("   No dimensions with upper/lower tolerances found.")
    
    print(f"\n3) DIMENSIONS WITH ISO FIT TOLERANCES ({len(iso_fit_tolerances_list)}):")
    print("-" * 60)
    if iso_fit_tolerances_list:
        for i, item in enumerate(iso_fit_tolerances_list, 1):
            tolerance_texts = ", ".join(item['tolerances'])
            
            # Show complete notation with symbols and dimensions if available
            complete_iso_notations = item.get('complete_iso_notations', [])
            has_symbols_or_dims = any(notation != tol for notation, tol in zip(complete_iso_notations, item['tolerances']))
            
            if has_symbols_or_dims:
                # Use the complete notation as the main dimension display
                complete_iso_display = ", ".join(complete_iso_notations)
                print(f"   {i}) {complete_iso_display} → {tolerance_texts} (Page {item['page']})")
            else:
                print(f"   {i}) {item['dimension']} → {tolerance_texts} (Page {item['page']})")
            
            # Try to calculate actual tolerance values if isofits is available
            for j, tol_text in enumerate(item['tolerances']):
                # Use the complete ISO notation for tolerance calculation if available
                complete_notation = complete_iso_notations[j] if j < len(complete_iso_notations) else tol_text
                
                # Extract nominal size from complete notation (remove symbols, keep numbers)
                clean_dim = re.sub(r'[⌀⌵⌴↧▱⌓°]', '', complete_notation)
                
                # Look for number that appears before the ISO fit code
                # This handles cases like "16 g6", "10 H7", "M6-2x" 
                iso_fit_pattern = r'(\d+(?:[.,]\d+)?)'
                nominal_matches = re.findall(iso_fit_pattern, clean_dim)
                
                # Use the first/largest number found as nominal size
                nominal_size = None
                if nominal_matches:
                    # Convert to floats and pick the largest (handles cases like "M6-2x" where we want 6, not 2)
                    numbers = [float(m.replace(',', '.')) for m in nominal_matches]
                    nominal_size = max(numbers)
                
                if nominal_size and tol_text in ALL_SUPPORTED_ISO_FITS:
                    try:
                        tolerance_info = calculate_iso_fit_tolerance(nominal_size, tol_text)
                        
                        if tolerance_info:
                            print(f"      └─ {tol_text} ({tolerance_info['fit_type'].title()}): "
                                  f"{tolerance_info['upper_deviation_um']:+.0f}/{tolerance_info['lower_deviation_um']:+.0f} µm "
                                  f"→ {tolerance_info['tolerance_range']}")
                    except ValueError:
                        continue
                elif tol_text not in ALL_SUPPORTED_ISO_FITS:
                    print(f"      └─ {tol_text}: Pattern match only (not supported by isofits library)")
    else:
        print("   No dimensions with ISO fit tolerances found.")
    
    print(f"\n4) DIMENSIONS WITH GENERAL TOLERANCES ASSIGNED ({len(general_tolerances_list)}):")
    print("-" * 60)
    if general_tolerances_list:
        for i, item in enumerate(general_tolerances_list, 1):
            decimal_info = f"{item['decimal_places']} decimals" if item['decimal_places'] > 0 else "whole number"
            print(f"   {i}) {item['dimension']} → {item['tolerance']} ({decimal_info}) (Page {item['page']})")
    else:
        print("   No dimensions assigned general tolerances.")
    
    # Print summary
    total_dimensions = len(plus_minus_tolerances_list) + len(upper_lower_tolerances_list) + len(iso_fit_tolerances_list) + len(general_tolerances_list)
    print(f"\n" + "="*80)
    print("SUMMARY:")
    print(f"• Total dimensions processed: {total_dimensions}")
    print(f"• Dimensions with ± tolerances: {len(plus_minus_tolerances_list)}")
    print(f"• Dimensions with upper/lower tolerances: {len(upper_lower_tolerances_list)}")
    print(f"• Dimensions with ISO fit tolerances: {len(iso_fit_tolerances_list)}")
    print(f"• Dimensions assigned general tolerances: {len(general_tolerances_list)}")
    
    # Print unassigned tolerances if any
    if unassigned_tolerances or unassigned_signed_tolerances or unassigned_iso_fit_tolerances:
        print(f"\nUNASSIGNED TOLERANCES:")
        if unassigned_tolerances:
            print(f"• Unassigned regular tolerances: {len(unassigned_tolerances)}")
            for tol in unassigned_tolerances:
                print(f"  - '{tol['text']}' (Page {tol['page']})")
        if unassigned_signed_tolerances:
            print(f"• Unassigned signed tolerances: {len(unassigned_signed_tolerances)}")
            for tol in unassigned_signed_tolerances:
                print(f"  - '{tol['text']}' (Page {tol['page']})")
        if unassigned_iso_fit_tolerances:
            print(f"• Unassigned ISO fit tolerances: {len(unassigned_iso_fit_tolerances)}")
            for tol in unassigned_iso_fit_tolerances:
                extracted_fit = tol.get('extracted_fit', tol['text'])
                complete_iso_notation = tol.get('complete_iso_notation', extracted_fit)
                
                if complete_iso_notation != extracted_fit:
                    # Show the complete notation directly without "Complete:" prefix
                    print(f"  - '{complete_iso_notation}' (Page {tol['page']})")
                else:
                    print(f"  - '{extracted_fit}' (Page {tol['page']})")
    
    print("="*80)
    
    # Close the document
    doc.close()

def main():
    """Main function to run the tolerance analysis"""
    parser = argparse.ArgumentParser(
        description="Analyze PDF engineering drawings for tolerance and ISO fit detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tolerance_analyzer.py drawing.pdf
  python tolerance_analyzer.py "C:/path/to/drawing.pdf"
  python tolerance_analyzer.py /home/user/drawings/part.pdf
        """
    )
    
    parser.add_argument(
        'pdf_file',
        help='Path to the PDF file to analyze'
    )
    
    parser.add_argument(
        '--tolerance-font-ratio',
        type=float,
        default=0.7,
        help='Font size ratio for distinguishing main dimensions from tolerances (default: 0.7)'
    )
    
    parser.add_argument(
        '--max-distance',
        type=int,
        default=30,
        help='Maximum distance for associating tolerances with dimensions (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file not found at '{args.pdf_file}'")
        print("Please check the file path and try again.")
        return 1
    
    try:
        analyze_and_group_tolerances(args.pdf_file, args.tolerance_font_ratio, args.max_distance)
        return 0
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())