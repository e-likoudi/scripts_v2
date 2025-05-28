from typing import Dict, List, Set, Optional
import re

REPORT_TEMPLATE = """
STEM CELL PROTOCOL REPORT
=========================

PROTOCOL PHASE: {phase}
DETAIL LEVEL: {detail_level}

CELL LINE: {cell_line}

FOCUS AREAS:
{focus_areas}

TIMELINE:
{timeline}

REAGENTS:
{reagents}

MEDIA COMPOSITION:
{media}

CULTURE MATRICES:
{culture_matrices}

GENE MARKERS:
{gene_markers}

EFFICIENCY METRICS:
{efficiency_metrics}

CRITICAL STEPS:
{critical_steps}
"""

def format_list(items: Set[str]) -> str:
    """Format a set of items as a bulleted list"""
    return "\n".join(f"- {item}" for item in sorted(items)) if items else "Not specified"

def extract_cell_line(data: Dict) -> str:
    """Enhanced cell line detection with multiple validation layers"""
    # Known cell line patterns (expand as needed)
    CELL_LINE_PATTERNS = {
        # Standard formats
        r"(?:cell\s*line|clone):?\s*([A-Za-z0-9-]+)": 1,
        r"\b([A-Za-z]{2,3}\d+)\b": 0,  # e.g., H9, H1, RPE
        r"\b(iPSC|iPS-[A-Za-z0-9]+)\b": 0,  # iPSC lines
        r"\b(hESC|hES-\d+)\b": 0,  # Embryonic stem cells
        r"\b(WA\d+|H9|BGO1)\b": 0,  # Common lines
        # Manufacturer formats
        r"\b(ATCC [A-Z]{2,3}-[0-9]+)\b": 0,  # ATCC CRL-1234
        r"\b(CCL-\d+)\b": 0  # ATCC CCL series
    }

    # Priority sources to check (in order)
    SOURCES = [
        ("reagents", 0.9),  # Highest confidence
        ("timeline", 0.7),
        ("gene_markers", 0.6),
        ("media", 0.5)
    ]

    best_match = None
    highest_confidence = 0

    for source, base_confidence in SOURCES:
        for item in data.get(source, []):
            text = item.split("|")[0] if "|" in item else item
            
            # Skip empty or very short texts
            if len(text.strip()) < 3:
                continue
                
            for pattern, group_idx in CELL_LINE_PATTERNS.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    candidate = match.group(group_idx).strip()
                    current_confidence = base_confidence * (
                        1.0 if "cell line" in text.lower() else 0.8
                    )
                    
                    # Validate candidate
                    if is_valid_cell_line(candidate):
                        if current_confidence > highest_confidence:
                            best_match = candidate
                            highest_confidence = current_confidence

    return best_match or "Not specified"

def is_valid_cell_line(candidate: str) -> bool:
    """Validate detected cell line patterns"""
    # Basic checks
    if len(candidate) < 2:
        return False
    
    # Common false positives
    false_positives = {
        "medium", "media", "solution", "buffer", "day", "week", 
        "dmem", "rpmi", "pbs", "dapi", "edta"
    }
    
    if candidate.lower() in false_positives:
        return False
    
    # Structure validation
    has_letter = any(c.isalpha() for c in candidate)
    has_digit = any(c.isdigit() for c in candidate)
    
    # Valid formats:
    # 1. Starts with letters, ends with numbers (H9, WA01)
    # 2. Contains hyphen (ATCC-CRL-1234)
    # 3. Special cases (iPSC, hESC)
    return (
        (has_letter and has_digit) or
        ("-" in candidate) or
        candidate.upper() in {"IPSC", "HESC"}
    )

def format_timeline(data: Dict) -> str:
    """Format timeline steps based on phase"""
    steps = []
    for i, step in enumerate(data.get("timeline", []), 1):
        if "|" in step:  # Enhanced format
            day, details = map(str.strip, step.split("|", 1))
            steps.append(f"{i}. {day}: {details}")
        else:  # Simple format
            steps.append(f"{i}. {step}")
    return "\n".join(steps)

def format_reagents(data: Dict) -> str:
    """Format reagents with concentrations"""
    reagent_list = []
    for reagent in data.get("reagents", []):
        if "|" in reagent:
            name, concentration, *rest = map(str.strip, reagent.split("|"))
            reagent_list.append(f"- {name} ({concentration})")
        else:
            reagent_list.append(f"- {reagent}")
    return "\n".join(reagent_list)

def format_media(data: Dict) -> str:
    """Format media components"""
    media = set()
    for item in data.get("media", []):
        if "|" in item:
            media.add(item.split("|")[0].strip())
        else:
            media.add(item.strip())
    return format_list(media)

def format_culture_matrices(data: Dict) -> str:
    """Format culture matrices"""
    matrices = set()
    for item in data.get("culture_matrices", []):
        matrices.add(item.split("|")[0].strip() if "|" in item else item.strip())
    return format_list(matrices)

def format_gene_markers(data: Dict) -> str:
    """Format gene markers with expression data"""
    markers = []
    for item in data.get("gene_markers", []):
        if "|" in item:
            marker, expression = map(str.strip, item.split("|"))
            markers.append(f"- {marker}: {expression}")
        else:
            markers.append(f"- {item}")
    return "\n".join(markers)

def format_efficiency_metrics(data: Dict) -> str:
    """Format efficiency metrics"""
    metrics = []
    for item in data.get("efficiency_metrics", []):
        if "|" in item:
            metric, value = map(str.strip, item.split("|"))
            metrics.append(f"- {metric}: {value}")
        else:
            metrics.append(f"- {item}")
    return "\n".join(metrics)

def format_critical_steps(data: Dict) -> str:
    """Format critical steps with warnings"""
    steps = []
    for item in data.get("critical_steps", []):
        if "|" in item:
            step, warning = map(str.strip, item.split("|"))
            steps.append(f"- {step} ({warning})")
        else:
            steps.append(f"- {item}")
    return "\n".join(steps)

def format_protocol_report(
    extracted_data: Dict,
    cell_line: Optional[str] = None,
    phase: str = "unclassified",
    detail_level: str = "unclassified",
    focus_areas: List[str] = None
) -> str:
    """Format protocol data into final report using new classification system
    
    Args:
        extracted_data: Dictionary containing extracted protocol data
        cell_line: Identified cell line (from identify_cell_line)
        phase: Protocol phase (early/mid/late)
        detail_level: Detail level (low/medium/high)
        focus_areas: List of focus areas from classification
    """
    if focus_areas is None:
        focus_areas = ["unclassified"]
    
    return REPORT_TEMPLATE.format(
        phase=phase.capitalize(),
        detail_level=detail_level.capitalize(),
        cell_line = extract_cell_line(extracted_data),
        focus_areas=format_list(set(focus_areas)),
        timeline=format_timeline(extracted_data),
        reagents=format_reagents(extracted_data),
        media=format_media(extracted_data),
        culture_matrices=format_culture_matrices(extracted_data),
        gene_markers=format_gene_markers(extracted_data),
        efficiency_metrics=format_efficiency_metrics(extracted_data),
        critical_steps=format_critical_steps(extracted_data)
    )