import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from collections import defaultdict

def format_report(grouped_data):
    """Converts grouped data into final report using MAIN_PROMPT template"""

    MAIN_PROMPT = """
    You are an expert research assistant synthesizing a stem cell protocol report.

    === REPORT STRUCTURE ===
    CELL LINE DETAILS:
    {cell_lines}

    DIFFERENTIATION STEPS:
    {timeline_steps}

    BASAL MEDIA:
    {basal_media}

    SERUMS & SUPPLEMENTS:
    {serums_supplements}

    GROWTH FACTORS:
    {growth_factors}

    CULTURE MATRICES:
    {culture_matrices}

    GENE MARKERS:
    {gene_markers}

    === FORMATTING RULES ===
    1. Use "-" for lists in CELL LINE DETAILS, BASAL MEDIA, etc.
    2. Use "1., 2., ..." numbering for DIFFERENTIATION STEPS
    3. For growth factors with multiple concentrations: "GF_NAME (X ng/mL, Y ng/mL)"
    4. Always maintain original units (ng/mL, %, etc.)
    5. If data is missing/unclear: "Not specified"
    """

    # --- Prepare Data Sections ---
    
    # CELL LINE DETAILS
    cell_lines = set()
    for reagent in grouped_data["factual"]["reagents"].keys():
        base_name = reagent.split('|')[0].strip()
        if base_name and base_name != "Not specified":
            cell_lines.add(base_name)
    cell_lines_str = "\n".join(f"- {line}" for line in sorted(cell_lines)) if cell_lines else "Not specified"
    
    # DIFFERENTIATION STEPS
    steps = []
    for i, step in enumerate(grouped_data["factual"]["timeline"], 1):
        if "|" in step:  # Enhanced format
            day, duration, action = map(str.strip, step.split("|"))
            steps.append(f"{i}. {day}: {duration} - {action}")
        else:  # Simple format
            steps.append(f"{i}. {step}")
    steps_str = "\n".join(steps) if steps else "Not specified"
    
    # BASAL MEDIA
    media = set()
    for entries in grouped_data["factual"]["reagents"].values():
        for entry in entries:
            if "DMEM" in entry or "RPMI" in entry:  # Media detection
                media.add(entry.split("|")[0].strip())
    media_str = "\n".join(f"- {m}" for m in sorted(media)) if media else "Not specified"
    
    # SERUMS & SUPPLEMENTS
    supplements = set()
    for entries in grouped_data["factual"]["reagents"].values():
        for entry in entries:
            if "FBS" in entry or "Serum" in entry:
                supplements.add(entry.split("|")[0].strip())
    supplements_str = "\n".join(f"- {s}" for s in sorted(supplements)) if supplements else "Not specified"
    
    # GROWTH FACTORS
    growth_factors = defaultdict(list)
    for reagent, entries in grouped_data["factual"]["reagents"].items():
        for entry in entries:
            if "ng/mL" in entry:
                name = entry.split("|")[0].strip()
                conc = entry.split("|")[1].strip()
                growth_factors[name].append(conc)
    gf_str = "\n".join(
        f"- {name} ({', '.join(concs)})"
        for name, concs in growth_factors.items()
    ) if growth_factors else "Not specified"
    
    # CULTURE MATRICES
    matrices = set()
    for step in grouped_data["factual"]["timeline"]:
        if "Matrigel" in step or "gel" in step.lower():
            matrices.add(step.split(":")[-1].strip())
    matrices_str = "\n".join(f"- {m}" for m in matrices) if matrices else "Not specified"
    
    # GENE MARKERS
    markers = set()
    for entry in grouped_data["analytical"]["efficiency"]:
        if "CD" in entry or "OCT" in entry:
            markers.add(entry.split(":")[-1].split("|")[0].strip())
    markers_str = "\n".join(f"- {marker}" for marker in sorted(markers)) if markers else "Not specified"
    
    # --- Apply to MAIN_PROMPT ---
    return MAIN_PROMPT.format(
        cell_lines=cell_lines_str,
        timeline_steps=steps_str,
        basal_media=media_str,
        serums_supplements=supplements_str,
        growth_factors=gf_str,
        culture_matrices=matrices_str,
        gene_markers=markers_str
    )