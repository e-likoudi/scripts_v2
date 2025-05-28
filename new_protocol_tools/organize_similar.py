import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from collections import defaultdict

def group_by_category(summaries):
    """Groups classified summaries by their phase, detail level, and focus tags"""
    grouped = {
        "phases": defaultdict(list),
        "detail_levels": defaultdict(list),
        "focus_areas": defaultdict(list)
    }
    
    for summary in summaries:
        # Group by phase
        grouped["phases"][summary.get("phase", "unclassified")].append(summary)
        
        # Group by detail level
        grouped["detail_levels"][summary.get("detail_level", "unclassified")].append(summary)
        
        # Group by focus areas (each summary can have multiple)
        for focus_area in summary.get("focus", []):
            grouped["focus_areas"][focus_area].append(summary)
    
    return grouped


def organize_by_priority(summaries):
    """Organizes summaries by protocol phase priority (early → mid → late)"""
    phase_priority = ["early", "mid", "late"]
    
    # Create base structure
    organized = {
        phase: {
            "low": [],
            "medium": [],
            "high": []
        } for phase in phase_priority
    }
    
    # Populate the structure
    for summary in summaries:
        phase = summary.get("phase", "unclassified")
        detail = summary.get("detail_level", "unclassified")
        
        if phase in phase_priority and detail in ["low", "medium", "high"]:
            organized[phase][detail].append(summary)
    
    return organized