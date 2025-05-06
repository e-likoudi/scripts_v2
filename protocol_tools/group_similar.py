import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import re
from collections import defaultdict

def group_by_first_token(items, key):
    """Groups extracted data by the first part before '|' in formatted strings"""
    groups = defaultdict(list)
    for item in items:
        for entry in item.get(key, []):
            if "|" in entry:
                first_token = entry.split("|")[0].strip()
                groups[first_token].append(entry)
    return dict(groups)


def safe_grouping(extracted_data):
    """Organizes data by category with type-safe grouping"""
    grouped = {
        "factual": {
            "reagents": group_by_first_token(extracted_data, "reagents"),
            "timeline": sorted(
                [item for data in extracted_data for item in data.get("timeline", [])],
                key=lambda x: int(re.search(r'Day (\d+)', x).group(1)) if "Day" in x else 999
            )
        },
        "analytical": {
            "efficiency": [item for data in extracted_data for item in data.get("efficiency", [])]
        },
        "contextual": {
            "critical_steps": [item for data in extracted_data for item in data.get("critical_steps", [])]
        }
    }
    return grouped