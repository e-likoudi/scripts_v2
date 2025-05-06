# Add to classify_protocol.py
CRITICAL_COMPONENTS = {
    'serums_supplements': [
        'fetal bovine serum', 'knockout serum replacement', 'b27', 'n2',
        'glutamax', 'penicillin-streptomycin', 'serum-free', 'kosp'
    ],
    'growth_factors': [
        'bmp4', 'activin a', 'fgf2', 'vegf', 'egf', 'wnt', 'tgf-β', 'noggin',
        'shh', 'bfgf', 'pdgf', 'igf'
    ],
    'culture_matrices': [
        'matrigel', 'gelatin', 'laminin', 'fibronectin', 'collagen', 'poly-l-ornithine',
        'vitronectin', 'ecm'
    ],
    'gene_markers': [
        'oct4', 'sox2', 'nanog', 'pax6', 'sox1', 'nestin', 'brachyury', 'foxa2',
        'ssea4', 'tra-1-60', 'cdx2', 'sox17'
    ]
}

# In critical_components.py
def contains_critical_components(text: str) -> dict:
    """Returns a structured dictionary of detected components"""
    text_lower = text.lower()
    return {
        'serums_supplements': [
            kw for kw in CRITICAL_COMPONENTS['serums_supplements'] 
            if kw in text_lower
        ],
        'growth_factors': [
            kw for kw in CRITICAL_COMPONENTS['growth_factors'] 
            if kw in text_lower
        ],
        'culture_matrices': [
            kw for kw in CRITICAL_COMPONENTS['culture_matrices'] 
            if kw in text_lower
        ],
        'gene_markers': [
            kw for kw in CRITICAL_COMPONENTS['gene_markers'] 
            if kw in text_lower
        ]
    }

def ensure_critical_components(report: str, components_info: list) -> str:
    """Add missing critical components to the report"""
    # Initialize coverage tracking
    coverage = {
        'serums_supplements': False,
        'growth_factors': False,
        'culture_matrices': False,
        'gene_markers': False
    }
    
    # Check what's already in the report
    report_lower = report.lower()
    for component in coverage.keys():
        coverage[component] = any(
            kw in report_lower 
            for kw in CRITICAL_COMPONENTS[component]
        )
    
    # Prepare missing components section
    missing_sections = []
    
    for component, is_covered in coverage.items():
        if not is_covered:
            # Find the first occurrence in original data
            for info in components_info:
                if isinstance(info, dict) and 'components' in info:
                    component_list = info['components'].get(component, [])
                    if component_list:  # If we found any
                        example = component_list[0]  # Take first example
                        missing_sections.append(
                            f"\n\n{component.replace('_', ' ').title()}:\n"
                            f"Protocol uses {example} (reference: step {info.get('id', 'N/A')})"
                        )
                        break
    
    if missing_sections:
        report += "\n\n=== CRITICAL COMPONENTS ===" + "".join(missing_sections)
    
    return report