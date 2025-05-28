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

def detect_critical_components(text):
    """Identify critical components in protocol text"""
    text_lower = text.lower()
    return {
        category: [kw for kw in keywords if kw in text_lower]
        for category, keywords in CRITICAL_COMPONENTS.items()
    }

def check_component_coverage(components_info):
    """Check which critical components are present in the data"""
    coverage = {category: False for category in CRITICAL_COMPONENTS.keys()}
    
    for info in components_info:
        if isinstance(info, dict) and 'components' in info:
            for category in coverage.keys():
                if info['components'].get(category):
                    coverage[category] = True
                    
    return coverage

def generate_missing_components_report(report, components_info):
    """Add missing critical components section to report"""
    coverage = check_component_coverage(components_info)
    missing_sections = []
    
    for category, is_present in coverage.items():
        if not is_present:
            # Find first example in original data
            for info in components_info:
                if isinstance(info, dict) and 'components' in info:
                    if info['components'].get(category):
                        example = info['components'][category][0]
                        missing_sections.append(
                            f"\n- {category.replace('_', ' ').title()}: "
                            f"{example} (see Step {info.get('id', 'N/A')})"
                        )
                        break
    
    if missing_sections:
        report += "\n\n=== MISSING CRITICAL COMPONENTS ===\n" + "\n".join(missing_sections)
    
    return report