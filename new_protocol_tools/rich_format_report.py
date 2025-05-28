from typing import Dict
from datetime import datetime

def generate_data_rich_report(extracted_data: Dict, classification: Dict) -> str:
    """Generates a report populated with actual extracted protocol data"""
    
    # Header with timestamp (using forward slashes)
    report = f"""
STEM CELL PROTOCOL REPORT
=========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Phase: {classification.get('phase', 'unclassified').upper()}
Detail Level: {classification.get('detail_level', 'unclassified').title()}
"""

    # Timeline Section
    if extracted_data.get('timeline'):
        report += """
TIMELINE
--------""" + "\n".join(f"• Day {i+1}: {step}" 
                      for i, step in enumerate(extracted_data['timeline']))
    
    # Reagents Section
    if extracted_data.get('reagents'):
        report += """
        
REAGENTS & CONCENTRATIONS
-------------------------""" + "\n".join(
            f"• {name.strip()}: {conc.strip()}" if "|" in reagent else f"• {reagent}"
            for reagent in extracted_data['reagents']
            for name, conc in [reagent.split("|")[:2]] if "|" in reagent
        )
    
    return report + "\n\nEND OF REPORT\n============="