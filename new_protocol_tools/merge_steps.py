def merge_similar_steps(steps_text):
    merged = {}
    
    for step in steps_text.split('\n\n'):
        step = step.strip()
            
        try:
            # Split step into lines
            step_lines = step.split('\n')
            
            # Process header line (e.g., "Step 0: Name")
            header = step_lines[0]
            step_num = int(header.split(':')[0].replace('Step ', '').strip())
            name = header.split(':')[1].strip()
            
            # Get description (all lines after header and duration)
            desc_lines = step_lines[2:] if len(step_lines) > 2 else ['']
            description = '\n'.join(desc_lines).strip()
            
            # Keep first occurrence of each step number
            if step_num not in merged:
                merged[step_num] = {
                    'step_num': step_num,
                    'name': name,
                    'description': description
                }
        except (IndexError, ValueError):
            continue

    return [merged[k] for k in sorted(merged.keys())]
