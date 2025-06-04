def sorted_steps(steps_text):
    steps = {}
    
    for step in steps_text:
        step = step.strip()
        if not step or not step.startswith('Step '):
            continue
            
        # Extract step number (first digit after "Step ")
        step_num = int(step.split(':')[0].split('Step ')[1].strip())
        
        # Keep only the first occurrence of each step number
        if step_num not in steps:
            steps[step_num] = step
    
    # Sort by step number and join
    return '\n\n'.join(steps[num] for num in sorted(steps.keys()))