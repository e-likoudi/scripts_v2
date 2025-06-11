def merge_similar_steps(stage_dict):
    merged = {}  
    order_keys = []   
    
    for s in stage_dict:
        stage = s.get('stage', '')
        reason = s.get('reason', '')
        specific = s.get('specific_step', '')

        key = (stage, specific)

        if key not in merged:
            order_keys.append(key)
            merged[key] = {
                'stage': stage,
                'specific_step': specific,
                'reasons': [reason]
            }
        elif reason not in merged[key]['reasons']:
            merged[key]['reasons'].append(reason)

    stage_order = [
        'Undifferentiated cells',
        'Differentiation Process',
        'Differentiated cells'
    ]

    merged_list = [
        {
            'stage': merged[key]['stage'],
            'specific_step': merged[key]['specific_step'],
            'reason': ' '.join(merged[key]['reasons'])
        }
        for key in order_keys
    ]

    # Sort using the custom order
    merged_list_sorted = sorted(
        merged_list,
        key=lambda x: stage_order.index(x['stage']) if x['stage'] in stage_order else len(stage_order)
    )

    return merged_list_sorted