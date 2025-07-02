def process_stages(stage_list):
   
    stage_order = [
        'Undifferentiated cells',
        'Differentiation Process',
        'Differentiated cells',
        'No differentiation step'
    ]
    
    # Sort by the 'stage' key using the custom order
    sorted_stages = sorted(
        stage_list,
        key=lambda x: stage_order.index(x['stage']) if x['stage'] in stage_order else len(stage_order)
    )
    return sorted_stages