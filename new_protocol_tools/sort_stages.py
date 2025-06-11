def process_stages(stage_dict):
   
    stage_order = [
        'Undifferentiated cells',
        'Differentiation Process',
        'Differentiated cells'
    ]
    # Sort by the 'stage' key using the custom order
    sorted_stages = sorted(
        stage_dict,
        key=lambda x: stage_order.index(x['stage']) if x['stage'] in stage_order else len(stage_order)
    )
    return sorted_stages