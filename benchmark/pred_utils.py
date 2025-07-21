def convertTarDep2TrueDep(data_item):
    if 'true_target_dependency' in data_item:
        data_item['target_dependency'] = data_item['true_target_dependency']
    elif 'true_dependency' in data_item:
        data_item['dependency'] = data_item['true_dependency']
    return data_item