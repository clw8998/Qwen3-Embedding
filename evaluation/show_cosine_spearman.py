import glob, json, os

dir_path = 'results/Qwen/Qwen3-Embedding-0.6B/Qwen__Qwen3-Embedding-0.6B/no_revision_available'
json_files = glob.glob(os.path.join(dir_path, '*.json'))

for fp in json_files:
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    task = data.get('task_name', os.path.basename(fp).replace('.json',''))
    for split, entries in data.get('scores', {}).items():
        for entry in entries:
            score = entry.get('cosine_spearman')
            langs = entry.get('languages', [])
            lang = langs[0] if langs else 'unknown'
            print(f"{task} | {split}({lang}) → cosine_spearman: {score}")