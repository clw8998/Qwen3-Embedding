from mteb import MTEB
from sentence_transformers import SentenceTransformer, models

import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
os.environ["HF_TRUST_REMOTE_CODE"] = "true"

import mteb
tasks = mteb.get_benchmark("MTEB(cmn, v1)").tasks

t = [
    "STS22",
    "ATEC",
    "BQ",
    "LCQMC",
    "PAWSX",
    "STSB",
    "AFQMC",
    "QBQTC",
]

target_tasks = []

for task in tasks:
    if task.metadata.name in t:
        target_tasks.append(task)

model = SentenceTransformer("clw8998/ABRSS", trust_remote_code=True)
model.add_module("normalize", models.Normalize())

evaluation = MTEB(tasks=target_tasks)
results = evaluation.run(model, output_folder="results", datasets_kwargs={"trust_remote_code": True},
      encode_kwargs={
        "batch_size": 8,
        "max_length": 8192,
        "truncation": True
    })

for result in results:
    name = result.task_name
    try:
        cosine_spearman = result.scores['test'][0]['cosine_spearman']
    except:
        cosine_spearman = result.scores['validation'][0]['cosine_spearman']

    rounded = round(cosine_spearman * 100, 3)
    print(f"{name}: {rounded}")