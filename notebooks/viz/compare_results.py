import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

wandb.login()

# highlight-start
run = wandb.init(
    # Set the project where this run will be logged
    project="aneurysm-logging",
    id="res",
    name="res",
    reinit=True,
)

path_results = "./results"
datasets = ["internal_test", "external"]

models_dict_path = "./notebooks/viz/model_dict.json"
with open(models_dict_path) as f:
    models_dict = json.load(f)

models = list(models_dict.keys())
names = list(models_dict.values())
iou = "0.2"
list_fields = []
for dataset in datasets:
    for model in models:
        path_base = Path(path_results) / dataset / model
        # get all subfolders matching pattern
        results = sorted(list(path_base.glob(f"iou{iou}_*")))
        for result in results:
            iterations = result.name.split("_")[-1]
            if iterations == "final":
                iterations = "final"
            else:
                iterations = int(iterations.replace("k", "")) * 1000
            csv_file = result / "froc.csv"
            df = pd.read_csv(csv_file)
            list_fields.append([dataset, model, iterations, df])


for j, dataset in enumerate(datasets):
    examples_dataset = [l for l in list_fields if l[0] == dataset]
    for i, model in enumerate(models):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        examples = [l for l in examples_dataset if l[1] == model]
        if len(examples) == 0:
            continue
        for example in examples:
            x = [0.1, 0.2, 0.5, 1.0, 2.0]
            y = example[-1].iloc[0, 1:].values[0:5]
            ax.plot(x, y, label=str(example[2]) + " it")
            ax.set_title(dataset + ": " + names[i])
            ax.set_xlabel("FPs per image")
            ax.set_ylabel("Sensitivity")
            ax.legend()
            ax.set_ylim(0.5, 1)
            # grid
            # more frequent y ticks
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
            dataset = example[0]
            ax.grid(True)
        plt.savefig(f"./results/{dataset}/{model}/{dataset}_{model}.png")
        image = wandb.Image(f"./results/{dataset}/{model}/{dataset}_{model}.png")
        run.log({f"{dataset}_{model}": image})
        print(dataset, model)
        plt.show()
