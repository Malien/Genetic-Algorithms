from matplotlib import pyplot as plt
import os
import polars
import matplotlib
from functools import partial
import multiprocessing as mp


matplotlib.use("Agg")


def process_population(path):
    table = polars.read_csv(os.path.join(path, "population.csv"))
    if table.null_count()["phenotype"][0] == 0:
        plt.hist(table["phenotype"], 64)
        plt.savefig(os.path.join(path, "phenotype_dist.png"))
        plt.clf()
    plt.hist(table["ones_count"], table["ones_count"].max() + 1)
    plt.savefig(os.path.join(path, "ones_count_dist.png"))
    plt.clf()
    plt.hist(table["fitness"], 64)
    plt.savefig(os.path.join(path, "fitness_dist.png"))
    plt.clf()


def config_search(path: str): 
    for entry in os.scandir(path):
        if entry.is_file() and entry.name == "runs.csv":
            yield path
            return
    for entry in os.scandir(path):
        if entry.is_dir():
            yield from config_search(entry.path)


def simple_plot(table: polars.DataFrame, path: str, column: str):
    plt.plot(table[column])
    plt.grid()
    plt.savefig(os.path.join(path, column + ".png"))
    plt.clf()


configs = list(config_search("data"))


def process_config(path: str):
    for i in range(1, 6):
        process_run(os.path.join(path, str(i)))


def process_run(path: str):
    for entry in os.scandir(path):
        if entry.is_dir():
            process_iteration(entry.path)
    table = polars.read_csv(os.path.join(path, "iterations.csv"), infer_schema_length=None)
    plot = partial(simple_plot, table, path)

    if "optimal_specimen_count" in table:
        plot("selection_intensity")
        plot("optimal_specimen_count")

        plot("growth_rate")
        plot("selection_diff")
        plt.plot(table["selection_diff"])
        plt.plot(table["selection_intensity"])
        plt.grid()
        plt.savefig(os.path.join(path, "selection.png"))
        plt.clf()

    plot("avg_fitness")
    plot("best_fitness")
    plot("fitness_std_dev")

    plt.plot(table["rr"])
    plt.plot(table["theta"])
    plt.grid()
    plt.savefig(os.path.join(path, "rr_and_theta.png"))
    plt.clf()


def process_iteration(path: str):
    table = polars.read_csv(os.path.join(path, "population.csv"))
    if table.null_count()["phenotype"][0] == 0:
        plt.hist(table["phenotype"], 64)
        plt.savefig(os.path.join(path, "phenotype_dist.png"))
        plt.clf()
    plt.hist(table["ones_count"], table["ones_count"].max())
    plt.savefig(os.path.join(path, "ones_count_dist.png"))
    plt.clf()
    plt.hist(table["fitness"], 64)
    plt.savefig(os.path.join(path, "fitness_dist.png"))
    plt.clf()


def process_fn(params):
    try:
        idx, config_path = params
        process_config(config_path)
        return (idx, config_path)
    except Exception as e:
        print(config_path, "errored out")
        print(e)
        print(e.__traceback__)
        raise e


if __name__ == '__main__':
    mp.freeze_support()
    with mp.Pool() as pool:
        res = pool.imap_unordered(process_fn, enumerate(config_search("data")))
        count = 0
        for idx, config_path in res:
            count += 1
            print(f"Processed #{idx} {count}/192 ({count * 100 // 192}%)\tConfig: {config_path}")

# for idx, config_path in enumerate(config_search("data")):
#     process_fn((idx, config_path))


# def recursive_csv_search(path: str):
#     for entry in os.scandir(path):
#         if entry.is_file() and entry.name in filetypes:
#             print("Yes, process a file @", entry.path)
#             filetypes[entry.name](path)
#         elif entry.is_dir():
#             recursive_csv_search(entry.path)


# recursive_csv_search("data/Pow2/binary/crossover/mutation/100/stochastic-tournament/with-replacement/1.0")
