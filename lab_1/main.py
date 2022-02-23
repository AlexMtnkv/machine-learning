from processing import DataInit
from time import perf_counter

if __name__ == "__main__":
    t = perf_counter()
    di = DataInit()
    di.initialize_random_seed()
    di.read_target_variable()
    di.read_data()
    di.model_results()
    print("time: ", perf_counter() - t)