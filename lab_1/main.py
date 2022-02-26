from processing import DataInit

if __name__ == "__main__":
    di = DataInit()
    di.initialize_random_seed()
    di.read_target_variable()
    di.read_data()
    di.training_SGD(normal=False)
    di.training_SGD(normal=True)
    di.training_RF()