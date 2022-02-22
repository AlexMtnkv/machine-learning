from processing import DataInit


if __name__ == "__main__":
    di = DataInit()
    di.initialize_random_seed()
    di.read_target_variable()
    di.read_data()
    di.model_training()
    # print(di.target_df)
    # print(di._features)
    # print(di.target)