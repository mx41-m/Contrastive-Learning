# Contrastive-Learning
This repository includes the generationg codes "generate_data.py" for Moving MNIST and the training codes "pretrain_encoders.py" for encoders. 

# Dependencies
To run the codes, you need PyTorch (2.0.0)

# Running an experiment
To run the experiment on Toy Moving Mnist:
You need first generate the moving mnist dataset: "python generate_data.py --data_path MNIST_dataset_path --generate_data_path Moving_MNIST_save_path"
Then you could pretrain encoders: "python pretrain_encoders.py --generate_data_path Moving_MNIST_save_path --encoders_res encoders_results_generate_features_save_path"

