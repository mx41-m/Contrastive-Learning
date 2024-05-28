# Contrastive Learning for Clinical Outcome Prediction with Partial Data Sources
This repository contains the code for the paper [Contrastive Learning for Clinical Outcome Prediction with Partial Data Sources](https://openreview.net/forum?id=elCOPIm4Xw&invitationId=ICML.cc/2024/Conference/Submission5763/-/Camera_Ready_Revision&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2024%2FConference%2FAuthors%23author-tasks)). The 'generate_data.py' script is used to generate data for Moving MNIST, and the 'pretrain_encoders.py' script is used to train the encoders.

# Dependencies
To run the codes, you need PyTorch (2.0.0)

# Running an experiment
To run the experiment on Toy Moving Mnist:
You need first generate the moving mnist dataset: "python generate_data.py --data_path MNIST_dataset_path --generate_data_path Moving_MNIST_save_path"
Then you could pretrain encoders: "python pretrain_encoders.py --generate_data_path Moving_MNIST_save_path --encoders_res encoders_results_generate_features_save_path"
