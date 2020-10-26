# Recommender-System-for-OHC-CLIR-CLLIR

Source code for the paper
> Li M, Gao W, Chen Y. A Topic and Concept Integrated Model for Thread Recommendation in Online Health Communities[C]//Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2020: 765-774.

# Folders:
- final_code: the code for CLIR and CLLIR
- data: the datasets involved in the paper, including raw data and experiment data
- model: the saved models for experiments

# Instruction:
- generate data: run `get_post.py` to generate experiment dataset from raw data. Skip this step if you use experiment data directly.
- run model: set path for the dataset/model folder in `thread_recommender_system.py` and `thread_recommender_system_with_interest_shifting.py`, then run `thread_recommendation_model_training_script.py` to train and test. If you want to load pre-trained model, set `load_model=True`, otherwise keep `load_model=False`.



