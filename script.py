import os

# 1: Verify System requirements

# 2: Split dataset for training, cross validation and testing
os.system('python split_train_validation.py')

# 3: Create the layers
mydir_new = os.chdir('similarities')
os.system('python similarities/compute_all_similarities.py')

# 4: Run training of the model

# 5: Run test

# 6: Analyze predictions

# 7: Format data for evaluation

# 8: Print output CSV

mydir_new = os.chdir('../features')
os.system('python case_typo.py')
os.system('python linked_id_popularity.py')
os.system('python test_name_length.py')

# now create the dataframe for LightGBM
mydir_new = os.chdir('..')
os.system('python create_expanded_dataset.py')

# finally run LightGBM
os.system('python LightGBM_full.py')
os.system('sub_evaluation.py')