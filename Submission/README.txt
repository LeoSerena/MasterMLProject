# TEAM DISPATCHED: Nicolas Brandt, Leo Serena, Rodrigo Granja

## DESCRIPTION

    This folder contains the result file of the EPFL Machine Learning Higgs 2019 contest.
    Website: https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019.
    It contains the following files:
        
        - run.py:
            script used to generate the submitted file of team dispatched
            
        - implementations.py:
            contains all the methods used in the run.py script and some mandatory basic machine learning functions
            
        - proj1_helpers.py:
            contains methods regarding data import and csv generation
            
        - MLproj1
            report giving all detail informations on the results
        
## USAGE
    
    Assuming python is installed, install numpy (if not already installed):
        
        -pip install numpy-
        
        or
        
        -conda install numpy-
        
    depending on your python setting, and make sure datasets, implementations.py and proj1_helpers.py are in the same folder as run.py,
    then run:
        
        -python run.py-
        
    on a command prompt. It will generate the submission.csv. It took us 40-60 min approximately to generate the submission file.
        
## RESULTS

    We coded a fully connected neural network to train our dataset and used regularized logistic regression to test out our ideas on data 
    expansion. We achieved a categorical accuracy of 0.844 and a f1 score of 0.764 on the website. All the details are in the report.
    
