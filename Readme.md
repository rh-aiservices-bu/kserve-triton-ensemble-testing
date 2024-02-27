# Ensemble Models Example with Triton and Kserve

Notes:

The ensemble.zip contains a folder which you can upload to object storage. The folder contains 3 models, ensemble_merger_google_xgb(ensembling logic), feature_merger(python backend) and the superset_google_xgb(xgboost – fil backend). The configuration file for all models in inside them so you can see the ensembling logic and how first one is dependent on second and third.

I have replaced the xgboost model with a dummy one as model binaries could have confidential data. So, there is a chance it can run into ‘wrong parameter shapes’ error but if you reach to that stage, it would be a good error. Hopefully it will serve the purpose for you to test the ensemble models.


The yaml you can use the same as any other model, just change the path to the directory where you have 3 models placed.



