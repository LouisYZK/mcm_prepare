# Ch9:Embedding a Machine Learning Model into a Web Application
> 前面的ML模型都是在本地运行计算的，此chapter介绍如何将模型应用在Web app上以获得实时学习、应用

- Saving the current state of a trained machine learning model
- Using SQLite databases for data storage 
- Developing a web application using the popular **Flask** web framework
- Deploying a machine learning application to a public web server

## Serializing fitted scikit-learn estimators
> 这里应用的是 sentiment analysis 中的out_of_core.py中训练的线上模型为例