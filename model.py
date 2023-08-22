from transformers import pipeline
import mlflow

class Model:

    def get_experiment_id(self, name):
        exp = mlflow.get_experiment_by_name(name)
        print(exp)
        if exp is None:
            exp_id = mlflow.create_experiment(name)
            return exp_id
        return exp.experiment_id

    def sentimentAnalysis(self, text: str, expName: str):
        exp_id = self.get_experiment_id(expName)

        sentiment_task = pipeline("sentiment-analysis", model='./model', tokenizer='./model')

        # text = "Covid cases are increasing fast!"
        result = sentiment_task(text)
        print(result)

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.transformers.log_model(
                transformers_model= sentiment_task,
                artifact_path= "my_model",
                input_example= text,
                output= result
            )
            mlflow.log_param("result", result)

        mlflow.end_run()
        return result
