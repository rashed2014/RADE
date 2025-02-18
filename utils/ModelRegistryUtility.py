import os
import mlflow
import mlflow.pyfunc
from mlflow.models import ModelSignature
from databricks.sdk import WorkspaceClient  # Databricks SDK for authentication
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from gliner import GLiNER
from colpali_engine.models import ColPali, ColPaliProcessor  # Correct import

class ModelRegistryUtility:
    """
    A utility class for downloading, saving, and registering models in MLflow on Databricks.
    """

    def __init__(self, tracking_uri: str = None):
        """Initialize Databricks MLflow authentication and setup tracking."""
        
        # Load Databricks credentials from environment variables
        self.databricks_host = os.getenv("DBX_ENDPOINT")
        self.databricks_token = os.getenv("DBX_TOKEN")

        if not self.databricks_host or not self.databricks_token:
            raise ValueError("Missing Databricks credentials. Set DATABRICKS_HOST and DATABRICKS_TOKEN.")

        # Authenticate with Databricks using the SDK
        self.databricks_client = WorkspaceClient(host=self.databricks_host, token=self.databricks_token)
        print(f"Authenticated with Databricks: {self.databricks_client.config.host}")

        # Set up MLflow Tracking URI
        if tracking_uri:
            self.tracking_uri = tracking_uri
        else:
            self.tracking_uri = "databricks"  # Uses the default Databricks MLflow tracking server

        mlflow.set_tracking_uri(self.tracking_uri)
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        os.environ["MLFLOW_TRACKING_TOKEN"] = self.databricks_token

        print(f"MLflow Tracking URI set to: {self.tracking_uri}")

    def register_colpali(self):
        """Download, save, and register the ColPali model in MLflow."""
        print("Downloading ColPali model...")
        colpali_model = ColPali.from_pretrained("vidore/colpali-v1.3")
        colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")

        # Save locally before registering
        local_path = "dbfs:/mlflow/models/ColPali"
        colpali_model.save_pretrained(local_path)
        colpali_processor.save_pretrained(local_path)

        print("Logging ColPali model in MLflow...")
        with mlflow.start_run():
            model_info = mlflow.transformers.save_model(
                transformers_model=colpali_model,
                path=local_path
            )
            model_uri = model_info.model_uri

        print("Registering ColPali model in MLflow Unity Catalog...")
        mlflow.register_model(
            model_uri=model_uri,
            name="dopdatabricks.ColPali"
        )
        print("ColPali model registered successfully!")

    def register_roberta_qa(self):
        """Download, save, and register Roberta Large model in MLflow."""
        print("Downloading Roberta QA model (Large)...")
        model_name = "deepset/roberta-large-squad2"
        qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Save locally
        local_path = "dbfs:/mlflow/models/RobertaLargeQA"
        qa_model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)

        print("Logging Roberta QA model in MLflow...")
        with mlflow.start_run():
            model_info = mlflow.transformers.save_model(
                transformers_model=qa_model,
                path=local_path
            )
            model_uri = model_info.model_uri

        print("Registering Roberta QA model in MLflow Unity Catalog...")
        mlflow.register_model(
            model_uri=model_uri,
            name="dopdatabricks.RobertaLargeQA"
        )
        print("Roberta QA model registered successfully!")

    def register_gliner(self):
        """Download, save, and register GLiNER in MLflow."""
        print("Downloading GLiNER model...")
        gliner_model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")

        # Save locally
        local_path = "dbfs:/mlflow/models/GLiNER"
        gliner_model.save_pretrained(local_path)

        print("Logging GLiNER model in MLflow...")
        with mlflow.start_run():
            model_info = mlflow.transformers.save_model(
                transformers_model=gliner_model,
                path=local_path
            )
            model_uri = model_info.model_uri

        print("Registering GLiNER model in MLflow Unity Catalog...")
        mlflow.register_model(
            model_uri=model_uri,
            name="dopdatabricks.GLiNER"
        )
        print("GLiNER model registered successfully!")

    def register_all_models(self):
        """Register all models in MLflow."""
        print("Registering all models in Unity Catalog...")
        self.register_colpali()
        self.register_roberta_qa()
        self.register_gliner()
        print("All models registered successfully in Unity Catalog!")

if __name__ == "__main__":
    registry = ModelRegistryUtility()
    registry.register_all_models()
