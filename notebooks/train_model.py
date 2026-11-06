
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# A MODIFIER avec ton catalogue et ton schema Databricks
MODEL_NAME = "catalog.schema.IrisRFModel"


def train_and_evaluate(random_state=42):

    # 1. Charger les donnees Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2. Diviser les donnees
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    # 3. Entrainer le modele
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # 4. Evaluer le modele
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, X_test[:5]


def main():

    model, accuracy, input_example = train_and_evaluate()

    # 5. Demarrer une experience MLflow
    with mlflow.start_run() as run:

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)

        # 6. Utiliser Unity Catalog
        mlflow.set_registry_uri("databricks-uc")

        # 7. Enregistrer le modele et sa version
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            registered_model_name=MODEL_NAME,
            input_example=input_example
        )

        print("Accuracy :", accuracy)
        print("Run ID :", run.info.run_id)
        print("Modele enregistre :", MODEL_NAME)


if __name__ == "__main__":
    main()
