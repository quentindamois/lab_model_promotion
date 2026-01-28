import os
import json
import sys
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.80"))

import ml
# uselless update 2
import os, mlflow
from mlflow.tracking import MlflowClient

def main():
    # Accept JSON from stdin or argv for simplicity
    if not sys.stdin.isatty():
        payload = sys.stdin.read().strip()
    else:
        payload = sys.argv[1]
    data = json.loads(payload)
    acc = float(data["accuracy"])
    passed = acc >= ACCURACY_THRESHOLD
    result = {
        "passed": passed,
        "accuracy": acc,
        "threshold": ACCURACY_THRESHOLD,
        "model_version": data["model_version"],
        "run_id": data["run_id"],
    }
    print(json.dumps(result))
    if not passed:
        sys.exit(2)
    if passed:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        token = os.environ["MLFLOW_TRACKING_TOKEN"]
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        client = MlflowClient()
        name = os.environ["MODEL_NAME"]
        version = data["model_version"]
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage="Staging",
            archive_existing_versions=True
        )
        print(f"Promoted {name} v{version} to Staging")

if __name__ == "__main__":
    main()