import matplotlib.pyplot as plt
import json

def plot_cv_metrics(json_path="models/cv_model_metrics.json"):
    # 假设你在训练后将 CrossValidator 的 avgMetrics 保存成了一个 JSON
    with open(json_path, "r") as f:
        data = json.load(f)
    metrics = data["avgMetrics"]

    plt.figure()
    plt.plot(metrics, marker='o')
    plt.title("Cross-Validation RMSE for Hyperparameter Grid")
    plt.xlabel("Parameter Set Index")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_cv_metrics()
