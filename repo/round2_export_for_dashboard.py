# --- Add this as a NEW CELL at the end of Round2_NLP_Notebook.ipynb, after cell 17 ---
# Dumps y_test/y_pred for both tasks so the dashboard can build confusion
# matrices + per-class F1 without re-running training.
import os
os.makedirs("outputs", exist_ok=True)

for task_name, (X_test, y_test, y_pred) in [("sentiment", sent_test), ("topic", topic_test)]:
    pd.DataFrame({"text": X_test, "true": y_test, "pred": y_pred}) \
      .to_csv(f"outputs/round2_{task_name}_predictions.csv", index=False)

print("Saved outputs/round2_sentiment_predictions.csv and outputs/round2_topic_predictions.csv")
