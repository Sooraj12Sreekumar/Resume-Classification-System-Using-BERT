from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import accuracy_score
from model_BERT import config, utils

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

def main():
    X_train, y_train = utils.load_data(config.TRAIN_FILE, fit_label_encoder=True)
    X_val, y_val = utils.load_data(config.VAL_FILE)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(utils.label_encoder.classes_)
    )
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    training_args = TrainingArguments(
        output_dir=config.MODEL_SAVE_PATH,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  
        greater_is_better=True,
        logging_dir=config.LOG_DIR,
        logging_steps=100,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=utils.ResumeDataset(X_train, y_train),
        eval_dataset=utils.ResumeDataset(X_val, y_val),
        compute_metrics=compute_metrics, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)]
    )

    try:
        print("Starting training...")
        trainer.train()
        trainer.save_model(config.MODEL_SAVE_PATH)
        print(f"Model saved to {config.MODEL_SAVE_PATH}")
    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()