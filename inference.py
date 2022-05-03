import argparse as argparse

from wrench.dataset import TextDataset
from wrench.endmodel import EndClassifierModel

# MODEL_PATH = './saved_model_weibo_base.pkl'
id2label = {
    "0": "Positive",
    "1": "Negative"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    model_path = args.model_path
    data = {
        "text": args.text
    }
    dataset = TextDataset()
    dataset.ids = [0]
    dataset.labels = [0]
    dataset.examples = [data]
    dataset.id2label = id2label

    model = EndClassifierModel()
    model.load(model_path)
    pred = model.predict_proba(dataset)[0]
    print(pred[0], pred[1])