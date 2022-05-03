import logging
import torch
from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.endmodel import EndClassifierModel


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)
device = torch.device('cuda')

#### Load dataset
dataset_path = './dataset/'
data = 'weibo_chn'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=False
)

#### Run end model: BERT
model = EndClassifierModel(
    batch_size=32,
    real_batch_size=32,
    test_batch_size=256,
    n_steps=1000,
    backbone='BERT',
    backbone_model_name='hfl/chinese-roberta-wwm-ext',
    backbone_max_tokens=512,
    backbone_fine_tune_layers=-1,
    optimizer='AdamW',
    optimizer_lr=2e-5,
    optimizer_weight_decay=0.0,
)
model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='acc',
    patience=10,
    device=device
)
model.save('./saved_model.pkl')
acc = model.test(test_data, 'acc')
logger.info(f'end model (BERT) test acc: {acc}')

