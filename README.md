# transformer-word-segmenter

This is a sequence labelling model base on [Universal Transformer (Encoder)](https://arxiv.org/abs/1807.03819) + Bi-LSTM + CRF which can be used for Chinese word segmentation. The role Universal Transformer (Encoder) plays similar to Elmo Embedding.

## Install

Just use `setup.sh` to install.

## Usage

You can simplely use factory method `get_or_create` to get model.

```python
from tf_segmenter import get_or_create, TFSegmenter

if __name__ == '__main__':
    segmenter: TFSegmenter = get_or_create("../data/default-config.json",
                                           src_dict_path="../data/src_dict.json",
                                           tgt_dict_path="../data/tgt_dict.json",
                                           weights_path="../models/weights.129-0.00.h5")
```

It accepts four params:

- config: which indicates the configuration used by the model
- src_dict_path: which indicates the dictionary file for texts.
- tgt_dict_path: which indicates the dictionary file for tags.
- weights_path: weights file model used.

And then, call `decode_texts` to cut setences.

```python
texts = [

        "巴纳德星的名字起源于一百多年前一位名叫爱德华·爱默生·巴纳德的天文学家。"
        "他发现有一颗星在夜空中划过的速度很快，这引起了他极大的注意。"
        ,
        "印度尼西亚国家抗灾署此前发布消息证实，印尼巽他海峡附近的万丹省当地时间22号晚遭海啸袭击。"
    ]

for sent, tag in segmenter.decode_texts(texts):
    print(sent)
    print(tag)
```

Results:

```python
['巴纳德', '星', '的', '名字', '起源于', '一百', '多年前', '一位', '名叫', '爱德华·爱默生·巴纳德', '的', '天文学家', '。', '他', '发现', '有', '一颗', '星', '在', '夜空', '中', '划过', '的', '速度', '很快', '，', '这', '引起', '了', '他', '极大', '的', '注意', '。']
['nrf', 'n', 'ude1', 'n', 'v', 'm', 'd', 'mq', 'v', 'nrf', 'ude1', 'nnd', 'w', 'rr', 'v', 'vyou', 'mq', 'n', 'p', 'n', 'f', 'v', 'ude1', 'n', 'd', 'w', 'rzv', 'v', 'ule', 'rr', 'a', 'ude1', 'vn', 'w']

['印度尼西亚国家抗灾署', '此前', '发布', '消息', '证实', '，', '印尼巽他海峡', '附近', '的', '万丹省', '当地时间', '22号', '晚', '遭', '海啸', '袭击', '。']
['nt', 't', 'v', 'n', 'v', 'w', 'ns', 'f', 'ude1', 'ns', 'nz', 'mq', 'tg', 'v', 'n', 'vn', 'w']


```

It can also identify PEOPLE, ORG or PLACE such as `印度尼西亚国家抗灾署`、`万丹省` and so on.

config, weigts and dictionaries link:

https://pan.baidu.com/s/1iHADmnSEywoVqq_-nb0bOA password: v34g

## Dataset Process

baidu: https://pan.baidu.com/s/1EtXdhPR0lGF8c7tT8epn6Q password: yj9j

### Convert dataset format

The data format in dataset as follow is not what we liked.

> 嫌疑人\n 赵国军\nr 。\w

We convert it by command:

```python
python ner_data_preprocess.py <src_dir> 2014_processed -c True
```

Where `<src_dir>` indicates training dataset dir, such as `./2014-people/train`.

Now, the data in file `2014_processed` can be seen as follow:




> 嫌 疑 人 赵 国 军 。    B-N I-N I-N B-NR I-NR I-NR S-W

### Make dictionaries

After data format converted, we expect to make dictionaries:

```python
python tools/make_dicts.py 2014_processed -s src_dict.json -t tgt_dict.json
```

This will generate two file:

- src_dict.json
- tgt_dict.json

### Convert to hdf5 (Optional)

In order to speed up performance, you can convert pure txt `2014_processed` to hdf5 file.

```python
python tools/convert_to_h5.py 2014_processed 2014_processed.h5 -s src_dict.json -t tgt_dict.json
```

## Training Result

The config used as follow:

```json
{
  "src_vocab_size": 6864,
  "tgt_vocab_size": 259,
  "max_seq_len": 150,
  "max_depth": 2,
  "model_dim": 128,
  "lstm_units": 128,
  "confidence_penalty_weight": 0.1,
  "l2_reg_penalty": 1e-06,
  "embedding_dropout": 0.0,
  "residual_dropout": 0.0,
  "attention_dropout": 0.1,
  "compression_window_size": null,
  "num_heads": 8,
  "use_crf": true,
  "label_smooth": true
}
```

And with:

```python
batch_size = 32
steps_per_epoch = 2000
validation_steps = 20
```

The training data is divided into training set and verification set according to the ratio of 8:2.

see more: `examples\train_example.py`

After 32 epochs, the accuracy of the verification set reached 98 %, and the model parameters are only half of the Bi-LSTM+CRF model with the same effect.

<div>
    <img src="assets/accuracy.png">
    <img src="assets/loss.png">
</div>

Test set (`2014-people/test`) evaluation results for word segmetion:

```python
result-(epoch:32):
Num of words：20744，accuracy rate：0.948419，error rate：0.049364
Num of lines：317，accuracy rate：0.378549，error rate：0.621451
Recall: 0.948419
Precision: 0.950527
F MEASURE: 0.949472
ERR RATE: 0.049364
====================================
result-(epoch:35):
Num of words：20744, accuracy rate：0.951600, error rate：0.048882
Num of lines：317, accuracy rate：0.425868, error rate：0.574132
Recall: 0.951600
Precision: 0.951142
F MEASURE: 0.951371
ERR RATE: 0.048882
====================================
```