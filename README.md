# transformer-word-segmenter

这是一个基于 [Universal Transformer (Encoder)](https://github.com/GlassyWing/keras-transformer) (https://arxiv.org/abs/1807.03819) + CRF 的序列标注模型，可以用于分词。

## 安装

使用 `setup.sh` 进行安装

## 使用

简单的使用工厂函数 `get_or_create` 获得模型

```python
from tf_segmenter import get_or_create, TFSegmenter

if __name__ == '__main__':
    segmenter: TFSegmenter = get_or_create("../data/default-config.json",
                                           src_dict_path="../data/src_dict.json",
                                           tgt_dict_path="../data/tgt_dict.json",
                                           weights_path="../models/weights.50--0.18.h5")
```

它接收4个参数:

- config: 指定模型使用的配置文件路径或配置字典
- src_dict_path: 指定文字字典
- tgt_dict_path: 指定标签字典
- weights_path: 使用的权重文件.

然后, 调用 `decode_texts` 切分语句：

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

它也可以识别人民(nr[f])、地名(ns)、组织机构名(nt[o])等，如 `印度尼西亚国家抗灾署`、`万丹省` 等。

配置, 权重和字典链接:

https://pan.baidu.com/s/1iHADmnSEywoVqq_-nb0bOA password: v34g

## 数据集处理

数据集: https://pan.baidu.com/s/1EtXdhPR0lGF8c7tT8epn6Q 验证码: yj9j

### 转换数据集格式

如下的数据集格式**不是**我们所需要的：

> 嫌疑人\n 赵国军\nr 。\w

通过如下命令转换格式:

```python
python ner_data_preprocess.py <src_dir> 2014_processed -c True -s True
```

 `<src_dir>` 指定训练集路径, 如 `./2014-people/train`.

现在`2014_processed`中的数据类似如下：

> 嫌 疑 人 赵 国 军 。    B-N I-N I-N B-NR I-NR I-NR S-W

### 制作字典

数据格式转换后, 制作字典:

```python
python tools/make_dicts.py 2014_processed -s src_dict.json -t tgt_dict.json
```

这会生成如下两个文件:

- src_dict.json
- tgt_dict.json

### 转为 hdf5 

为了加速训练，将纯文本 `2014_processed` 转换为 hdf5 文件.

```python
python tools/convert_to_h5.py 2014_processed 2014_processed.h5 -s src_dict.json -t tgt_dict.json
```

## 训练效果

使用的配置:

```json
{
    "src_vocab_size": 5649,
    "tgt_vocab_size": 301,
    "max_seq_len": 150,
    "max_depth": 2,
    "model_dim": 256,
    "embedding_size_word": 300,
    "embedding_dropout": 0.0,
    "residual_dropout": 0.1,
    "attention_dropout": 0.1,
    "output_dropout": 0.0,
    "l2_reg_penalty": 1e-6,
    "confidence_penalty_weight": 0.1,
    "compression_window_size": None,
    "num_heads": 2,
    "use_crf": True
}
```

其它参数:

| 参数             | 值   |
| ---------------- | ---- |
| batch_size       | 32   |
| steps_per_epoch  | 2000 |
| validation_steps | 50   |
| warmup           | 6000 |

训练集占比0.975.

参考: `examples\train_example.py`

50次迭代后, 验证集精度达到 98 %, 随后精确度几乎不再增长。收敛时长与BiLSTM+CRF几乎一样，但参数数量减少了约20万：

<div>
    <img src="assets/accuracy.png">
    <img src="assets/loss.png">
</div>

以词为单位进行测试集(`2014-people/test`)评估:

```python
result-(epoch:50):
Num of words：20744, accuracy rate：0.958639, error rate：0.046712
Num of lines：317, accuracy rate：0.406940, error rate：0.593060
Recall: 0.958639
Precision: 0.953536
F MEASURE: 0.956081
ERR RATE: 0.046712
====================================
```

## 参考

1. Universal Transformer [https://github.com/GlassyWing/keras-transformer](https://github.com/GlassyWing/keras-transformer)
2. Transformer [https://github.com/GlassyWing/transformer-keras](https://github.com/GlassyWing/transformer-keras)
