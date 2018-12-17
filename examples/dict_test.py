from segmenter.tools import make_dictionaries, load_dictionaries

if __name__ == '__main__':
    # make_dictionaries("../data/2014",
    #     #                   src_dict_path="../data/src_dict.json",
    #     #                   tgt_dict_path="../data/tgt_dict.json",
    #     #                   filters="\t\n",
    #     #                   oov_token="<UNK>",
    #     #                   min_freq=1)

    src_tokenizer, tgt_tokenizer = load_dictionaries("../data/src_dict.json",
                                                     "../data/tgt_dict.json")
    sequnces = src_tokenizer.texts_to_sequences([list(
        "叶依姆的家位于仓山区池后弄6号，属于烟台山历史风貌区，"
        "一家三代五口人挤在五六十平方米土木结构的公房里，屋顶逢雨必漏，居住环境不好。"
        "2013年11月，烟台山历史风貌区地块房屋征收工作启动，叶依姆的梦想正逐渐变为现实。")])
    print(src_tokenizer.sequences_to_texts(sequnces))

    sequnces = tgt_tokenizer.texts_to_sequences(['B-NR I-NR I-NR S-W B-V I-V B-V I-V B-N I-N S-W'])
    print(sequnces)
    print(tgt_tokenizer.sequences_to_texts(sequnces))
