from segmenter.data_loader import DataLoader

if __name__ == '__main__':
    data_loader = DataLoader("../data/src_dict.json", "../data/tgt_dict.json",
                             batch_size=1,
                             max_len=300,
                             sparse_target=False)

    generator = data_loader.generator("../data/2014/valid")
    for _ in range(1):
        data = next(generator)
        sent, tag = data
        print(sent)
        print(tag)