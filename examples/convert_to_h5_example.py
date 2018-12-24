from segmenter.data_loader import DataLoader

if __name__ == '__main__':
    data_loader = DataLoader("../data/src_dict.json", "../data/tgt_dict.json",
                             batch_size=32,
                             max_len=256,
                             sparse_target=False)

    data_loader.load_and_dump_to_h5("E:/data/2014_processed", "E:/data/2014_processed.h5", encoding='utf-8')
