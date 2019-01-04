from keras.callbacks import EarlyStopping

from tf_segmenter import get_or_create
from tf_segmenter.custom.callbacks import SingleModelCK
from tf_segmenter.data_loader import DataLoader

if __name__ == '__main__':
    segmenter = get_or_create("../data/default-config.json",
                              weights_path="../models/weights.08--0.06.h5")

    dataloader = DataLoader(src_dict_path="../data/src_dict.json",
                            tgt_dict_path="../data/tgt_dict.json",
                            batch_size=7,
                            max_len=256)

    es = EarlyStopping(monitor='acc', mode='min', baseline=1.0)
    ck = SingleModelCK("../models/fine_tuned.h5",
                       model=segmenter.model,
                       save_best_only=True,
                       save_weights_only=True,
                       monitor='acc',
                       verbose=0)

    segmenter.model.fit_generator(dataloader.generator("../data/fine_tune.txt"),
                                  epochs=5,
                                  steps_per_epoch=15,
                                  callbacks=[es, ck])
