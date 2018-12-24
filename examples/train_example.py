from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from segmenter.custom.callbacks import LRFinder, LRSchedulerPerStep, SingleModelCK, SGDRScheduler
from segmenter import get_or_create, save_config
from segmenter.data_loader import DataLoader

if __name__ == '__main__':
    h5_file_path = "../data/2014_processed.h5"  # data path
    config_save_path = "../data/default-config.json"  # config path
    weights_save_path = "../models/weights.{epoch:02d}-{val_loss:.2f}.h5"  # weights path to save
    init_weights_path = "../models/weights.17-0.07.h5"  # weights path to load

    src_dict_path = "../data/src_dict.json"  # source dict
    tgt_dict_path = "../data/tgt_dict.json"  # target dict
    batch_size = 128
    epochs = 128
    num_gpu = 4
    max_seq_len = 256

    # import os
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

    data_loader = DataLoader(src_dict_path=src_dict_path,
                             tgt_dict_path=tgt_dict_path,
                             max_len=max_seq_len,
                             batch_size=batch_size,
                             sparse_target=False)

    steps_per_epoch = 500
    validation_steps = 50

    config = {
        'src_vocab_size': data_loader.src_vocab_size,
        'tgt_vocab_size': data_loader.tgt_vocab_size,
        'max_seq_len': max_seq_len,
        'max_depth': 2,
        'model_dim': 320,
        'residual_dropout': 0.2,
        'attention_dropout': 0.0,
        'l2_reg_penalty': 1e-6,
        'confidence_penalty_weight': 0.1,
        'compression_window_size': None,
        'use_masking': True,
        'num_heads': 8,
        'use_crf': True,
        'label_smooth': True
    }

    segmenter = get_or_create(config,
                              optimizer=Adam(1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                              src_dict_path=src_dict_path,
                              weights_path=init_weights_path,
                              num_gpu=num_gpu)

    save_config(segmenter, config_save_path)

    segmenter.model.summary()

    ck = SingleModelCK(weights_save_path,
                       model=segmenter.model,
                       save_best_only=False,
                       save_weights_only=True,
                       monitor='val_loss',
                       verbose=0)
    log = TensorBoard(log_dir='../logs',
                      histogram_freq=0,
                      batch_size=data_loader.batch_size,
                      write_graph=True,
                      write_grads=False)

    # Use LRFinder to find effective learning rate
    lr_finder = LRFinder(1e-6, 1e-2, steps_per_epoch, epochs=1)  # => (2e-4, 3e-4)
    lr_scheduler = LRSchedulerPerStep(segmenter.model_dim)
    # lr_scheduler = SGDRScheduler(min_lr=1e-5, max_lr=1e-4, steps_per_epoch=steps_per_epoch,
    #                              cycle_length=15,
    #                              lr_decay=0.98,
    #                              mult_factor=1.5)

    X_train, Y_train, X_valid, Y_valid = data_loader.load_data(h5_file_path, frac=0.8)

    segmenter.parallel_model.fit_generator(data_loader.generator_from_data(X_train, Y_train),
                                           epochs=epochs,
                                           steps_per_epoch=steps_per_epoch,
                                           validation_data=data_loader.generator_from_data(X_valid, Y_valid),
                                           validation_steps=validation_steps,
                                           callbacks=[ck, log, lr_scheduler])

    # lr_finder.plot_lr()
    # lr_finder.plot_loss()
