# NLI models (EMNLP 2018)

This repository contains the TensorFlow Implementation of:

1. "Compare, Compress and Propagate: Enhancing Neural Architectures with Alignment Factorization for Natural Language Inference" (EMNLP 2018)
2. "Co-Stack Residual Affinity Networks with Multi-level Attention Refinement for Matching Text Sequences" (EMNLP 2018)

This repository implements and supports:
1. CAFE model and CSRAN (EMNLP'18)
2. CAFE + ELMo model (88.7-89.0) / CSRAN + ELMo model (88.9-89.0)
3. Using ELMo and CoVe (McCann et al.) in NLI models.
4. Using CuDNN versions of CoveLSTM and also BiLSTMs. (much faster)

For CAFE, you should be able to get 88.1-88.3 score *easily* on SNLI. (88.5 was the best we obtained). CSRAN can be thought of a better version of CAFE (since it's multi-level CAFE plus residual attention), the best score obtained was 88.65 (i.e., 88.7 reported in the paper) on SNLI. To give a better idea of the training process, the exact log file can be found at `saved_logs/csran.txt`. With ELMo only at the embedding layer, CSRAN gets about 88.9-89.0 and CAFE gets about 88.8-89.0. We did not tune the ELMo models much (only 1-2 runs max and only after paper acceptance, so it probably can get higher performance if we tune some HPs).

# Setup

We provide a preprocessed SNLI file that works nicely with our input format. We host it on dropbox for your convenience.

```
bash ./setup/setup_snli.sh
```

We use POS embeddings, Char Embeddings and EM features. You can also use `setup_mnli.sh` and `setup_scitail` to download pre-processed versions of the other NLI datasets.

### Running the code

An example run to train our CSRAN model can be found:
```
python train_acc.py --rnn_type SOFT_HP_CHAR_CAFE_RESFM_HIGH_LSTM --char_enc RNN --char_max 16 --use_cudnn 1 --batch-size 128 --gpu 0 --opt Adam --lr 1E-3 --epochs 100 --early_stop 0 --num_dense 2 --dataset SNLI --decay_lr 0.96 --decay_epoch 10 --l2_reg 1E-8 --cnn_size 64 --translate_proj 1 --num_proj 1 --rnn_size 200 --hdim 300 --rnn_layers 3 --sort_batch 1 --use_pos 1 --pos_emb_size 10 --clip_norm 5 --aggr_layers 2
```

For CAFE, please switch `--rnn_type SOFT_HP_CHAR_FF2_MM_LSTM`.


### Using ELMo and CoVE

For CoVe please use `dl_cove.sh`. For ELMo, you need Tensorflow Hub. We ported CoVe into CuDNN LSTMs for faster speed


## Some random (possibly helpful) notes

1. The original naming of CAFE was FF2 (FactorFlow) and also RESFM was the code for CSRAN. It might be helpful if you're looking at the model code.
2. Even though both models have been accepted in EMNLP'18, experiments for CAFE was conducted in Nov 2017 and CSRAN was conducted around April 2018. (This is relevant if we want to consider for different TF versions.) CAFE was tested using non-cudnn LSTM while CSRAN mainly used CuDNN LSTMs.

## References

If you find our repository useful, please consider citing our work! Thanks!

```
@inproceedings{emnlp2018,
  author    = {Yi Tay and
               Luu Anh Tuan and
               Siu Cheung Hui},
  title     = {Compare, Compress and Propagate: Enhancing Neural Architectures with Alignment Factorization for Natural Language Inference},
  booktitle = {Proceedings of EMNLP 2018},
  year      = {2018}
}

@inproceedings{emnlp2018,
  author    = {Yi Tay and
               Luu Anh Tuan and
               Siu Cheung Hui},
  title     = {Co-Stack Residual Affinity Networks with Multi-level Attention Refinement for Matching Text Sequences},
  booktitle = {Proceedings of EMNLP 2018},
  year      = {2018}
}
```
