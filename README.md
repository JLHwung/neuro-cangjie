# Deep Learning for Cangjie

## Backgrounds

[Cangjie](https://en.wikipedia.org/wiki/Cangjie_input_method) is a Chinese input method, based on the visual structure of the ideographs. It assigns 1-5 letters from a-z to various sub-components in the character, from left to right, top to down.

18% of the characters have multiple Cangjie codes. This is because Unicode unifies minor variants of the 'same' ideographs, whose strokes and structures might differ slightly but are significant enough to be considered different in Cangjie codes.

Ideographs in Unicode is still in expansion. Most of the newly adopted ideographs don't have a definite pronunciation in many East Asian languages, phonetic inputs are not viable for them, leaving Cangjie one of the viable visually based input methods. Each time a new Unicode extension comes out, it is dreadful to encode those new ones with Cangjie codes. So this neural network aims at assigning Cangjie codes to new ideographs automatically.

## The Model

In this task, I use standard Transformer encoder-decoder structure with a feed forward layer as image enbedding layer. The structure is exactly the same as in [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the image encoder is a 2-layer feed forward block, the same as the one in Transformer.

An image is 128x128 matrix, first it is divided into 16x16 patches, 64 in total, non-overlapping. For each patch, the 16x16 data is flattened to an array of 128 length. This step of transformation changes the image from 128x128 to 8x8x128. Then the last dimention goes through the feed farward block as in Transformer, added by 8x8 positional encoding, then flattened again, resulting in a 64x`embed_dim` tensor as input to Transformer.

The Cangjie code is tokenized, with added <start> and <end> special tokens, as well as a separator. Since one character can have multiple Cangjie codes, those different codes are grouped togather with the separator in between. For example: `令: osl`, `令: oni` is rewritten as `令: osl,oni`, following simple alphabetic order. Then the tokens go through embedding, resulting in `token_len`x`embed_dim` A typical padding mask (0-padded) is generated in embedding step as well. The token as input is shifted to the right, so that it's always that the previous token is used to predict the next.

Embedded image is fed into encoders, embedded tokens are fed into decoders with both padding mask and causal mask. After the transformer is done, a simple fully connected layer is used to transform the output to probabilities over the token space.

## Training the Model

To train this model yourself, you may need to download [TH-Tshyn](http://cheonhyeong.com/English/download.html) fonts, the 3 fonts (0, 1, 2) are the only fonts covering all Unicode ideographs. A copy of Cangjie code is already included in the data dir ([source](https://github.com/rime-aca/rime-cangjie6)).

The model is written with [TensorFlow](https://www.tensorflow.org) 2.10, other framework requirements are included in `requirements.txt` file. It is highly recommended to use GPU to train this model. The default model has 178_059_806 parameters, then pruned with 9_198_297 non-zero ones.

To train the model, first install all the requirements in `requirements.txt`, then download the font mentioned above, then run `base_train.py`, `prune.py` and `tflite.py`. A trained tflite model is provided in the **Release** page.

The training and validation results from the latest run is shown in this figure below. <span style="color: #425066">dark</span> and <span style="color: #f9ab00">yellow</span> lines are on training set, <span style="color: #12b5cb">cyan</span> and <span style="color: #9334e6">purple</span> lines are on validation set. The main session is 100 epochs, and the tailing session is pruning, which takes 20 epochs.

<p align="middle">
  <img src="/Figures/log.png" alt="Accuracy" title="Accuracy and Loss" width="700"/>
</p>

*Accuracy shown above is calculated on **per code basis**. Full-code accuracy is 99% in training and 85% on validation.*

## Pretrained Model

A pretrained model is ready in the release page, which can be used directly with `evaluate.ipynb`. The model file is a tflite binary, it can also be deployed in other programing languages and environments.

## License

GPL 3.0
