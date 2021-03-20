# Deep Learning for Cangjie

## Backgrounds

[Cangjie](https://en.wikipedia.org/wiki/Cangjie_input_method) is a Chinese input method, based on the visual structure of the ideographs. It assigns 1-5 letters from a-z to various sub-components in the character, from left to right, top to down. 

18% of the characters have multiple Cangjie codes. This is because Unicode unifies minor variants of the 'same' ideographs, whose strokes and structures might differ slightly but are significant enough to be considered different in Cangjie codes.

Ideographs in Unicode is still in expansion. Most of the newly adopted ideographs don't have a definite pronunciation in many East Asian languages, phonetic inputs are not viable for them, leaving Cangjie one of the viable visually based input methods. Each time a new extension comes out, it is dreadful to encode those new ones with Cangjie codes. So this neural network aims at assigning Cangjie codes to new ideographs automatically.

## The Model

In this task, I adopt a revised encoder-decoder model with attention. The graph of the model is depicted below: (Note, all activations are ignored)

![Model Graph](/Figures/graph.svg "Model Graph")

4 models are used, an encoder which is a CNN network that converts a 64 × 64 × 1 bitmap into features. The other 3 models rely on the features generated by encoder. Among them are 2 dense layer based models that predict code length and potential multiple code representations count respectfully. These 2 models are trained independently from the encoder, since the features from the encoder is rich enough to include the necessary information to make such predictions. The outputs from these 2 models, together with  the features from encoder are fed into the decoder.

The decoder is a RNN, starting from an indication of 'start', it predict the next code using the current code ('start' at the beginning) and the above mentioned inputs, and then the prediction is fed beck as the current code to continue this process, until the maximum code length is reached. Codes shorter than the maximum length are padded with 'end' to the tail. The decoder will learn that 'end' is only followed with 'end', so no special treatment is needed for early stoping with shorter codes. The decoder and encoder are trained together.

There is an extra 5th model (not depicted in the figure above), which is a reduced decoder, used in the first step training. The whole training process is divided into 2 steps, during the first, the reduced decoder is used to focus the training of encoder. After 20 epochs, the full decoder starts to train, replacing the reduced one. This 2 steps training process increases the robustness and reduces the total training time.

Activation function is a smoothed P-ReLU like function shown below, inc which alpha is trainable.
<img src="https://render.githubusercontent.com/render/math?math=\frac{1}{2}\left(\log\left(e^{2\alpha x}+e^{2x}\right)-\log 2\right)">

## Training the Model

To train this model yourself, you may need to download [Hanazono](https://fonts.jp/hanazono/) fonts, the 2 fonts (A and B) are the only fonts covering all Unicode ideographs. Although the current model is trained on a alternative to Hanazono fonts. A copy of Cangjie code is already included in the data dir ([source](https://github.com/rime-aca/rime-cangjie6)).

The model is written with [TensorFlow](https://www.tensorflow.org)   2.3, other framework requirements are included in requirements.txt file. It is highly recommended to use a GPU to train this model. Training time with GPU is usually 3-6 hours, and can be longer than 2 days without GPU. In the latest training, 1 GPUs were used, and multiple GPUs is supported with the mirrored strategy adopted.

The training and validation result from the last run is shown in this figure below as below. During epoch 30-90, teacher forcing fade away gradually, the effect is noticeble in the loss and accuracy figures below. Cyan lines are on training set, magenta lines are on validation set. Accuracies are on the left, and losses are on the right.

<p align="middle">
  <img src="/Figures/accuracy.svg" alt="Accuracy" title="Accuracy" width="350"/>
  <img src="/Figures/loss.svg" alt="Loss" title="Loss" width="350"/>
</p>

In the end of this latest run, accuracy on the training set reached 98%, on the validation set (which was not used in the training process in any form) reached 85% after 150 epochs. This is good enough to put into actual use.

Among those wrong predictions in validation, around 80% correspond to predicted probabilities of 90% or lower. Whereas among those correct predictions, only less than 20% correspond to predicted probabilities of 90% or lower. So, in addition to a predictions themselves, the predicted probabilities can be a good indicator of the correctness of predictions.

## Pretrained Model

A pretrained model is ready in the release page, which can be used directly with `Cangjie6_Evaluate.ipynb`, which .generates Cangjie code and prediction probability for given charactor list.

## License

GPL 3.0
