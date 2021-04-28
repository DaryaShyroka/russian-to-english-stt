# Russian-to-English Automatic Speech Subtitling System
### Group 7: Darya Shyroka, Mukhamedali Zhadigerov, Ladan Naimi, Shreyas Shankar

## ***Abstract***
In this project we implement a proof-of-concept pipeline for Russian-to-English speech-to-text translation, with the future intent of constructing a robust automatic speech subtitling system. This was implemented as a two-stage pipeline: automatic speech recognition (ASR) of Russian speech to Russian text using a pre-trained XLSR-Wav2Vec2 ASR model fine-tuned on additional Russian speech data, followed by a pre-trained Russian text to English text machine translation (MT) model trained on Russian subtitling data. This project builds on recent advances in the field of ASR, primarily coming from the Wav2Vec2 architecture released in 2020 by Facebook AI Research, implemented by HuggingFace.

Complications with compute infrastructure led to many hurdles in the training process, leading to the ASR model being trained on a small fraction of the originally intended dataset. This led to the ASR model’s word error rate being somewhat poor, at about 40%, which is far below what is considered state-of-the-art in Russian ASR. However, we were still able to recover some of this loss by applying an off-the-shelf spelling correction model. Feeding the corrected silver-standard Russian texts into the pre-trained Russian-to-English text machine translation model OPUS-MT, we were able to produce promising translations that, with some refinements to the ASR model and through the addition of a language model prior to the MT model, could be deployed in a full-fledged pipeline for automatic Russian-to-English subtitling.

## ***Introduction***
In this project we attempted to implement a proof-of-concept pipeline for a Russian-to-English speech-to-text translation system. This entailed automatic speech recognition (ASR) of Russian speech audio to Russian text using the XLSR-Wav2Vec2 model fine-tuned on Russian speech data, followed by machine translation (MT) of the Russian transcriptions to English. Due to the complex and cascading nature of this task, we decided that the priority of the project would be to construct a robust ASR model before pursuing the MT model.

Our motivation for a pipeline approach is to see how well a model trained on XLSR-Wav2Vec2 can do on basic ASR (Russian audio to Russian text), to see if we can get a better word error rate (WER) than the fine-tuned Russian-53 model. We also want to see if the silver-standard (model output) transcriptions do well in a machine translation model, and see if the final output is at all correct and usable. If we can achieve good results in both ASR and MT, we can use Watson TTS or any other (better?) word to speech models to build a `Russian speech to English speech` system. The idea for this project was motivated by [`ili - instant translation device for travelers`](https://iamili.com/us/#:~:text=ili%20is%20made%20for%20your,ili%20to%20anywhere%20you%20want).

We experimented by fine-tuning an XLSR-Wav2Vec2 model on CommonVoice Russian speech data following the [tutorial on Turkish data](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb#scrollTo=ZJy7p04j78c3). A few functions needed to be changed in the tutorial to make it compatible with Russian data, more RAM was added, and a faster processor was used in order to be able to run the notebook, since the Russian dataset was very large. In order to train the model, we started by using a fraction of the data and increased the data in the next attempts of running the model. The result of these experiments is summarized in the following sections.

## ***Related Works***
Research in the domain of Russian ASR has been of great interest for several decades now. The history of Russian ASR goes as far back as the Soviet Union with KGB backing for speech recognition research. A descendent of this state-backed research is the organization known in the West as [SpeechPro](https://www.wikiwand.com/ru/%D0%A6%D0%B5%D0%BD%D1%82%D1%80_%D1%80%D0%B5%D1%87%D0%B5%D0%B2%D1%8B%D1%85_%D1%82%D0%B5%D1%85%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D0%B9) (AKA., the Speech Technology Centre (STC)), which is known for creating what has been regarded as the best solution on the market for Russian ASR [[1](https://doi.org/10.1007/978-3-319-23132-7_29), [2](https://doi.org/10.1007/978-3-319-43958-7_13)].

STC focuses heavily on spontaneous speech recognition of conversational Russian language, widely regarded as one of the most difficult areas of ASR as conversational speech is very noisy and error-prone. Conversational speech varies greatly based on the diversity of individual speakers' speech style, accent, speech rate variability, and the speaker's emotions. It might also contain informal words, such as slang, and might have slurred and poorly articulated words. Background noise and music might also have an effect. There are existing highly effective spontaneous speech recognition systems for English that achieve a WER between 8.0% and 14.1%.

In 2016, Prudnikov et. al implemented a speaker-independent system for Russian spontaneous speech recognition. This is more challenging for Russian as compared to English for several reasons, two of which are the lack of available datasets, and because Russian is an inflective language with much more unique words than English. They implemented an i-vector based deep neural network adaptation and speaker-dependent bottleneck features to achieve a WER of 25.1% (11.9% relative word error rate reduction over the baseline system).

The very next year, Prudnikov et. al [[2](https://link.springer.com/chapter/10.1007%2F978-3-319-43958-7_13)] presented further improvements to the Russian spontaneous speech recognition system developed in the Speech Technology Center (STC).In this experiment, they improved upon the previous word error rate to achieve 16.4% by using acoustic modelling approaches, combined with deep BLSTM acoustic models and hypothesis rescoring with RNN-based language models.

Research in this domain has really skyrocketed only within the last 5-6 years, so it is still a hotbed for novel techniques with much room for improvement. Some other notable implementations from outside of Russia in recent years have been based on Mozilla's/Baidu's DeepSpeech model [[3](https://github.com/GeorgeFedoseev/DeepSpeech), [4](http://ceur-ws.org/Vol-2267/470-474-paper-90.pdf)], though they are not able to boast of results even half as good as those from STC.

The most recent addition to the domain of ASR is the XSLR Wav2Vec2 architecture, which was released for public use only a few months ago. Considering its recency, there are not yet many publications illustrating the cutting-edge applications of this model on the Russian language yet. The most notable resource is Huggingface's pre-trained cross-lingual speech recognition model `Wav2Vec2-Large-XLSR-53-Russian` which boasts a word error rate of 17.39% on the CommonVoice Russian test set, which is comparable to other SOTA implementations.

## ***Data***
For the ASR model, we used the Russian CommonVoice dataset. It contains 111 hours of Russian text read by native speakers, all recorded in a quiet environment. We are loading this data straight into the tutorial notebook using load_dataset("common_voice", "ru") with the appropriate train, validation and test splits, so we do not need to store the dataset anywhere. Due to disk space limitations, we used subsets of the data in our experiments, with the biggest being the first 4100 sentences in the training set, and 820 in the validation and test sets.

For the spell checker model and the Machine Translation model, we used the outputs of the ASR model as inputs to the models. Thus, these were silver-standard audio transcriptions made by our pretrained model, with a WER score of 0.44. There were XXX sentences used.

## ***Engineering***
For this project we used Google Colab Pro as our primary computing infrastructure for developing models. As described in the `Data` section above, the datasets were loaded directly into the Colab runtime environment upon execution of the notebook and thus there was no need for a shared dataset storage solution like Google Drive. We used the PyTorch framework to develop our models, with extensive use of Huggingface's Transformer and Datasets libraries. A primary objective of this project was to investigate the performance of Huggingface's cutting-edge ASR system, [`Wav2Vec2-Large-XLSR-53-Russian`](https://huggingface.co/anton-l/wav2vec2-large-xlsr-53-russian), a pre-trained cross-lingual speech recognition model. This model formed one half of the speech translation pipeline.

We evaluated our ASR using word error rate (WER). We could not use Sentence Error Rate (SER) because it is not a suitable metric for our dataset. Intent/entity recognition rate is also irrelevant, as we were not aiming to detect intents of the speech. An alternative metric was character error rate (CER), but in recent years WER has become the standard metric of evaluation for most ASR models (in the Russian domain especially). Therefore, to allow for easier comparison with other models, we believed that WER was the most suitable metric for our project.

We calculated WER as follows: 

Word Error Rate = (Substitutions + Insertions + Deletions) / Number of Words Spoken

* Substitutions are anytime a word gets replaced (for example, “twinkle” is transcribed as “crinkle”).
* Insertions are anytime a word gets added that wasn’t said (for example, “trailblazers” becomes “tray all blazers”).
* Deletions are anytime a word is omitted from the transcript (for example, “get it done” becomes “get done”).

The other half of the pipeline involved a pre-trained translation model, which took the silver-standard output of the ASR model as its input. We used the EasyNMT library’s pre-trained `opus-mt` model to translate Russian text to English text. We had initially tried to use HuggingFace's Helsinki-NLP/opus-mt-ru-en but could not make it work. Searching the internet we found that there might be version incompatibility issues with internal libraries, like sentencepiece. Since we had a relatively small dataset and were not using language modelling, our Russian text outputs had many spelling errors. That is why we decided to post-process the output texts by correcting misspellings using the spellchecker library. It returns a set of the most likely corrections for misspelled words (ordered by probability) and we selected the most probable corrections. We understand that this is not a perfect solution because spellchecker might substitute named entities as well, for example, 'вася' is a russian name and the spell checker substitutes it with 'вас', which means 'you'. This issue could be prevented if the named entities were capitalized, however our ASR model doesn't support NE capitalization.

## ***Challenges*** TODO - update with more recent MT stuff
We faced many hurdles through the course of this project, the majority of which came at the time of training. Some challenges we faced when trying to train the model were as follows:

* Modifying the OpenSLR data to be compatible with the HuggingFace Notebook was too difficult and would take too much time, so we used the CommonVoice dataset instead, which worked with the notebook seamlessly.
* The Russian dataset had some configurations that were different from the Turkish dataset. We had to modify the `speech_file_to_array_fn` to work with the batches in the dataset. We were also unable to work with batch sizes greater than 1, which made the data loading and training really slow. Only towards the last days of the project were we finally able to use larger batch sizes, enabling relatively faster training times.
* The notebook would often crash due to RAM shortage. We had to upgrade to Colab Pro in order to increase RAM, and once we did that, we still ran out of disk space when using a CPU or GPU runtime. Finally, the TPU runtime had enough disk space to load all of the data and start training, but by default the TPU runtime trains on CPU which was much too slow (an estimated 279 hours). We attempted to set the device to TPU for the model in an attempt to reduce training time. We were unfamiliar with the parallelization of TPU cores on Colab and tried to train using one core, which resulted in a memory error for that TPU core, since the entire ASR model could not be loaded onto a single TPU core. Therefore, we switched back to using GPUs but with just a fraction of the data.

## ***Experiments and Results***  #TODO: Update with MT results
Once we were able to train our model, we performed several experiments:
* At first, we ran the model on only 100 training examples, 20 validation examples, and 20 test examples. We got a WER score of 100%, and the predictions made by the model were all blank (just a series of spaces).
* At 500 examples, the model started making predictions, but they were mostly jumbled and the spaces were in the wrong places. WER: still 100%.
* At 1000, WER: 0.6, predictions got better
* At 4100, WER: around 0.44.

Final prediction set: XXX utterances, WER: XXX. 

Since the training of the model for 4100 examples took around 10 hours, we were only able to run 3 experiments of training the model, so we could not explore different hyperparameters. We only changed the logging time from 400 to 100 so that we could see the WER score every 100 utterances.

4100 training examples:
| Step | WER |
|:----:|:---:|
100 | 1.000000
200 | 1.000000
300 | 1.000000
400 | 1.000000
500 | 1.000000
600 | 1.000000
700 | 1.000000
800 | 0.902783
900 | 0.773992
1000 | 0.686201
1100 | 0.615786
1200 | 0.623623
1300 | 0.574787
1400 | 0.567973
1500 | 0.555821
1600 | 0.549461
1700 | 0.529131
1800 | 0.522203
1900 | 0.515957
2000 | 0.504713
2100 | 0.502555
2200 | 0.496309
2300 | 0.489743
2400 | 0.471623
2500 | 0.475581
2600 | 0.475581
2700 | 0.461314
2800 | 0.465271
2900 | 0.455691
3000 | 0.451526
3100 | 0.458503
3200 | 0.464751
3300 | 0.449651
3400 | 0.452879
3500 | 0.445173
3600 | 0.445798
3700 | 0.445590
3800 | 0.441841
3900 | 0.448402
4000 | 0.439238
4100 | 0.437676

The following graph depicts how the WER improves every 100 iterations:

![WER declining over 4100 training iterations](./images/step_wer.png?raw=true)

We see a drop starting from about 800, when the WER first dips below 100%. Then, we observe a sharp decline until around 1100, where it continues to decrease, but at a much slower rate. After 2000, the rate of decrease is very small, but WER does continue to decrease. We think that if we were able to train it on more data, we could get the WER down to less than 40%, but it is impossible to tell for sure. The lowest the WER got, as we can see from the table, is after 4100 iterations for a final WER score of 0.437.

However, despite the WER being pretty high, we believe the model does pretty well at guessing the sounds, based on our manual inspection of the outputs. Russian is a very phonetic language, so words are generally spelled the way they sound. Since our model does not have a language model, it does not know what combinations of letters form a valid word. Thus, it is just guessing the letters based on the sounds it hears, and the spaces based on the silence in between. If it messes up some of the spelling, or where it places the spaces, the WER will go down. However, this does not mean that the model isn't doing well, as a simple repositioning of the space, or the fixing of a typo, would fix the issue.

Other graphs:

![Training loss declining over 4100 training iterations](./images/step_TrainingLoss.png?raw=true)
![Validation loss declining over 4100 training iterations](./images/step_ValidationLoss.png?raw=true)

Next, we passed the results from our final experiment (trained on XXX sentences with a WER of XXX) into the next step of our pipeline, which is the translation step. We divided this into two experiments: using a spell checker first, and not using a spell checker at all. The results were as follows: ## TODO

## ***Future Work***
In order to improve our model, we would first of all need to improve the WER score. It is high because we were not able to train the model on enough data, due to disk space and time limitations. In order to overcome these limitations, we would need to train the model on all of the data available on CommonVoice. Thus, the first thing we would do is figure out how to get more disk space on Google Colab Pro (or get access to a supercomputer), parallelize the data and model to run on multiple TPU cores, or save checkpoints of a pretrained model, and then train the pretrained model on more data (train in batches).

Additionally, to make sure that we get a sensible output from an MT model, we could improve the output of the ASR model by adding a language model, so that the ASR model knows what a valid word in Russian is, and what is not. Currently, it is just guessing the sounds, and while it would likely perform well after more training (and guess more words correctly), a language model could help it to generate valid words, which would be better output to a machine translation model. With a language model, it would have different probabilities for possible outputs, and outputs that are valid words would get higher probabilities, so the model would output a real word. This would do better in an MT model, as those are trained to translate words, not sounds.

Another thing that could help correct the output of the model, in the case of a letter in a word being misrecognized, which often happens to vowels in Russian, would be to apply a Russian grammatical error correction model to the output before passing it to the machine translation model, instead of the comparatively rudimentary spell checker model that we used. An example of such a model could be the [MAGEC error correction system](https://github.com/grammatical/magec-wnut2019/tree/master/models). This way, any evident typos are likely to be corrected, and the machine translation model will have to deal with much fewer out-of-vocabulary words.

## ***Conclusion***
The ultimate goal for the future of this project is to create a tool that automatically translates Russian speech to English text, focusing primarily on the ASR component of the task. From this project we understood that if we had more computing power and more time we could increase the amount of training data which would likely decrease our WER on the ASR task. Other things that would have likely improved our WER score are incorporating a language model into our ASR model, and using a Grammatical Error Correction model after the ASR step. Working through all of the challenges we faced and performing the experiments along the way taught us a great deal about working with speech data and training speech recognition models. It also gave us a good base to work with for our ultimate goal of an automatic subtitling system for Russian, and we intend to continue working on this system.
