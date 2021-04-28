## Project proposal

### *Introduction:*

For this project, we performed Russian Automatic Speech Recognition, as part of a pipeline in an end-to-end system for a Russian Speech to English text translation system. This would entail Automatic Speech Recognition using XLSR-Wav2Vec2 fine-tuned on Russian speech data, followed by Machine Translation of the Russian transcriptions to English. We also intended to fine-tune the vanilla XLSR-Wav2Vec2 model on a different Russian speech dataset (other than Common Voice, which it's already been fine-tuned on) to see how it performs. We were going to use the OpenSLR dataset from Librispeech to do this, however since it is not in the same format as Common Voice, it would take really long to fix the format and we don't think this would have added much value, so we decided to stick with Common Voice. Additionally, unanticipated challenges arose through working with speech data, so we will likely need to reduce our original scope. 

We have experiemented with fine-tuning an XLSR-Wav2Vec2 model on Common Voice Russian speech data following the tutorial on Turkish data (https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb#scrollTo=ZJy7p04j78c3). We have had to change some of the functions in the tutorial to allow it to work with Russian data, and we had to get more RAM and a faster processor in order to be able to run the notebook, since the Russian dataset is very large. Following this week, once we get the Russian data working with the code given in the HuggingFace XLSR-Wav2Vec2 tutorial, we intend to use the fine-tuned model in a pipeline with a Machine Translation model to translate the output of XLSR-Wav2Vec2 from Russian to English, and report on the results.

### *Motivation:*

Our motivation for our End-to-end pipeline approach is to see how well a model trained on XLSR-Wav2Vec2 can do on basic ASR (Russian audio to Russian text), to see if we can get a better WER score than the Russian-53 model. We also want to see if the silver-standard (model output) transcriptions do well in a machine translation model, and see if the final output is at all correct and usable.

Overall, we will be learning to train a Machine Learning model to do a Speech Recognition and/or translation task, which would be a learning experience for all of us.

### *Data:*

We will use the Common Voice dataset for the Russian ASR. The Common Voice dataset for Russian contains 111 hours of Russian text read by native speakers, all recorded in a quiet environment. We are loading this data straight into the tutorial notebook using `load_dataset("common_voice", "ru")` with the appropriate train, validation and test splits, so we do not need to store the dataset anywhere. If we decide to use a Machine Translation model, we will likely use a pretrained model so we will not need a dataset for training, but we will store the outputs of the XLSR-Wav2Vec2 model in a file and use that as input to the MT model. 

### *Engineering:*

In this project we use Google Colab as our primary computing infrastructure for developing models. The datasets are stored on a shared Google Drive directory. We use the PyTorch framework. A primary objective of this project is to investigate the performances of two cutting-edge ASR/speech translation systems. On one hand we have the Fairseq Speech2Text model which follows the end-to-end paradigm of converting source language audio to target language text in one self-contained model. On the other hand we use Huggingface's pre-trained cross-lingual speech recognition model `Wav2Vec2-Large-XLSR-53-Russian`, which is based on the XLSR `Wav2Vec2forCTC` model, forming one half of a speech translation pipeline. The other half of the pipeline involves a manually trained Russian text to English text translation model, which takes the output of the ASR model as its input.

### *Previous Works:*

Research in the domain of Russian ASR has been of great interest for several decades now. The history of Russian ASR goes as far back as the Soviet Union with KGB backing for speech recognition research. A descendent of this state-backed research is the organization known in the West as [SpeechPro](https://www.wikiwand.com/ru/%D0%A6%D0%B5%D0%BD%D1%82%D1%80_%D1%80%D0%B5%D1%87%D0%B5%D0%B2%D1%8B%D1%85_%D1%82%D0%B5%D1%85%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D0%B9) (AKA., the Speech Technology Centre (STC)), which is known for creating what has been regarded as the best solution on the market for Russian ASR [[1](https://doi.org/10.1007/978-3-319-23132-7_29), [2](https://doi.org/10.1007/978-3-319-43958-7_13)].

STC focuses heavily on spontaneous speech recognition of conversational Russian language, widely regarded as one of the most difficult areas of ASR as conversational speech is very noisy and error-prone. Conversational speech varies greatly based on the diversity of individual speakers' speech style, accent, speech rate variability, and the speaker's emotions. It might also contain informal words, such as slang, and might have slurred and poorly articulated words. Background noise and music might also have an effect. There are existing highly effective spontaneous speech recognition systems for English that achieve a WER between 8.0% and 14.1%.

In 2016, Prudnikov et. al implemented a speaker-independent system for Russian spontaneous speech recognition. This is more challenging for Russian as compared to English for several reasons, two of which are the lack of available datasets, and because Russian is an inflective language with much more unique words than English. They implemented an i-vector based deep neural network adaptation and speaker-dependent bottleneck features to achieve a WER of 25.1% (11.9% relative word error rate reduction over the baseline system).

The very next year, Prudnikov et. al [[2](https://link.springer.com/chapter/10.1007%2F978-3-319-43958-7_13)] presented further improvements to the Russian spontaneous speech recognition system developed in the Speech Technology Center (STC).In this experiment, they improved upon the previous word error rate to achieve 16.4% by using acoustic modelling approaches, combined with deep BLSTM acoustic models and hypothesis rescoring with RNN-based language models.

Research in this domain has really skyrocketed only within the last 5-6 years, so it is still a hotbed for novel techniques with much room for improvement. Some other notable implementations from outside of Russia in recent years have been based on Mozilla's/Baidu's DeepSpeech model [[3](https://github.com/GeorgeFedoseev/DeepSpeech), [4](http://ceur-ws.org/Vol-2267/470-474-paper-90.pdf)], though they are not able to boast of results even half as good as those from STC.

Considering the recency of the XSLR Wav2Vec2 architecture, there are not yet many publications illustrating the cutting edge applications of this model on the Russian language yet. The most notable resource is Huggingface's pre-trained cross-lingual speech recognition model `Wav2Vec2-Large-XLSR-53-Russian` which boasts a word error rate of 17.39% on the Common Voice Russian test set, which is comparable to other SOTA implementations.

Extending the domain from ASR to speech translation (ST), we see Facebook AI Research's very new Speech-to-Text modelling system, as described in Wang et. al [[5] (https://arxiv.org/abs/2007.10310)] which implements an end-to-end multilingual ST system using the CoVoST project datasets. It uses two Transformer encoder-decoder architectures for ASR followed by MT. While the WER scores for Russian to English (31.4%) are not as stellar as those seen in dedicated ASR models, this is still a novel approach to speech translation and is worth exploring in greater depth.

We hope to achieve comparable results in our project.

### *Evaluation:*

We will evaluate our ASR using word error rate (WER). We will not use Sentence Error Rate (SER), because it is not a suitable metric for our dataset. Intent/entity recognition rate is also irrelevant, as we are not aiming to detect intents of the speech. An alternative metric is character error rate (CER), but it is becoming that WER is the standard metric of evaluation for most ASR models in the Russian domain especially. Therefore, to allow for easier comparison with other models, we believe that WER is the most suitable metric for our project.

We will calculate WER as follows: 

Word Error Rate = (Substitutions + Insertions + Deletions) / Number of Words Spoken

Substitutions are anytime a word gets replaced (for example, “twinkle” is transcribed as “crinkle”).
Insertions are anytime a word gets added that wasn’t said (for example, “trailblazers” becomes “tray all blazers”).
Deletions are anytime a word is omitted from the transcript (for example, “get it done” becomes “get done”).

`wer_metric = load_metric("wer")`

### *Challenges:*

Some challenges we faced this week when trying to train the model were as follows:

- Modifying the OpenSLR data to be compatible with the HuggingFace Notebook was too difficult and would take too much time, so we used the Common Voice dataset instead, which worked with the notebook seamlessly.
- The Russian dataset had some configurations that were different from the Turkish dataset. We had to modify the `speech_file_to_array_fn` to work with the batches in the dataset. We were also unable to work with batch sizes greater than 1, which made the data loading and training really slow. (finally we were able to use larger batch sizes)
- The notebook would often crash due to RAM shortage. We had to upgrade to Colab Pro in order to increase RAM, and once we did that, we still ran out of disk space when using a CPU or GPU runtime. Finally, the TPU runtime had enough disk space to load all of the data and start training, but training on CPU was way too slow (an estimated 279 hours), and we could not access the GPU in a TPU runtime. Then, we attempted to set the device to TPU for the model in an attempt to reduce training time. We were unfamiliar with the parallelization of TPUs and tried to train using one core, which resulted in a memory error for that TPU core. We are currently investigating how to parallelize our dataset and the training onto multiple cores.


## *Experiments and Results:*

Once we were able to train our model, we performed several experiments.

- At first, we ran the model on only 100 training examples, 20 validation examples, and 20 test examples. We got a WER score of 100%, and the predictions made by the model were all blank (just series of spaces). 
- At 500 examples, the model started making predictions, but they were mostly jumbled and the spaces were in the wrong places. WER:
- At 1000, WER:
- At 4100, WER: around 0.44.

Since the training of the model for 4100 examples took around 10 hours, we were only able to run the model 3 times, so we did not explore different hyperparameters. We only tried to chance the logging time (from 400. 

- Analysis: our WER score is low because we were not able to train the model on enough data. If we could have trained the model on all of the data available in the Common Voice Russian dataset, I'm sure the WER would have gone down further. (future work)

2200 training examples:


| Step     | WER          | 
| ------------- |:-------------:| 
| 100 | 1.000000 | 
| 200 | 1.000000 | 
| 300 | 1.000000 | 
| 400 | 1.000000 |
| 500 | 1.000000 |
| 600 | 1.000000 |
| 700 | 1.000000 |
| 800 | 0.902783 |
| 900 | 0.773992 |
| 1000 | 0.686201 |
| 1100 | 0.615786 |
| 1200 | 0.623623 |
| 1300 | 0.574787 |
| 1400 | 0.567973 |
| 1500 | 0.555821 |
| 1600 | 0.549461 |
| 1700 | 0.529131 |
| 1800 | 0.522203 |
| 1900 | 0.515957 |
| 2000 | 0.504713 |
| 2100 | 0.502555 |
| 2200 | 0.496309 |
| 2300 | 0.489743 |
| 2400 | 0.471623 |
| 2500 | 0.475581 |
| 2600 | 0.475581 |
| 2700 | 0.461314 |
| 2800 | 0.465271 |
| 2900 | 0.455691 |
| 3000 | 0.451526 |
| 3100 | 0.458503 |
| 3200 | 0.464751 |
| 3300 | 0.449651 |
| 3400 | 0.452879 |
| 3500 | 0.445173 |
| 3600 | 0.445798 |
| 3700 | 0.445590 |
| 3800 | 0.441841 |
| 3900 | 0.448402 |
| 4000 | 0.439238 |
| 4100 | 0.437676 |


The following graph depicts how the WER improves every 100 iterations:

![WER decrease graph](https://github.ubc.ca/sshank00/COLX_585_Project/edit/darya/step_wer.png "WER decrease per 100 utterances")

We see a drop starting from about 800, when the WER first dips below 100%. Then, we observe a sharp decline until around 1100, where it continues to decrease, but at a much slower rate. After 2000, the rate of decrease is very small, but WER does continue to decrease. We think that if we were able to train it on more data, we could get the WER down to less than 40%, but it is impossible to tell for sure. 

However, despite the WER being pretty high, we believe the model does pretty well at guessing the sounds, based on our manual inspection of the outputs. Russian is a very phonetic language, so words are generally spelled the way they sound. Since our model does not have a language model, it does not know what combinations of letters form a valid word. Thus, it is just guessing the letters based on the sounds it hears, and the spaces based on the silence in between. If it messes up some of the spelling, or where it places the spaces, the WER will go down. However, this does not mean that the model isn't doing well, as a simple repositioning of the space, or the fixing of a typo, would fix the issue. 


Next, we passed the results from our final experiment (trained on XXX sentences with a WER of XXX) into the next step of our pipeline, which is the translation step. We divided this into two experiments: using a spell checker first, and not using a spell checker at all. The results were as follows:



### *Future Work:*

In order to improve our model, we would first of all need to improve the WER score. In order to do this, we would need to train the model on all of the data available. Thus, the first thing we would do is figure out how to get more disk space on Google Colab (or get access to a supercomputer), parallelize the data and model to run on multiple TPU cores, or save checkpoints of a pretrained model, and then train the pretrained model on more data (train in batches).

Additionally, to make sure that we get a sensible output from a Machine Translation model, we could improve the output of the ASR model by adding a language model, so that the ASR model knows what a valid word in Russian is, and what is not. Currently, it is just guessing the sounds, and while it would likely perform well after more training (and guess words correctly), sometimes it gets the sounds correct but words wrong. With a language model, it would have different probabilities for outputs, and output that is a word would get a higher probability, so the model would output a real word. This would do better in a Machine Translation model, as those are trained to translate words, not sounds. 

Another thing that could help correct the output of the model, in the case of a letter in a word being misrecognized (happens to vowels often), would be to apply a russian grammatical error correction model to the output before passing it to the machine translation model, instead of a simple spell checker model that we used. An example could be https://github.com/grammatical/magec-wnut2019/tree/master/models. This way, any evident typos are likely to be corrected, and the machine translation model will have to deal with much less OOV words.

We looked into Grammatical Error Correction for Russian in ![Rozovskaya & Roth, 2019](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00251/43532/Grammar-Error-Correction-in-Morphologically-Rich). Rozovskaya & Roth, 2019, attempted to perform the correction of writing mistakes of the Russian language which is among the morphologically rich languages. They used minimal supervision methods as there was not a large amount of annotated training data available in Russian language. The corpus they used consisted of essays and papers written at university level by student who were learning Russian as a foreign language and students who had exposure to Russian at home. Two annotators were corrected the sentences and categorized the mistakes to different classes. The inter annotator agreement was calculated and the errors were also classified based on foreign and native speakers. Then they used phrase-based Machine Translation (MT) system. They then performed error analysis and compared the output of the MT with the annotated data. The result of the MT also was later annotated by annotators. They hope that the release of the annotated data can help the upcoming studies to have a better base for their performance.


### *Conclusion:*

The ultimate goal is to create a tool that automatically translates Russian speech to English text in the future. If we succeed with the ASR portion, this project will be a great starting point for that goal.
