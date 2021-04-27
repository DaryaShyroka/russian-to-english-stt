## Project proposal

### *Introduction:*

For this project, we planed to perform Russian Automatic Speech Recognition, as part of an end-to-end pipeline for a Russian Speech to English text translation system. This would entail Automatic Speech Recognition using XLSR-Wav2Vec2 fine-tuned on Russian speech data, followed by Machine Translation of the Russian transcriptions to English. We also intended to fine-tune the vanilla XLSR-Wav2Vec2 model on a different Russian speech dataset (other than Common Voice, which it's already been fine-tuned on) to see how it performs. We were going to use the OpenSLR dataset from Librispeech to do this, however since it was not in the same format as Common Voice, it would take a considerable amount of time to adapt the format to work with the existing model and we didn't think this would have added much value, so we decided to stick with Common Voice. Additionally, unanticipated challenges arose through working with speech data, so we reduced our original scope.

We have experimented with fine-tuning an XLSR-Wav2Vec2 model on Common Voice Russian speech data following the [tutorial on Turkish data](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb#scrollTo=ZJy7p04j78c3). We have had to change some of the functions in the tutorial to allow it to work with Russian data, and we had to get more RAM and a faster processor in order to be able to run the notebook, since the Russian dataset was very large. This week we managed the Russian data to work with the code given in the HuggingFace XLSR-Wav2Vec2 tutorial, we intended to use the fine-tuned model in a pipeline with a Machine Translation model to translate the output of XLSR-Wav2Vec2 from Russian to English, and reported on the results.

### *Motivation:*

Our motivation for an End-to-end pipeline approach was to see how well a model trained on XLSR-Wav2Vec2 could do on basic ASR (Russian audio to Russian text), to see if we could get a better WER score than the fine-tuned Russian-53 model. We also wanted to see if the silver-standard (model output) transcriptions do well in a machine translation model, and if the final output was at all correct and usable. We anticipated that if we can achieve good results in both ASR and MT, we can use Watson TTS or any other (better?) word to speech models to build a `Russian speech to English speech` system. The idea for this project was motivated by [`ili - instant translation device for travelers`](https://iamili.com/us/#:~:text=ili%20is%20made%20for%20your,ili%20to%20anywhere%20you%20want).

Overall, as a result we expected to learn how to train a Machine Learning model to do a Speech Recognition and/or translation task.

### *Data:*

We used the Common Voice dataset for the Russian ASR. The Common Voice dataset for Russian contained 111 hours of Russian text read by native speakers, all recorded in a quiet environment. We were loading this data straight into the tutorial notebook using `load_dataset("common_voice", "ru")` with the appropriate train, validation and test splits, so we did not need to store the dataset anywhere; they got loaded straight into the Colab runtime environment. If we decided to use a Machine Translation model, we would likely use a pre-trained model so we did not need a dataset for training. We then would store the outputs of the XLSR-Wav2Vec2 model in a file and use that as input to the MT model.

### *Engineering:*

In this project we used Google Colab as our primary computing infrastructure for developing models. As described in the `Data` section above, the datasets were loaded directly into the Colab runtime environment upon execution of the notebook and thus there was no need for a shared dataset storage solution like Google Drive. We used the PyTorch framework to develop our models, with extensive use of Huggingface's Transformer and Datasets libraries. A primary objective of this project was to investigate the performance of Huggingface's cutting-edge ASR system, [`Wav2Vec2-Large-XLSR-53-Russian`](https://huggingface.co/anton-l/wav2vec2-large-xlsr-53-russian), a pre-trained cross-lingual speech recognition model. This model did form one half of a speech translation pipeline. The other half of the pipeline involved a manually trained Russian text to English text translation model, which took the output of the ASR model as its input. This Machine Translation half of the pipeline was built as a encoder-decoder Transformer model as well, either trained from scratch on Russian subtitling data (dataset TBD) that was relevant to the scope of our project, or used an off-the-shelf pre-trained Russian-to-English translation model.

### *Previous Works:*

Research in the domain of Russian ASR has been of great interest for several decades now. The history of Russian ASR wernt as far back as the Soviet Union with KGB backing for speech recognition research. A descendent of this state-backed research was the organization known in the West as [SpeechPro](https://www.wikiwand.com/ru/%D0%A6%D0%B5%D0%BD%D1%82%D1%80_%D1%80%D0%B5%D1%87%D0%B5%D0%B2%D1%8B%D1%85_%D1%82%D0%B5%D1%85%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D0%B9) (AKA., the Speech Technology Centre (STC)), which was known for creating what has been regarded as the best solution on the market for Russian ASR [[1](https://doi.org/10.1007/978-3-319-23132-7_29), [2](https://doi.org/10.1007/978-3-319-43958-7_13)].

STC focused heavily on spontaneous speech recognition of conversational Russian language, widely regarded as one of the most difficult areas of ASR as conversational speech was very noisy and error-prone. Conversational speech varies greatly based on the diversity of individual speakers' speech style, accent, speech rate variability, and the speaker's emotions. It might also contain informal words, such as slang, and might have slurred and poorly articulated words. Background noise and music might also have an effect. There are existing highly effective spontaneous speech recognition systems for English that achieve a WER between 8.0% and 14.1%.

In 2016, Prudnikov et. al implemented a speaker-independent system for Russian spontaneous speech recognition. This is more challenging for Russian as compared to English for several reasons, two of which are the lack of available datasets, and because Russian is an inflective language with much more unique words than English. They implemented an i-vector based deep neural network adaptation and speaker-dependent bottleneck features to achieve a WER of 25.1% (11.9% relative word error rate reduction over the baseline system).

The very next year, Prudnikov et. al [[2](https://link.springer.com/chapter/10.1007%2F978-3-319-43958-7_13)] presented further improvements to the Russian spontaneous speech recognition system developed in the Speech Technology Center (STC).In this experiment, they improved upon the previous word error rate to achieve 16.4% by using acoustic modelling approaches, combined with deep BLSTM acoustic models and hypothesis rescoring with RNN-based language models.

Research in this domain has really skyrocketed only within the last 5-6 years, so it is still a hotbed for novel techniques with much room for improvement. Some other notable implementations from outside of Russia in recent years have been based on Mozilla's/Baidu's DeepSpeech model [[3](https://github.com/GeorgeFedoseev/DeepSpeech), [4](http://ceur-ws.org/Vol-2267/470-474-paper-90.pdf)], though they were not able to boast of results even half as good as those from STC.

The most recent addition to the domain of ASR is the XSLR Wav2Vec2 architecture, which was released for public use only a few months ago. Considering its recency, there are not yet many publications illustrating the cutting-edge applications of this model on the Russian language yet. The most notable resource is Huggingface's pre-trained cross-lingual speech recognition model `Wav2Vec2-Large-XLSR-53-Russian` which boasts a word error rate of 17.39% on the Common Voice Russian test set, which was comparable to other SOTA implementations.

We hope to achieve comparable results in our project.

### *Evaluation:*

We evaluated our ASR using word error rate (WER). We did not use Sentence Error Rate (SER), because it was not a suitable metric for our dataset. Intent/entity recognition rate was also irrelevant, as we were not aiming to detect intents of the speech. An alternative metric was character error rate (CER), but from the previous work we realized that WER was the standard metric of evaluation for most ASR models in the Russian domain especially. Therefore, to allow for easier comparison with other models, we believed that WER was the most suitable metric for our project.

WER was calculated as follows: 

Word Error Rate = (Substitutions + Insertions + Deletions) / Number of Words Spoken

Substitutions are anytime a word gets replaced (for example, “twinkle” is transcribed as “crinkle”).
Insertions are anytime a word gets added that wasn’t said (for example, “trailblazers” becomes “tray all blazers”).
Deletions are anytime a word is omitted from the transcript (for example, “get it done” becomes “get done”).

In our project we used `load_metric` from `datasets`. 

First we defined the metric as:

`wer_metric = load_metric("wer")`

Then, we called it's `compute()` function within a separate function for calculating `WER`.

### *Challenges:*

Some challenges we faced when trying to train the model were as follows:

- Modifying the OpenSLR data to be compatible with the HuggingFace Notebook was too difficult and would take too much time, so we used the Common Voice dataset instead, which worked with the notebook off-the-shelf.
- The Russian dataset had some configurations that were different from the Turkish dataset. We had to modify the `speech_file_to_array_fn` to work with the batches in the dataset. We were also unable to work with batch sizes greater than 1, which made the data loading and training really slow. (finally we were able to use larger batch sizes)
- The notebook would often crash due to RAM shortage. We had to upgrade to Colab Pro in order to increase RAM, and once we did that, we still ran out of disk space when using a CPU or GPU runtime. Finally, the TPU runtime had enough disk space to load all of the data and start training, but we were unable to leverage the TPU cores for training and inadvertently trained on CPU; training on CPU was too slow (an estimated 279 hours). Then, we attempted to set the device to TPU for the model in an attempt to reduce training time. We were unfamiliar with the parallelization of TPUs and tried to train using one core, which resulted in a memory error for that TPU core. 


### *Conclusion:*

The ultimate goal is to create a tool that automatically translates Russian speech to English text in the future. If we succeed with the ASR portion, this project will be a great starting point for that goal.
