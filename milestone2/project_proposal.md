## Project proposal

### *Introduction:*

Originally for this project we had planned to perform Russian Automatic Speech Recognition, as part of a pipeline in an end-to-end system for a Russian Speech to English text translation system. This would entail Automatic Speech Recognition using XLSR-Wav2Vec2 fine-tuned on Russian speech data, followed by Machine Translation of the Russian transcriptions to English. However, further reading has brought to our attention a one-step Speech Translation system by Fairseq [[1](https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/docs/covost_example.md)] which is a very recent model that can perform both steps (text creation and translation) in an end-to-end model. We will compare the performance of these two models on a new dataset. We will also experiment with fine-tuning the vanilla XLSR-Wav2Vec2 model on a different Russian speech dataset (other than Common Voice, which it's already been fine-tuned on) to see how it performs. Finally, we are considering using subtitled movie/TV series data to see how a Speech Translation model might perform on a noisier dataset from a different domain.

### *Motivation:*

Our motivation for the different directions are as follows:

- If we used subtitled data, we would be creating a new dataset from a different domain that could be used for training future ASR and ST systems for Russian, which is a significant contribution because Russian does not have as many speech resources as English.
- If we compare different models we would be providing an analysis of which system works better for our applications, which is a novel endeavour.
- If we compare a model on a new dataset, we would be providing an analysis of how an existing system works on new data, and discuss its effectiveness in new domains. 

Overall, we will be learning to train a Machine Learning model to do a Speech Recognition and/or translation task, which would be a learning experience for all of us.

### *Data:*

Originally, when we had planned to do just Russian ASR without translation, we planned to use the OpenSLR Russian LibriSpeech (RuLS) dataset to fine-tune an XLSR-Wav2Vec2 model. The OpenSLR Russian LibriSpeech (RuLS) dataset contains 98 hours of Russian speech data from public domain audiobooks. The audio is all recorded in quiet environment with clear speech and no interference from background noise or music. The data is available on the site https://openslr.org/96/. 

However, now that we are doing End-To-End Speech Translation using Fairseq, we might also use the CoVost dataset, which is a multilingual speech-to-text translation corpus from 21 languages into English. We would use the subset of Russian speech to English text. This dataset contains 10.2 hours of training data, 9.0 hours of development data, and 8.2 hours of test data. This is a dataset of read speech data - participants were recorded reading donated sentences. The transcriptions of these sentences were sent to translators to translate into English. There are some sentences that were read out by multiple different speakers, presumably to reduce overfitting to a single speaker. The data is available for download on https://github.com/facebookresearch/covost.

We are also considering using subtitling data as training data. We currently have about 14 hours of a Russian show with manually written, gold standard English subtitles, which we could use as the training data. We already have the video files and subtitles downloaded as they were created by one of our team members a few years ago. We could also find more subtitle data online. In the case of subtitled data, we would need to extract the audio from the video files, and run a script that would separate the one large audio file into small audio files that match the timestamps in the subtitles. In this way, we would create a dataset which contains around one to two sentences in each audio file, with the target being the subtitle transcription. This would be similar to the OpenSLR RuLS dataset. However, this dataset would be much noisier than a read speech dataset. Speech in movies and TV is often combined with background noise and music. One of the interesting things we could investigate in this project would be the effect of noise in the dataset on speech recognition and how it affects WER.

In each case, we will store the data on our personal laptops and/or on a share Google Drive directory.

### *Engineering:*

In this project we use Google Colab as our primary computing infrastructure for developing models. The datasets are stored on a shared Google Drive directory. We use the PyTorch framework. A primary objective of this project is to investigate the performances of two cutting-edge ASR/speech translation systems. On one hand we have the Fairseq Speech2Text model which follows the end-to-end paradigm of converting source language audio to target language text in one self-contained model. On the other hand we use Huggingface's pre-trained cross-lingual speech recognition model `Wav2Vec2-Large-XLSR-53-Russian`, which is based on the XLSR `Wav2Vec2forCTC` model, forming one half of a speech translation pipeline. The other half of the pipeline involves a manually trained Russian text to English text translation model, which takes the output of the ASR model as its input.

### *Previous Works:*

Research in the domain of Russian ASR has been of great interest for several decades now. The history of Russian ASR goes as far back as the Soviet Union with KGB backing for speech recognition research. A descendent of this state-backed research is the organization known in the West as [SpeechPro](https://www.wikiwand.com/ru/%D0%A6%D0%B5%D0%BD%D1%82%D1%80_%D1%80%D0%B5%D1%87%D0%B5%D0%B2%D1%8B%D1%85_%D1%82%D0%B5%D1%85%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D0%B9) (AKA., the Speech Technology Centre (STC)), which is known for creating what has been regarded as the best solution on the market for Russian ASR [[1](https://doi.org/10.1007/978-3-319-23132-7_29), [2](https://doi.org/10.1007/978-3-319-43958-7_13)].

STC focuses heavily on spontaneous speech recognition of conversational Russian language, widely regarded as one of the most difficult areas of ASR as conversational speech is very noisy and error-prone. Conversational speech varies greatly based on the diversity of individual speakers' speech style, accent, speech rate variability, and the speaker's emotions. It might also contain informal words, such as slang, and might have slurred and poorly articulated words. Background noise and music might also have an effect. There are existing highly effective spontaneous speech recognition systems for English that achieve a WER between 8.0% and 14.1%.

In 2016, Prudnikov et. al implemented a speaker-independent system for Russian spontaneous speech recognition. This is more challenging for Russian as compared to English for several reasons, two of which are the lack of available datasets, and because Russian is an inflective language with much more unique words than English. They implemented an i-vector based deep neural network adaptation and speaker-dependent bottleneck features to achieve a WER of 25.1% (11.9% relative word error rate reduction over the baseline system).

The very next year, Prudnikov et. al [[2] (https://link.springer.com/chapter/10.1007%2F978-3-319-43958-7_13)] presented further improvements to the Russian spontaneous speech recognition system developed in the Speech Technology Center (STC).In this experiment, they improved upon the previous word error rate to achieve 16.4% by using acoustic modelling approaches, combined with deep BLSTM acoustic models and hypothesis rescoring with RNN-based language models.

Research in this domain has really skyrocketed only within the last 5-6 years, so it is still a hotbed for novel techniques with much room for improvement. Some other notable implementations from outside of Russia in recent years have been based on Mozilla's/Baidu's DeepSpeech model [[3](https://github.com/GeorgeFedoseev/DeepSpeech), [4](http://ceur-ws.org/Vol-2267/470-474-paper-90.pdf)].

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

### *Conclusion:*

The ultimate goal is to create a tool that automatically translates Russian speech to English text in the future. If we succeed with the ASR portion, this project will be a great starting point for that goal.
