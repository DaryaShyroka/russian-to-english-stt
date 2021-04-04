## Project proposal

### *Introduction:* 

Originally, we wanted to create an end-to-end system for automatic subtitle generation. We wanted to try out two approaches:  

1) Training an ASR system to first convert the Russian audio to Russian text, and then doing Neural Machine Translation on the Russian text to translate it to English.

2) Training a system to generate English subtitles straight from the Russian audio, without translating it in between.

However, we would need a lot of time and data to train such a system. We would need about 10,000 hours of clean audio, which we do not have. Given the time constraints on this project, and the lack of sufficient audio, we will not be attempting to complete the full end-to-end pipeline. *Instead, we plan to do the first step, which will be Russian Automatic Speech Recognition*. This task will entail taking clean Russian audio from OpenSLR and outputting text that corresponds to the words in the audio. In the future beyond this course, we intend to continue this project and complete the full pipeline.

### *Motivation and Contributions/Originality:*

The motivation for this project is to improve existing automatic subtitling systems. Currently, auto-translated subtitles on YouTube produce very poor results, so we wanted to see if we could do better. There is a lot of media in Russian that should be accessible to speakers of other languages and is not, because it is not translated. A high-quality automatic subtitling system would make content accessible without human translators having to spend many many hours. Our work on the ASR part of the pipeline will add to existing trained models, making more resources for those working with Russian speech data and ASR models. It will also help us complete part of the pipeline on the way to our overarching goal. 

### *Data:*

We will be using data from the OpenSLR organization. The OpenSLR Russian LibriSpeech (RuLS) dataset contains 98 hours of Russian speech data from public domain audiobooks. The audio is all recorded in quiet environment with clear speech and no interference from background noise or music. We will download the data from the site https://openslr.org/96/ and we will store it on our personal laptops or on Google Drive.

### *Engineering:*

We will be using Google Colab as our primary computing infrastructure for training and testing. We may supplement Colab with Jupyter notebooks on our personal computers. The datasets will be stored on a shared personal Google Drive folder. We will use the PyTorch framework. We will use Huggingface's pre-trained cross-lingual speech recognition model `Wav2Vec2-Large-XLSR-53-Russian`, which is based on the XLSR `Wav2Vec2forCTC` model, as the backbone of the ASR pipeline, and fine-tune it for our use case, as demonstrated in the [ASR tutorial notebook for fine-tuning a similar model for Turkish](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb).

### *Previous Works (minimal):*

Research in the domain of Russian ASR has been of great interest for several decades now. The history of Russian ASR goes as far back as the Soviet Union with KGB backing for speech recognition research. A descendent of this state-backed research is the organization known in the West as [SpeechPro](https://www.wikiwand.com/ru/%D0%A6%D0%B5%D0%BD%D1%82%D1%80_%D1%80%D0%B5%D1%87%D0%B5%D0%B2%D1%8B%D1%85_%D1%82%D0%B5%D1%85%D0%BD%D0%BE%D0%BB%D0%BE%D0%B3%D0%B8%D0%B9) (AKA., the Speech Technology Centre), which is regarded as the best solution on the market for Russian ASR [[1](https://doi.org/10.1007/978-3-319-23132-7_29), [2](https://doi.org/10.1007/978-3-319-43958-7_13)]. Research in this domain has really skyrocketed only within the last 5-6 years, so it is still a hotbed for novel techniques with much room for improvement. Some notable implementations in recent years have been based on Mozilla's/Baidu's DeepSpeech model [[3](https://github.com/GeorgeFedoseev/DeepSpeech), [4](http://ceur-ws.org/Vol-2267/470-474-paper-90.pdf)]. Considering how new the XSLR Wav2Vec2 architecture is, there are not yet many publications illustrating the cutting edge applications of this model on the Russian language yet. The most notable resource is Huggingface's pre-trained cross-lingual speech recognition model `Wav2Vec2-Large-XLSR-53-Russian` which boasts a word error rate of 17.39% on the Common Voice Russian test set, which is comparable to other SOTA implementations. We hope to achieve comparable results in our project.

### *Evaluation:*

We will evaluate our ASR using word error rate (WER). We will not use Sentence Error Rate (SER), because it is not a suitable metric for our dataset. Intent/entity recognition rate is also irrelevant, as we are not aiming to detect intents of the speech. An alternative metric is character error rate (CER), but it is becoming that WER is the standard metric of evaluation for most ASR models in the Russian domain especially. Therefore, to allow for easier comparison with other models, we believe that WER is the most suitable metric for our project.

### *Conclusion (optional):*

The ultimate goal is to create a tool that automatically translates Russian speech to English text in the future. If we succeed with the ASR portion, this project will be a great starting point for that goal.
