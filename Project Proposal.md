## Project proposal

Describe your proposed project. Your should include the following information and anything else you deem relevant:

### *Introduction:* 

- Where you introduce the task/problem you will work on. This answers the question: ``What is the nature of the task?`` 
(e.g., sentiment analysis, machine translation, language generation, style transfer, etc.?). Please explain ``what the task entails`` 
(e.g., taking in as input a sequence of text from a source language and turning it into a sequence of sufficiently equivalent meaning in target language). 

Originally, we wanted to create an end-to-end system for automatic subtitle generation. We wanted to try out two approaches:  

1) Training an ASR system to first convert the Russian audio to Russian text, and then doing Neural Machine Translation on the Russian text to translate it to English.

2) Training a system to generate English subtitles straight from the Russian audio, without translating it in between.

However, we would need a lot of time to train such a system. We would also need about 10,000 hours of clean audio, which we do not have. Given the time constraints on this project, and the lack of sufficient audio, we will not be attempting to complete the full end-to-end pipeline. Instead, we plan to do the first step, which will be Russian Automatic Speech Recognition. This task will entain taking clean Russian audio from OpenSLR and outputting text that corresponds to the words in the audio. In the future beyond this course, we intend to continue this project and complete the full pipeline.

### *Motivation and Contributions/Originality:*
- ``What is the motivation for pursuing this project?`` In other words, ``why is the project important``. This could be because this is a ``(relatively) new problem`` where you are using an existing method on (e.g., translating tweets where the language is noisy and doesn't usually obey `standard` rules). This could also be because the problem is ``timely`` (e.g., carrying out ``sentiment analysis on COVID-19`` data, given the negative impact of the pandemic). Further, this could be because the problem is ``socially motivated`` and/or ``remains unsolved`` (e.g., ``toxic`` and/or ``racist`` comments on social media, given their pervasively harmful impact).  
- What do you hope your ``contribution`` will be? Here, you could aim at providing a ``better system`` than what exists (e.g., more robust MT), an application on new data (possibly within a new domain) (e.g., ``tweet intent and topic detection on COVID-19 data``), a system that delivers insights on a new topic (e.g., ``scale and sentiment in tweets in different location as to COVID-19``), etc. 

The motivation for this project is to improve existing automatic subtitling systems. Currently, auto-translated subtitles on YouTube produce very bad results, so we wanted to see if we could do better. 

Our contribution: ASR for Russian (adding to existing data), completion of part of the pipeline

### *Data:*
- What kind of ``data`` will you be using? ``Describe the corpus``: genre, size, language, style, etc. Do you have the data? Will you acquire the data? How? Where will you ``store`` your data? 

OpenSLR has data - all audio recorded in quiet environment with clear speech (no interference from background music): 98 hours.


### *Engineering:*
- What ``computing infrastructure`` will you use? Personal computers? Google Colab? Google Cloud TPUs?
- What ``deep learning of NLP (DL-NLP)`` methods will you employ? For example, will you do ``text classification with BERT?``, ``MT with attention-based BiLSTMs``, ``language generation with transformers``, etc.? 
- ``Framework`` you will use. Note: You *must* use PyTorch. Identify any ``existing codebase`` you can start off with. Provide links.

We will be using Google Colab as well as Jupyter notebooks on our personal computers. We will use Wav2Vec2 and OpenSLR for ASR. We will use the PyTorch framework.

### *Previous Works (minimal):*
- Refer to one or more projects that people have carried out that may be somewhat relevant to your project. This will later be expanded to some form of ``literature review``. For the sake of the proposal, this can be very brief. You are encouraged to refer to previous work here as a way to alert you to benefiting from existing projects. Provide links to any such projects.

https://ieeexplore.ieee.org/abstract/document/8001709
https://link.springer.com/chapter/10.1007/978-981-15-3514-7_1
https://www.aclweb.org/anthology/W19-5209.pdf

### *Evaluation:*
- How will you ``evaluate`` your system? For example, if you are going to do MT, you could evaluate in ``BLEU``. For text classification, you can use ``accuracy`` and ``macro F1`` score. If your projects involves some interpretability, you could use ``visualization`` as a vehicle of deriving insights (and possibly some form of ``accuracy`` as approbriate).

We will evaluate our ASR using Word Error Rate (WER). 

### *Conclusion (optional):*
- You can have a very brief conclusion just summarizing the goal of the proposal. (2-3 sentences max).
