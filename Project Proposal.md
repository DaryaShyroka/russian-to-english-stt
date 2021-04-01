## Project proposal

Describe your proposed project. Your should include the following information and anything else you deem relevant:

### *Introduction:* 

- Where you introduce the task/problem you will work on. This answers the question: ``What is the nature of the task?`` 
(e.g., sentiment analysis, machine translation, language generation, style transfer, etc.?). Please explain ``what the task entails`` 
(e.g., taking in as input a sequence of text from a source language and turning it into a sequence of sufficiently equivalent meaning in target language). 

The problem we will be working on is automatic subtitle generation. We will be attempting to add English subtitles onto Russian videos. We will be trying out two approaches:  

1) Training an ASR system to first convert the Russian audio to Russian text, and then doing Neural Machine Translation on the Russian text to translate it to English.

2) Attempting to learn the English subtitles from the Russian audio directly.

### *Motivation and Contributions/Originality:*
- ``What is the motivation for pursuing this project?`` In other words, ``why is the project important``. This could be because this is a ``(relatively) new problem`` where you are using an existing method on (e.g., translating tweets where the language is noisy and doesn't usually obey `standard` rules). This could also be because the problem is ``timely`` (e.g., carrying out ``sentiment analysis on COVID-19`` data, given the negative impact of the pandemic). Further, this could be because the problem is ``socially motivated`` and/or ``remains unsolved`` (e.g., ``toxic`` and/or ``racist`` comments on social media, given their pervasively harmful impact).  
- What do you hope your ``contribution`` will be? Here, you could aim at providing a ``better system`` than what exists (e.g., more robust MT), an application on new data (possibly within a new domain) (e.g., ``tweet intent and topic detection on COVID-19 data``), a system that delivers insights on a new topic (e.g., ``scale and sentiment in tweets in different location as to COVID-19``), etc. 
### *Data:*
- What kind of ``data`` will you be using? ``Describe the corpus``: genre, size, language, style, etc. Do you have the data? Will you acquire the data? How? Where will you ``store`` your data? 
### *Engineering:*
- What ``computing infrastructure`` will you use? Personal computers? Google Colab? Google Cloud TPUs?
- What ``deep learning of NLP (DL-NLP)`` methods will you employ? For example, will you do ``text classification with BERT?``, ``MT with attention-based BiLSTMs``, ``language generation with transformers``, etc.? 
- ``Framework`` you will use. Note: You *must* use PyTorch. Identify any ``existing codebase`` you can start off with. Provide links.
### *Previous Works (minimal):*
- Refer to one or more projects that people have carried out that may be somewhat relevant to your project. This will later be expanded to some form of ``literature review``. For the sake of the proposal, this can be very brief. You are encouraged to refer to previous work here as a way to alert you to benefiting from existing projects. Provide links to any such projects.
### *Evaluation:*
- How will you ``evaluate`` your system? For example, if you are going to do MT, you could evaluate in ``BLEU``. For text classification, you can use ``accuracy`` and ``macro F1`` score. If your projects involves some interpretability, you could use ``visualization`` as a vehicle of deriving insights (and possibly some form of ``accuracy`` as approbriate).
### *Conclusion (optional):*
- You can have a very brief conclusion just summarizing the goal of the proposal. (2-3 sentences max).
