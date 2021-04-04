## Project proposal

### *Introduction:* 

Originally, we wanted to create an end-to-end system for automatic subtitle generation. We wanted to try out two approaches:  

1) Training an ASR system to first convert the Russian audio to Russian text, and then doing Neural Machine Translation on the Russian text to translate it to English.

2) Training a system to generate English subtitles straight from the Russian audio, without translating it in between.

However, we would need a lot of time to train such a system. We would also need about 10,000 hours of clean audio, which we do not have. Given the time constraints on this project, and the lack of sufficient audio, we will not be attempting to complete the full end-to-end pipeline. Instead, we plan to do the first step, which will be Russian Automatic Speech Recognition. This task will entail taking clean Russian audio from OpenSLR and outputting text that corresponds to the words in the audio. In the future beyond this course, we intend to continue this project and complete the full pipeline.

### *Motivation and Contributions/Originality:*

The motivation for this project is to improve existing automatic subtitling systems. Currently, auto-translated subtitles on YouTube produce very bad results, so we wanted to see if we could do better. There is a lot of media in Russian that should be accessible to speakers of other languages and is not, because it is not translated. A high-quality automatic subtitling system would make content accessible without human translators having to spend many many hours. Our work on the ASR part of the pipeline will add to existing trained models, making more resources for those working with Russian speech data and ASR models. It will also help us complete part of the pipeline on the way to our overarching goal. 

### *Data:*

We will be using data from OpenSLR. It contains 98 hours of Russian speech data from public domain audiobooks. The audio is all recorded in quiet environment with clear speech and no interference from background noise or music. We will download the data from the site https://openslr.org/96/ and we will store it on our personal laptops or on Google Drive.

### *Engineering:*
- What ``computing infrastructure`` will you use? Personal computers? Google Colab? Google Cloud TPUs?
- What ``deep learning of NLP (DL-NLP)`` methods will you employ? For example, will you do ``text classification with BERT?``, ``MT with attention-based BiLSTMs``, ``language generation with transformers``, etc.? 
- ``Framework`` you will use. Note: You *must* use PyTorch. Identify any ``existing codebase`` you can start off with. Provide links.

We will be using Google Colab as well as Jupyter notebooks on our personal computers. We will use Wav2Vec2 for ASR. We will use the PyTorch framework.

### *Previous Works (minimal):*
- Refer to one or more projects that people have carried out that may be somewhat relevant to your project. This will later be expanded to some form of ``literature review``. For the sake of the proposal, this can be very brief. You are encouraged to refer to previous work here as a way to alert you to benefiting from existing projects. Provide links to any such projects.

https://ieeexplore.ieee.org/abstract/document/8001709
https://link.springer.com/chapter/10.1007/978-981-15-3514-7_1
https://www.aclweb.org/anthology/W19-5209.pdf

### *Evaluation:*

We will evaluate our ASR using Word Error Rate (WER). We will not use Sentence Error Rate (SER), because it is not a suitable metric for our dataset. Intent/entity recognition rate is also irrelevant, as we are not aiming to detect intents of the speech. Therefore we believe that WER is the most suitable metric for our task.

### *Conclusion (optional):*

We are aiming to create a tool that automatically translates Russian speech to English text in the future. If we succeed with the ASR portion, this project will be a great starting point for that goal.
