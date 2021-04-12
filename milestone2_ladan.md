

# COLX 585 - Trends in CL | Milestone 2
---


## Project Progress Report
rubric={reasoning:5,writing:3}


We are going to perform Russian Automatic Speech Recognition following by translation to English. This takes clean Russian audio from OpenSLR and outputting text that corresponds to the words in the audio, then translating it to audio. We are going to explore Fairseq [[1]https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/docs/covost_example.md which is a recent model that can perform both steps (text creation and translation) together.]

Discussion on the feedbacks received for Milestone 1:

- Why do automatic subtitling systems do poorly for Russian?

1. According to Prudnikov  et. al. (2017) high channel and speaker variance, additive and non-linear distortions accents and emotional speech, diversity of speaking styles, speech rate variance, reductions and weakened articulation are the challenges on processing Russian language.
2. In subtitling we are dealing with two stages one is ASR the other is translation. There are still major challenges in dealing with the translation stage. Also, in the end-to-end process where a language model can improve the results.
3. There is also the problem of error propagation, when an error happens during the process, the impact of this error is conveyed to other steps of processing also.

- How will you be calculating WER?

The WER is the word level of the Levenshtein distance, comparing to the origin of it where it was at the phoneme level. Using WER we can compare different systems also make improvements within one system. It is important to know that this metric do not provide any information on the nature of translation errors and we need to identify the main source(s) of error. Based on WER first we align the recognized word sequence with the reference (spoken) word sequence. Word error rate (WER) can be computed as: WER = (S + D + I) / N = (S + D + I) / (S + D + C) where S is the number of substitutions, D is the number of deletions, I is the number of insertions, C is the number of correct words, N is the number of words in the reference (N=S+D+C). WER's output is always a number between 0 and 1. [[2]https://huggingface.co/metrics/wer]


- How will you be pre-processing the transcripts? Does Russian require any specific text processing procedures?

In our case as we are using the OpenSLR Russian LibriSpeech (RuLS) dataset contains 98 hours of Russian speech data from public domain audiobooks data the data is already processed.

- Will you be using a language model on your outputs?

For the primary analysis we won’t use language model but time permitting we explore the implication of a language model.

- Will you use any decoding strategies on the output? (e.g., beam search)

This depends on the model we choose.

- What is the gender / dialect breakdown of the RuLS dataset? Do you have other info about it?

The data consists of mixed female and male audios and there is no specific dialect.





# Summary of paper 1:

In this paper [[3](https://doi.org/10.1007/978-3-319-23132-7_29)] Prudnikov  et. al. (2017) investigated the spontaneous speech recognition of Russian language. This is the most difficult area in automatic speech recognition (ASR). The corpus that is used for English language is Switchboard-1 corpus (SWB), which comprises 300 hours of telephone conversations. Kaldi Switchboard is usually used to model these data. When the model is used for English data the word error rate (WER) was low, around 10.4 %.  However, when the same method was used on Russian data it was less than this baseline. In order to improve the result two methods including i-vector based deep neural network adaptation and speaker-dependent bottleneck features were used that provided 8.6% and 11.9% relative word error rate reduction over the baseline system respectively. The challenges faced in processing Russian language is due to the properties of Russian spontaneous conversational speech: high channel and speaker variance, additive and non-linear distortions accents and emotional speech, diversity of speaking styles, speech rate variance, reductions and weakened articulation. One of the most obvious ways to deal with such variability is enlarging the training dataset. However, this method is expensive and labour-intensive. 


