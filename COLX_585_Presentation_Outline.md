### COLX 585 
### Presentation Slides

1. Slide 1:
   - Project title (Team members): Russian Automatic Subtitling System
   
2. Slide 2:
   - Presentation Outline
   
3. Slide 3:
   - Motivation & Objective: 
   a. Developing an end-to-end pipeline for a model that is trained on XLSR-Wav2Vec2 to perform a basic ASR (Russian audio to Russian text), with the objective to improve the WER score compared to the Russian-53 model. 
   
   b. Assessing if silver-standard (model output) transcriptions do well in a machine translation model, and see if the final output is at all correct and usable.
   
4. Slide 4:
   - Literature Review
   a. The history of Russian ASR goes back to the Soviet Union with KGB backing the speech recognition research
   
   b. In A speaker-independent system for spontaneous speech recognistion was implemented (Apeech Technology Center, STC)
   
   c. A follow-up study by the same research group developed a model that has further improvement by increasing the WER where they combinded BLSTM and RNN
   
   d. Another research in the areas was based on Mozilla's/Baidu's DeepSpeech model, though they were not able to improve the result nearlly half as good as the result from STC
   
   e. The XSLR Wav2Vec2 architecture is based on Huggingface's pre-trained cross-lingual speech recognition model Wav2Vec2-Large-XLSR-53-Russian increased word error rate of 17.39% from on the Common Voice Russian test set compared to the previous highest WER which was 16.4%
   
   f. The most recent development is presented by Facebook AI Research's very new Speech-to-Text modelling system, which implements an end-to-end multilingual ST system using the CoVoST project datasets. The WER scores for Russian to English was around 31.4%.
   
   
  
5. Slide 5:
   - Challenges
   
   a. Modifying the OpenSLR data to be compatible with the HuggingFace Notebook was too difficult and would take too much time, so we used the Common Voice dataset instead, which worked with the notebook seamlessly.
   
   b. The Russian dataset had some configurations that were different from the Turkish dataset. We had to modify the speech_file_to_array_fn to work with the batches in the dataset. We were also unable to work with batch sizes greater than 1, which made the data loading and training really slow. (finally we were able to use larger batch sizes)
   
   e. The notebook would often crash due to RAM shortage. We had to upgrade to Colab Pro in order to increase RAM, and once we did that, we still ran out of disk space when using a CPU or GPU runtime. 
   
   f. The TPU runtime had enough disk space to load all of the data and start training, but training on CPU was way too slow (an estimated 279 hours), and we could not access the GPU in a TPU runtime. The device was set  to TPU for the model in an attempt to reduce training time. We were unfamiliar with the parallelization of TPUs and tried to train using one core, which resulted in a memory error for that TPU core. 

   
6. Slide 6:
   - Project status
   
   We are currently investigating how to parallelize our dataset and the training onto multiple cores.
   
7. Slide 7:
   - Outcomes and Results
   
8. Slide 8:
   - Acknowledgment

        
       