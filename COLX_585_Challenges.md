### COLX 585
### Project Challenges: Weeks 1-2-3


Objective 1:

Training an ASR system to convert the Russian audio to Russian text and translating the Russian text to English text using Neural Machine.

Training a system to generate English subtitles straight from the Russian audio, eliminating the translation step.

Challenge One: 10,000 hours of clean audio, and a long time is needed for training such system. The time limit of this project and lack of this extensive data were barriers in pursuing these objectives.

 
Modified Objective 2: 

We plan to do the first step, which will be Russian Automatic Speech Recognition. This task will entail taking clean Russian audio from OpenSLR and outputting text that corresponds to the words in the audio.

The data downloaded and saved in a shared file in google Doc. The HuggingFace Notebook on Turkish Automatic Speech Recognition was used to process the OpenSLR data.

Challenge Two: The HuggingFace Notebook was using common_voice data which was accessed using 🤗 Datasets' simple API to download the data. After many attempts modifying this dataset loader to be compatible to using new datasets was not possible.

Modified Objective 3:

We planned to use the common voice Russian dataset which is avaialable from HuggingFace Datasets and modify the HuggingFace notebook to work with the new dataset.

Challenge Three: 

- Russian dataset had a new configuration that was different from Turkish dataset. We modified the code to become compatible in reading the new dataset.

- At this stage the Notebook ran to 4- 25% (on different attempts) and crashed due to RAM shortage.

   
   
 
