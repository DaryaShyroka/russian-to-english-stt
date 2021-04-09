# summary of the lab discussion:

Peter is also going to upload the topics of the Lab. 

1. to use the data from the shared google drive:
  - right click on the folder
  - click on the "Add shortcut to dive"
  - make a folder with a suitable name and store the data there
  - when opening the "Fine-Tune XLSR-Wav2Vec2 on Turkish ASR with 🤗 Transformers.ipynb" in you google colab you will see the folder with the data  
  
2. Using a language model helps however maybe it is a good idea starting without a language model, then if you have time add a language model. If you end up not using a language model then you can explain the reason.

3. If you are using a language model the downside is it should be incorporated with the decoder

4. Remember to save cahckpoints in order to keep track.

5. Expect minimum of one day of training.

6. If you are planning to train over time use the trick in the tutorial??.

7. Make sure to reach out to your mentor for a half an hour feedback.

8. What should we expect for the WER for our basic model?

9. A code to use when we want to use data that is not available on the hugging face:


```def map_to_array(batch):
     speech, _ = sf.read(batch["file"])
     batch["speech"] = speech
     return batch

 # load dummy dataset and read soundfiles
 ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
 ds = ds.map(map_to_array)
```


10. A baseline for Russian WER:

https://huggingface.co/anton-l/wav2vec2-large-xlsr-53-russian

It reported a WER of 17.39%.
