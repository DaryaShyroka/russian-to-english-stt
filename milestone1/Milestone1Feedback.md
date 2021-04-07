
# Milestone 1 Feedback:

### Like:

- I like that you have addressed the issue of scope in your proposal. You have clear areas to expand if things prove easy!
- Thorough team contract!
- This may seem somewhat silly, but thank you very much for using the given template, your project proposal is well formatted and offers good coverage of all the things you should consider in the project!
- I like that you've started looking at the HuggingFace(VonPlatten) Wav2vec2 tutorial.
- Great start to the background research section!


### Questions:

- Why do automatic subtitling systems do poorly for Russian?
- How will you be calculating WER?
- How will you be pre-processing the transcripts? Does Russian require any specific text processing procedures?
- Will you be using a language model on your outputs?
- Will you use any decoding strategies on the output? (e.g. beam search)
- What is the gender / dialect breakdown of the RuLS dataset? Do you have other info about it?

### Recommended Next Steps / Direct Feedback:  

- The model that you have identified Wav2Vec2-Large-XLSR-53-Russian *is* fine-tuned. It has a 17% WER on test. consider using this just as a comparison, but not fine-tuning it further (maybe this is reflective of an ambiguity in your proposal, are you starting with vanilla XLSR as your starting point?)


### Asides
- RE: WER vs. CER. You're right that WER is definitely the go to. CER is still common for some languages though (e.g. Chinese).
