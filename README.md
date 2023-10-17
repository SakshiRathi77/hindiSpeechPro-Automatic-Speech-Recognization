# hindiSpeechPro

# Objective

The primary objective of this project is to build a robust and accurate ASR system that can effectively transcribe spoken Hindi into text.The motivation behind creating an ASR system for the Hindi language stems from the need to bridge the gap between technology and the diverse linguistic communities in India. Hindi, being one of the most widely spoken languages in the country, deserves a robust voice recognition system that can cater to the needs of millions of native speakers.


### Workflow of project
<img width="920" alt="flowchart" src="https://github.com/SakshiRathi77/hindiSpeechPro-Automatic-Speech-Recognization/assets/78577141/60ae0bbc-c5e8-4485-bb64-8c9bf90562f5">

### Technology Utilized:
Used cutting-edge technologies, specifically Facebook's Wav2Vec2 and OpenAI's Whisper, which are renowned for their proficiency in speech recognition tasks <br>
###  Dataset :[My dataset](https://huggingface.co/datasets/SakshiRathi77/ASR_CV15_Hindi_wav_16000)<br>

# Wav2vec2
<img width="920" alt="flowchart" src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/xls_r.png">
For the Facebook Wav2Vec2 model fine-tuned on the provided dataset, the following key performance metrics were achieved on the evaluation set:<br>
Loss: 0.3691<br>
Word Error Rate (WER): 32.85%<br>
Character Error Rate (CER): 8.75%<br>

<a href="https://huggingface.co/SakshiRathi77/wav2vec2-large-xlsr-300m-hi-kagglex">
  <img width="824" alt="wav2vec2 model" src="https://github.com/SakshiRathi77/hindiSpeechPro-Automatic-Speech-Recognization/assets/78577141/cdb74b4b-f6d4-448e-886b-86b7c1d06be8">
</a>

# whisper
<img width="920" alt="flowchart" src="https://raw.githubusercontent.com/openai/whisper/main/approach.png">
The fine-tuned Whisper ASR model, on the other hand, exhibited exceptional performance, achieving the following metrics:<br>
Word Error Rate (WER): 13.9913%<br>
Character Error Rate (CER): 5.8844%<br>
<br>
Through this project, significant advancements have been made in the domain of Hindi language speech recognition, 
contributing to improved communication technologies, accessibility, and usability for Hindi speakers across various applications and platforms

<a href="https://huggingface.co/SakshiRathi77/whisper-hindi-kagglex">
  <img width="824" alt="wav2vec2 model" src="https://github.com/SakshiRathi77/hindiSpeechPro-Automatic-Speech-Recognization/assets/78577141/c5f974d1-0fdc-4538-8696-52a4687a1222">
</a>

# Live Demo
[Whisper].(https://huggingface.co/spaces/SakshiRathi77/SakshiRathi77-Wishper-Hi-Kagglex)
<br>
[Wav2vec2](https://huggingface.co/spaces/SakshiRathi77/SakshiRathi77-Wav2Vec2-hi-kagglex)
# Reference
#### CTC 
[ctc](https://distill.pub/2017/ctc/)
##### Transformer
[transformers](https://github.com/huggingface/transformers)
##### Research Paper:<br>
[wav2vec-2-0-a-framework-for-self-supervised](https://paperswithcode.com/paper/wav2vec-2-0-a-framework-for-self-supervised)<br>
        [illustrated-wav2vec-2.html](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html)<br>
               [fine-tune-wav2vec2-english](https://huggingface.co/blog/fine-tune-wav2vec2-english)<br>
               [wav2vec2-with-ngram](https://huggingface.co/blog/wav2vec2-with-ngram)<br>
               [/attention-is-all-you-need-discovering-the-transformer-paper](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634)<br>
               [Whisper](https://cdn.openai.com/papers/whisper.pdf)<br>
               


