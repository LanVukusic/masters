# Research overiew

## Online speech modeli

- [Enformer STT](https://docs.pytorch.org/audio/main/tutorials/online_asr_tutorial.html)
- [Enformer - paper](https://arxiv.org/pdf/2010.10759)
- [Emformer github](https://github.com/pytorch/audio/tree/main/examples/asr/emformer_rnnt)

Hubert je bert like clustering/clasification za audio

- [Huber](https://huggingface.co/docs/transformers/en/model_doc/hubert)

Moonshine je nek model ka se uporablja za realtime SR

- [Moonshine](https://arxiv.org/pdf/2410.15608)
- Kao optimizirajo speed - lahko bi se kej naučil od teh rotary encodingov. zelo speech centric

- Moshi - za realtime speech dialogue? mogoce interesting
- 160ms latence
- audio to semantic representation
- [Paper](https://arxiv.org/pdf/2410.00037)

- Parakeet
- Nvidia NeMo fora
- Fast Conformer - conecctionist temporal classification
- [Fast confomer - nvidia](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#fast-conformer)
- [Paper](https://arxiv.org/pdf/2509.14128), par dobrih for, par optimizacij ...
- [ctc, mmybe useful. boljsi w2vec](https://arxiv.org/pdf/2109.06870)

- [t5 basically...malo meh](https://arxiv.org/pdf/2110.07205)

- [eat](https://github.com/cwx-worst-one/EAT)
- [eat paper](https://arxiv.org/abs/2401.03497)

Spectorgrame samo daš v neke vision modele in je done.... i dont like this.  

[AST - audio spectrogram](https://huggingface.co/docs/transformers/en/model_doc/audio-spectrogram-transformer)

UNI - speech

- labeled in unlabeled pretraining
- [UniSpeech paper](https://huggingface.co/papers/2101.07597)

Multilingual Expressive and Streaming Speech Translation  

- [Paper](https://scontent-vie1-1.xx.fbcdn.net/v/t39.2365-6/406941874_247486308347770_2317832131512763077_n.pdf?_nc_cat=102&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=C7rVsNh8AXYQ7kNvwGn6r2M&_nc_oc=AdlmBeOgQA8Z9-XrDMy7VmG439ZBUiikJe6FkQ9C_eOPvPNSNneLr6sxOoRK2MTELRr2dXD5Aezd0jJ5eUYuPeh9&_nc_zt=14&_nc_ht=scontent-vie1-1.xx&_nc_gid=eRuFEpxhfZ9OII06wVvyfQ&oh=00_AfiXbQqfhk311jqr68O6LG3PiyinIWXbO6LXoeSH50kkSQ&oe=69150869)
- [Efficient monotonic multiheaded attention](https://arxiv.org/abs/2312.04515)
  - to generate low-latency target translations without waiting for complete source utterance... to bi znal bit usefull princip?
  - predicta token al pa čaka in consuma več contexta. uporabno za streaming

WavLM - še en self supervised approach

- [Paper](https://arxiv.org/pdf/2110.13900)

## music segmentation / semantic analysis

- [DEEP EMBEDDINGS AND SECTION FUSION IMPROVE MUSIC SEGMENTATION - paper](https://archives.ismir.net/ismir2021/paper/000074.pdf)

- Ruptures  je nek lib ka dela segmentation, sam je malo meh
- [Ruptures getting started website](https://centre-borelli.github.io/ruptures-docs/examples/music-segmentation/)

- [INA speech segmenter](https://github.com/ina-foss/inaSpeechSegmenter?tab=readme-ov-file)
  - Detecta music, noise, speech: male / female

- TLDR tega res ni tok kokr bi si mislu in sploh ni tega tko... dobrega

## Music diffusion

Ideja je, da online generiraš muziko in denoisas sample. Sampli ki so bližje tebi so bolj denoisani - 100% denoisani ko so LIVE in 100%noisy ko so "MAX" in the future  

- Baje je DDIM scheduling boljsi in hitrejsi od DDPM ja, kar omogoča diffusion v majn steppinh. probbly better?
- [DDPM vs DDIM](https://sachinruk.github.io/blog/2024-02-11-DDPM-to-DDIM.html)
  - [Matematika](https://sachinruk.github.io/blog/2024-02-11-DDPM-to-DDIM.html)
- [Zanimiv DIY audio difussion repo](https://github.com/teticio/audio-diffusion)
- [Go to članek za audio diffusion](https://stability.ai/research/stable-audio-efficient-timing-latent-diffusion)
  - 95 sec, stereo 44khz audio v < 1 sec
- [Zelo good audio codec - 44khz wav - latent](https://github.com/descriptinc/descript-audio-codec)
- [Music gen - Mousai](https://arxiv.org/pdf/2301.11757)
- [Audio X - karkoli v audio diffusion](https://arxiv.org/pdf/2503.10522)

## new stuff

- [Meta ASR model 96x realtime](https://github.com/facebookresearch/omnilingual-asr?tab=readme-ov-file)
- [CALM - continous token prediction](https://github.com/shaochenze/calm)
- [Magenta realtime](https://arxiv.org/pdf/2301.11325)
- [magenta](https://github.com/magenta/magenta-realtime)
- [Magenta paper - POGLEJ](https://arxiv.org/pdf/2508.04651)
- [LeVo - generation model in codec](https://arxiv.org/html/2506.07520v1)
  - Zanimiv codec - preveč lyrics focused

## Codeci

- [enCodec](https://arxiv.org/abs/2210.13438)
  - from [Audiocraft](https://github.com/facebookresearch/audiocraft/)
- [To je baje sam bolš od vsega. in EnCodeca in AudioStreama (google in fb)](https://arxiv.org/pdf/2411.19842)
- [MIMI - codec](https://huggingface.co/kyutai/mimi)
- [StableCodec - stableAI](https://github.com/Stability-AI/stable-codec)

## Ostalo

- [Differentiable DSP](https://github.com/magenta/ddsp)
  - za kalkulirat razne fine featurje iz modela.


