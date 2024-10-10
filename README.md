# LLM_Embedding_Attack

This is the repository of "[Adversarial Attacks and Defenses in Large Language Models: Old and New Threats
](https://arxiv.org/abs/2310.19737)" by [Leo Schwinn](https://schwinnl.github.io/), David Dobre, [Stephan Günnemann](https://www.professoren.tum.de/guennemann-stephan), and [Gauthier Gidel](https://gauthiergidel.github.io/) which was accepted at the *NeurIPS 2023 ICBINB Workshop* for spotlight presentation.

See ```embedding_attack_toxic.py```

## Unlearning

This repository also contains the code of "[Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space](https://arxiv.org/pdf/2402.09063)" by [Leo Schwinn](https://schwinnl.github.io/), David Dobre, Sophie Xhonneux, [Gauthier Gidel](https://gauthiergidel.github.io/), and [Stephan Günnemann](https://www.professoren.tum.de/guennemann-stephan).

See ```embedding_attack_unlearning.py``` and ```unlearning_utils.py```
The script can also be used for toxicity experiments with individual and universal attacks.

## New Version compatible with HarmBench

A stronger version of embedding space attacks integrated into the Harmbench framework can be found here:

[https://github.com/SchwinnL/circuit-breakers-eval](https://github.com/SchwinnL/circuit-breakers-eval)

## Disclamimer 

In this work, we want to highlight the safety vulnerabilities of LLMs. As powerful LLM assistants are readily available and machine-learning robustness has been an unsolved research problem for the last decade, we believe that the best way to approach this problem is through culminating awareness. 

## Content

The repository contains the code to conduct embedding space attacks on LLMs. 
Typically, adversarial attacks in the embedding space of LLMs are not considered, as most threat models concentrate on attacks that can be transferred to closed-source models utilized through an API which usually demand natural language input. However, in the case of open-source LLMs, an attacker is not restricted to attack the model in natural language space. 

A range of malicious actions can be performed without the need to use closed-source models through restricted APIs, or interact with users of LLM-integrated apps. This includes the distribution of hazardous knowledge (e.g. instructions for creating malicious software), promoting harmful biases, spreading misinformation, building ``troll'' bots to respond to real users on social media, etc. 

Within embedding space attacks we exploit that once an LLM starts giving an affirmative response, it is likely to remain in that ``mode'' and continue to provide related outputs.

## Experiments

The experiments in "Adversarial attacks and defenses in large language models: Old and new threats" can be reproduced by running 
```
embedding_attack_toxic.py
```

Some code snippets are taken from https://github.com/llm-attacks/llm-attacks

For experiments related to "Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space" see

```
embedding_attack_unlearning.py
```

## Cite

If you use this repository in your research, please consider citing:

```	
@article{schwinn2023adversarial,
  title={Adversarial attacks and defenses in large language models: Old and new threats},
  author={Schwinn, Leo and Dobre, David and G{\"u}nnemann, Stephan and Gidel, Gauthier},
  journal={arXiv preprint arXiv:2310.19737},
  year={2023}
}

@article{schwinn2024soft,
  title={Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space},
  author={Schwinn, Leo and Dobre, David and Xhonneux, Sophie and Gidel, Gauthier and Gunnemann, Stephan},
  journal={arXiv preprint arXiv:2402.09063},
  year={2024}
}
```
