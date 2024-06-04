# LLM_Embedding_Attack

This is the repository of "[Adversarial Attacks and Defenses in Large Language Models: Old and New Threats
](https://arxiv.org/abs/2310.19737)" by [Leo Schwinn](https://schwinnl.github.io/), David Dobre, [Stephan Günnemann](https://www.professoren.tum.de/guennemann-stephan), and [Gauthier Gidel](https://gauthiergidel.github.io/) which was accepted at the *NeurIPS 2023 ICBINB Workshop* for spotlight presentation.

## Unlearning

Unlearning experiments are now provided in 'embedding_attack_unlearning.py'
The script can also be used for toxicity experiments with universal attacks.

## Disclamimer 

In this work, we want to highlight the safety vulnerabilities of LLMs. As powerful LLM assistants are readily available and machine-learning robustness has been an unsolved research problem for the last decade, we believe that the best way to approach this problem is through culminating awareness. 

## Content

The repository contains the code to conduct embedding space attacks on LLMs. 
Typically, adversarial attacks in the embedding space of LLMs are not considered, as most threat models concentrate on attacks that can be transferred to closed-source models utilized through an API which usually demand natural language input. However, in the case of open-source LLMs, an attacker is not restricted to attack the model in natural language space. 

A range of malicious actions can be performed without the need to use closed-source models through restricted APIs, or interact with users of LLM-integrated apps. This includes the distribution of hazardous knowledge (e.g. instructions for creating malicious software), promoting harmful biases, spreading misinformation, building ``troll'' bots to respond to real users on social media, etc. 

Within embedding space attacks we exploit that once an LLM starts giving an affirmative response, it is likely to remain in that ``mode'' and continue to provide related outputs.

## Experiments

The embedding space attack experiment from the paper can be reproduced by running 
```
embedding_attack_submission.py
```

Some code snippets are taken from https://github.com/llm-attacks/llm-attacks

## Cite

If you use this repository in your research, please consider citing:

```	
@misc{schwinn2023adversarial,
      title={Adversarial Attacks and Defenses in Large Language Models: Old and New Threats}, 
      author={Leo Schwinn and David Dobre and Stephan G\"unnemann and Gauthier Gidel},
      year={2023},
      eprint={2310.19737},
      archivePrefix={arXiv},
}
```
