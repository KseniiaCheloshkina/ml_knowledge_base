## "Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference" 
Paper: https://aclanthology.org/2021.eacl-main.20.pdf

Inspiration:
- solving a task from only a few examples becomes much easier when we also have a task description, i.e., a textual explanation that helps us understand what the task is about. So far, this idea has mostly been considered in zero-shot scenarios where no training data is available at all.
- PET works in three steps: First, for each pattern a separate PLM is finetuned on a small training set T . The ensemble of all models is then used to annotate a large unlabeled dataset D with soft labels. Finally, a standard classifier is trained on the soft-labeled dataset. 

Algorithm:
- We assume access to a small training set T and a (typically much larger) set of unlabeled examples D
- We define a number of `patterns P`
![fwc1](pics/fwc1.png "fwc 1")
- We define a `verbalizer` that converts token to classification label
![fwc2](pics/fwc2.png "fwc 2")
- Then for each pattern P we train a separate MLM model that predicts at `[MASK]` position a token converted to label with the help of `verbalizer`. The loss function used is MLM loss combined with cross-entropy loss. 
- At the next step the big unlabeled set is scored by all models/patterns separately and soft-labels are assigned by aggregation of predictions.
- A standard classifier if trained on these soft labels which is a final model to be used.
![fwc](pics/fwc.png "fwc p")

Conclusions:
- Works better than classic fine-tuning (RoBERTa base compared) from 10 to 100 examples per class with score from 45% to 89% on 10 examples for different datasets
- Have an iterative alternative iPET which is slightly performant but much slower
- Need a big unlabeled set a hand-crafted patterns

## "Making Pre-trained Language Models Better Few-shot Learners"
Paper: https://aclanthology.org/2021.acl-long.295.pdf
Implementation: https://github.com/princeton-nlp/LM-BFF
Inspiration: 
- Prompt-based prediction treats the downstream task as a (masked) language modeling problem, where the model directly generates a textual response (referred to as a label word) to a given prompt defined by a taskspecific template. We address this issue by introducing automatic prompt generation, including a pruned brute-force search to identify the best working label words, and a novel decoding objective to automatically generate templates using the generative T5 model — all of which only require the few-shot training data.
![fsl](pics/lm_bff.png "lm_bff 1")
- Inspired by GPT-3  we adopt the idea of incorporating demonstrations as additional context. We develop a more refined strategy, where, for each input, we randomly sample a single example at a time from each class to create multiple, minimal demonstration sets. We also devise a novel sampling strategy that pairs inputs with similar examples, thereby providing the model with more discriminative comparisons.
![fsl2](pics/lm_bff1.png "lm_bff 2")

Algorithm:
- 

Conclusions: 
- only use a few annotated examples as supervision
- experiment with RoBERTa-large and 16 training examples for each clas

## Sentence embeddings and ZMap
Paper: https://few-shot-text-classification.fastforwardlabs.com/

## Induction Networks for Few-Shot Text Classification
Paper: https://arxiv.org/pdf/1902.10482.pdf

Общая идея:
- Учимся извлекать знание о том, как из эмбеддингов саппорт сета извлечь описание класса. Основано на идее meta learning - learn several tasks (datasets) of the same type (classification). Relation Induction Networks
Мы не пытаемся научиться разделять классы, а учимся выделять общее из саппорт сета, то есть итоговый алгоритм будет принимать на вход саппорт сет + пример для скоринга и выдавать ответ, похожи или нет

Conclusions:
- need a big dataset of similar tasks on which to pretrain model for few shot classification (for example, for a task of sentiment analysis for freezers we need labeled sentiment sets for > 10 other categories

## Общие мысли
Базовые подходы:
1) Задача в получении эмбеддинга класса на основании маленького количества примеров (саппорт сет - набор примеров одного класса). Затем для нового примера измеряем близость до эмбеддингов классов

2) Пример от Тинькофф: https://www.youtube.com/watch?v=m0zv3cRk1qA&list=PLLrf_044z4JrM_7YvA0oTgIrZMbrXYisA&index=35
