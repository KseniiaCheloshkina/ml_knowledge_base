## "Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference" 
Paper: https://aclanthology.org/2021.eacl-main.20.pdf
Key points:
- solving a task from only a few examples becomes much easier when we also have a task description, i.e., a textual explanation that helps us understand what the task is about. So far, this idea has mostly been considered in zero-shot scenarios where no training data is available at all.
- PET works in three steps: First, for each pattern a separate PLM is finetuned on a small training set T . The ensemble of all models is then used to annotate a large unlabeled dataset D with soft labels. Finally, a standard classifier is trained on the soft-labeled dataset. 

Algorithm:
- We assume access to a small training set T and a (typically much larger) set of unlabeled examples D

## "Making Pre-trained Language Models Better Few-shot Learners"
