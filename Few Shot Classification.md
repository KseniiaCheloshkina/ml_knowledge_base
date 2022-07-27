## "Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference" 
Paper: https://aclanthology.org/2021.eacl-main.20.pdf
Key points:
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


## "Making Pre-trained Language Models Better Few-shot Learners"
