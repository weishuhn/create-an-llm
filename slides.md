---
theme: dracula

title: LLMs from Scratch
info: |
  Understanding AI chatbots from raw data to working model.
  Presented by [Your Name]

class: text-center
drawings: 
  persist: false
transition: slide-left
mdc: true
---

# LLMs from Scratch

---

# Summary

This is where I unsuccessfully try to explain how LLMs are built. 

We'll start with raw data and finish with a "working" model. 

We'll cover:

- Why they make up facts
- Why they can't do math
- Why billions of parameters is considered 'intelligence'

But we'll also touch on what they are good at and how you can best put them to use.

---

# Goals

Only a few "true" AI companies exist, most are just "AI enabled". 

I want you to:

- Understand the basics of LLMs and how they work
- Be able to intelligently reason about how best to use them
- Avoid wasting time and money on "AI" that doesn't work

---

# Caveats

This presentation is going to be a lot like the output of an LLM:

<v-clicks>

- Contains factual errors and probably hallucinations
- Filled with platitudes and gross generalizations
- Has a knowledge cutoff date of about a year ago


<div class="mt-15">

# Questions 
Feel free to ask questions and correct me. My likely responses will be:

  - "It depends" - because context matters
  - "I don't know" - because, well, I don't know
  - "That's interesting, I didn't know that" - because I'm always learning too

</div>
</v-clicks>

---

# What is Chat**GPT**?

It's all in the name GPT:

<div grid="~ cols-3 gap-4">
<div>

## Generative

- Creates rather than classifies
- Produces novel outputs
- Text, images, code, etc.

</div>
<div>

## Pre-trained

- Trained on massive datasets
- Learns patterns from existing content
- Base model for further fine-tuning

</div>
<div>

## Transformer

- "Attention is all you need" - 2017 paper from Google
- Specific type of neural network
- New variants are emerging constantly

</div>
</div>

<v-clicks>

<div class="mt-25">

**Learning**: Almost all LLMs are based on the Transformer architecture.

</div>

</v-clicks>

---
layout: section
---

# Neural Networks: How They Work
 
---
layout: two-cols-header 
---
# Neural Networks: How They Work
At its core, it's multiple layers of linear functions sandwiched between non-linear functions
::left::
- **Linear function**: 
  - $y = mx + b$
- **Non-linear functions:** 
  - $y = x^2$
  - $y = 2^x$
  - ReLU: $y = \max(0 , x)$
::right::

<div class="flex justify-center h-full items-center">
<img src="/NN.jpg" class="w-full max-h-100" />
</div>

---
layout: two-cols-header
---
# Neural Networks: How They Work
::left::
<v-clicks>

- Every node has a bias and weights which are learned during training
	- $y = w_1x_1 + w_2x_2 + ... + w_nx_n + b$
- The weights are how much each input contributes to the output
- The bias is how much the output is shifted i.e. valued more or less
- The final output passes through a non-linear function before next layer
- Without non-linear functions, models can only represent linear relationships

</v-clicks>

::right::
<div class="flex justify-center h-full items-center">
<img src="/NN-zoom.jpg" class="w-full max-h-100 pl-6" />
</div>

---
layout: two-cols-header
---
# Neural Networks: How They Work
::left::
<v-clicks>

- A model is a list of floating-point numbers
- These are the weights and biases we calculated for each node
  - $y = w_1x_1 + w_2x_2 + ... + w_nx_n + b$
- The number of connections between nodes and the number of layers drives the size of the model
- llama-3.1-70b has 70B parameters
- The larger the model, the more parameters it can use to learn the patterns in the data.
- Larger models are take longer to train and use more computational resources in inference

Learning: The **model size** = number of parameters = the number of weights and biases

</v-clicks>

::right::

<div class="flex justify-center h-full items-center">
<img src="/NN-zoom.jpg" class="w-full max-h-100 pl-6" />
</div>

---
layout: section
---

# How to Train an LLM

---

# Training vs Inference

<v-clicks>

- Training is when the model learns the patterns in the data
- Inference is when the model uses the patterns it learned to make predictions
- Training is done on a large dataset and can take days to months to complete
- Inference is done when the model is deployed and is used to make predictions on new data

**Now onto training!**
</v-clicks>

---

# Data Gathering

<v-clicks>

- Sources:
  - Common Crawl 
  - Books, articles, research papers
  - Code repositories
  - Websites
  - Social media
  - And more...
- Volume: Typically hundreds of terabytes of data
- You have to stop collecting data at some point

<div class="mt-15">

Learning: The **knowledge cutoff** date is when you stop collecting data!

</div>

</v-clicks>

---

# Tokenization

<v-clicks>

- Process of converting text into integers for computation
- Words and subwords become tokens represented by integers
- Different LLMs use different tokenization algorithms:

<div class="flex justify-center pb-15 pt-3">
<div class="w-1/2">
OpenAI Tokenizer
<img src="/token-os-oai.jpg" class="w-100" />
</div>
<div class="w-1/2">
Anthropic Tokenizer
<img src="/token-os-anthropic.jpg" class="w-100" />
</div>
</div>

Learning: The **tokenization** method is specific to the LLM!  They cannot "talk" to each other.
</v-clicks>

---

# Tokenization: Numbers

<v-clicks>

- Because everything is converted to tokens, numbers are also converted. 
- LLMs don't have a "token" for every number, only the most common ones.
- They break large numbers into tokens. For example 94569 + 32458 = 127027:

<div class="flex justify-center pb-15 pt-10">
 <img src="/token-numbers.jpg" class="w-75" />
</div>

Learning: This is why **LLMs are bad at math**!

</v-clicks>

---
layout: section
---

# Training

---

# Types of Training in Machine Learning
<v-clicks>

- **Supervised Learning**
  - Model given direct feedback
  - Correct outputs specified
  - Common in image classification

- **Unsupervised Learning**
  - Model discovers patterns itself
  - No explicit correct answers
  - Common in clustering tasks

- **Reinforcement Learning**
  - Model learns to follow preferences
  - Human or other model provides feedback on outputs
  - Most common term is RLHF (Reinforcement Learning from Human Feedback)

</v-clicks>

---

# Training an LLM

The three stages of training are: 
 
<v-click>

1. **Pre-training** is where the LLM "learns" the language and the relationships between words (tokens)
   - This is unsupervised
</v-click>

<v-click>

2. **Instruction-tuning** is where the LLM is trained on a specific task
   - This is supervised
</v-click>

<v-click>

3. **Reward Modelling** is where the LLM is trained to follow human (or other model's) preferences
   - This is a mix of supervised and reinforcement learning

</v-click>

<v-click>

<div class="mt-15">

**Learning**: Training an LLM is a multi-stage process that uses three different ML training techniques.

</div>

</v-click>

---

# Pre-Training (Unsupervised)

The core of an LLM, the probabilistic sentence completion model, is built here.

<v-click>

- Core technique, Next-token prediction
  - Feed in part of a text sequence
  - Model predicts next token
  - Check against actual next token
  - Adjust weights and biases to improve accuracy
  - Repeat until the model is good at predicting the next token

</v-click>

<v-click>

- After training:
  - Model probabilistically predicts next token in a sequence
  - Can generate text by repeatedly predicting the next token

</v-click>

<v-click>

<div>

**Learning 1**: An LLM is fundamentally "just" a probabilistic sentence completion program.

</div>

</v-click>

<v-click>
<div>

**Learning 2**: There is no place for storing "facts" - just statistical patterns/relationships between tokens, there be hallucinations.

</div>

</v-click>

---

# Instruction Fine-tuning (Supervised)

<v-clicks>

- Feed examples of prompts (input) and desired responses (output)
- Measure how closely model's responses match desired outputs
- Adjust weights to improve alignment with expected behavior
- Example:
<div v-click class="pl-5 text-sm">

- **System Prompt**: You are an AI assistant. Generate a detailed answer.
- **Task**: Generate a fifteen-word sentence describing: Midsummer House
	```json
	{ 
		"eatType": "restaurant", 
		"food": "Chinese", 
		"priceRange": "moderate", 
		"customer rating": "3/5", 
		"near": "All Bar One" 
	}
	```
- **Output**: Midsummer House is a moderately priced Chinese restaurant with a 3/5 customer rating, located near All Bar One.
</div>

</v-clicks>

---

# Reward Modelling (Reinforcement Learning)

<v-clicks>

- Model solves a problem multiple ways
- Human or stronger model ranks the solutions
- Model learns to prefer higher-ranked outputs

<div class="mt-10">

## RLHF (Reinforcement Learning from Human Feedback)
A common buzzword in the industry.

- Humans rank model outputs by quality/helpfulness
- Model trained to maximize predicted human preference
- Helps align model with human values and expectations
</div>

</v-clicks>

---
layout: two-cols-header
---

# Prompt "Training" (BONUS)
This is where we use the prompt to guide the model's behavior. It's not really training, as it happens at inference time, but it's another way to get the model to do what we want.

::left::

## System Instructions
- Instructions that guide model behavior
- Prevents answering harmful questions
- Companies hope users can't bypass (but they usually can)

<div class="mb-40"></div>
::right::
## One (or few) shot "learning"
- Provide example(s) of the desired behavior in the prompt
- Model will try to mimic the behavior of the examples

---

# What Do We Know?

<v-clicks>

- Most LLMs are based on the Transformer architecture
- There is a cutoff date for information - new events and terms aren't in the model
- Tokenization makes LLMs bad at math - numbers get split into tokens
- LLMs cannot talk to each other - they have different tokenization schemes
- Model size is just the number of weights and biases in the model
- The larger the model, the more parameters it can use to learn the patterns in the data, but it's more computationally expensive to train and use
- They are just (really good) probabilistic sentence completion models that have been  trained on a lot of data to predict the next token
- There is no place for storing "facts" - just statistical patterns
- Training an LLM is a multi-stage process that uses multiple training techniques to create a final model
  
</v-clicks>

---

# What Makes for an "Open" Model?

Most "open" models today are not truly open. To be truly open, we need full transparency in:

<v-clicks>

- Training data sources and selection criteria
- Tokenization method
- Model architecture
- Training methodology, including:
  - Fine-tuning approaches and datasets
  - Reinforcement Learning evaluation metrics
- System instructions and safety measures
- Complete weights and parameters
- Unfettered ability to use the model for any purpose
- and more...

</v-clicks>

---
layout: center
class: text-center
---

# Thank You!

Questions?