# Pinokio
RL natural language translation

Pinokio uses a finite state machine constructing a RL environment for processing input to output.

The idea is that it will process a very literal translation from one human language to another using a manually compiled dictionary.

Each step currently controlled by the PPO algorithm.

The RL environment has the following actions
- Pushing and pulling from a stack for word reordering
- Pushing and pulling from a dictionary for word translation
- Pulling from the input
- Pushing to the output

## Installation

Clone repository

```sh
$ cd install/path
$ git clone https://github.com/JEdward7777/Pinokio.git
$ cd Pinokio
```

Install pytorch: 
https://pytorch.org/

Install Pinokio:

```sh
$ pip install stable-baselines3[extra]
```

## Run training
```sh
$ cd pinokio
$ python3 pinokio3.py
```

Kill it with control-c while it isn't saving the save file.

## View current output capabilities
```sh
$ python3 pinokio3_test.py
$ cat pinokio3_test_output.txt
```

## Short description of internals
pinokio2.py is the basic RL environment for the translation, however to aid the RL algorithm in finding the correct answers pinokio3.py is a layer on top of it which brute-forces the optimum solution from input to output so that it can drop point breadcrumbs for the RL to pick up on instead of it having to find the correct output directly from the input.

The sentence pairs which are being learned are in the file ```./spa-eng/spa_edited.txt``` .
The dictionary is in the json file ```./pinokio/words.json``` .