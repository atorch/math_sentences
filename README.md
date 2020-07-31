# math_sentences

A recurrent character-by-character neural network that answers basic math questions
(inspired by [this example](https://youtu.be/0VH1Lim8gL8?t=1417)).

```bash
sudo docker build ~/math_sentences --tag=math_docker
sudo docker run -it -v ~/math_sentences:/home/math_sentences math_docker bash
python fit_model.py
```

Predictions on training generator sentences:

```bash
Input sentence 'My question is: what's fourteen minus fourteen =', correct result 0, prediction -0.5
Input sentence 'Hey forty-one plus eighteen =', correct result 59, prediction 59.06999969482422
Input sentence 'Um, barf! I hate math... forty-two - forty-nine =', correct result -7, prediction -6.510000228881836
```

Extrapolation, i.e. sentences that are outside of the training distribution
(the last sentence is nearly impossible for this model, since it has never seen negative inputs!):

```bash
Input sentence 'What is one + one =', correct result 2, prediction 3.9800000190734863
Input sentence 'Can you answer one + one =', correct result 2, prediction 5.880000114440918
Input sentence 'Never seen before: what is one + one =', correct result 2, prediction 3.9600000381469727
Input sentence 'What is one + one?', correct result 2, prediction 2.009999990463257
Input sentence 'Sentence you've never seen before: what is one + one??', correct result 2, prediction 0.3799999952316284
Input sentence 'This is a sentence you've never seen before. What is negative one plus negative two?', correct result -3, prediction 6.150000095367432
```

The maximum number of possible (unique) sentences simulated by the training generator is 495,000.