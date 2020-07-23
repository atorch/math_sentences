# math_sentences

A neural network that answers basic math questions
(inspired by [this example](https://youtu.be/0VH1Lim8gL8?t=1417)).

The training set in this repo is _very_ small, so we're likely overfitting.

```bash
sudo docker build ~/math_sentences --tag=math_docker
sudo docker run -it -v ~/math_sentences:/home/math_sentences math_docker bash
python fit_model.py
```

```bash
Input sentence 'Did you know that three plus four =', correct result 7, prediction 7.732294082641602
Input sentence 'What do you think seven added to five is', correct result 12, prediction 12.400672912597656
Input sentence 'two plus one equals', correct result 3, prediction 1.5675246715545654
```

