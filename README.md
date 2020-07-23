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
Input sentence 'I think ten added to two is', correct result 12, prediction 13.812738418579102
Input sentence 'twenty-one plus ten =', correct result 31, prediction 33.2834358215332
Input sentence 'nineteen plus three equals', correct result 22, prediction 25.56694793701172
```
