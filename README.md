# Simple Neural Network

This is simple neural network for predict output of the function from given training data set.This neural network consists of 3 node as input and 1 node as output. The function can be represented as


```
y = a * x1  + b * x2  + c * x3
```

![alt text](https://cdn-images-1.medium.com/max/400/1*HDWhvFz5t0KAjIAIzjKR1w.png)

Refer to article https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1.

## How to use
### 1. Generate data set
Edit secret function with above format at function secret_function() and generate practical data set into data_set.csv file

```
python run.py generate
```
### 2. Training data set and predict output
You can remove some rows in data_set.csv file and use them as testing data.

Input value testing input at function get_test_data() and run below command.

```
python run.py
```
