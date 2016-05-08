# kaggle DigitRecognizer
## TODO
1. read train data √
 a. use panda lib √
2. signle image visualization √
 a. use matplot lib √
3. handle data √
 a. vectorize label √
4. simple bp nerual network 
 a. sigmoid function √
  aa. sigmoid gradient √
 b. feedforward √
 c. cost function √
  aa. regularized √
 d. back propagation
  aa. random start √
  ab. Set the input layer’s values (a(1)) to the t-th training example x(t) √
  ac. err3 √
  ad. err2
  ae. Accumulate the gradient
  af. Obtain the (unregularized) gradient
 e. gradient checking
 f. regularized back propagation
 g. learning parameters
5. evaluate algorithm
6. cnn rnn or dnn

## bp
input 28*28 = 784 + 1(bias) row vector 1x785 marked as x
hidden 1000
output 10 = [0..9] marked as y

theta1 is 785x1000
theta2 is 1001x10

### forward
z1 = x*theta1
a1 = sigmoid(z1)
z2 = a1*theta2
y = a2 = sigmoid(z2)
label = max index of y
