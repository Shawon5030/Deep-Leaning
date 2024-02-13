# Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function

# Understanding-and-Implementing-the-Activation-Function

## Objective:

**1.**	To comprehend the conceptual and mathematics underpinnings of the Activation Function.

**2.**	To execute the Activation Function in a programming language (such as Python).

**3.**	The objective is to examine the attributes and consequences of using the Activation Function inside neural networks.

**Tasks:**

## **Theoretical Understanding: Explain the Activation Function, including its equation and graph.**

Ans: Activation functions play a crucial role in artificial neural networks by introducing non-linearity, which enables the network to learn complex mappings between input and output spaces. An activation function is a crucial component of a neural network, responsible for introducing non-linearity into the model. It operates on the weighted sum of inputs and biases of a neuron, determining whether it should be activated or not. The output of the activation function decides the input for the next layer in the neural network.

### **What is an activation function and why use them?** 

The activation function decides whether a neuron should be activated or not by calculating the weighted sum and further adding bias to it. The purpose of the activation function is to introduce non-linearity into the output of a neuron. 

**Explanation:** 

We know, the neural network has neurons that work in correspondence with weight, bias, and their respective activation function. In a neural network, we would update the weights and biases of the neurons on the basis of the error at the output. This process is known as back-propagation. Activation functions make the back-propagation possible since the gradients are supplied along with the error to update the weights and biases. 

### **Why do we need Non-linear activation function?**

A neural network without an activation function is essentially just a linear regression model. The activation function does the non-linear transformation to the input making it capable to learn and perform more complex tasks. 

### **Mathematical proof**

Suppose we have a Neural net like this :- 

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/ae814e08-9076-474f-a78c-ba524d68b635)

Elements of the diagram are as follows: 

**Hidden layer i.e. layer 1**:

z(1) = W(1)X + b(1) a(1)

Here,

z(1) is the vectorized output of layer 1

W(1) be the vectorized weights assigned to neurons of hidden layer i.e. w1, w2, w3 and w4

X be the vectorized input features i.e. i1 and i2

b is the vectorized bias assigned to neurons in hidden layer i.e. b1 and b2

a(1) is the vectorized form of any linear function.

(Note: We are not considering activation function here)

**Layer 2 i.e. output layer** :-

**Note : Input for layer 2 is output from layer 1**

z(2) = W(2)a(1) + b(2)  

a(2) = z(2) 

Calculation at Output layer

z(2) = (W(2) * [W(1)X + b(1)]) + b(2)

z(2) = [W(2) * W(1)] * X + [W(2)*b(1) + b(2)]

Let, 

    [W(2) * W(1)] = W

    [W(2)*b(1) + b(2)] = b

Final output : z(2) = W*X + b

which is again a linear function.

This observation results again in a linear function even after applying a hidden layer, hence we can conclude that, doesn’t matter how many hidden layer we attach in neural net, all layers will behave same way because the composition of two linear function is a linear function itself. Neuron can not learn with just a linear function attached to it. A non-linear activation function will let it learn as per the difference w.r.t error. Hence we need an activation function. 

## **Theoretical Understanding: Discuss why activation functions are used in neural networks, focusing on the role of the Activation function.**

**Answer:** Activation functions are a fundamental component of neural networks, playing a crucial role in their ability to model complex relationships within data. The primary reasons for using activation functions include:

1. **Introducing Non-linearity**: Activation functions introduce non-linearity into the network, enabling it to learn and represent non-linear relationships between input features and output labels. Without activation functions, neural networks would merely be a composition of linear transformations, severely limiting their expressive power. By applying non-linear activation functions like sigmoid, ReLU, or tanh, neural networks can approximate and learn complex mappings from inputs to outputs.

2. **Enabling Complex Representations**: The non-linear properties introduced by activation functions allow neural networks to represent highly complex and intricate relationships within data. This enables them to solve tasks that involve complex decision boundaries or patterns, such as image recognition, natural language processing, and speech recognition.

3. **Facilitating Feature Learning**: Activation functions facilitate feature learning by enabling neurons to learn and activate specific features within the data. Different activation functions emphasize different features and help the network extract relevant information from the input data. For instance, ReLU activation tends to activate features that have a positive influence on the prediction, while sigmoid activation squashes the output to a range between 0 and 1, useful for tasks like binary classification where probabilities are desired.

4. **Controlling Neuron Activation**: Activation functions also control the activation level of neurons within the network. They determine whether a neuron should be activated (fire) or remain inactive based on the input it receives. This allows the network to regulate the flow of information and selectively focus on relevant features during the learning process.

5. **Gradient Propagation**: Activation functions play a crucial role in gradient propagation during the backpropagation algorithm, which is used to update the weights of the network during training. The derivative of the activation function affects how gradients are backpropagated through the network layers, influencing the learning dynamics and convergence speed of the training process.

Overall, activation functions are essential components of neural networks, enabling them to learn complex patterns and representations from data by introducing non-linearity, facilitating feature learning, controlling neuron activation, and influencing gradient propagation during training.


## **Variants of Activation Function**: (**Mathematical Exploration: Derive the Activation function formula and demonstrate its output range.**)


Several types of activation functions have been proposed and utilized in neural network architectures, each with its characteristics, advantages, and disadvantages. Below are some widely studied activation functions:

1. Sigmoid Function (Logistic Function)
2. Hyperbolic Tangent Function (tanh)
3. Rectified Linear Unit (ReLU)
4. Leaky ReLU
5. Exponential Linear Unit (ELU)
6. Scaled Exponential Linear Unit (SELU)
7. Softmax Function
8. Swish Function
9. Linear Function
10. Parametric ReLU Function
11. Gaussian Error Linear Unit (GELU) Function**


### **1. Sigmoid Function (Logistic Function)**:

The sigmoid function, defined as ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/ba2fe442-da1f-4bd8-bd28-98ccd3ae5b7a), maps the input to the range (0, 1). Historically popular, it is mainly used in the output layer of binary classification tasks. However, its tendency to saturate and the vanishing gradient problem limit its effectiveness in deeper networks.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/13f1c962-6017-4037-b3ae-85958b89a702)

It is a function which is plotted as ‘S’ shaped graph.

**Equation** : A = 1/(1 + e-x)

**Nature** : Non-linear. Notice that X values lies between -2 to 2, Y values are very steep. This means, small changes in x would also bring about large changes in the value of Y.

**Value Range** : 0 to 1

**Uses** : Usually used in output layer of a binary classification, where result is either 0 or 1, as value for sigmoid function lies between 0 and 1 only so, result can be predicted easily to be 1 if value is greater than 0.5 and 0 otherwise.


### **2. Hyperbolic Tangent Function (tanh)**:

Tanh function, ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/587140ac-dae6-43c8-be0c-13e7427a4602) ,outputs values in the range (-1, 1). Similar to the sigmoid function, tanh is also prone to the vanishing gradient problem but is preferred for its zero-centered output, which aids in faster convergence.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/d5ebc422-1d60-426c-8551-6aa95de54874)

The activation that works almost always better than sigmoid function is Tanh function also known as Tangent Hyperbolic function. It’s actually mathematically shifted version of the sigmoid function. Both are similar and can be derived from each other.

**Equation**:-

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/9f366d3c-3bc7-42b9-aef5-bf33f4fa1ab6)

**Value Range**:- -1 to +1

**Nature** :- non-linear

**Uses** :- Usually used in hidden layers of a neural network as it’s values lies between -1 to 1 hence the mean for the hidden layer comes out be 0 or very close to it, hence helps in centering the data by bringing mean close to 0. This makes learning for the next layer much easier.


### **3. Rectified Linear Unit (ReLU)**:

The ReLU function, ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/20a43ac9-dd85-4c7f-a437-cfbca2eee71b), is currently the most widely used activation function due to its simplicity and effectiveness. It avoids the vanishing gradient problem and accelerates convergence, making it suitable for deep neural networks.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/540af06a-aef9-4fa9-ae53-bb0e590bf96a)

It Stands for Rectified linear unit. It is the most widely used activation function. Chiefly implemented in hidden layers of Neural network.

**Equation** :- A(x) = max(0,x). It gives an output x if x is positive and 0 otherwise.

**Value Range** :- [0, inf)

**Nature** :- non-linear, which means we can easily backpropagate the errors and have multiple layers of neurons being activated by the ReLU function.

**Uses** :- ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations. At a time only a few neurons are activated making the network sparse making it efficient and easy for computation.

In simple words, RELU learns much faster than sigmoid and Tanh function.


### **4. Leaky ReLU**:

Leaky ReLU addresses the "dying ReLU" problem by allowing a small gradient for negative inputs. It is defined as ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/fc386701-40ae-4f93-96d2-ccd38e860974) ,where ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/41f63c8e-e227-450d-bc4b-34805b12adfb) is a small positive constant.

Using this function, we can convert negative values to make them close to 0 but not actually 0, solving the dying ReLU issue that arises from using the standard ReLU function during neural network training.

**Note**: If we set the value of alpha to 0 this function will act as the standard ReLU function.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/ec3b3af4-5e81-4e86-ac86-25b0c134371a)

Let's suppose we use a small constant value for the variable alpha. In that case, all negative values on the x-axis representing the input to the function get mapped close to zero while the positive values remain unchanged.

**Note**: The value of the constant (alpha) is determined before training, i.e. it is not learned during training.

### **5. Exponential Linear Unit (ELU)**:

ELU function, ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/06979617-c781-4a3b-8c2d-37d7e054b142) ,offers improved robustness to the saturation problem observed in ReLU. It ensures smooth gradients for both positive and negative inputs.

**Mathematical Definition of ELU**

The mathematical definition of this algorithm is:

Given an input x, the output f(x) is such that:

f(x) = x, for x> 0

f(x) = 𝜇(exp(x)- 1), for x ≤ 0

f'(x) = 1, for x>0

f'(x) = 𝜇(exp(x)), for x ≤ 0

where 𝜇 > 0

Exponential Linear Units are different from other linear units and activation functions because ELU's can take negetive inputs. Meaning, the ELU algorithm can process negetive inputs(denoted by x), into usefull and significant outputs.

If x keeps reducing past zero, eventually, the output of the ELU will be capped at -1, as the limit of exp(x) as x approaches negetive infinity is 0. The value for 𝜇 is chosen to control what we want this cap to be regardless of how low the input gets. This is called the saturation point. At the saturation point, and below, there is very little difference in the output of this function(approximately 𝜇), and hence there’s little to no variation(differential) in the information delivered from this node to the other node in the forward propagation.


In contrast to other linear unit activation functions, ELUs give negative outputs(i.e activations). These allow for the mean of the activations to be closer to 0, which is closer to the natural gradient, so the outputs are more accurate. This reduced difference in the unit gradient and the natural gradient makes learning more efficient as the training of the model will hence converge faster.

**Learning Using the Derivative of ELU**

Convolutional Neural Networks employs the use of back propagation algorithms during learning. Basically, the algorithm is going to go back into the neurons to learn the historical steps taken to reach an outcome.

Forward propagation is the steps taken to reach an outcome from input to output. The error of the algorithm is calculated by the (actual value - the outcome) sqaured / 2. Essentially, what back propagation does is to go back and optimize the weights of each node. It does this by finding the effect on the error when you change the weights by a small value(i.e d(error)/d(weight)).So for the node that uses the ELU activation function, the differential of the ELU is needed and will be used in reference to the differential of the output error.

Now let's focus on the derivative function of ELU.

f'(x) = 1, for x>0

f'(x) = 𝜇(exp(x)), for x ≤ 0

for x ≤ 0,

f(x) = 𝜇(exp(x)-1)

hence,

f'(x) = 𝜇 * (exp(x)-1)' + 𝜇' * (exp(x)-1), Product Rule

f'(x) = 𝜇 * (exp(x)) + 0

f'(x)= 𝜇(exp(x)

futher more,

f'(x)= 𝜇(exp(x) - 𝜇 + 𝜇

f'(x) = 𝜇(exp(x) - 1) + 𝜇

therefore,

f'(x) = f(x) + 𝜇

Since back propagation and forward propagation is done simultaneously, we need a function for the derivative of f(x) to have low computational cost. Since the value of f(x) and 𝜇 is already stored you can get f'(x) by finding the sum of f(x) and 𝜇 at a lower computational cost.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/7edcab21-caca-4aab-9916-19036eecf59f)

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/26f18a24-f4c4-4d1c-a1ea-5c4cd415fc5f)


### **6. Scaled Exponential Linear Unit (SELU)**:

SELU is designed to maintain the mean and variance of the activations across layers, promoting self-normalization in deep networks. It is defined as
![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/c1d3cf0a-ddfd-4cac-8939-bc5dad92a811) , with carefully chosen constants ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/e1c2f4d0-a7f3-46f4-8a80-b65fbf892e08) and ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/6f098de2-1a58-48d0-8317-a03f759e5ae3).

Where 

λ and α are the following approximate values:

λ≈ 1.0507009873554804934193349852946

a≈1.6732632423543772848170429916717

If x is larger than 0, the output result is x multiplied by lambda lambda. If the input value x is less than or equal to zero, we have a function that goes up to 0, which is our output y, when x is zero. Essentially, when x is smaller than zero, we take the exponential of the x-value minus 1, then we multiply it with alpha α and lambda λ.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/822a01ef-8e08-410c-89ff-47011bc2ab5e)

Unlike ReLU, it can get below 0, allowing the system to have a zero average output. As a result, the model may converge faster.

**SELU is known to be a self-normalizing function, but what is normalization?**

Normalization is a data preparation technique that involves changing the values of numeric columns in a dataset to a common scale. This is usually used when the attributes of the dataset have different ranges.

**There is 3 types of normalization:**

**Input normalization**: One example is scaling the pixel values of grey-scale photographs (0–255) to values between zero and one

**Batch normalization**: Values are changed between each layer of the network so that their mean is zero and their standard deviation is one.

**Internal normalization**: this is where SELU's magic happens. The key idea is that each layer keeps the previous layer's mean and variance.

So, **how does SELU make this possible?** 

More precisely, How can it adjust the mean and variance? Let's take another look at the graph:

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/a866296f-588d-4dbc-a2f3-5d6b79e3fa08)

For y to change the mean, the activation function needs both positive and negative values. Both options are available here. It is also why ReLU ReLU isn't a good option for a self-normalizing activation function since it can not output negative values.


### **7. Softmax Function**:

Softmax function is utilized in the output layer for multi-class classification tasks. It transforms raw scores into probabilities, ensuring that the sum of probabilities across classes is 1.

SoftMax function turns logits value into probabilities by taking the exponents of each output and then normalizing each number by the sum of those exponents so that the entire output vector adds up to one. Logits are the raw score values produced by the last layer of the neural network before applying any activation function on it.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/de2943eb-bba7-41df-a59b-7d26765047a5)

The softmax function is similar to the sigmoid function, except that here in the denominator we sum together all of the things in our raw output. In simple words, when we calculate the value of softmax on a single raw output (e.g. z1) we cannot directly take the of z1 value alone. We have to consider z1, z2, z3, and z4 in the denominator.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/ae886ae7-c1fc-4554-b2cc-4782f0854a2e)

The softmax function is also a type of sigmoid function but is handy when we are trying to handle multi- class classification problems.

**Nature** :- non-linear

**Uses** :- Usually used when trying to handle multiple classes. the softmax function was commonly found in the output layer of image classification problems.The softmax function would squeeze the outputs for each class between 0 and 1 and would also divide by the sum of the outputs. 

**Output**:- The softmax function is ideally used in the output layer of the classifier where we are actually trying to attain the probabilities to define the class of each input.

### **8. Swish Function**:

Swish activation, proposed by Ramachandran et al., is defined as ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/404e07ad-a896-45b1-8e76-08a325b75d72). It has been shown to outperform ReLU in certain scenarios, offering smoother gradients.

It is a self-gated activation function developed by researchers at Google. 

Swish consistently matches or outperforms ReLU activation function on deep networks applied to various challenging domains such as image classification, machine translation etc. 

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/9487d8f1-caf3-4e3b-85f7-498ad1e6e8c0)

This function is bounded below but unbounded above i.e. Y approaches to a constant value as X approaches negative infinity but Y approaches to infinity as X approaches infinity.


### **9. Linear Function**:

The linear activation function, also known as "no activation," or "identity function" (multiplied x1.0), is where the activation is proportional to the input.

The function doesn't do anything to the weighted sum of the input, it simply spits out the value it was given. 

Linear function has the equation similar to as of a straight line i.e. y = x

No matter how many layers we have, if all are linear in nature, the final activation function of last layer is nothing but just a linear function of the input of first layer.

**Range** : -inf to +inf

**Uses** : Linear activation function is used at just one place i.e. output layer.

**Issues** : If we will differentiate linear function to bring non-linearity, result will no more depend on input “x” and function will become constant, it won’t introduce any ground-breaking behavior to our algorithm.

**For example** : Calculation of price of a house is a regression problem. House price may have any big/small value, so we can apply linear activation at output layer. Even in this case neural net must have any non-linear function at hidden layers. 

Mathematically it can be represented as:

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/7ddeafb4-e951-4158-9a9a-5ded1df99bb1)


![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/07f64c54-da70-447c-b060-3d941305c085)

### **10. Parametric ReLU Function**:

Parametric ReLU is another variant of ReLU that aims to solve the problem of gradient’s becoming zero for the left half of the axis. 

This function provides the slope of the negative part of the function as an argument a. By performing backpropagation, the most appropriate value of a is learnt.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/e97795bf-ba0b-480d-a9ac-4c7a727cf235)

Mathematically it can be represented as:

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/5006b3a1-1788-4043-ac68-38713e79d73d)

Where "a" is the slope parameter for negative values.

The parameterized ReLU function is used when the leaky ReLU function still fails at solving the problem of dead neurons, and the relevant information is not successfully passed to the next layer. 

This function’s limitation is that it may perform differently for different problems depending upon the value of slope parameter a.

### **11. Gaussian Error Linear Unit (GELU) Function**:

The Gaussian Error Linear Unit (GELU) activation function is compatible with BERT, ROBERTa, ALBERT, and other top NLP models. This activation function is motivated by combining properties from dropout, zoneout, and ReLUs. 

ReLU and dropout together yield a neuron’s output. ReLU does it deterministically by multiplying the input by zero or one (depending upon the input value being positive or negative) and dropout stochastically multiplying by zero. 

RNN regularizer called zoneout stochastically multiplies inputs by one. 

We merge this functionality by multiplying the input by either zero or one which is stochastically determined and is dependent upon the input. We multiply the neuron input x by 

m ∼ Bernoulli(Φ(x)), where Φ(x) = P(X ≤x), X ∼ N (0, 1) is the cumulative distribution function of the standard normal distribution. 

This distribution is chosen since neuron inputs tend to follow a normal distribution, especially with Batch Normalization.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/2a2c766a-02c8-4eb4-ad3a-6b1f1382f36d)

Mathematically it can be represented as:

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/dd2282eb-52de-4123-b74f-507766c0ed52)

GELU nonlinearity is better than ReLU and ELU activations and finds performance improvements across all tasks in domains of computer vision, natural language processing, and speech recognition.

**The basic rule of thumb is if you really don’t know what activation function to use, then simply use RELU as it is a general activation function in hidden layers and is used in most cases these days.**

**If your output is for binary classification then, sigmoid function is very natural choice for output layer.**

**If your output is for multi-class classification then, Softmax is very useful to predict the probabilities of each classes.**

## **Mathematical Exploration: Calculate the derivative of the Activation function and explain its significance in the backpropapation process.**

**Answer:**

Let's calculate the derivatives of the mentioned activation functions one by one and discuss their significance in the backpropagation process:

### 1. **Sigmoid Function (Logistic Function)**:

   - Equation: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/fe955ea7-d336-4ab4-aec1-d53b182822e1)

   - Derivative: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/88f73067-c7fd-454c-9af9-debcaf4c4aa3)

   - Significance: In the backpropagation process, the derivative of the sigmoid function helps in calculating the gradient of the loss function with respect to the weights of the neural network. It is crucial for adjusting the weights during training, as it indicates how much the output of a neuron should be adjusted to minimize the error.

### 2. **Hyperbolic Tangent Function (tanh)**:

   - Equation: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/4c40edda-0065-423b-81b7-61a8e1530863)

   - Derivative: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/b79c2795-5cd6-44bd-8ed2-c536bd88c5b8)

    - Significance: Similar to the sigmoid function, the derivative of the tanh function is used in backpropagation to compute gradients. It helps in updating the weights of the network to minimize the error. Tanh is often preferred over sigmoid due to its zero-centered output range, which can make optimization easier.

### 3. **Rectified Linear Unit (ReLU)**:

    - Equation: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/a9215745-bd78-44d1-a989-c0c79eddd53e)

   - Derivative: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/c992827b-36b6-4c76-8708-54bc327239da)

    
   - Significance: The derivative of ReLU is straightforward, either 0 for negative inputs or 1 for positive inputs. In the backpropagation process, it helps in addressing the vanishing gradient problem by allowing gradients to flow for positive inputs. However, it can suffer from the dying ReLU problem where neurons may become inactive and stop learning.

### 4. **Leaky ReLU**:
   - Equation: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/16be1316-b4ab-40de-ab32-2eee0665dad0)
where ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/66aca9b5-438c-4902-ac21-a30bb3ba89f9) is a small constant (typically 0.01).

   - Derivative: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/3eeb5896-b711-4347-8bcd-646da38e4c66)

   - Significance: Leaky ReLU addresses the dying ReLU problem by allowing a small gradient for negative inputs, which prevents neurons from becoming completely inactive. Its derivative helps in backpropagation, allowing gradients to flow for both positive and negative inputs.

### 5. **Exponential Linear Unit (ELU)**:
   - Equation: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/299f9a12-053e-49e9-ba44-88900ef2ceb6)
 where ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/d1efd55a-3438-425c-8d27-654ceb46352a) is a small positive constant (typically 1).

   - Derivative: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/fb9a66d0-de34-4ebd-9e95-f4491c662c16)


   - Significance: ELU is another alternative to ReLU, which addresses the dying ReLU problem by allowing negative inputs to have non-zero gradients. Its derivative helps in backpropagation, allowing gradients to flow smoothly for both positive and negative inputs.

### 6. **Scaled Exponential Linear Unit (SELU)**:
   - Equation: SELU is a scaled version of the ELU function with specific parameters.
   - Derivative: The derivative of SELU depends on its parameters and can be complex.
   - Significance: SELU is designed to ensure self-normalization of the hidden layers, which can lead to more stable training dynamics and improved convergence properties. Its derivative is essential in the backpropagation process for updating the weights.

### 7. **Softmax Function**:
   - Equation: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/aee3e51a-0d76-42fd-9823-4db02c87305a) for i = 1, 2, ..., n

    - Derivative: The derivative of the softmax function is complex and typically not used explicitly in backpropagation. Instead, the cross-entropy loss function is often used in conjunction with softmax, and its derivative is computed during backpropagation.
 
   - Significance: Softmax is commonly used in the output layer of neural networks for multi-class classification tasks, as it converts the raw scores into probability distributions. While its derivative is not directly used in backpropagation, the softmax function itself is crucial for obtaining meaningful class probabilities.

### 8. **Swish Function**:
   - Equation: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/00f5ceaf-38e0-467a-9728-eac17ca6ff38)

   - Derivative: The derivative of the Swish function is not as straightforward as some other functions and involves the derivative of the sigmoid function.
  
   - Significance: Swish is a recently proposed activation function that combines features of ReLU and sigmoid functions. It has shown promising performance in some scenarios, and its derivative is essential for updating weights during backpropagation.

### 9. **Linear Function**:
   - Equation: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/591666b7-b416-483a-b77a-41965551bb6f)

   - Derivative: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/10b7768e-eb1a-4b0c-8aac-e203fe1ae0b4)

   - Significance: The linear function is simple and does not introduce non-linearity. It is typically used in the output layer for regression tasks, where the network needs to directly predict continuous values. Its derivative of 1 simplifies the backpropagation process, allowing gradients to flow unchanged.

### 10. **Parametric ReLU Function**:
    - Equation: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/80ababc1-3543-4d86-9912-1b2a1ddc3a63)
 where ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/6b98b982-dff4-41ad-ba09-9f43bad99a48) is a learnable parameter.
  
    - Derivative: ![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/bd07e1c5-d683-42f2-948c-1da793403a68)

    - Significance: Parametric ReLU extends ReLU by introducing a learnable parameter \( \alpha \), which allows the network to learn the optimal slope for negative inputs. Its derivative helps in backpropagation, enabling adaptive adjustments to the slope based on the input data.

### 11. **Gaussian Error Linear Unit (GELU) Function**:
    - Equation: GELU is a smooth approximation of the ReLU function based on the Gaussian cumulative distribution function.
    - Derivative: The derivative of GELU is not as simple as some other functions and involves the Gaussian probability density function.
    - Significance: GELU has shown promising performance in certain scenarios, particularly in transformer architectures. Its derivative is crucial for backpropagation, allowing gradients to be computed and weights updated during training.

In summary, the derivatives of activation functions play a vital role in the backpropagation process of neural networks, enabling the computation of gradients and the adjustment of weights to minimize the error between predicted and actual outputs. Each activation function has its characteristics and significance in training neural networks, influencing the learning dynamics and performance of the model.

**3.	Programming Exercise:**

## o **Implement the Activation Activation Function in Python. Use the following prototype for your function:**

**def Activation_Function_Name(x): # Your implementation**

**Answer**: Please check the 'Activation Functions of Neural Network.py' file.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/1014468d-8f0e-432b-9afd-a1812332dab8)

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/fc95e65c-d941-43aa-be0b-c252c41fe654)

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/72476894-c78d-4e3b-b387-aa608e3c97db)

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/c7e7a1f0-fd58-484f-987f-704b3814080c)

## o **Create a small dataset or use an existing one to apply your function and visualize the results.**

**Answer**: Please check the 'Activation Functions of Neural Network.py' file.

![image](https://github.com/TITHI-KHAN/Nazmun_Assignment-1_Understanding-and-Implementing-the-Activation-Function/assets/65033964/41040a98-db10-455c-bec1-07f30797d805)

 
**4.	Analysis:**

## o	**Analyze the advantages and disadvantages of using the Activation Function in neural networks**.

**Answer**: 

Let's analyze the advantages and disadvantages of using activation functions in neural networks:

**Advantages:**

1. **Introducing Non-linearity**: Activation functions introduce non-linearity into neural networks, enabling them to learn and model complex relationships within data. This non-linearity allows neural networks to approximate arbitrary functions, making them powerful function approximators.

2. **Feature Learning**: Activation functions facilitate feature learning by enabling neurons to selectively activate and learn relevant features from the input data. Different activation functions emphasize different aspects of the data, allowing the network to capture various patterns and representations.

3. **Gradient Propagation**: Activation functions play a crucial role in gradient propagation during backpropagation, enabling efficient learning by providing meaningful gradients for weight updates. This allows the network to adjust its parameters to minimize the error between predicted and actual outputs.

4. **Addressing Vanishing Gradient**: Certain activation functions, such as ReLU and its variants, help mitigate the vanishing gradient problem by allowing gradients to flow more effectively during backpropagation. This promotes faster convergence and more stable training dynamics, especially in deep neural networks.

5. **Normalization**: Some activation functions, like SELU, incorporate normalization properties, leading to self-normalization of hidden layers. This can improve training stability and convergence, particularly in architectures like recurrent neural networks (RNNs).

**Disadvantages:**

1. **Dying Neurons**: Activation functions like ReLU suffer from the "dying ReLU" problem, where neurons become inactive and stop learning if their inputs fall below zero. This can lead to dead neurons, resulting in a reduction of the model's capacity to learn and generalize.

2. **Gradient Saturation**: Certain activation functions, such as sigmoid and tanh, exhibit saturation behavior, where gradients become very small (close to zero) for large input values. This can slow down learning, especially in deeper networks, due to vanishing gradients.

3. **Non-Zero-Centeredness**: Activation functions like ReLU are not zero-centered, which can lead to issues during optimization, particularly when using gradient-based methods. This may require additional normalization techniques or careful weight initialization strategies to mitigate.

4. **Complexity and Computation**: Some activation functions, especially those with complex formulations like ELU and GELU, can introduce additional computational complexity during both forward and backward passes. This may impact training speed and memory requirements, especially in large-scale neural networks.

5. **Sensitivity to Hyperparameters**: The performance of activation functions can be sensitive to hyperparameters such as learning rate, initialization methods, and network architecture. Choosing the appropriate activation function for a specific task often requires empirical experimentation.

In summary, while activation functions are essential components of neural networks that enable non-linear transformations and efficient learning, they also come with their limitations and challenges. Understanding these advantages and disadvantages is crucial for selecting the appropriate activation functions and designing effective neural network architectures.

## o	**Discuss the impact of the Activation function on gradient descent and the problem of vanishing gradients**.

**Answer**: 

The choice of activation function in neural networks has a significant impact on the gradient descent optimization algorithm and directly affects the problem of vanishing gradients. Let's delve deeper into this relationship:

**Impact on Gradient Descent:**

1. **Gradient Calculation:** Activation functions influence how gradients are computed during backpropagation, which is a critical step in the gradient descent optimization algorithm. The derivative of the activation function determines the magnitude of the gradient and affects how weights are updated to minimize the loss function.

2. **Learning Dynamics:** Different activation functions lead to varying learning dynamics in neural networks. Some activation functions, like ReLU, allow for faster convergence due to their simple derivative properties, while others, like sigmoid and tanh, can introduce saturation effects that slow down learning.

3. **Stability and Convergence:** The choice of activation function can impact the stability and convergence speed of the optimization process. Activation functions with non-saturating derivatives, such as ReLU and its variants, often lead to more stable training dynamics and faster convergence compared to functions with saturating derivatives, like sigmoid and tanh.

**Impact on Vanishing Gradients:**

1. **Definition:** The vanishing gradient problem occurs when gradients become extremely small as they propagate backward through many layers of a neural network during training. This phenomenon can hinder the learning process, especially in deep networks, as it leads to negligible updates to the weights in early layers.

2. **Saturated Activation Functions:** Activation functions like sigmoid and tanh are prone to saturation, particularly for large positive or negative inputs. When inputs are in the saturated regions of these functions, their derivatives approach zero, causing gradients to vanish during backpropagation.

3. **Effect on Depth:** The vanishing gradient problem becomes more pronounced in deeper neural networks due to the cumulative effect of gradients diminishing as they propagate backward through multiple layers. This limits the ability of deep networks to effectively learn hierarchical representations of data.

4. **Mitigation with ReLU and Variants:** Activation functions like ReLU and its variants (e.g., Leaky ReLU, ELU) address the vanishing gradient problem by providing non-saturating derivatives for positive inputs. This allows gradients to flow more freely during backpropagation, facilitating more effective weight updates and mitigating the issue of vanishing gradients.

In summary, the choice of activation function profoundly impacts the behavior of gradient descent optimization and directly influences the occurrence of the vanishing gradient problem. Activation functions with non-saturating derivatives, such as ReLU and its variants, are often preferred in practice for deep neural networks due to their ability to mitigate the vanishing gradient problem and facilitate faster convergence.

