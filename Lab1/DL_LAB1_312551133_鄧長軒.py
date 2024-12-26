import numpy as np
import matplotlib.pyplot as plt

data = 0
hsize1 = 4
hsize2 = 4
epochs = 100000
learning_rate = 0.1

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)

def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0]>pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21,1)

def show_result(x,y,pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

def show_learning_curve(x, y):
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('line chart')
    plt.show()


class NeuralNetwork:
    def __init__(self, hsize1, hsize2):
        self.hsize1 = hsize1
        self.hsize2 = hsize2
        self.w1 = np.random.rand(2, self.hsize1)
        self.bias1 = np.random.rand(1, self.hsize1)
        self.w2 = np.random.rand(self.hsize1, self.hsize2)
        self.bias2 = np.random.rand(1, self.hsize2)
        self.w3 = np.random.rand(self.hsize2, 1)
        self.bias3 = 0.1
        
    def forward(self, x):
        self.z1 = np.matmul(x, self.w1)+self.bias1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.matmul(self.a1, self.w2)+self.bias2
        self.a2 = sigmoid(self.z2)
        
        self.z3 = np.matmul(self.a2, self.w3)+self.bias3
        self.output = sigmoid(self.z3)
        return self.output
    
    def backward(self, x, y, learning_rate):
        error = y-self.output
        d_output = error*derivative_sigmoid(self.output)
        
        error_h2 = np.matmul(d_output, self.w3.T)
        d_h2 = error_h2*derivative_sigmoid(self.a2)
        
        error_h1 = np.matmul(d_h2,self.w2.T)
        d_h1 = error_h1*derivative_sigmoid(self.a1)
        
        self.w3 += np.matmul(self.a2.T, d_output)*learning_rate
        self.bias3 += np.sum(d_output, axis=0, keepdims=True)*learning_rate
        
        self.w2 += np.matmul(self.a1.T, d_h2)*learning_rate
        self.bias2 += np.sum(d_h2, axis=0, keepdims=True)*learning_rate
        
        self.w1 += np.matmul(x.T, d_h1)*learning_rate
        self.bias1 += np.sum(d_h1, axis=0, keepdims=True)*learning_rate

    def train(self, x, y, epochs, learning_rate):
        e = []
        l = []
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, learning_rate)
            loss = np.mean(np.square(y-output))
            if epoch % 100 == 0:
                e.append(epoch)
                l.append(loss)
            if epoch % 5000 == 0:
                print(f'epoch {epoch} loss : {loss}')
        show_learning_curve(e, l)
                
    def predict(self, x):
        return self.forward(x)
    
if data == 0:
    x,y = generate_linear(n = 100)
else:
    x,y = generate_XOR_easy()


nn = NeuralNetwork(hsize1, hsize2)
nn.train(x, y, epochs, learning_rate)

if data == 0:
    x,y = generate_linear(n = 100)
else:
    x,y = generate_XOR_easy()

pred_y = nn.predict(x)
correct = 0
for i in range(len(x)):
    print(f'Iter{i} |  Ground truth: {y[i][0]} |  prediction: {pred_y[i][0]:.5f} |')
    if(y[i] == np.round(pred_y[i])):
        correct+=1

loss = np.mean(np.square(y-pred_y))
accuracy=correct/len(x)*100
print(f'loss={loss:.5f} accuracy={accuracy}%')
show_result(x,y,np.round(pred_y))
