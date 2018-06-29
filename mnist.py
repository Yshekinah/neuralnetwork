import numpy
import os
import scipy.special
import matplotlib.pyplot as plt
from PIL import Image

# erste Gehversuche mit neuronalen Netzwerken
class NeuralNetwork:

    # Initialisierung
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # die Gewichtungsmatrizen
        # wih = weight input hidden
        # who = weight hidden output
        # die Gewichte im Array ssind w_i_j, mit der Verbindung i zu j im nächsten Layer
        # w11, w21
        # w12, w22 etc
        # self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        #self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        # die gleiche Initialisierung der Gewichte, ABER normalverteilt
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Aktivierungsfunktion ist die sigmoid Funktion
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # das Netzwerk lernt
    def train(self, inputs_list, targets_list):

        # Die inputs_list in ein zweidimensionales Array umwandeln
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # berechne die Signale in den hidden Layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # berechne die Signale beim Austreten aus dem hidden Laywe
        hidden_outputs = self.activation_function(hidden_inputs)

        # berechne die Eingangssignale in den final Layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # berechne die Signale beim Austreten aus dem final Layer
        final_outputs = self.activation_function(final_inputs)

        ########################
        # Fehlerbackpropabation#
        ########################

        # Output Fehler = target - actual
        output_errors = targets - final_outputs

        # hidden layer error = output_errors durch Summe der Gewichte des hidden layer
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update der Gewichte für die Verbindungen zwischen hidden und Output Layer
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # Update der Gewichte für die Verbindungen zwischen input und hidden layer
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    # das Netz abfragen
    def query(self, inputs_list):

        # inputs in ein 2d Array transformieren
        inputs = numpy.array(inputs_list, ndmin=2).T

        # berechne Signale in den hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # berechne Signale aus dem hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # berechne Signale in den finale output layer
        final_outputs = numpy.dot(self.who, hidden_outputs)
        # berechne Signale aus dem finale output layer
        final_outputs = self.activation_function(final_outputs)

        return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#n.query([1.0, 0.5, -1.5])

# absolute dir the script is in
script_dir = os.path.dirname(__file__)

#training_data_file = open("D:/Stuff/Django_projects/NeuronalesNetzwerk/testdaten/mnist_train_100.csv", 'r')
training_data_file = open(os.path.join(script_dir,"testdaten/mnist_train_100.csv"), 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')
    # Eingabewerte skalieren und den ersten Wert (die gesuchte Zahl) rausnehmen
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99 ) + 0.01

    # erstelle den Zielzahlenstrahl mit allen Werten 0.01 bis auf den Zielwert mit 0.99
    targets = numpy.zeros(output_nodes) + 0.01
    # all_values[0] beinhaltet das Ziel
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)


#test_data_file = open("D:/Stuff/Django_projects/NeuronalesNetzwerk/testdaten/mnist_test_10.csv", 'r')
test_data_file = open(os.path.join(script_dir,"testdaten/mnist_test_10.csv"), 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

correct_answer = 0
wrong_answer = 0
testsample = len(test_data_list)

for line in test_data_list:
    all_values = line.split(',')
    print("Number in the picture is: ", all_values[0])

    result = n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    print("The network guesses: ", numpy.argmax(result))

    if(int(all_values[0]) == int(numpy.argmax((result)))):
        correct_answer += 1
    else:
        wrong_answer += 1

print(len(training_data_list), " training sets of data were used")
print(testsample, " tests have been made")
print("Percentage of correct guesses: ", (correct_answer * 1.0 / testsample * 1.0) * 100, ' %')