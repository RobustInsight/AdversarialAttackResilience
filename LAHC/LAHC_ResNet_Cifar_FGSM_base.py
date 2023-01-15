# -*- coding: utf-8 -*-
from __future__ import print_function
import lahc
import math
import random
import torch

from simanneal import Annealer
import random
import torch.nn as nn
import time
from Utils.Tools import read_data_from_file, string_to_number

from Networks import AF_Tools
from Networks.GetData import *
import os.path
import torch.optim as optim

from Utils.Tools import write_data_in_file

from Networks.ResNet18_Multiple import resnet18

from torchattacks import FGSM, PGD



Activation_Functions = AF_Tools.Activation_Functions_
operators = AF_Tools.operators_
operators_p = AF_Tools.operators_p_

EPOCHS = 15

LAYERS = 4
source_file = '../Results/results_ResNet_Cifar_FGSM_total.csv'
destination_path = '../save_lahc/res18_cifar/res18_cifar_'
result_file_name ='results_lahc_ResNet_Cifar_FGSM_base.csv'
trace = False

def find_best_state(eps):
    pd_data = read_data_from_file(source_file)
    pd_data = pd_data.drop_duplicates()

    best_chromosome_eps = pd_data.loc[pd_data['eps'] == (eps)]
    top_record_per_eps = best_chromosome_eps.nlargest(1, 'rob_acc')
    ch = top_record_per_eps.iloc[0]['chromosome']
    ch = ch.split(',')

    return string_to_number(ch)



def train_model(model, data_loaders, dataset_sizes, criterion, optimizer, device):
    model = model.to(device)
    model.train()  # set train mode


    # for each epoch
    for epoch in range(EPOCHS):

        epoch_time = time.time()
        running_loss, running_corrects, total_checked = 0.0, 0, 0
        error_found = False
        # for each batch
        for indx, (inputs, labels) in enumerate(data_loaders['train']):
            #btch_time = time.time()
            # print('indx', indx, len(inputs))

            try:
                inputs, labels = inputs.to(device), labels.to(device)

                # making sure all the gradients of parameter tensors are zero
                optimizer.zero_grad()  # set gradient as 0

                # get the model output
                outputs = model(inputs)



                # get the prediction of model
                _, preds = torch.max(outputs, 1)

                # calculate loss of the output
                loss = criterion(outputs, labels)

                # backpropagation
                loss.backward()

                # update model parameters using optimzier
                optimizer.step()

                batch_loss_total = loss.item() * inputs.size(0)  # total loss of the batch
                running_loss += batch_loss_total  # cumluative sum of loss
                running_corrects += torch.sum(preds == labels.data)  # cumulative sum of correct count
                total_checked += inputs.size(0)

            except Exception as e:
                error_found = True
                print(' ****************************                  error      ****************************')
                print(str(e))
                break
            '''
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, indx * len(inputs), len(data_loaders['train'].dataset),
                       100. * indx / len(data_loaders['train']), loss.item()))
            '''

            #print(indx, 'btch time :', time.time()-btch_time)


        #print('----------------------------------------------epo time :', time.time() - epoch_time)
        if error_found:
            return -1
        # calculating the loss and accuracy for the epoch
        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() * 100 / total_checked
        print('correct   :  ', running_corrects, '         dataset size  :   ', dataset_sizes['train'])
        # if trace:
        print('Epoch {}/{} -  Train Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, EPOCHS, epoch_loss, epoch_acc))
        print('time : ', time.time()-epoch_time )

    return 0


def test_model(model, data_loaders, criterion, device, dataset_sizes):
    # after training epochs, test epoch starts
    model.eval()  # set test mode
    running_loss, running_corrects, total_checked = 0.0, 0, 0

    # for each batch
    for indx, (inputs, labels) in enumerate(data_loaders['test']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # same with the training part.
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)  # cumulative sum of loss
        running_corrects += torch.sum(preds == labels.data)  # cumulative sum of corrects count
        total_checked += inputs.size(0)


    # calculating the loss and accuracy
    test_loss = running_loss / dataset_sizes['test']
    test_acc = running_corrects.double() * 100 / total_checked

    return test_loss, test_acc.item()
    # print('<Test Loss: {:.4f} Acc: {:.4f}>'.format(test_loss, test_acc))



def get_attacks():
    return {'FGSM': FGSM,
            'PGD': PGD, }


def atk_by_name(atkname):
    attaks = get_attacks()
    return attaks[atkname]



class RobustProblem(lahc.LateAcceptanceHillClimber):
    """Defines the Rosenbrock problem.

        f(x, y) = (a-x)**2 + b(y-x**2)**2

    Global minima is located at (x, y) = (a, a**2) for b > 0
    """
    def __init__(self, initial_state, data_loaders, dataset_sizes, atk_):
        self.dataloaders, self.dataset_sizes, (self.atk, self.eps) = data_loaders, dataset_sizes, atk_
        self.p1 = range(0, LAYERS)
        self.p2 = range(LAYERS, LAYERS * 2)
        self.p3 = range(LAYERS * 2, LAYERS * 3)

        super(RobustProblem, self).__init__(initial_state=initial_state)


    def move(self):
        moves_count = 2
        for i in range(moves_count):
            loc = random.randint(0, len(self.state) - 1)
            if loc in self.p1:  # the first part, which is for the operator
                self.state[loc] = random.choices(operators, weights=operators_p)[0]
                if self.state[loc] < 2:  # if operator is a.f(x) then remove the second activation function
                    self.state[loc + LAYERS*2] = 0

            elif loc in self.p2:  # first activation function
                self.state[loc] = random.randint(0, len(Activation_Functions) - 1)
            elif loc in self.p3:  # second activation function
                self.state[loc] = random.randint(0, len(Activation_Functions) - 1)
                if self.state[loc - LAYERS*2] < 2:  # if operator is a.f(x) there is no need to change the chromosome
                    self.state[loc - LAYERS*2] = random.randint(2, 4)
            '''
            for i in range(0, LAYERS):
                if self.state[i] == 2 and self.state[i+LAYERS] == 0 and self.state[i+LAYERS*2] == 2:
                    self.state[i + LAYERS] = 1
            '''

        print(self.state)




        # return self.energy() #- initial_energy

    def energy(self):
        total_time = time.time()
        """Calculates the length of the route."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'

        if trace:
            print('Individual    : ', self.state)

        model = resnet18(num_classes=10, states_=self.state).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        fname = destination_path + '_'.join(map(str, self.state)) + '.pth'
        train_time = time.time()
        if os.path.isfile(fname):
            model.load_state_dict(torch.load(fname))
        else:
            res = train_model(model, self.dataloaders, self.dataset_sizes, criterion, optimizer, device)
            if res < 0:
                return 100
            torch.save(model.state_dict(), fname)

        train_time = time.time() - train_time

        test_time = time.time()
        test_loss, test_acc = test_model(model, self.dataloaders, criterion, device, self.dataset_sizes)

        test_time = time.time() - test_time

        correct = 0
        total = 0

        eval_time = time.time()
        model.eval()

        attack = atk_by_name(self.atk)

        a_t_k = attack(model, eps=self.eps)

        for indx, (images, labels) in enumerate(self.dataloaders['test']):
            perturb_images = a_t_k(images, labels).to(device)
            # images = images.to(device)
            outputs = model(perturb_images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()


        eval_time = time.time() - eval_time
        total_time = time.time() - total_time
        # print(total,correct )
        if trace:
            print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
            print('_' * 20)

        rob_acc = 100 * float(correct) / total
        write_data_in_file(result_file_name, (self.state, self.atk, self.eps, rob_acc, test_acc))
        # print(100 * float(correct) / total)
        print(self.state, self.atk, self.eps, test_acc, rob_acc)
        print('total time : ', total_time, ' train time : ', train_time, ' eval time : ', eval_time, 'test time :', test_time)

        return 100 * (1 - rob_acc)


if __name__ == '__main__':
    # initial state, a randomly-ordered itinerary
    eps = 8/255


    torch.autograd.set_detect_anomaly(True)
    data_loaders_, dataset_sizes_ = get_cifar10_data_227(200, 100)

    #initial_state = find_best_state(eps)
    initial_state = [3, 0.25, 1, 0.25, 2, 0, 4, 4, 2, 0, 0, 0]
    print('initial state : ', initial_state)

    #initial_state = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 79.61


    atk_eps_s = ("FGSM", eps)
    # adjust the history length and update interval
    lahc.LateAcceptanceHillClimber.steps_minimum = 100
    lahc.LateAcceptanceHillClimber.history_length = 50
    lahc.LateAcceptanceHillClimber.updates_every = 1

    # initializing the problem
    prob = RobustProblem(initial_state, data_loaders_, dataset_sizes_, atk_eps_s)
    prob.copy_strategy = 'slice'
    # and run the Late Acceptance Hill Climber for a solution
    prob.run()

    # compare found solution to exact solution
    print()
    print(prob.state, prob.exact_solution)
    print(
        "Solution with `slice` strategy in: {0:.1f}s".format(
            prob.time_end - prob.time_start))
    print()
