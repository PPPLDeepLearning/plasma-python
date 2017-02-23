#!/usr/bin/env python
from functools import partial
from itertools import tee

class Loader():
    def example_batch_generator(self,n):
        for batch in range(n):
            yield batch

class MPIModel():
    def __init__(self,batch_generator):
        self.batch_iterator = batch_generator

    def train_epochs(self,M):
        num_total = 8
        for epoch in range(M):
            num_so_far = 0
            print ("Batch iter. summary: {}{}".format(self,self.batch_iterator))
            for batch in self.batch_iterator():
                num_so_far += 1

                whatever=batch
                print ("Next batch id: {}".format(batch))
                if num_so_far > num_total: break
            print "+++++++"


class MPIModel_default():
    def __init__(self,batch_generator):
        self.batch_iterator = batch_generator

    def train_epochs(self,M):
        num_total = 8 #number of samples per epoch
        batch_generator_func = self.batch_iterator()

        for iepoch in range(M):
            #print ("Batch iter. summary: {}{} epoch: {}".format(self,self.batch_iterator,iepoch))
            num_so_far = 0

            while num_so_far < num_total:
                num_so_far += 1

                try:
                    batch = batch_generator_func.next()
                except:
                    batch_generator_func = self.batch_iterator()
                    batch = batch_generator_func.next()
                print ("Next batch id: {}".format(batch))

            print "+++++++"



def main():
    num_batches = 10
    epochs = 3

    loader = Loader()
    batch_generator = partial(loader.example_batch_generator,n=num_batches)
    my_example_class = MPIModel(batch_generator)
    my_example_class.train_epochs(epochs)

def main_default():
    num_batches = 10
    epochs = 3

    loader = Loader()
    batch_generator = partial(loader.example_batch_generator,n=num_batches)
    my_example_class = MPIModel_default(batch_generator)
    my_example_class.train_epochs(epochs)

if __name__=='__main__':
    import timeit
    #print min(timeit.Timer(setup=main).repeat(7, 1000))
    print min(timeit.Timer(setup=main_default).repeat(7, 1000))
