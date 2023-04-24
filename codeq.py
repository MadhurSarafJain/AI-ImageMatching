from typing import List
import numpy as np
import matplotlib as mplib
from matplotlib.image import imread
from PIL import Image
from itertools import islice
from numpy.random.mtrand import random
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


image_array_1 = imread('groupGray.jpg')
rows_1,columns_1=image_array_1.shape



# print("Number of rows in big Picture are ", rows_1 )
# print("Number of columns in big Picture are ", columns_1 )
# print(len(image_array_1))

image_array_2 = imread('boothiGray.jpg')
rows_2,columns_2=image_array_2.shape

# print("Number of rows in Small Picture are ", rows_2 )
# print("Number of columns in Small Picture are ", columns_2 )
# print(image_array_2)

def initilizePop(Rows,Columns,Size):
    Row_List=np.random.randint(Rows,size=(Size))
    Col_List=np.random.randint(Columns,size=(Size)) 
    pop= [(Row_List[i],Col_List[i]) for i in range (0,len(Row_List))]
    return(pop)



Row_List=list(np.random.randint(rows_1,size=(100)))
Col_List=list(np.random.randint(columns_1,size=(100)))

# Find corelation value of random points
def FitnessEvaluation(image_array_1,image_array_2,pop):
    fitness_val = []
    for i in range (len(pop)):
        x=pop[i][0]
        y=pop[i][1]
        if x+29<1024 and y+35<512:
    
    
            num=image_array_1[pop[i][1]:pop[i][1]+35,pop[i][0]:pop[i][0]+29]
            
            co_relation=np.mean((num-num.mean())*(image_array_2-image_array_2.mean()))/(num.std()*image_array_2.std())
            
            fitness_val.append(round(co_relation,2))
        else:
            fitness_val.append(round(-1,2))
        
    return fitness_val

# Sort Ranked_pop
def Selection(list1,list2):
    Selected_List=[]
    pairs=[]
    pairs = sorted(zip(list1, list2),reverse=True)
    for i in pairs:
        Selected_List.append(i[0])
    return (Selected_List)


#CrossOver Process
def CrossOver(Ranked_Pop):

    # Ranked pop without corelation
    Binary=[]
    for i in range(0,len(Ranked_Pop)):
        # print(Ranked_Pop,"aaa")
        x=Ranked_Pop[i][0]
        x=np.binary_repr(x,10)
        y=Ranked_Pop[i][1]
        y=np.binary_repr(y,10)
        Binary.append((x,y))
    
    New_Gen=[]
    for i in range(0,len(Binary),2):
        a_x=list(Binary[i][0])
        a_y=list(Binary[i][1])

        b_x=list(Binary[i+1][0])
        b_y=list(Binary[i+1][1])
        a=[]
        b=[]
        a.extend(a_x)
        a.extend(a_y)
        b.extend(b_x)
        b.extend(b_y)
        p=np.random.randint(1,len(a)-1)
        for i in range(p,len(a)):
            a[i],b[i]=b[i],a[i]

        #Swapping Process
        a,b = ''.join(a),''.join(b)
        
        New_Gen.append((int(a[0:10],2),int(a[10:],2)))
        New_Gen.append((int(b[0:10],2),int(b[10:],2)))

    return New_Gen

def Mutation(Evolved_1):
    
    Binary_Gen_2=[]
    Mutated_Gen=[]

    for i in range(0,len(Evolved_1)):
        
        x=Evolved_1[i][0]
        x=np.binary_repr(x,10)
        y=Evolved_1[i][1]
        y=np.binary_repr(y,10)
        Binary_Gen_2.append((x,y))
       
    # covert touples into list
      
    List_Gen_1=[i for t in Binary_Gen_2 for i in t]
    
    for i in range(0,len(List_Gen_1)):
        a=list(List_Gen_1[i])
    
        p=np.random.randint(1,len(a)-1)
        One='1'
        Zero='0'
        for j in range(0,len(a)):
            if (j==p):
                if (a[j]==One):
                    a[j]=Zero
                    a=''.join(a)
                    Mutated_Gen.append((int(a[0:10],2)))
                    
                else:
                    a[j]=One
                    a=''.join(a)
                    Mutated_Gen.append((int(a[0:10],2)))
    Mutated_Gen=[(Mutated_Gen[i],Mutated_Gen[i+1]) for i in range(0,len(Mutated_Gen),2)]
    return Mutated_Gen



            
pop=initilizePop(1024-29,512-35,100)
print("Population is", pop)

fitness=FitnessEvaluation(image_array_1,image_array_2,pop)
print("Fitness of each random is",fitness)

Ranked_Pop=Selection(pop,fitness)

print("Ranked Pop is",Ranked_Pop)

Gen_1=CrossOver(Ranked_Pop)
print("CrossOver Evolved is: ",Gen_1)
Gen_2=Mutation(Gen_1)
print("Mutation Evolved is: ", Gen_2)

fitness_2=FitnessEvaluation(image_array_1,image_array_2,Gen_2)
print("Fitness of mutated random is", fitness_2)

def hillClimbing(image_array_1, image_array_2, pop, stopping_iter=1000):
    # Evaluate the fitness of the initial population
    fitness = FitnessEvaluation(image_array_1, image_array_2, pop)
    best_fitness = max(fitness)
    best_solution = pop[fitness.index(best_fitness)]
    
    # Initialize the iteration counter
    iteration = 0
    
    while iteration < stopping_iter:
        # Select a random individual from the population
        idx = np.random.randint(len(pop))
        current_solution = pop[idx]
        current_fitness = fitness[idx]
        
        # Generate a new candidate solution by randomly perturbing the current solution
        candidate = (np.random.randint(current_solution[0]-10, current_solution[0]+10), 
                     np.random.randint(current_solution[1]-10, current_solution[1]+10))
        
        # Evaluate the fitness of the candidate solution
        candidate_fitness = FitnessEvaluation(image_array_1, image_array_2, [candidate])[0]
        
        # Check if the candidate solution is better than the current solution
        if candidate_fitness > current_fitness:
            pop[idx] = candidate
            fitness[idx] = candidate_fitness
            
            if candidate_fitness > best_fitness:
                best_fitness = candidate_fitness
                best_solution = candidate
                
        # Update the iteration counter
        iteration += 1
    
    return best_solution, best_fitness


def simulatedAnnealing(image_array_1, image_array_2, pop, T=1.0, alpha=0.95, stopping_T=1e-4, stopping_iter=1000):

    # Evaluate the fitness of the initial population
    fitness = FitnessEvaluation(image_array_1, image_array_2, pop)
    best_fitness = max(fitness)
    best_solution = pop[fitness.index(best_fitness)]

    # Initialize the temperature and the iteration counter
    iteration = 0

    while T > stopping_T and iteration < stopping_iter:

        # Select a random individual from the population
        idx = np.random.randint(len(pop))
        current_solution = pop[idx]
        current_fitness = fitness[idx]

        # Generate a new candidate solution by randomly perturbing the current solution
        candidate = (np.random.randint(current_solution[0]-10, current_solution[0]+10), 
                     np.random.randint(current_solution[1]-10, current_solution[1]+10))

        # Evaluate the fitness of the candidate solution
        candidate_fitness = FitnessEvaluation(image_array_1, image_array_2, [candidate])[0]

        # Compute the change in fitness and decide whether to accept or reject the candidate solution
        delta_fitness = candidate_fitness - current_fitness
        if delta_fitness > 0 or np.exp(delta_fitness / T) > np.random.rand():
            pop[idx] = candidate
            fitness[idx] = candidate_fitness

            if candidate_fitness > best_fitness:
                best_fitness = candidate_fitness
                best_solution = candidate

        # Update the temperature and the iteration counter
        T *= alpha
        iteration += 1

    return best_solution, best_fitness

temp = True
mean=[]
max=[]
for i in range(5000):
    for j in range(0,len(fitness_2)):

        if (fitness_2[j]>0.85):

            print("Person recognized at" , i,"th" , "generation")
            temp = j

            # Stopping Criteria
            break
    if temp != True:
        break


    Ranked_Gen_2=Selection(Gen_2,fitness_2)
    
    mean.append(sum(fitness_2)/len(fitness_2))
    c=sorted(fitness_2, reverse=True)
    max.append(c[0])



    Gen_1_1=CrossOver(Ranked_Gen_2)

    Gen_2=Mutation(Gen_1_1)

    fitness_2=FitnessEvaluation(image_array_1,image_array_2,Gen_2)


if temp != True:
    x=Gen_2[temp][0]     
    y=Gen_2[temp][1] 

    
print(c, "loop")
plt.plot(max)
plt.plot(mean)
plt.show()

plt.imshow(Image.open("groupGray.jpg"))
plt.gca().add_patch(Rectangle((x,y),29,35,linewidth=1,edgecolor='r',facecolor='none'))
plt.show()
