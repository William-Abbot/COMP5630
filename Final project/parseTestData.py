#parse test data
#white = 1
#black = 2
import numpy as np

data_dir = 'E:\\Documents\\School\\Comp 6630\\Final project\\testData'

counter = 0
total_counter = 0

dir_size = 53



def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif

for c in range(dir_size):
    data_file = open(data_dir + '\\test_data_' + str(counter) + '.txt')
    
    lines  = data_file.readlines()
    
    line_counter = 0
    
    current_array = np.array([[0]*5]*5)
    X_array = list()
    Y_array = list()
    
    for i in range(len(lines)):
        current_array = np.array([[0]*5]*5)
        #Y_array = [0]*25
        line = lines[i]
        if 'black' in line or 'white' in line:
            if 'wins' not in line:
                for j in range(12):
                    line = lines[i+j]
                    for k in range(len(line)):
                        if line[k] == '@':
                            current_array[int((k-5)/4)][int((j-3)/2)] = 2
                        if line[k] == 'O':
                            current_array[int((k-5)/4)][int((j-3)/2)] = 1
                if line_counter%2 == 0:
                    X_array.append((np.transpose(current_array).flatten()).tolist())
                    past_array = (np.transpose(current_array).flatten()).tolist()
                else:
                    #position = ((int((k-5)/4)+1)+(int((j-3)/2)+1)*5)-1
                    Y_array.append((np.transpose(current_array).flatten()).tolist())
                
                line_counter += 1
        total_counter += 1
        line = lines[i] 


    data_file.close()
    
    save_file = open(data_dir + '\\parsed_' + str(counter) + '.txt', 'w')
    
    for z in range(len(X_array)):
        if z == len(X_array)-1:
            save_file.write(str(X_array[z]))
        else:
            save_file.write(str(X_array[z])+',\n')

    save_file.write(' \n\n')
    
    for q in range(len(Y_array)):
        if q == len(X_array)-1:
            save_file.write(str(Y_array[q]))
        else:
            save_file.write(str(Y_array[q])+',\n')
    
    
    save_file.close()
    
    counter += 1
    total_counter += 1

print(total_counter)

'''
read each stone placemet string and add it to the array
at the proper location. after that 
'''