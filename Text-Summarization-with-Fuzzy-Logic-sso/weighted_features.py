import os

folder = './2.feature/'
new_folder = './6.weighted_feature/'
file_names = []

weights_file = './weights.txt'
weights_float = []
weights_string = open(weights_file).read().split()
for weight in weights_string:
	weights_float.append(float(weight))

index = 0
for enum, file in enumerate(os.listdir(folder)):
	f = open(folder + file).read()
	file_names.append(file)
	oneD_list = f.split()
	twoD_list = [oneD_list[i:i+9] for i in range(0, len(oneD_list), 9)]
	for i in range(len(twoD_list)):
		for j in range(1, 9):
			twoD_list[i][j] = weights_float[j-1] * float(twoD_list[i][j])
			twoD_list[i][j] = str(twoD_list[i][j])
	s = ''
	for i in range(len(twoD_list)):
		s =  s + '      '.join(twoD_list[i])
		s = s + '\n'
	f = open( new_folder + file_names[index], "w+")
	f.write(s[:-1])
	f.close()
	index = index + 1

print(file_names)