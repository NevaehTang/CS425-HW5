import sys
import pickle

example_dict = {1:"6",2:"2",3:"f"}

program_name = sys.argv[0]
command = sys.argv[1]

pickleout = open("qtable.pickle","wb")
pickle.dump(example_dict, pickleout)
pickleout.close()

picklein = open("qtable.pickle", "rb")
now = pickle.load(picklein)

print now

# if command == "train":
