import os
for i in range(100):
   '''
   parallel experiments.
   '''
   cmp="python main.py --input1 data/mouse/mouse_pos.edgelist --input2 data/mouse/mouse_neg.edgelist --output embeddings/mouse --seed "+str(i)
   os.system(cmp)





