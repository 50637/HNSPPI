HNSPPI: A Hybrid Computational Model Combing Network and Sequence Information for Predicting Protein-Protein Interaction
HNSPPI is a novel model for PPI predictions which comprehensively characterizes the intrinsic relationship between two proteins by integrating protein sequence and PPI network connection properties.

Scripts:
The scripts require PPI in edgelist Format.
main.py - This script is the entry for the program to run.
parallel.py - This script is used to perform parallel experiments.
utils.py - This scripts are used to prepare features of proteins.
evaluation.py script is used to evalutate the performance of our method.

Running:
Download the data file from http://cdsic.njau.edu.cn/data/PPIDataBankV1.0
run main.py script with --input1 <positive edgelist> --input2 <negative edgelist> --output <output file> --seed <seed>
for example:python main.py --input1 data/mouse/mouse_pos.edgelist --input2 data/mouse/mouse_neg.edgelist --output embeddings/mouse --seed 0
See the results in results.tsv file

Datasets
This dataset includes sequence information and interaction information of 6 species.
## Sharing/access Information
Links to other publicly accessible locations of the data:
PMID: 25657331
PMID: 11196647
PMID: 19171120
PMID: 34536380
http://www.csbio.sjtu.edu.cn/bioinf/LR_PPI/Dara.htm

