# Method
You need to follow the steps below to extract features:
1. Prepare the raw data (.csv file), protein sequence file (.fasta file), and predicted protein spatial structure file (.pdb file)
2. Extract features using ESM-C and DSSP models.
3. Extract global physicochemical property characteristics of proteins.
4. Execute code to extract full features.

It should be noted that the edge attribute features and some node features related to the protein graph are not extracted here, but rather during the training process. You only need to run `../train.py` or `../predict.py` as described in the project root directory to extract features while training the model or making predictions.


# Data preparation
Regarding the data we used, it has been stored in `../dataset`, so you can proceed directly to the next step. Below, we only describe the processing method for feature extraction using other data.
In the `../dataset` folder, train_data stores training set data, eval_data stores validation set data, test_data stores test set data, and val1_scere_data stores external test set data (i.e., the Saccharomyces cerevisiae dataset).

1. Place your raw protein data tables in `../dataset/csvFile/predict.csv`. The tables must include at least a "name" column (containing Uniprot IDs or corresponding gene names) and a "sequence" column (recording protein sequences). If you intend to train with your own data, they should also contain a "solubility" column to record solubility labels.
2. Convert the data in the table to .fasta files. We provide a conversion script in `./csv_to_fasta.py`. The example input path is `../dataset/csvFile/predict.csv`, and the example output path is `../dataset/prediction/fasta`.
3. Use AlphaFold3 to input protein sequences and obtain protein spatial structure data. We obtained .pdb files by accessing the website https://alphafoldserver.com/. Alternatively, you can deploy the model on a server to get more accurate predictions. If your task is prediction, it is recommended to place the obtained .pdb files in `../dataset/prediction/pdb`.

Additionally, it is important to ensure that the values in the "name" column of the .csv file match the filenames of the corresponding .fasta and .pdb files (excluding their extensions).


# Extract ESM-C and DSSP features
You can download the model code and weights of ESM-C according to the description in `../README.md`, and the code can be stored in `./ESMC`.  `/extract_ESMC.py` is a sample code we provided for extracting ESMC features. You can adjust the input and output according to your specific situation to correctly extract the features.

Within the current directory, we provide the mkdssp.exe tool, which can only be executed on Linux systems. This tool enables the generation of .dssp files. Additionally, we offer the script `./extract_DSSP.py` as an example for extracting features from the .dssp files. Users may modify the input and output parameters to accommodate their specific requirements.

We also provide pre-extracted ESMC and DSSP feature files. If you encounter difficulties during feature extraction or wish to verify whether your extracted features match ours, you can access them via the link: https://drive.google.com/drive/folders/1g-rAdBmCbskx37Q9PMY1DoveopYo8UKp?usp=sharing.

# Extract physicochemical characteristics of proteins.
To extract the physicochemical features of proteins, please follow these steps:
1. Navigate to the "extract_extra_features" directory and run "prediction_pdb_to_pqr.bat" (for user-prepared datasets) or "pdb_to_pqr.bat" (employed in our work for processing all training and testing datasets) to convert .pdb data to .pqr format for subsequent processing.

If you wish to change the paths, you can check the `.bat` file, `add_feature1.py`, `construction_features.py`, and `csv_to_npy.py` files to make modifications. Note that the input file paths in the `.bat` file must be absolute paths. To run the `.bat` file, simply double-click it. When running extract_extra_feat.sh, first activate the virtual environment, then enter the `extract_extra_features` directory, and execute the command 
```shell
./extract_extra_feat.sh
````

2. Open the ChimeraX tool (download and installation instructions are detailed in `../README.md`). In the command line, navigate to the current directory (i.e., `/feature_extraction/extract_extra_features`), execute the command
```shell
runscript pqr_chimera.py
```
or
```shell
runscript prediction_pqr_chimera.py
```
 And then enter the registration command to initiate the process.
```shell
pqr_processor
```

3. Run the script "extract_extra_feat.sh" or "prediction_extract_extra_feat.sh" to obtain .npy files that store the physicochemical features of proteins.

We also provide pre-extracted extra_feat feature files. If you encounter difficulties during feature extraction or wish to verify whether your extracted features match ours, you can access them via the link provided in the ../README.md file.

# Extract full features
Return to the current directory. For our dataset, run the script `./extract_all.sh`. For the dataset you intend to use for prediction, run the script `./extract_prediction_features.py`. You will obtain .pkl files in the corresponding directory; these files contain all the features we need to extract and can be directly used for model training.
