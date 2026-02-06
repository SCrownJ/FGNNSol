@echo off
REM Batch script to convert PDB files to PQR format using pdb2pqr30

set "TRAIN_PATH=D:\Programs\dev\workspace\github_code\FGNNSol-adj\dataset\train_data\pdbTrain"
set "TEST_PATH=D:\Programs\dev\workspace\github_code\FGNNSol-adj\dataset\test_data\pdbTest"
set "EVAL_PATH=D:\Programs\dev\workspace\github_code\FGNNSol-adj\dataset\eval_data\pdbEval"
set "VAL_PATH=D:\Programs\dev\workspace\github_code\FGNNSol-adj\dataset\val1_scere_data\pdbVal1_scere"

REM Process training dataset
echo Processing training dataset...
if not exist "./train_pqr" mkdir "./train_pqr"
for %%f in ("%TRAIN_PATH%\*.pdb") do (
    pdb2pqr30 --ff=CHARMM "%%f" "./train_pqr/%%~nf.pqr"
)

REM Process test dataset
echo Processing test dataset...
if not exist "./test_pqr" mkdir "./test_pqr"
for %%f in ("%TEST_PATH%\*.pdb") do (
    pdb2pqr30 --ff=CHARMM "%%f" "./test_pqr/%%~nf.pqr"
)

REM Process evaluation dataset
echo Processing evaluation dataset...
if not exist "./eval_pqr" mkdir "./eval_pqr"
for %%f in ("%EVAL_PATH%\*.pdb") do (
    pdb2pqr30 --ff=CHARMM "%%f" "./eval_pqr/%%~nf.pqr"
)

REM Process validation dataset
echo Processing validation dataset...
if not exist "./val_pqr" mkdir "./val_pqr"
for %%f in ("%VAL_PATH%\*.pdb") do (
    pdb2pqr30 --ff=CHARMM "%%f" "./val_pqr/%%~nf.pqr"
)

echo All PDB to PQR conversions completed.
pause
