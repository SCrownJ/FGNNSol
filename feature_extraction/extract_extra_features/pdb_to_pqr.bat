@echo off
REM Batch script to convert PDB files to PQR format using pdb2pqr30
REM Processes train, test, and validation datasets sequentially

REM Process training dataset
echo Processing training dataset...
if not exist "./train_pqr" mkdir "./train_pqr"
for %%f in ("../../dataset/train_data/pdbTrain/*.pdb") do (
    pdb2pqr30 --ff=CHARMM "%%f" "./train_pqr/%%~nf.pqr"
)

REM Process test dataset
echo Processing test dataset...
if not exist "./test_pqr" mkdir "./test_pqr"
for %%f in ("../../dataset/test_data/pdbTest/*.pdb") do (
    pdb2pqr30 --ff=CHARMM "%%f" "./test_pqr/%%~nf.pqr"
)

REM Process validation dataset
echo Processing validation dataset...
if not exist "./val_pqr" mkdir "./val_pqr"
for %%f in ("../../dataset/val1_scere_data/pdbVal1_scere/*.pdb") do (
    pdb2pqr30 --ff=CHARMM "%%f" "./val_pqr/%%~nf.pqr"
)

echo All PDB to PQR conversions completed.
pause
