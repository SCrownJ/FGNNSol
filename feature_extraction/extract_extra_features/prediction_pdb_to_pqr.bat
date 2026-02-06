@echo off
REM Batch script to convert PDB files to PQR format using pdb2pqr30
REM Processes prediction dataset

REM Set absolute path for prediction dataset
set "PREDICTION_PATH=D:\Programs\dev\workspace\github_code\FGNNSol-adj\dataset\prediction\pdb"

REM Process prediction dataset
echo Processing prediction dataset...
if not exist "./prediction_pqr" mkdir "./prediction_pqr"
for %%f in ("%PREDICTION_PATH%\*.pdb") do (
    pdb2pqr30 --ff=CHARMM "%%f" "./prediction_pqr/%%~nf.pqr"
)

echo All PDB to PQR conversions completed.
pause
