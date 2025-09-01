@echo off
REM Batch script to convert PDB files to PQR format using pdb2pqr30
REM Processes prediction dataset

REM Process prediction dataset
echo Processing training dataset...
if not exist "./prediction_pqr" mkdir "./prediction_pqr"
for %%f in ("../../dataset/prediction/pdb/*.pdb") do (
    pdb2pqr30 --ff=CHARMM "%%f" "./prediction_pqr/%%~nf.pqr"
)

echo All PDB to PQR conversions completed.
pause
