./main -camPos 0,22,150 -focal 50 -vdbIn /path/to/vdb/dir/exploSim.vdb -exrOut /path/to/render/dir/exploSim.exr -light dir:45:1,0.89,0.79:0.7:-48.80,-311.86,0.0 -light env:0:1,1,1:0.35:3:/path/to/env/map/dir/ibl.exr -ptsPerKernel 2 -scatterScale 3.0 -scatterDensityScale 1.0 -scatterDensityMin 0.025 -densityScale 0.5 -iter 500 -cutoff 0.0025 -primaryStep 1 -shadowStep 2 -emissionScale 3 -scatterDensityMaskPower 1.0 -volumeColor 0.1 -tempColorRamp 0.0,0.33,0.66,1.0/0,0,0:1.0,0.076,0.0:1.0,0.322,0.0:1.0,0.45,0.15/0.35/1.5 -scattering 1.0,1.25,1.5 -samples 3;
