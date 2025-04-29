set terminal pngcairo size 900,400 enhanced font 'Arial,10'
set output output_file
set grid
set key top left

plot data_file using 1:2 with lines title graph_title
