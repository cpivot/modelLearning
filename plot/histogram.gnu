set terminal x11 size 1000,400
set multiplot layout 1,2

set logscale y
plot "../bin/prog/save.dat" u 1:2 w d


unset logscale y
set logscale x
set format x "%g"
set boxwidth 0.8 relative
set style fill transparent solid 0.5 noborder
plot "../bin/prog/histo.dat" u 1:2 with boxes

unset multiplot

pause -1
