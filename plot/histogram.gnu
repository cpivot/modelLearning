set terminal x11 size 1000,800
set multiplot layout 2,3

set logscale y
plot "../bin/prog/rosslerModel.dat" u 1:2 w d

unset logscale y
plot "../bin/prog/rosslerModel.dat" u 1:3 w d

set logscale x
set format x "%g"
set boxwidth 0.8 relative
set style fill transparent solid 0.5 noborder
plot "../bin/prog/histogram.dat" u 1:2 with boxes

unset logscale x
splot "../bin/prog/representation.dat" u 1:2:4:4 w d lc palette

set logscale z
splot "../bin/prog/representation.dat" u 1:2:5:5 w d lc palette

unset multiplot

pause -1
