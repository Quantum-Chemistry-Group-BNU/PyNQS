#!/bin/bash
eFe2S2=-116.605609043
eH50=-26.92609
eH12=-5.669548398125276
eH36=-17.01897043
eH18=-8.50739691691
eN2=-108.971457
eref=$eN2
# eref=$eH36
# eref=$eFe2S2
# eref=$eH12

for file in $@;do
  if [ ! -f $file ]; then
     continue
  fi
  m=$(grep "<E>" $file -c)
  if [ -z $m ] || [ $m -eq 0 ]; then
    continue
  fi
  echo -n ${file}: " " ;
  grep "^Total energy" $file | tail -n 50 | awk -v E="$eref" '{sum+=$3} END {printf "%.4f" " " "'$m'" "-iters.\n", ((sum/NR - E) * 1000)}'
  # grep "^Total energy" $file | tail -n 50 | awk -v E="$eref" '{sum+=$3} END {printf "%.6f\n", ((sum/NR - E) * 1000)}'
  # grep "<E>" $file | tail -n 50 | awk '{print $3}' | sed ':a;N;$!ba;s/\n/, /g'
  # grep "reduce rate" $file |  awk '{sum+=$(NF-1)} END {printf "%.6f\n", ((sum/NR - E))}'
done
