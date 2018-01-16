set -x
for((i=0; i<31; i++))
do
    python3 -m  doctools.scripts.cost -c doctools/configs/Hindi.json -l hi -b $i -o doctools/outdir
done

