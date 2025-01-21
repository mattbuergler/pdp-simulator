path=$1
echo $path
#check if run exists
cp -r $path/input $path/run
vel=$(sed -n '/mean_velocity/p' $path/run/config.json)
vel=$(cut -d "[" -f2- <<< "$vel")
vel=$(cut -d "," -f1 <<< "$vel")
python sbg.py -r all -tsa $path/run > $path/run/run.out
python mssrc.py -vel $vel $path/run >> $path/run/run.out
python evaluate.py $path/run >> $path/run/run.out